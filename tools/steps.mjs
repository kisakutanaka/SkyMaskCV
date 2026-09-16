// segmentSky の処理を1段ずつ取り出して書き出す。理解のための可視化が目的。
//
// skyMask.ts には手を入れず、公開されている関数を同じ順で呼び直す。
// ただし「参照実装を別に持つと実装ドリフトが起きる」ので、最後に segmentSky の
// 出力と1バイト単位で一致することを必ず検証する（一致しなければ異常終了）。
//
// 使い方: node tools/steps.mjs <prep.py の出力先> <id> <出力先> [見出し]
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const here = fileURLToPath(new URL('.', import.meta.url));
const S = await import(`${here}../dist/skyMask.js`);

const [dir, id, out, label = id] = process.argv.slice(2);
mkdirSync(out, { recursive: true });

const manifest = JSON.parse(readFileSync(`${dir}/manifest.json`, 'utf8'));
const { w: fw, h: fh } = manifest.find((m) => m.id === id);
const buf = readFileSync(`${dir}/${id}.bin`);
const fine = { data: new Uint8ClampedArray(buf.buffer, buf.byteOffset, buf.length), width: fw, height: fh };

const opts = S.DEFAULT_OPTIONS;
const steps = [];
/** 1段を書き出す。kind: 'field'（連続値）| 'mask'（0/1）。scale は粗解像度なら 2。 */
function dump(key, label, note, kind, data, w, h, scale, extra = {}) {
  const name = `${String(steps.length).padStart(2, '0')}_${key}`;
  writeFileSync(`${out}/${name}.bin`, Buffer.from(
    kind === 'field' ? new Float32Array(data).buffer : Uint8Array.from(data),
  ));
  steps.push({ name, label, note, kind, w, h, scale, ...extra });
}

// ---- 粗パス ----
const coarse = S.downsample2x(fine);
const w = coarse.width, h = coarse.height;

dump('src', '元画像（処理解像度）', `${fw}×${fh}。画素数だけを 180×320 相当に固定し、縦横比は入力のまま。`,
     'rgba', fine.data, fw, fh, 1);

const blue = S.blueDominance(coarse);
const gray = S.toGray(coarse);
const tex = S.localStdDev(gray, w, h, opts.texRadius);
const rel = new Float32Array(tex.length);
for (let i = 0; i < tex.length; i++) rel[i] = tex[i] / (gray[i] + 1);

dump('blue', '手がかり1: 青優勢度 (B-R)/(B+R)', '比なので照明の強さに影響されない。薄暮では空と地上がここで分かれ、曇天では両者ほぼ同じ値になり「黙る」。',
     'field', blue, w, h, 2, { lo: -0.5, hi: 0.5 });
dump('gray', '手がかり2: 輝度', '曇天ではここだけが効く（空165 / ビル115）。薄暮では空も地上も約73で使えない。',
     'field', gray, w, h, 2, { lo: 0, hi: 255 });
dump('rel', '手がかり3: 相対σ（局所σ / 輝度）', `窓は半径${opts.texRadius}。輝度で割るのは、ざらつきの振幅が明るさに比例するから。絶対値で切ると明るい空ほど不利になる。しきい値は ${opts.texMaxRel}。`,
     'field', rel, w, h, 2, { lo: 0, hi: 0.09 });

const band = new Uint8Array(w * h);
band.fill(1, 0, Math.max(1, Math.round(h * opts.seedRowRatio)) * w);
const seedPixels = new Uint8Array(w * h);
for (let i = 0; i < band.length; i++) if (band[i] && rel[i] < opts.texMaxRel) seedPixels[i] = 1;
const seed = S.skySeedIn(blue, gray, tex, band, [], opts.minSeedCount, opts);
if (seed === null) { console.error('シードが取れなかった'); process.exit(1); }

dump('seed', `シード画素（上端${opts.seedRowRatio * 100}% のうち滑らかなもの）`,
     `ここから「この空の見え方」を中央値で推定する → 青優勢度 ${seed.blue.toFixed(3)} / 輝度 ${seed.gray.toFixed(1)}。平均でなく中央値なのは街灯の光芒が上端行に混ざるため。`,
     'mask', seedPixels, w, h, 2);

const walls = S.straightEdges(gray, w, h, opts);
dump('walls', '人工物の直線', `勾配 ${opts.edgeMin} 超を2値化し、縦横それぞれ半径${opts.edgeLen}のオープニングで長い直線だけ残す。検出された直線の85.8%が建物の上にあり、雲の縁は3.2%。`,
     'mask', walls, w, h, 2);

const seeds = [seed];
const build = () => {
  let m = S.classifyBySeed(blue, gray, tex, seeds[0], opts);
  for (let k = 1; k < seeds.length; k++) {
    const part = S.classifyBySeed(blue, gray, tex, seeds[k], opts);
    for (let i = 0; i < m.length; i++) if (part[i]) m[i] = 1;
  }
  m = S.morphSquare(m, w, h, opts.openRadius, 'erode');
  m = S.morphSquare(m, w, h, opts.openRadius, 'dilate');
  m = S.morphSquare(m, w, h, opts.closeRadius, 'dilate');
  m = S.morphSquare(m, w, h, opts.closeRadius, 'erode');
  m = S.keepTopComponent(m, w, h);
  m = S.fillHoles(m, w, h);
  for (let i = 0; i < m.length; i++) if (walls[i]) m[i] = 0;
  return m;
};

// 1周目だけは build の中身を開いて1段ずつ見せる
let m = S.classifyBySeed(blue, gray, tex, seeds[0], opts);
dump('classify', '判定: シードからの距離', `|青優勢度−${seed.blue.toFixed(2)}|<${opts.bdTol} かつ |輝度−${seed.gray.toFixed(0)}|<${opts.vTol} かつ 相対σ<${opts.texMaxRel}。3つを AND で繋ぐと、差のない手がかりは素通りして黙る＝どれが効くかをシーンが自分で決める。`,
     'mask', m, w, h, 2);

m = S.morphSquare(m, w, h, opts.openRadius, 'erode');
m = S.morphSquare(m, w, h, opts.openRadius, 'dilate');
dump('open', `オープニング（半径${opts.openRadius}）`, '孤立した誤検出を削る。先にオープニングをかけるのが必須で、逆順だと膨張でノイズが空本体と地続きになり落とせなくなる。',
     'mask', m, w, h, 2, { diff: true });

m = S.morphSquare(m, w, h, opts.closeRadius, 'dilate');
m = S.morphSquare(m, w, h, opts.closeRadius, 'erode');
dump('close', `クロージング（半径${opts.closeRadius}）`, '空に入った細い切れ込みを橋渡しする。ここを bandRadius より大きくすると境界がブロック状に残る。',
     'mask', m, w, h, 2, { diff: true });

m = S.keepTopComponent(m, w, h);
dump('top', '上端に接する連結成分だけ残す', '空は画面上端に接している、という前提を使う。窓ガラスや看板など、空と地続きでない「空っぽく見える面」がここで落ちる。',
     'mask', m, w, h, 2, { diff: true });

m = S.fillHoles(m, w, h);
dump('holes', '穴埋め', '外周から背景を塗りつぶし、届かなかった背景＝囲まれた穴を空に倒す。空の中のアンテナや鳥が抜けるのを防ぐ。',
     'mask', m, w, h, 2, { diff: true });

for (let i = 0; i < m.length; i++) if (walls[i]) m[i] = 0;
dump('wall', '直線を差し引く', '引く位置が決定的。判定の直後に引くとクロージングが切れ込みを埋め戻し、穴埋めが塞いでしまい出力が一切変わらない。形状処理の後に引いて初めて効く。',
     'mask', m, w, h, 2, { diff: true });

// ---- モードの継ぎ足し ----
let mask = m;
for (let round = 0; round < opts.modeRounds; round++) {
  const rim = S.morphSquare(mask, w, h, opts.modeReach, 'dilate');
  for (let i = 0; i < rim.length; i++) if (mask[i]) rim[i] = 0;
  const extra = S.skySeedIn(blue, gray, tex, rim, seeds, opts.modeMinCount, opts);
  if (extra === null) {
    dump(`rim${round + 1}`, `モード探索 ${round + 1}回目: 追加なし`, `いま空と判った領域の縁（半径${opts.modeReach}）を探したが、既知のモードから離れた滑らかな画素が${opts.modeMinCount}個に届かなかった。ここで収束。`,
         'mask', rim, w, h, 2);
    break;
  }
  seeds.push(extra);
  mask = build();
  dump(`mode${round + 1}`, `モード${round + 1 + 1}を足す（青優勢度 ${extra.blue.toFixed(3)} / 輝度 ${extra.gray.toFixed(1)}）`,
       '空は1組の代表値では表せない。青空と白い雲は離れすぎていて同時に囲えないので、空と判った領域の縁から次のモードを拾って足す。縁に接していることを求めるのが肝で、これがないと地続きでない白い壁を拾う。',
       'mask', mask, w, h, 2, { diff: true });
}

// ---- 精緻化パス ----
const grown = S.morphSquare(mask, w, h, opts.bandRadius, 'dilate');
const shrunk = S.morphSquare(mask, w, h, opts.bandRadius, 'erode');
const bandMask = new Uint8Array(w * h);
for (let i = 0; i < bandMask.length; i++) bandMask[i] = grown[i] & ~shrunk[i];
dump('band', `境界帯（半径${opts.bandRadius}）`, '粗マスクの境界の周りだけを帯状に切り出す。ここだけを実解像度で判定し直す。帯の外は粗マスクをそのまま使うので、連結性と穴埋めの結果が保たれる。',
     'mask', bandMask, w, h, 2);

let fineMask = S.refineBoundary(fine, mask, w, h, seed, opts);
dump('refine', '境界を実解像度で引き直す', `帯の中では覆す向きで基準を変える（ヒステリシス）。非空→空 の覆しは81.8%正しいのに 空→非空 は38.5%しか正しくないので、空を削る側にだけ ${opts.keepTol}倍の強い証拠を要求する。`,
     'mask', fineMask, fw, fh, 1, { diff: true, prevScale: 2 });

fineMask = S.despeckle(fineMask, fw, fh);
fineMask = S.despeckle(fineMask, fw, fh);
dump('final', 'despeckle ×2 → 最終マスク', '精緻化は局所σを見ないぶん判定が緩く、孤立画素が散る。3×3の多数決なら被覆率を変えずに消せる＝境界を後退させない。オープニングより1/5の費用。',
     'mask', fineMask, fw, fh, 1, { diff: true });

// ---- ドリフト検証 ----
const truth = S.segmentSky(fine);
let diff = 0;
for (let i = 0; i < truth.length; i++) if (truth[i] !== fineMask[i]) diff++;
if (diff !== 0) { console.error(`segmentSky と ${diff} 画素ずれた。再生の順序が食い違っている`); process.exit(1); }

writeFileSync(`${out}/steps.json`, JSON.stringify({ id, label, fw, fh, w, h, steps }, null, 1));
console.log(`${steps.length} 段を ${out} に書き出した（segmentSky と完全一致）`);
