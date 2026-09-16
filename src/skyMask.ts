/**
 * 空マスク抽出（古典CV）
 *
 * すべて純粋関数。DOM・Canvas・Node API に依存しない。
 */

/** RGBA 画像。`ImageData` とそのまま噛み合う形。 */
export interface Image {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

/**
 * 2×2 の箱平均で半分に縮小する。
 *
 * `getImageData` の呼び出しを1回に抑えるため、精緻化パス用の 180×320 を1枚だけ
 * 読み出し、粗パス用の 90×160 はこの関数で JS 側で作る。
 *
 * 幅・高さが奇数の場合は切り捨てた偶数部分だけを使う（端の1行/1列は捨てる）。
 */
export function downsample2x(src: Image): Image {
  const dw = src.width >> 1;
  const dh = src.height >> 1;
  const out = new Uint8ClampedArray(dw * dh * 4);

  for (let y = 0; y < dh; y++) {
    // 元画像の 2 行分の先頭オフセット
    let a = (y * 2 * src.width) * 4;
    let b = a + src.width * 4;
    let d = (y * dw) * 4;

    for (let x = 0; x < dw; x++, a += 8, b += 8, d += 4) {
      out[d]     = (src.data[a]     + src.data[a + 4] + src.data[b]     + src.data[b + 4]) >> 2;
      out[d + 1] = (src.data[a + 1] + src.data[a + 5] + src.data[b + 1] + src.data[b + 5]) >> 2;
      out[d + 2] = (src.data[a + 2] + src.data[a + 6] + src.data[b + 2] + src.data[b + 6]) >> 2;
      out[d + 3] = 255;
    }
  }

  return { data: out, width: dw, height: dh };
}

/**
 * 青優勢度 `(B - R) / (B + R)` を画素ごとに求める。値域 [-1, 1]。
 *
 * 比なので照明の強さに不変。これが薄暮シーンで空と地上を分ける主力になる
 * （実測 f001: 空 +0.47 / 地上 -0.26 に対し、輝度は 73 / 72 で分離不能）。
 *
 * 真っ黒な画素では B+R=0 になるが、ε により 0（中立）に落ちる。
 */
export function blueDominance(src: Image): Float32Array {
  const n = src.width * src.height;
  const out = new Float32Array(n);

  for (let i = 0, j = 0; i < n; i++, j += 4) {
    const r = src.data[j];
    const b = src.data[j + 2];
    out[i] = (b - r) / (b + r + 1e-3);
  }

  return out;
}

/**
 * 輝度（グレースケール）を画素ごとに求める。値域 [0, 255]。
 *
 * 係数は ITU-R BT.601。OpenCV の `COLOR_RGB2GRAY` と同じなので、
 * Python 参照実装との突き合わせで係数差による食い違いが出ない。
 *
 * これが曇天シーンで空と建物を分ける主力になる
 * （実測 cloudy01: 空 165 / RESOLA 115 に対し、青優勢度は +0.021 / +0.014 で分離不能）。
 */
export function toGray(src: Image): Float32Array {
  const n = src.width * src.height;
  const out = new Float32Array(n);

  for (let i = 0, j = 0; i < n; i++, j += 4) {
    out[i] = 0.299 * src.data[j] + 0.587 * src.data[j + 1] + 0.114 * src.data[j + 2];
  }

  return out;
}

/**
 * 正方カーネルのボックス平均。半径によらず O(画素数)。
 *
 * 分離可能なので 横パス → 縦パス の2回で済み、移動和で1画素あたり
 * 加算1・減算1に落ちる（素直に書くと O(画素数 × 半径) になる）。
 *
 * 画像の外側は端の値を複製して埋める（replicate）。OpenCV の `boxFilter` の
 * 既定は reflect-101 なので最外周1画素だけ僅かに値が異なるが、この値を使う
 * シード推定は中央値で集計するため影響しない。
 */
export function boxBlur(field: Float32Array, width: number, height: number, radius: number): Float32Array {
  const tmp = new Float32Array(width * height);
  const out = new Float32Array(width * height);
  const n = 2 * radius + 1;

  // 横パス
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let sum = 0;
    for (let k = -radius; k <= radius; k++) sum += field[row + clamp(k, width)];

    for (let x = 0; x < width; x++) {
      tmp[row + x] = sum / n;
      sum -= field[row + clamp(x - radius, width)];
      sum += field[row + clamp(x + radius + 1, width)];
    }
  }

  // 縦パス
  for (let x = 0; x < width; x++) {
    let sum = 0;
    for (let k = -radius; k <= radius; k++) sum += tmp[clamp(k, height) * width + x];

    for (let y = 0; y < height; y++) {
      out[y * width + x] = sum / n;
      sum -= tmp[clamp(y - radius, height) * width + x];
      sum += tmp[clamp(y + radius + 1, height) * width + x];
    }
  }

  return out;
}

/** 座標を [0, size-1] に丸める。 */
function clamp(v: number, size: number): number {
  return v < 0 ? 0 : v >= size ? size - 1 : v;
}

/**
 * 局所標準偏差（テクスチャの粗さ）。`Var[x] = E[x²] - E[x]²` を
 * `boxBlur` 2回で求め、平方根を取る。
 *
 * 空は滑らかで小さく、建物は窓のエッジで大きくなる
 * （実測: 空 0.8〜4.1 / RESOLA 8〜13 / 茶色ビル 23〜35）。
 * 色と輝度が空と一致してしまう平らな壁を弾くのが役割。
 *
 * 桁落ちは float32 で最悪 6e-4 と実測済み。判定しきい値 6.0 に対し4桁の余裕がある。
 */
export function localStdDev(field: Float32Array, width: number, height: number, radius: number): Float32Array {
  const sq = new Float32Array(field.length);
  for (let i = 0; i < field.length; i++) sq[i] = field[i] * field[i];

  const mean = boxBlur(field, width, height, radius);
  const meanSq = boxBlur(sq, width, height, radius);

  const out = new Float32Array(field.length);
  for (let i = 0; i < out.length; i++) {
    // 桁落ちで僅かに負へ振れることがあるので 0 で止める
    const variance = meanSq[i] - mean[i] * mean[i];
    out[i] = variance > 0 ? Math.sqrt(variance) : 0;
  }

  return out;
}

/** `morphSquare` の動作。`erode` は近傍の最小値、`dilate` は最大値を取る。 */
export type MorphMode = 'erode' | 'dilate';

/**
 * 正方カーネルの収縮 / 膨張。マスク（0 か 1）の形を整える。
 *
 * 正方カーネルは分離可能なので 横パス → 縦パス の2回で済む
 * （2次元をそのまま舐めると 1画素あたり (2r+1)² 回になる）。
 *
 * この2つを組み合わせて使う:
 *   オープニング  = erode → dilate : 孤立した小さな塊を消す（ノイズ除去）
 *   クロージング  = dilate → erode : 小さな切れ込みや穴を埋める（街灯の光芒対策）
 * どちらも面積をおおむね保ったまま形だけを整えるのが要点。
 */
/**
 * 1方向だけの収縮/膨張。窓の合計を累積和の差で取るので半径に依存しない。
 * 範囲外は端の値が続くものとして数える。
 */
function morphAxis(
  src: Uint8Array, width: number, height: number, radius: number,
  dilate: boolean, horizontal: boolean,
): Uint8Array {
  const span = 2 * radius + 1;
  const out = new Uint8Array(width * height);
  const outer = horizontal ? height : width;
  const inner = horizontal ? width : height;
  const step = horizontal ? 1 : width;
  const sum = new Int32Array(inner + 1);

  for (let o = 0; o < outer; o++) {
    const base = horizontal ? o * width : o;
    for (let i = 0; i < inner; i++) sum[i + 1] = sum[i] + src[base + i * step];
    const first = src[base];
    const last = src[base + (inner - 1) * step];
    for (let i = 0; i < inner; i++) {
      const lo = i - radius, hi = i + radius;
      let total = sum[hi < inner ? hi + 1 : inner] - sum[lo > 0 ? lo : 0];
      if (lo < 0) total += -lo * first;
      if (hi > inner - 1) total += (hi - inner + 1) * last;
      out[base + i * step] = (dilate ? total > 0 : total === span) ? 1 : 0;
    }
  }
  return out;
}

/** 1方向のオープニング（収縮→膨張）。その向きに長く伸びた構造だけが残る。 */
function openAxis(
  src: Uint8Array, width: number, height: number, radius: number, horizontal: boolean,
): Uint8Array {
  return morphAxis(morphAxis(src, width, height, radius, false, horizontal),
                   width, height, radius, true, horizontal);
}

export function morphSquare(
  mask: Uint8Array, width: number, height: number, radius: number, mode: MorphMode,
): Uint8Array {
  const dilate = mode === 'dilate';
  return morphAxis(morphAxis(mask, width, height, radius, dilate, true),
                   width, height, radius, dilate, false);
}

/**
 * マスクを最近傍で拡大（または縮小）する。
 *
 * 精緻化パスで2回使う: 粗マスクを帯の判定用に引き伸ばすときと、
 * 帯の外側で粗マスクの値をそのまま採用するとき。
 *
 * 補間しないのは、出力が 0 か 1 のマスクだから。中間値を作ってしまうと
 * `morphSquare` の 0/1 前提が崩れる。表示用の滑らかな拡大は GPU 側
 * （`drawImage`）に任せる。
 */
export function upscaleNearest(
  mask: Uint8Array, srcW: number, srcH: number, dstW: number, dstH: number,
): Uint8Array {
  const out = new Uint8Array(dstW * dstH);
  const sx = srcW / dstW;
  const sy = srcH / dstH;

  for (let y = 0; y < dstH; y++) {
    const srcRow = ((y * sy) | 0) * srcW;
    const dstRow = y * dstW;
    for (let x = 0; x < dstW; x++) {
      out[dstRow + x] = mask[srcRow + ((x * sx) | 0)];
    }
  }

  return out;
}

/** 画面上端から推定した「この空の見え方」。判定の基準点になる。 */
export interface SkySeed {
  /** 空の青優勢度の代表値 */
  blue: number;
  /** 空の輝度の代表値 */
  gray: number;
}

/** 中央値。偶数個のときは中央2つの平均を返す。 */
function median(values: number[]): number {
  values.sort((a, b) => a - b);
  const mid = values.length >> 1;
  return values.length & 1
    ? values[mid]
    : (values[mid - 1] + values[mid]) / 2;
}

/**
 * 与えられた探索範囲から、空のモードを1つ拾う。足りなければ `null`。
 *
 * 固定しきい値をやめてこれを基準にするのが、薄暮と曇天を1本の規則で扱える理由。
 * 薄暮は色で分離でき輝度では分離できず、曇天はその逆になる（実測）。
 * どちらの軸が効くかを決め打ちせず、空自身の値を基準点として両方を持つ。
 *
 * 探索範囲のうち、テクスチャが滑らかで、`seeds` のどの箱にも入らない画素を集める。
 * 滑らかさで絞るのは、上端に建物が写り込むフレーム（実測 f004）で窓のエッジを
 * 除くため。既知の箱を除くのは、2回目以降に同じモードを拾い直さないため。
 *
 * 代表値は平均でなく中央値。街灯の光芒が上端行に混入するフレーム
 * （実測 f006 / f007）では平均だと基準が引きずられて破綻する。
 *
 * 呼び出し方は2通りある。探索範囲が違うだけで規則は同じ。
 *   1回目 … 探索範囲＝画面上端のバンド、`seeds` は空
 *   2回目以降 … 探索範囲＝いま空と判った領域の縁、`seeds` に既知のモード
 *
 * 2回目以降が要るのは、空が1組の代表値では表せないことがあるから。青空に白い雲が
 * 浮かぶ構図では、青空（実測 輝度113 / 青優勢度0.37）と雲（輝度189 / 0.07）が
 * 離れすぎていて同時には囲えない。雲の局所σは 0.033 と滑らかなので、落として
 * いたのは箱そのものだった。縁に接していることを求めるのは、空と地続きでない
 * 明るい面（白い壁など）を拾わないため。
 */
export function skySeedIn(
  blue: Float32Array, gray: Float32Array, tex: Float32Array,
  search: Uint8Array,
  seeds: readonly SkySeed[],
  minCount: number,
  opts: { texMaxRel: number; bdTol: number; vTol: number },
): SkySeed | null {
  // PERF: 上端10% = 最大1,440画素を number[] に push している。実測 2.92ms には
  //       含まれた上での数字なので現状は許容。実機計測で GC やここが効いてきたら、
  //       事前確保した Float32Array + 件数カウンタ + subarray(0, n).sort() に差し替える。
  const blues: number[] = [];
  const grays: number[] = [];

  for (let i = 0; i < search.length; i++) {
    if (!search[i]) continue;
    if (tex[i] / (gray[i] + 1) >= opts.texMaxRel) continue;
    const known = seeds.some((s) => Math.abs(blue[i] - s.blue) < opts.bdTol
                                 && Math.abs(gray[i] - s.gray) < opts.vTol);
    if (known) continue;
    blues.push(blue[i]);
    grays.push(gray[i]);
  }

  if (blues.length < minCount) return null;
  return { blue: median(blues), gray: median(grays) };
}

/**
 * シードからの距離で「空らしさ」を判定する。判定規則はこの関数だけに置く。
 *
 *   |青優勢度 - seed.blue| < bdTol  かつ
 *   |輝度     - seed.gray| < vTol   かつ
 *   （tex があれば）tex / 輝度 < texMaxRel
 *
 * 各軸を独立に見る（ユークリッド距離を取らない）のは、2軸のスケールに
 * 共通の物差しがないため。青優勢度は無次元の比、輝度は 0〜255 の絶対値で、
 * 正規化係数を決める根拠がない。軸ごとの許容値なら実測から直接決められる。
 *
 * 薄暮では青優勢度の差（空 +0.47 / 地上 -0.26）が効き、曇天では輝度の差
 * （空 165 / 建物 115）が効く。どちらが効くかはシーンが決める。
 *
 * `tex` に `null` を渡すとテクスチャ条件を省く。精緻化パスで使う:
 * 連結性は粗パスで確定済みなので、高解像度で局所σを計算する価値がない。
 */
export function classifyBySeed(
  blue: Float32Array, gray: Float32Array, tex: Float32Array | null,
  seed: SkySeed,
  opts: { bdTol: number; vTol: number; texMaxRel: number },
): Uint8Array {
  const out = new Uint8Array(blue.length);

  for (let i = 0; i < out.length; i++) {
    const nearColour = Math.abs(blue[i] - seed.blue) < opts.bdTol;
    const nearValue = Math.abs(gray[i] - seed.gray) < opts.vTol;
    const smooth = tex === null || tex[i] / (gray[i] + 1) < opts.texMaxRel;
    out[i] = nearColour && nearValue && smooth ? 1 : 0;
  }

  return out;
}

/**
 * 人工物の輪郭に出る直線を拾う。空の判定から差し引くために使う。
 *
 * 直線性は局所σとは独立した手がかりになる。実測（val 200枚）では検出された
 * 直線の 85.8% が建物の上にあり、雲の縁に乗るのは 3.2% だけだった。σ では
 * 雲の縁（中央値 0.077）と建物の縁（0.230）が重なって分けられなかったが、
 * 直線性なら分かれる。
 *
 * 勾配を2値化し、縦横それぞれの向きにオープニングをかける。その向きに
 * 長く伸びた構造だけが残り、雲のような曲がった輪郭は落ちる。
 * 斜め方向の棒も試したが結果は変わらなかったので入れていない。
 *
 * 注意: これは 0°/90° に近い線しか拾わない。浅い角度（実測15°）の稜線は
 * どの棒にも収まらず取りこぼす。エッジを太らせれば拾えるが、val で悪化が
 * 改善を上回った（改善42/悪化50、最悪 -0.411）ので採っていない。
 */
export function straightEdges(
  gray: Float32Array, width: number, height: number,
  opts: { edgeMin: number; edgeLen: number },
): Uint8Array {
  const edge = new Uint8Array(width * height);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const gx = Math.abs(gray[i + 1] - gray[i - 1]);
      const gy = Math.abs(gray[i + width] - gray[i - width]);
      edge[i] = gx + gy > opts.edgeMin ? 1 : 0;
    }
  }

  const horizontal = openAxis(edge, width, height, opts.edgeLen, true);
  const vertical = openAxis(edge, width, height, opts.edgeLen, false);

  const out = new Uint8Array(width * height);
  for (let i = 0; i < out.length; i++) out[i] = horizontal[i] | vertical[i];
  return out;
}

/**
 * 画面上端に接する十分大きな連結成分だけを残す。
 *
 * 判定は画素ごとに独立なので、空と同じ色・輝度・滑らかさを持つ地上の領域
 * （舗装、水面、平らな壁）が飛び地として残る。空は必ず画面上端に接している
 * という前提を使ってそれらを落とす。
 *
 * かつては面積の下限も持っていたが、実測で回帰セットの出力が完全に同一だった
 * ため外した。小さな塊はオープニングと多モードの整形が先に落としている。
 *
 * 4近傍の幅優先探索。`queue` に添字を積むので再帰せず、
 * スタックを溢れさせない（90×160 でも最悪 14,400 段になる）。
 */
export function keepTopComponent(
  mask: Uint8Array, width: number, height: number,
): Uint8Array {
  const n = width * height;
  const out = new Uint8Array(n);
  const visited = new Uint8Array(n);
  const queue = new Int32Array(n);

  // 上端の空画素それぞれから探索する。成分を一度なめて面積を測り、
  // 合格したものだけ書き戻す。
  for (let seed = 0; seed < width; seed++) {
    if (!mask[seed] || visited[seed]) continue;

    let head = 0;
    let tail = 0;
    queue[tail++] = seed;
    visited[seed] = 1;

    while (head < tail) {
      const p = queue[head++];
      const x = p % width;

      if (x > 0 && mask[p - 1] && !visited[p - 1]) { visited[p - 1] = 1; queue[tail++] = p - 1; }
      if (x < width - 1 && mask[p + 1] && !visited[p + 1]) { visited[p + 1] = 1; queue[tail++] = p + 1; }
      if (p >= width && mask[p - width] && !visited[p - width]) { visited[p - width] = 1; queue[tail++] = p - width; }
      if (p < n - width && mask[p + width] && !visited[p + width]) { visited[p + width] = 1; queue[tail++] = p + width; }
    }

    for (let i = 0; i < tail; i++) out[queue[i]] = 1;
  }

  return out;
}

/**
 * 完全に囲まれた穴を空で埋める。
 *
 * 外周の非空画素から背景を塗りつぶし、塗り残った非空画素＝どこからも
 * 外に出られない穴、と判定する。
 *
 * クロージングと役割が相補的:
 *   クロージング : カーネルより小さい穴だけ / 外に繋がっていても埋まる
 *   fillHoles    : 大きさは問わない        / 完全に囲まれている必要がある
 * 空に浮かぶ構造物（クレーン、電線の交差、街灯の傘）が作る穴はここで消える。
 */
// LIMITATION: 穴の大きさを問わないので、空に完全に囲まれた大きな構造物
//             （遠景の塔、クレーン、上部だけ見えているビル）も空として塗り潰す。
//             現素材では未発生（建物は下端か他の建物に必ず繋がる）。カメラを
//             上に向ける実運用で発生したら opts に maxHoleAreaRatio を足して
//             「大きすぎる穴は埋めない」とする。実例を見るまで入れない。
export function fillHoles(mask: Uint8Array, width: number, height: number): Uint8Array {
  const n = width * height;
  const outside = new Uint8Array(n);
  const queue = new Int32Array(n);
  let head = 0;
  let tail = 0;

  const push = (p: number) => {
    if (!mask[p] && !outside[p]) { outside[p] = 1; queue[tail++] = p; }
  };

  // 外周の非空画素を起点にする
  for (let x = 0; x < width; x++) { push(x); push((height - 1) * width + x); }
  for (let y = 0; y < height; y++) { push(y * width); push(y * width + width - 1); }

  while (head < tail) {
    const p = queue[head++];
    const x = p % width;

    if (x > 0) push(p - 1);
    if (x < width - 1) push(p + 1);
    if (p >= width) push(p - width);
    if (p < n - width) push(p + width);
  }

  // 空 か、外に出られなかった非空画素（＝穴）を空とする
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = mask[i] || !outside[i] ? 1 : 0;

  return out;
}

/**
 * 粗マスクの境界付近だけを、高い解像度で判定し直す。
 *
 * 粗パスは 90×160 なので、そのまま引き伸ばすと境界が階段状になり実際の稜線に
 * 乗らない。かといって全画素を高解像度で処理すると予算に入らない
 * （実測 8.17ms 対 2.92ms）。そこで境界の帯だけを実画素で引き直す。
 *
 * 帯は粗マスクの 膨張 − 収縮 で作る。帯の外は粗マスクをそのまま使うので、
 * 連結成分の選別と穴埋めの結果は保たれる。帯の中だけを塗り替えるため、
 * 高解像度側で孤立した誤判定が出ても帯の幅を超えて広がらない。
 *
 * 判定には粗パスと同じ `seed` と同じ規則を使う。局所σは渡さない
 * （連結性は確定済みで、高解像度で計算する価値がない）。
 *
 * ただし覆す方向で許容値を変える（ヒステリシス）。帯の中は形状処理を通さず
 * 生の判定をそのまま採用するため、雲や空の諧調で許容値を外れた画素が
 * そのまま境界になる。実測（val 335枚）では方向で精度がまるで違う:
 *   非空→空 の覆し 718,856件 … 81.8% が正しい
 *   空→非空 の覆し  25,842件 … 38.5% しか正しくない
 * そこで空を削るほうにだけ `keepTol` 倍の強い証拠を要求する。建物は倍にしても
 * 許容外に出るので、稜線に吸着する働きは保たれる。
 */
export function refineBoundary(
  fine: Image,
  coarse: Uint8Array, coarseW: number, coarseH: number,
  seed: SkySeed,
  opts: { bandRadius: number; bdTol: number; vTol: number; texMaxRel: number; keepTol: number },
): Uint8Array {
  // 帯は粗解像度で作ってから引き伸ばす。高解像度で膨張・収縮するより安い。
  const grown = morphSquare(coarse, coarseW, coarseH, opts.bandRadius, 'dilate');
  const shrunk = morphSquare(coarse, coarseW, coarseH, opts.bandRadius, 'erode');

  const band = new Uint8Array(coarse.length);
  for (let i = 0; i < band.length; i++) band[i] = grown[i] - shrunk[i];

  const fineBand = upscaleNearest(band, coarseW, coarseH, fine.width, fine.height);
  const fineBase = upscaleNearest(coarse, coarseW, coarseH, fine.width, fine.height);

  const blue = blueDominance(fine);
  const gray = toGray(fine);
  const fresh = classifyBySeed(blue, gray, null, seed, opts);
  // 粗マスクが既に空と言っている画素はこちらで見る。判定規則は
  // `classifyBySeed` に置いたまま、許容値だけ差し替えて2回呼ぶ。
  const keep = classifyBySeed(blue, gray, null, seed, {
    ...opts, bdTol: opts.bdTol * opts.keepTol, vTol: opts.vTol * opts.keepTol,
  });

  const out = new Uint8Array(fine.width * fine.height);
  for (let i = 0; i < out.length; i++) {
    if (!fineBand[i]) { out[i] = fineBase[i]; continue; }
    out[i] = fineBase[i] ? keep[i] : fresh[i];
  }

  return out;
}

/** `segmentSky` の調整値。すべて実測から較正した。 */
export interface SkyMaskOptions {
  /** 局所σのカーネル半径（粗解像度） */
  texRadius: number;
  /**
   * これ以上ざらついた画素は空としない。局所σを輝度で割った相対値。
   *
   * 絶対値で切ると明るい空ほど不利になる。テクスチャの振幅は明るさに比例
   * するため。青優勢度を (B-R)/(B+R) にしたのと同じ理由で、ここも比を取る。
   * 実測（Open Images 4236枚）: 絶対σで適合率を 0.908 に揃えたとき再現率
   * 0.795、相対σなら 0.821。判定とシード抽出の両方を相対にすると
   * val で mIoU 0.740→0.753 / 適合率 0.926→0.932 / 再現率 0.776→0.795。
   */
  texMaxRel: number;
  /** シードを取る上端の割合 */
  seedRowRatio: number;
  /**
   * 第1モードのシードがこれ未満なら「空が写っていない」と判断して全0を返す。
   * 粗画像の面積に対する比で持つ（`modeMinRatio` と同じ理由。絶対数だと
   * 処理解像度を変えたときに黙って挙動が変わる）。
   *
   * 本来はカメラを下に向けた場合のための安全弁で、val 335枚では
   * しきい値を 1 から 100 まで振っても出力が1画素も変わらない（発火しない）。
   * 一方 train 3901枚では5枚で発火しており、いずれも空が小さく候補が43〜49個の
   * 構図だった。比にして43に下がることでこの5枚が全0マスクから救われる
   * （IoU 0.000 → 0.008〜0.040、適合率はいずれも 1.000）。
   */
  minSeedRatio: number;
  /** 青優勢度がシードからこれ以上離れたら空でない */
  bdTol: number;
  /** 輝度がシードからこれ以上離れたら空でない */
  vTol: number;
  /**
   * オープニングの半径（孤立ノイズ除去）。
   * 大きくすると 2*r+1 画素より細い空の筋が消える。粗解像度 90×160 では
   * r=2 が「5画素未満を消す」＝ 表示幅 1958px 換算で約109px より細い隙間が消える。
   * 実測: r=2 だと cloudy01 のビル間の空（粗解像度で3画素幅）が丸ごと落ち、
   * さらに上端との連結が切れて keepTopComponent にも捨てられた。r=1 で72%回復し、
   * 孤立画素はむしろ減った（7→6）。
   */
  openRadius: number;
  /** クロージングの半径（切れ込みの橋渡し） */
  closeRadius: number;
  /**
   * 境界帯の半径（粗解像度）。`closeRadius` 以上にすること。
   * クロージングは境界を最大 closeRadius だけ動かすので、帯がそれより狭いと
   * 真の境界が帯の外に出て粗マスクのまま残り、表示解像度で 8px のブロックになる。
   * 実測: bandRadius=2 だと境界画素の 33〜47% が未精緻化、4 で 3〜8% に落ちる。
   */
  bandRadius: number;
  /**
   * 境界帯で「空を維持する側」の許容値の倍率。
   *
   * 1 にすると両方向が同じ基準になり、雲の縁や空の諧調で境界が切れる。
   * 1.5〜∞ で集計指標はほぼ動かない（削る覆しは帯の 1% しかなく IoU に
   * 出ない）ので、効くのは見た目のほう。∞（絶対に削らない）にしないのは、
   * 粗パスのクロージングが建物側にはみ出した分を戻す経路を残すため。
   */
  keepTol: number;
  /** 次のモードを探す範囲。マスクの縁から粗解像度で何画素まで見るか */
  modeReach: number;
  /**
   * 次のモードを認めるのに必要な候補画素数。粗画像の面積に対する比で持つ。
   *
   * `minSeedCount` と分けてあるのは、雲の縁は遷移部で局所σが上がって滑らかさの
   * 条件を落ちるため、候補が数十画素しか残らないことがあるから。実測の
   * 1b8d1d161bf4396b では候補が22画素で、絶対数 50 だと第2モードが作られず
   * 雲が取れないままだった。
   *
   * 絶対数で持っていたときに実際に壊れた。較正時の粗面積 16640 画素で決めた
   * 20 をそのまま使っていたところ、処理解像度を入力の縦横比に合わせた結果
   * 粗面積が 14400 になり、候補が16〜19個で 20 に届かず第2モードが消えた
   * （同じ画像で IoU 0.864 → 0.559）。候補数は面積に比例するので比で持つ。
   */
  modeMinRatio: number;
  /** モードを継ぎ足す最大回数。実測では3回で収束した */
  modeRounds: number;
  /** この勾配を超えた画素を輪郭の候補にする */
  edgeMin: number;
  /** 直線と見なすのに必要な長さ（半径）。これより短い輪郭は落ちる */
  edgeLen: number;
}

export const DEFAULT_OPTIONS: SkyMaskOptions = {
  texRadius: 2,
  texMaxRel: 0.045,
  seedRowRatio: 0.10,
  minSeedRatio: 0.003,    // 較正時の 50/16640。粗面積 14400 なら43画素
  bdTol: 0.25,
  vTol: 42.0,
  openRadius: 1,
  closeRadius: 4,
  bandRadius: 4,
  keepTol: 2.0,
  modeReach: 4,
  modeMinRatio: 0.0012,   // 較正時の 20/16640。粗面積 14400 なら17画素
  modeRounds: 3,
  edgeMin: 20.0,
  edgeLen: 6,
};

/**
 * 空マスクを求める。入力と同じ解像度の 0/1 マスクを返す。
 *
 * 入力は 180×320 程度を想定する（粗パスはその半分で走る）。呼び出し側は
 * `drawImage` で video をその大きさの canvas に縮小し、`getImageData` を
 * 1回だけ呼べばよい。表示解像度への拡大も `drawImage` に任せる。
 *
 * 空が写っていないと判断した場合は全0のマスクを返す。
 */
export function segmentSky(fine: Image, options: Partial<SkyMaskOptions> = {}): Uint8Array {
  const opts = { ...DEFAULT_OPTIONS, ...options };

  // 粗パス: どこが空かという「面」を決める
  const coarse = downsample2x(fine);
  const w = coarse.width;
  const h = coarse.height;

  const blue = blueDominance(coarse);
  const gray = toGray(coarse);
  const tex = localStdDev(gray, w, h, opts.texRadius);

  // 1回目のモードは画面上端のバンドから取る
  const band = new Uint8Array(w * h);
  band.fill(1, 0, Math.max(1, Math.round(h * opts.seedRowRatio)) * w);
  const minSeed = Math.max(1, Math.round(w * h * opts.minSeedRatio));
  const seed = skySeedIn(blue, gray, tex, band, [], minSeed, opts);
  if (seed === null) return new Uint8Array(fine.width * fine.height);

  // シードは1つとは限らない。青空と白い雲のように空が複数の見えを持つ構図では、
  // 空と地続きの領域から次のモードを拾って足していく（nextSkySeed）。
  const seeds: SkySeed[] = [seed];
  const walls = straightEdges(gray, w, h, opts);

  const build = (): Uint8Array => {
    // 判定規則は classifyBySeed に置いたまま、シードごとに呼んで重ねる。
    let m = classifyBySeed(blue, gray, tex, seeds[0], opts);
    for (let k = 1; k < seeds.length; k++) {
      const part = classifyBySeed(blue, gray, tex, seeds[k], opts);
      for (let i = 0; i < m.length; i++) if (part[i]) m[i] = 1;
    }

    // オープニングが先。逆順にすると孤立ノイズが膨張で周囲と繋がり、
    // 落とせない構造になったうえ keepTopComponent で空本体と地続きになる。
    m = morphSquare(m, w, h, opts.openRadius, 'erode');
    m = morphSquare(m, w, h, opts.openRadius, 'dilate');
    m = morphSquare(m, w, h, opts.closeRadius, 'dilate');
    m = morphSquare(m, w, h, opts.closeRadius, 'erode');

    m = keepTopComponent(m, w, h);
    m = fillHoles(m, w, h);

    // 直線を差し引くのは最後。判定の直後に引くと、クロージングが細い切れ込みを
    // 埋め戻し、fillHoles が囲まれた穴を塞いでしまう（実測で出力が変わらなかった）。
    for (let i = 0; i < m.length; i++) if (walls[i]) m[i] = 0;
    return m;
  };

  let mask = build();
  for (let round = 0; round < opts.modeRounds; round++) {
    // 2回目以降は、いま空と判った領域の縁を探索範囲にする
    const rim = morphSquare(mask, w, h, opts.modeReach, 'dilate');
    for (let i = 0; i < rim.length; i++) if (mask[i]) rim[i] = 0;

    const minCount = Math.max(1, Math.round(w * h * opts.modeMinRatio));
    const extra = skySeedIn(blue, gray, tex, rim, seeds, minCount, opts);
    if (extra === null) break;
    seeds.push(extra);
    mask = build();
  }

  // 精緻化パス: 境界だけを実画素に乗せ直す。
  // シードは第1モードだけを渡す。追加モードの領域は帯の中で許容外に出るが、
  // keepTol のヒステリシスが粗マスクの判断を保つので削られない。
  let fineMask = refineBoundary(fine, mask, w, h, seed, opts);

  // 緩い判定が残した孤立画素を落とす。2回かけると open(1) より綺麗で
  // 1/5 の費用で済む（実測 +0.27ms 対 +1.37ms）。
  fineMask = despeckle(fineMask, fine.width, fine.height);
  fineMask = despeckle(fineMask, fine.width, fine.height);

  return fineMask;
}

/**
 * 3×3 の多数決（2値のメディアン）で孤立画素を消す。1パス・確保1本。
 *
 * 精緻化パスは局所σを見ないぶん判定が緩く、許容値の境目にある暗い空で
 * 孤立した誤判定が散る。静止画では目立たないが、映像では毎フレーム位置が
 * 変わるためちらつきとして見える。
 *
 * 形態学的オープニングでも消せるが、180×320 では 4パス必要で +1.37ms かかる
 * （計算律速でバッファを使い回しても速くならないことを実測で確認済み）。
 * 多数決なら1パス +0.13ms で、しかも被覆率を一切変えずに消せる
 * ＝ 境界を後退させない。
 *
 * 外周1画素は近傍が揃わないのでそのまま通す。
 */
export function despeckle(mask: Uint8Array, width: number, height: number): Uint8Array {
  const out = new Uint8Array(width * height);

  for (let y = 1; y < height - 1; y++) {
    const row = y * width;
    for (let x = 1; x < width - 1; x++) {
      const i = row + x;
      const sum =
        mask[i - width - 1] + mask[i - width] + mask[i - width + 1] +
        mask[i - 1] + mask[i] + mask[i + 1] +
        mask[i + width - 1] + mask[i + width] + mask[i + width + 1];
      out[i] = sum >= 5 ? 1 : 0;
    }
  }

  for (let x = 0; x < width; x++) {
    out[x] = mask[x];
    out[(height - 1) * width + x] = mask[(height - 1) * width + x];
  }
  for (let y = 0; y < height; y++) {
    out[y * width] = mask[y * width];
    out[y * width + width - 1] = mask[y * width + width - 1];
  }

  return out;
}
