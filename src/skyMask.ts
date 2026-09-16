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
export function morphSquare(
  mask: Uint8Array, width: number, height: number, radius: number, mode: MorphMode,
): Uint8Array {
  const dilate = mode === 'dilate';
  const tmp = new Uint8Array(width * height);
  const out = new Uint8Array(width * height);

  // 横パス
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      let v = dilate ? 0 : 1;
      for (let k = -radius; k <= radius; k++) {
        const s = mask[row + clamp(x + k, width)];
        v = dilate ? (v | s) : (v & s);
      }
      tmp[row + x] = v;
    }
  }

  // 縦パス
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      let v = dilate ? 0 : 1;
      for (let k = -radius; k <= radius; k++) {
        const s = tmp[clamp(y + k, height) * width + x];
        v = dilate ? (v | s) : (v & s);
      }
      out[y * width + x] = v;
    }
  }

  return out;
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

/**
 * 画面上端の画素から「この空はどう見えているか」を推定する。
 *
 * 固定しきい値をやめてこれを基準にするのが、薄暮と曇天を1本の規則で扱える理由。
 * 薄暮は色で分離でき輝度では分離できず、曇天はその逆になる（実測）。
 * どちらの軸が効くかを決め打ちせず、空自身の値を基準点として両方を持つ。
 *
 * 上端 `seedRowRatio` の範囲のうち、テクスチャが滑らかな画素だけを集める。
 * 滑らかさで絞るのは、上端に建物が写り込むフレーム（実測 f004）で
 * 窓のエッジを除くため。
 *
 * 代表値は平均でなく中央値。街灯の光芒が上端行に混入するフレーム
 * （実測 f006 / f007）では平均だと基準が引きずられて破綻する。
 *
 * 十分な画素が集まらなければ `null` を返す（＝空が写っていないと判断する）。
 */
export function estimateSkySeed(
  blue: Float32Array, gray: Float32Array, tex: Float32Array,
  width: number, height: number,
  opts: { seedRowRatio: number; texMaxRel: number; minSeedCount: number },
): SkySeed | null {
  const rows = Math.max(1, Math.round(height * opts.seedRowRatio));
  const limit = rows * width;

  // PERF: 上端10% = 最大1,440画素を number[] に push している。実測 2.92ms には
  //       含まれた上での数字なので現状は許容。実機計測で GC やここが効いてきたら、
  //       事前確保した Float32Array + 件数カウンタ + subarray(0, n).sort() に差し替える。
  const blues: number[] = [];
  const grays: number[] = [];
  for (let i = 0; i < limit; i++) {
    if (tex[i] / (gray[i] + 1) < opts.texMaxRel) {
      blues.push(blue[i]);
      grays.push(gray[i]);
    }
  }

  if (blues.length < opts.minSeedCount) return null;

  return { blue: median(blues), gray: median(grays) };
}

/** 中央値。引数の配列を破壊的に並べ替える。 */
function median(values: number[]): number {
  values.sort((a, b) => a - b);
  const mid = values.length >> 1;
  return values.length & 1
    ? values[mid]
    : (values[mid - 1] + values[mid]) / 2;
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
 * 画面上端に接する十分大きな連結成分だけを残す。
 *
 * 判定は画素ごとに独立なので、空と同じ色・輝度・滑らかさを持つ地上の領域
 * （舗装、水面、平らな壁）が飛び地として残る。空は必ず画面上端に接している
 * という前提を使ってそれらを落とす。
 *
 * 面積が `minAreaRatio` に満たない成分も捨てる。上端に接していても、
 * 数画素の塊は空ではなくノイズと見なす。
 *
 * 4近傍の幅優先探索。`queue` に添字を積むので再帰せず、
 * スタックを溢れさせない（90×160 でも最悪 14,400 段になる）。
 */
export function keepTopComponent(
  mask: Uint8Array, width: number, height: number, minAreaRatio: number,
): Uint8Array {
  const n = width * height;
  const out = new Uint8Array(n);
  const visited = new Uint8Array(n);
  const queue = new Int32Array(n);
  const minArea = n * minAreaRatio;

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

    // tail がこの成分の画素数
    if (tail >= minArea) {
      for (let i = 0; i < tail; i++) out[queue[i]] = 1;
    }
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
 */
export function refineBoundary(
  fine: Image,
  coarse: Uint8Array, coarseW: number, coarseH: number,
  seed: SkySeed,
  opts: { bandRadius: number; bdTol: number; vTol: number; texMaxRel: number },
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

  const out = new Uint8Array(fine.width * fine.height);
  for (let i = 0; i < out.length; i++) out[i] = fineBand[i] ? fresh[i] : fineBase[i];

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
  /** シードがこれ未満なら空なしと判断する */
  minSeedCount: number;
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
  /** この割合に満たない連結成分は捨てる */
  minAreaRatio: number;
  /**
   * 境界帯の半径（粗解像度）。`closeRadius` 以上にすること。
   * クロージングは境界を最大 closeRadius だけ動かすので、帯がそれより狭いと
   * 真の境界が帯の外に出て粗マスクのまま残り、表示解像度で 8px のブロックになる。
   * 実測: bandRadius=2 だと境界画素の 33〜47% が未精緻化、4 で 3〜8% に落ちる。
   */
  bandRadius: number;
}

export const DEFAULT_OPTIONS: SkyMaskOptions = {
  texRadius: 2,
  texMaxRel: 0.045,
  seedRowRatio: 0.10,
  minSeedCount: 50,
  bdTol: 0.25,
  vTol: 42.0,
  openRadius: 1,
  closeRadius: 4,
  minAreaRatio: 0.01,
  bandRadius: 4,
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

  const seed = estimateSkySeed(blue, gray, tex, w, h, opts);
  if (seed === null) return new Uint8Array(fine.width * fine.height);

  let mask = classifyBySeed(blue, gray, tex, seed, opts);

  // オープニングが先。逆順にすると孤立ノイズが膨張で周囲と繋がり、
  // 落とせない構造になったうえ keepTopComponent で空本体と地続きになる。
  mask = morphSquare(mask, w, h, opts.openRadius, 'erode');
  mask = morphSquare(mask, w, h, opts.openRadius, 'dilate');
  mask = morphSquare(mask, w, h, opts.closeRadius, 'dilate');
  mask = morphSquare(mask, w, h, opts.closeRadius, 'erode');

  mask = keepTopComponent(mask, w, h, opts.minAreaRatio);
  mask = fillHoles(mask, w, h);

  // 精緻化パス: 境界だけを実画素に乗せ直す
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
