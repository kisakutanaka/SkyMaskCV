// 出荷している dist/skyMask.js をそのまま走らせて採点する。
// 参照実装を別に持つと実装ドリフトが起きるので、評価対象は本物のコードにする。
//
// 使い方: node tools/eval.mjs <prep.py の出力先> ['{"vTol":60}' ...]
//   第2引数以降に SkyMaskOptions の部分指定を JSON で並べると、その分だけ比較する。
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const here = fileURLToPath(new URL('.', import.meta.url));
const { segmentSky } = await import(`${here}../dist/skyMask.js`);

const dir = process.argv[2];
const manifest = JSON.parse(readFileSync(`${dir}/manifest.json`, 'utf8'));
const images = manifest.map(({ id, w, h }) => {
  const b = readFileSync(`${dir}/${id}.bin`);
  return {
    w, h,
    data: new Uint8ClampedArray(b.buffer, b.byteOffset, b.length),
    gt: new Uint8Array(readFileSync(`${dir}/${id}.gt`)),
  };
});

/**
 * 適合率と再現率を分けて出す。mIoU だけを見て較正しないこと。
 * このデータセットの平均は空が大きく写った易しい画像に支配されるので、
 * 判定を緩めれば mIoU は上がるが、建物を空と誤る誤りが増える。
 * 合成アプリでは後者のほうが遥かに目立つ。
 */
function evaluate(options) {
  const ious = [];
  let precision = 0, recall = 0;
  const t0 = performance.now();
  for (const im of images) {
    const mask = segmentSky({ data: im.data, width: im.w, height: im.h }, options);
    let tp = 0, fp = 0, fn = 0;
    for (let i = 0; i < mask.length; i++) {
      const a = mask[i], b = im.gt[i];
      if (a && b) tp++; else if (a) fp++; else if (b) fn++;
    }
    ious.push(tp + fp + fn ? tp / (tp + fp + fn) : 1);
    precision += tp + fp ? tp / (tp + fp) : 1;
    recall += tp + fn ? tp / (tp + fn) : 1;
  }
  const n = images.length;
  ious.sort((a, b) => a - b);
  return {
    miou: ious.reduce((a, b) => a + b, 0) / n,
    median: ious[n >> 1],
    precision: precision / n,
    recall: recall / n,
    ms: (performance.now() - t0) / n,
    over: (t) => ious.filter((v) => v >= t).length / n,
    dead: ious.filter((v) => v < 0.01).length / n,
  };
}

const cases = process.argv.length > 3
  ? process.argv.slice(3).map((a) => [a, JSON.parse(a)])
  : [['既定値', {}]];

for (const [name, options] of cases) {
  const r = evaluate(options);
  console.log(`${name}  (${images.length}枚)`);
  console.log(`  mIoU ${r.miou.toFixed(3)}  中央値 ${r.median.toFixed(3)}`
            + `  適合率 ${r.precision.toFixed(3)}  再現率 ${r.recall.toFixed(3)}  ${r.ms.toFixed(2)}ms/枚`);
  console.log(`  IoU>=0.9 ${(100 * r.over(0.9)).toFixed(1)}%   IoU>=0.7 ${(100 * r.over(0.7)).toFixed(1)}%`
            + `   IoU>=0.5 ${(100 * r.over(0.5)).toFixed(1)}%   完全失敗 ${(100 * r.dead).toFixed(1)}%`);
}
