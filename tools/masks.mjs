// 全画像に現手法のマスクを作って書き出す。比較画像の材料。
// 使い方: node tools/masks.mjs <prep.py の出力先>
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const here = fileURLToPath(new URL('.', import.meta.url));
const { segmentSky } = await import(`${here}../dist/skyMask.js`);

const dir = process.argv[2];
const manifest = JSON.parse(readFileSync(`${dir}/manifest.json`, 'utf8'));
let done = 0;
for (const { id, w, h } of manifest) {
  const b = readFileSync(`${dir}/${id}.bin`);
  const data = new Uint8ClampedArray(b.buffer, b.byteOffset, b.length);
  writeFileSync(`${dir}/${id}.mask`, Buffer.from(segmentSky({ data, width: w, height: h })));
  if (++done % 500 === 0) console.log(`  ${done}/${manifest.length}`);
}
console.log(`${done} 枚のマスクを書き出した`);
