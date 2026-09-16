"""データセットの JPEG をデコードして raw RGBA に展開する。

ブラウザでの経路に合わせ、長辺 320 に縮めてアスペクト比は保つ。
使い方: python3 tools/prep.py <split> <出力先> [枚数上限]
"""
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), '..', 'dataset')

split, out = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
os.makedirs(out, exist_ok=True)

ids = sorted(p[:-4] for p in os.listdir(f'{ROOT}/labels_skyseg/{split}') if p.endswith('.png'))
ids = ids[:limit]

manifest = []
for i in ids:
    im = Image.open(f'{ROOT}/images/{split}/{i}.jpg').convert('RGB')
    w, h = im.size
    scale = 320 / max(w, h)
    nw, nh = max(2, round(w * scale)), max(2, round(h * scale))
    a = np.asarray(im.resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
    np.dstack([a, np.full((nh, nw, 1), 255, np.uint8)]).tofile(f'{out}/{i}.bin')
    manifest.append({'id': i, 'w': nw, 'h': nh})

import json
json.dump(manifest, open(f'{out}/manifest.json', 'w'))
print(f'{len(manifest)} 枚を {out} に展開した')
