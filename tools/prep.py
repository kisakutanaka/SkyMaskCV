"""データセットの JPEG をデコードして raw RGBA に展開する。

デモ（index.html）と同じ規則で縮める。縦横比は入力のまま、画素数だけを
PROC_PIXELS に揃える。評価とアプリの条件を一致させるため、ここを変えるときは
index.html の procSize も一緒に直すこと。
使い方: python3 tools/prep.py <split> <出力先> [枚数上限]
"""
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), '..', 'dataset')
PROC_PIXELS = 180 * 320

split, out = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
os.makedirs(out, exist_ok=True)

ids = sorted(p[:-4] for p in os.listdir(f'{ROOT}/labels_skyseg/{split}') if p.endswith('.png'))
ids = ids[:limit]

manifest = []
for i in ids:
    im = Image.open(f'{ROOT}/images/{split}/{i}.jpg').convert('RGB')
    w, h = im.size
    ratio = w / h
    nw = max(2, round((PROC_PIXELS * ratio) ** 0.5))
    nh = max(2, round((PROC_PIXELS / ratio) ** 0.5))
    a = np.asarray(im.resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
    np.dstack([a, np.full((nh, nw, 1), 255, np.uint8)]).tofile(f'{out}/{i}.bin')
    manifest.append({'id': i, 'w': nw, 'h': nh})

import json
json.dump(manifest, open(f'{out}/manifest.json', 'w'))
print(f'{len(manifest)} 枚を {out} に展開した')
