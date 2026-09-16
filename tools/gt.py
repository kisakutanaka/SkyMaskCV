"""ラベル PNG を raw の 0/1 マスクに展開する。

ラベルは 256×256 に正方化されていて、R チャンネルが空マスク。
元のアスペクト比に戻してから 0/1 にする。Node 側だけで採点できるようにするため。
使い方: python3 tools/gt.py <split> <prep.py の出力先>
"""
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), '..', 'dataset')
split, d = sys.argv[1], sys.argv[2]

for m in json.load(open(f'{d}/manifest.json')):
    i, w, h = m['id'], m['w'], m['h']
    lb = np.asarray(Image.open(f'{ROOT}/labels_skyseg/{split}/{i}.png').resize((w, h), Image.BILINEAR))
    (lb[:, :, 0] > 127).astype(np.uint8).tofile(f'{d}/{i}.gt')

print(f'GT を {d} に書き出した')
