"""現手法のマスクと skyseg のラベルを並べた比較画像を作る。

    左: 元画像 / 中: 現手法 / 右: skyseg

目視で当たりを付けるためのものなので、IoU の低い順に並べた index.html も出す。
使い方: python3 tools/compare.py <prep.py と masks.mjs の出力先> <比較画像の出力先>
"""
import json
import os
import sys

import cv2
import numpy as np

SRC, DST = sys.argv[1], sys.argv[2]
H = 240                      # 1枚あたりの表示高さ
TINT = np.array((255, 0, 255), np.float32)   # BGR: 空と判定された領域に乗せる色。空の青や雲の白と紛れない色を選ぶ
ALPHA = 0.45
BAR = 18                     # 見出し帯の高さ

os.makedirs(DST, exist_ok=True)
manifest = json.load(open(f'{SRC}/manifest.json'))
rows = []

for m in manifest:
    i, w, h = m['id'], m['w'], m['h']
    rgba = np.fromfile(f'{SRC}/{i}.bin', np.uint8).reshape(h, w, 4)
    src = rgba[:, :, :3][:, :, ::-1].copy()          # RGB -> BGR
    pred = np.fromfile(f'{SRC}/{i}.mask', np.uint8).reshape(h, w) > 0
    gt = np.fromfile(f'{SRC}/{i}.gt', np.uint8).reshape(h, w) > 0

    inter, union = (pred & gt).sum(), (pred | gt).sum()
    iou = inter / union if union else 1.0

    def tint(mask):
        out = src.copy()
        out[mask] = (ALPHA * TINT + (1 - ALPHA) * out[mask]).astype(np.uint8)
        return out

    panels = [cv2.resize(p, (max(1, round(w * H / h)), H)) for p in (src, tint(pred), tint(gt))]
    body = np.hstack(panels)
    bar = np.zeros((BAR, body.shape[1], 3), np.uint8)
    for k, text in enumerate((i, f'ours IoU {iou:.3f}', 'skyseg')):
        cv2.putText(bar, text, (k * panels[0].shape[1] + 4, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.imwrite(f'{DST}/{i}.jpg', np.vstack([bar, body]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    rows.append((iou, i))

rows.sort()
with open(f'{DST}/index.html', 'w', encoding='utf-8') as fp:
    fp.write('<!doctype html><meta charset="utf-8"><title>空マスク比較</title>\n')
    fp.write('<style>body{background:#111;color:#ccc;font:13px system-ui;margin:0;padding:12px}'
             'img{display:block;margin:0 0 10px;max-width:100%}</style>\n')
    fp.write(f'<p>{len(rows)} 枚。左: 元画像 / 中: 現手法 / 右: skyseg。IoU の低い順。</p>\n')
    for iou, i in rows:
        fp.write(f'<img loading="lazy" src="{i}.jpg" alt="{i} IoU {iou:.3f}">\n')

print(f'{len(rows)} 枚を {DST} に書き出した（IoU 中央値 {np.median([r[0] for r in rows]):.3f}）')
