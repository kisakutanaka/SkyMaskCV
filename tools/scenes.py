"""`img/` の素材から回帰セットの元画像を作る。

静止画はそのまま、動画は `STEP` フレームおきに jpg として書き出す。
出力先はデータセットと同じ配置（split 名 = scenes）にしてあるので、
prep.py / gt.py がそのまま使える。

使い方: python3 tools/scenes.py
"""
import os

import cv2

ROOT = os.path.join(os.path.dirname(__file__), '..')
DST = f'{ROOT}/dataset/images/scenes'
STEP = 20
QUALITY = [cv2.IMWRITE_JPEG_QUALITY, 95]

os.makedirs(DST, exist_ok=True)

for name in ('cloudy01', 'cloudy02'):
    cv2.imwrite(f'{DST}/{name}.jpg', cv2.imread(f'{ROOT}/img/{name}.png'), QUALITY)

cap = cv2.VideoCapture(f'{ROOT}/img/IMG_4476.mp4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
kept = 0
for f in range(0, total, STEP):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imwrite(f'{DST}/f{f:03d}.jpg', frame, QUALITY)
    kept += 1

print(f'静止画2枚 + 動画{kept}枚 を {DST} に書き出した')
