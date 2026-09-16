"""回帰セットの元画像を作る。

`img/` の静止画はそのまま、動画は `STEP` フレームおきに jpg として書き出す。
加えて「青空にはっきりした白い雲」の構図をデータセットから足す（下記）。
出力先はデータセットと同じ配置（split 名 = scenes）にしてあるので、
prep.py / gt.py がそのまま使える。

使い方: python3 tools/scenes.py
"""
import os
import shutil

import cv2

ROOT = os.path.join(os.path.dirname(__file__), '..')
DST = f'{ROOT}/dataset/images/scenes'
STEP = 20
QUALITY = [cv2.IMWRITE_JPEG_QUALITY, 95]

# 「青空にはっきりした白い雲」は必達要件なのに、img/ の素材にこの構図がない。
# 指標に出ない要件は守れず、実際に処理解像度の変更で静かに壊れていたのを
# 数週間気づけなかった。そこでデータセットから足す。
#
# 選んだのは構図だけを見た結果で、現手法の出来は一切見ていない（自分の変更が
# 良く見えるテストを作らないため）。条件は、正解ラベルの空領域のうち
# 青空らしい画素（青優勢度>0.15）と雲らしい画素（青優勢度<0.08 かつ 輝度>170）が
# それぞれ15%以上あり、空が画面の15%以上を占めること。552枚が該当し、
# そのうち都市・建物が写っているものを選んだ。
# ラベルは既にある教師の出力をそのまま流用する（同じ教師なので作り直す意味がない）。
SKY_CLOUD = {
    'train': ['6618503f6b98d2c2', '468b1ef69e5bb8fa', 'abcdd19268ad5bd1',
              '0c074e1fa53d8623', '7b086b14000087b9', '0a4012883d88bdcc',
              '768842cae291f938'],
    'val': ['1b8d1d161bf4396b'],     # 多モードを入れる根拠にした画像
}
LBL = f'{ROOT}/dataset/labels_skyseg/scenes'

os.makedirs(DST, exist_ok=True)
os.makedirs(LBL, exist_ok=True)

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

added = 0
for split, ids in SKY_CLOUD.items():
    for i in ids:
        shutil.copyfile(f'{ROOT}/dataset/images/{split}/{i}.jpg', f'{DST}/{i}.jpg')
        shutil.copyfile(f'{ROOT}/dataset/labels_skyseg/{split}/{i}.png', f'{LBL}/{i}.png')
        added += 1

print(f'静止画2枚 + 動画{kept}枚 + 青空と白い雲{added}枚 を {DST} に書き出した')
