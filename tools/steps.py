"""steps.mjs が出した各段を画像にして、1枚の説明ページに並べる。

マスクの段は「結果」と「前の段との差分（緑=足した / 赤=消した）」を並べる。
段ごとに何をしたのかが直接見えるようにするのが目的。
複数のソースを渡すと、上のタブで切り替えられる1ページになる。

使い方: python3 tools/steps.py <出力先> <steps.mjs の出力先>...
"""
import html
import json
import os
import shutil
import sys

import cv2
import numpy as np

DST, SRCS = sys.argv[1], sys.argv[2:]

TINT = np.array((255, 0, 255), np.float32)   # BGR: 空と判定された領域に乗せる色
ALPHA = 0.45

# 段がパイプラインのどの局面かを示す見出し。順序そのものが情報なので番号を振る。
PHASE = {
    'src': '入力', 'blue': '手がかり', 'gray': '手がかり', 'rel': '手がかり',
    'seed': 'シード', 'walls': '手がかり', 'classify': '判定',
    'open': '形状', 'close': '形状', 'top': '形状', 'holes': '形状', 'wall': '形状',
    'band': '精緻化', 'refine': '精緻化', 'final': '精緻化',
}


def render(src, out):
    """1ソース分の PNG を書き出し、ページに差し込む行を返す。"""
    meta = json.load(open(f'{src}/steps.json'))
    fw, fh = meta['fw'], meta['fh']
    os.makedirs(out, exist_ok=True)

    rgba = np.fromfile(f"{src}/{meta['steps'][0]['name']}.bin", np.uint8).reshape(fh, fw, 4)
    base = rgba[:, :, :3][:, :, ::-1].copy()          # RGB -> BGR

    def up(a, scale):
        return a if scale == 1 else cv2.resize(a, (fw, fh), interpolation=cv2.INTER_NEAREST)

    def tinted(mask):
        o, m = base.copy(), mask > 0
        o[m] = (ALPHA * TINT + (1 - ALPHA) * o[m]).astype(np.uint8)
        return o

    def diffed(cur, prev):
        add, rem = (cur > 0) & (prev == 0), (cur == 0) & (prev > 0)
        o = (base * 0.22).astype(np.uint8)
        o[(cur > 0) & (prev > 0)] = (150, 80, 150)    # 変わらず空のまま
        o[add] = (80, 255, 80)
        o[rem] = (60, 60, 255)                        # BGR なので赤
        return o, int(add.sum()), int(rem.sum())

    def heat(field, lo, hi):
        return cv2.applyColorMap((np.clip((field - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8),
                                 cv2.COLORMAP_JET)

    rows, prev = [], None
    for s in meta['steps']:
        path, w, h, sc = f"{src}/{s['name']}.bin", s['w'], s['h'], s['scale']
        panels = []
        if s['kind'] == 'rgba':
            panels.append((base, f'{fw}×{fh}'))
        elif s['kind'] == 'field':
            field = np.fromfile(path, np.float32).reshape(h, w)
            panels.append((up(heat(field, s['lo'], s['hi']), sc), f"青 {s['lo']:g} ← → {s['hi']:g} 赤"))
        else:
            mask = up(np.fromfile(path, np.uint8).reshape(h, w), sc)
            panels.append((tinted(mask), f'被覆 {mask.mean() * 100:.1f}%'))
            if s.get('diff') and prev is not None:
                img, add, rem = diffed(mask, prev)
                panels.append((img, '変化なし' if add == rem == 0 else f'+{add} / −{rem}'))
            prev = mask

        imgs = []
        for k, (img, cap) in enumerate(panels):
            cv2.imwrite(f"{out}/{s['name']}_{k}.png", img)
            imgs.append((f"{os.path.basename(out)}/{s['name']}_{k}.png", cap))
        rows.append((s, imgs))
    return meta, rows


def section(meta, rows, first):
    key = html.escape(meta['id'])
    o = [f'<section class="src" id="src-{key}"{"" if first else " hidden"}>']
    for i, (s, imgs) in enumerate(rows):
        phase = PHASE.get(s['name'].split('_', 1)[1], 'モード')
        o.append('<article class="stage">')
        o.append(f'<header><span class="idx">{i:02d}</span>'
                 f'<span class="phase">{phase}</span>'
                 f'<h3>{html.escape(s["label"])}</h3></header>')
        if s.get('purpose'):
            o.append(f'<p class="purpose"><span>目的</span>{html.escape(s["purpose"])}</p>')
        o.append('<div class="plates">')
        for k, (path, cap) in enumerate(imgs):
            role = '結果' if k == 0 or s['kind'] != 'mask' else '前の段との差分'
            o.append(f'<figure><img loading="lazy" src="{path}" alt="{html.escape(s["label"])}">'
                     f'<figcaption><b>{role}</b><span>{html.escape(cap)}</span></figcaption></figure>')
        o.append('</div>')
        o.append(f'<p class="note">{html.escape(s["note"])}</p>')
        o.append('</article>')
    o.append('</section>')
    return '\n'.join(o)


os.makedirs(DST, exist_ok=True)
metas = []
for src in SRCS:
    out = f'{DST}/{json.load(open(f"{src}/steps.json"))["id"]}'
    shutil.rmtree(out, ignore_errors=True)
    metas.append(render(src, out))

CSS = """
:root{
  --ground:#FBFAFC; --surface:#FFFFFF; --rule:#E4E1E9; --text:#191520; --muted:#6A6478;
  --accent:#A9308C; --add:#1F8A4C; --rem:#C2354B; --plate:#0A0B0E;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --ground:#0E1014; --surface:#161922; --rule:#262B36; --text:#E8E6ED; --muted:#8E8899;
  --accent:#D45BB8; --add:#4ADE80; --rem:#FB7185; --plate:#0A0B0E;
}}
:root[data-theme="dark"]{
  --ground:#0E1014; --surface:#161922; --rule:#262B36; --text:#E8E6ED; --muted:#8E8899;
  --accent:#D45BB8; --add:#4ADE80; --rem:#FB7185; --plate:#0A0B0E;
}
body{background:var(--ground);color:var(--text);
  font-family:"Zen Kaku Gothic New","Hiragino Sans","Noto Sans JP",system-ui,sans-serif;
  font-size:15px;line-height:1.85;}
.wrap{max-width:940px;margin:0 auto;padding-block:40px 72px;padding-left:20px;padding-right:20px;}
.mono{font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;}
h1{font-size:30px;line-height:1.35;margin:0 0 10px;font-weight:700;text-wrap:balance;letter-spacing:.01em;}
.lede{color:var(--muted);margin:0 0 32px;max-width:62ch;}
.lede strong{color:var(--text);font-weight:500;}
h2{font-size:19px;margin:0 0 12px;font-weight:700;}
.intro{background:var(--surface);border:1px solid var(--rule);border-radius:10px;
  padding:22px 22px 6px;margin:0 0 36px;}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin:0 0 18px;}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--rule);}
th{color:var(--muted);font-weight:500;font-size:12px;letter-spacing:.06em;text-transform:uppercase;}
td.n{font-variant-numeric:tabular-nums;}
td b{color:var(--accent);}
.dead{color:var(--muted);}
.tabs{display:flex;gap:8px;flex-wrap:wrap;position:sticky;top:env(safe-area-inset-top,0px);
  background:var(--ground);padding-block:12px;margin:0 0 8px;z-index:5;border-bottom:1px solid var(--rule);}
.tabs button{font:inherit;font-size:13px;padding:7px 14px;border-radius:999px;cursor:pointer;
  border:1px solid var(--rule);background:transparent;color:var(--muted);}
.tabs button[aria-selected="true"]{border-color:var(--accent);color:var(--accent);font-weight:500;}
.tabs button:focus-visible{outline:2px solid var(--accent);outline-offset:2px;}
.stage{padding:26px 0;border-bottom:1px solid var(--rule);}
.stage header{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:14px;}
.idx{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:13px;color:var(--muted);}
.phase{font-size:10.5px;letter-spacing:.1em;color:var(--accent);border:1px solid var(--rule);
  border-radius:4px;padding:1px 7px;}
.stage h3{font-size:17px;margin:0;font-weight:700;flex:1 1 240px;}
.purpose{display:flex;gap:10px;align-items:baseline;margin:0 0 14px;font-size:14.5px;
  color:var(--text);border-left:2px solid var(--accent);padding-left:12px;max-width:74ch;}
.purpose span{flex:0 0 auto;font-size:10.5px;letter-spacing:.1em;color:var(--accent);
  transform:translateY(-1px);}
.plates{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,480px));gap:12px;}
figure{margin:0;}
figure img{width:100%;display:block;background:var(--plate);border-radius:6px;border:1px solid var(--rule);}
figcaption{display:flex;justify-content:space-between;gap:10px;padding-top:5px;font-size:11.5px;color:var(--muted);}
figcaption b{font-weight:500;color:var(--text);}
figcaption span{font-family:"JetBrains Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;}
.note{margin:14px 0 0;color:var(--muted);max-width:74ch;font-size:14px;}
.key{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:var(--muted);margin:0 0 28px;}
.key i{font-style:normal;display:inline-flex;align-items:center;gap:6px;}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block;}
@media (max-width:640px){ h1{font-size:24px;} .wrap{padding-block:28px 56px;} }
"""

CUES = """
<table>
<thead><tr><th>手がかり</th><th>薄暮: 空 / 地上</th><th>曇天: 空 / ビル</th></tr></thead>
<tbody>
<tr><td>青優勢度 (B−R)/(B+R)</td><td class="n"><b>+0.47 / −0.26</b></td>
    <td class="n dead">+0.021 / +0.014 — 差がない</td></tr>
<tr><td>輝度</td><td class="n dead">73 / 72 — 差がない</td><td class="n"><b>165 / 115</b></td></tr>
</tbody></table>
<p class="note">薄暮と曇天では、空を見分ける手がかりが入れ替わります。だから固定しきい値は必ずどちらかで破綻する。
代わりに画面上端の画素から「この空の見え方」を推定し、そこからの距離で判定します。
3つの条件を AND で繋ぐと、差のない手がかりは素通りして黙るので、
どれが効くかをシーンが自分で決めます。判定に分岐はありません。</p>
"""

with open(f'{DST}/index.html', 'w', encoding='utf-8') as fp:
    fp.write('<title>空マスクができるまで</title>\n')
    fp.write('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
             'family=Zen+Kaku+Gothic+New:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap">\n')
    fp.write(f'<style>{CSS}</style>\n<div class="wrap">\n')
    fp.write('<h1>空マスクができるまで</h1>\n')
    fp.write('<p class="lede">CNN を使わず、色・輝度・ざらつきの3つの手がかりだけで空を切り出しています。'
             '1フレームぶんの処理を、途中の絵をぜんぶ開いて1段ずつ追いかけます。'
             '各段には<strong>その段が何のためにあるか（目的）</strong>と、'
             'なぜそう作ったかの説明を付けてあります。</p>\n')
    fp.write(f'<div class="intro"><h2>なぜ固定しきい値ではだめなのか</h2>{CUES}</div>\n')
    fp.write('<div class="key">'
             '<i><span class="sw" style="background:#D45BB8"></span>空と判定された領域</i>'
             '<i><span class="sw" style="background:#4ADE80"></span>その段で足した画素</i>'
             '<i><span class="sw" style="background:#FB7185"></span>その段で消した画素</i>'
             '</div>\n')
    fp.write('<div class="tabs" role="tablist">\n')
    for k, (meta, _) in enumerate(metas):
        fp.write(f'<button role="tab" aria-selected="{"true" if k == 0 else "false"}" '
                 f'data-src="{html.escape(meta["id"])}">{html.escape(meta["label"])}</button>\n')
    fp.write('</div>\n')
    for k, (meta, rows) in enumerate(metas):
        fp.write(section(meta, rows, k == 0) + '\n')
    fp.write("""</div>
<script>
 const tabs = [...document.querySelectorAll('.tabs button')];
 tabs.forEach((b) => b.addEventListener('click', () => {
   tabs.forEach((o) => o.setAttribute('aria-selected', String(o === b)));
   document.querySelectorAll('.src').forEach((s) => { s.hidden = s.id !== 'src-' + b.dataset.src; });
   window.scrollTo({ top: 0 });
 }));
</script>
""")

print(f'{len(metas)} ソース・{sum(len(r) for _, r in metas)} 段を {DST}/index.html に書き出した')
