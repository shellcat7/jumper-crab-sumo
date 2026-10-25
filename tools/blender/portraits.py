"""The champions' portraits for the hall's walls (10/20), from portrait_renders.py's studio
renders: each a made-up champion -- a shikona of our own (crabs and claws, no real
wrestler's), the tournament it won -- the render cropped tall, graded a little warm or cool
like an old colour photograph, and the name written down its side in brush letters. 36
portraits (9 on each wall), from however many renders there are, none twice the same.

    python -I portraits.py RENDERS_DIR OUT_DIR
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

BRUSH = "C:/Windows/Fonts/HGRGY.TTC"        # HG行書体: a brush hand
MINCHO = "C:/Windows/Fonts/HGRME.TTC"

# made-up shikona (crab and claw words, mountains and seas): no real wrestler's
NAMES = ["鋏ノ峰", "鋏ノ嶺", "鋏ノ富士", "回ノ里", "総ノ山", "正面山", "甲羅錦", "八足山", "双鋏",
         "横歩川", "磯ノ花", "潮鋏", "大甲羅", "砂嵐", "蟹ヶ嶽", "波ノ鋏", "鋏錦", "岩甲",
         "千鋏", "赤甲山", "浜ノ里", "鋏龍", "雲上山", "机上海", "渦潮", "鋏浦", "潮騒",
         "甲斐鋏", "白砂", "岬ノ海", "蒼鋏", "朱甲", "海老名山", "磯風", "鋏光", "沖ノ鋏"]
BASHO = ["初場所", "春場所", "夏場所", "名古屋場所", "秋場所", "九州場所", "十月場所"]
KANJI = "〇一二三四五六七八九"


def kanji(n: int) -> str:
    t, o = divmod(n, 10)
    return ("" if t == 0 else ("" if t == 1 else KANJI[t]) + "十") + (KANJI[o] if o else "")


def portrait(src: Image.Image, name: str, caption: str, rng: random.Random, size=(360, 480)) -> Image.Image:
    W, H = size
    # a tall crop round the wrestler (a little off-centre, a little zoomed, varied)
    s = rng.uniform(0.78, 0.95)
    cw, ch = int(src.width * s), int(src.width * s * H / W)
    ch = min(ch, src.height)
    cw = int(ch * W / H)
    x0 = int((src.width - cw) * rng.uniform(0.35, 0.65))
    y0 = int((src.height - ch) * rng.uniform(0.3, 0.7))
    im = src.crop((x0, y0, x0 + cw, y0 + ch)).resize((W, H), Image.LANCZOS)
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    # graded as an old colour photograph: warmer or cooler, a little faded, a soft vignette
    r, g, b = im.split()
    warm = rng.uniform(-0.06, 0.10)
    r = r.point(lambda v: min(255, int(v * (1 + warm))))
    b = b.point(lambda v: min(255, int(v * (1 - warm))))
    im = Image.merge("RGB", (r, g, b))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.75, 0.95))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.9, 1.05))
    vig = Image.new("L", (W, H), 0)
    ImageDraw.Draw(vig).ellipse((-W * 0.25, -H * 0.15, W * 1.25, H * 1.15), fill=255)
    vig = vig.filter(ImageFilter.GaussianBlur(W * 0.15))
    im = Image.composite(im, Image.new("RGB", (W, H), (20, 14, 10)), vig)
    # the name down the right side in brush letters, the tournament in small letters beside it
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(BRUSH, int(W * 0.15))
    fs = ImageFont.truetype(MINCHO, int(W * 0.055))
    x, y = W * 0.86, H * 0.06
    for ch_ in name:
        d.text((x, y), ch_, font=f, fill=(250, 246, 236), anchor="mt", stroke_width=2, stroke_fill=(40, 30, 20))
        y += f.size * 1.02
    y = H * 0.07
    for ch_ in caption:
        d.text((W * 0.73, y), ch_, font=fs, fill=(235, 220, 170), anchor="mt", stroke_width=1, stroke_fill=(40, 30, 20))
        y += fs.size * 1.05
    return im


def main(src_dir: str, out_dir: str) -> None:
    srcs = sorted(Path(src_dir).rglob("portrait_*.png"))
    if not srcs:
        raise SystemExit(f"no renders in {src_dir}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261020)
    imgs = [Image.open(p).convert("RGB") for p in srcs]
    for i, name in enumerate(NAMES):
        cap = f"{rng.choice(BASHO)}優勝"
        portrait(imgs[i % len(imgs)], name, cap, rng).save(out / f"champion_{i:02d}.jpg", quality=92)
    print(f"{len(NAMES)} portraits from {len(srcs)} renders -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
