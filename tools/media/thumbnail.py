"""The post's thumbnails (10/24): 1280x720 (YouTube's size), palette B as the video's cards (white
and vermilion on sumi ink), the words few and large enough to read at a phone's thumbnail size.

    python -I tools/media/thumbnail.py OUT_DIR

A  the two finalists face to face (their front portraits), カニの大相撲 / CRAB SUMO over them, VS
B  the video review's close-up of the foot over the straw, 判定は？ / VIDEO REVIEW -- no result given
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_final as MF  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(os.environ.get("JUMPER_RUNS", str(ROOT.parent / "jumper-runs"))) / "blender"
W, H = 1280, 720
BOLD = "C:/Windows/Fonts/seguibl.ttf"          # Segoe UI Black


def _title(d, cx, top, size, rule=True):
    """カニの大相撲 in the white brush face, the vermilion rule under it; returns the rule's bottom."""
    f = ImageFont.truetype(MF.BRUSH, size)
    tw = d.textlength("カニの大相撲", font=f)
    d.text((cx - tw / 2, top), "カニの大相撲", font=f, fill=MF.INK, stroke_width=max(3, size // 22),
           stroke_fill=MF.PANEL)
    bb = d.textbbox((cx - tw / 2, top), "カニの大相撲", font=f)
    y = bb[3] + size * 0.14
    if rule:
        d.rectangle((cx - size * 0.9, y, cx + size * 0.9, y + max(5, size * 0.07)), fill=MF.ACCENT)
    return y + max(5, size * 0.07)


def thumb_versus(out: Path) -> None:
    img = Image.new("RGB", (W, H), MF.PANEL)
    half = (W // 2, H)
    for side, path, mirror in (("east", RUNS / "portraits_hero3_front" / "portrait_robot_95.png", False),
                               ("west", RUNS / "portraits_hero3_front" / "portrait_opponent_95.png", True)):
        p = MF._portrait(str(path), half, mirror=mirror)
        if p is None:
            continue
        p = MF._fade(p, "right" if side == "east" else "left", 0.30)
        img.paste(p, (0 if side == "east" else W // 2, 0))
    # darken the top for the title and the bottom for the line
    shade = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(shade)
    for y in range(H):
        a = max(0.0, 1 - y / (H * 0.42)) * 215 + max(0.0, (y - H * 0.72) / (H * 0.28)) * 200
        sd.line([(0, y), (W, y)], fill=int(min(235, a)))
    img = Image.composite(Image.new("RGB", (W, H), MF.PANEL), img, shade)
    d = ImageDraw.Draw(img)
    _title(d, W / 2, H * 0.02, int(H * 0.17))
    vs = ImageFont.truetype(BOLD, int(H * 0.20))
    d.text(((W - d.textlength("VS", font=vs)) / 2, H * 0.40), "VS", font=vs, fill=MF.INK, stroke_width=5,
           stroke_fill=(0, 0, 0))
    big = ImageFont.truetype(BOLD, int(H * 0.085))
    s = "CRAB SUMO"
    d.text(((W - d.textlength(s, font=big)) / 2, H * 0.78), s, font=big, fill=MF.INK, stroke_width=3,
           stroke_fill=(0, 0, 0))
    sub = ImageFont.truetype(MF.LATIN, int(H * 0.042))
    s = "AI-trained robots · the final bout"
    d.text(((W - d.textlength(s, font=sub)) / 2, H * 0.895), s, font=sub, fill=(214, 216, 224), stroke_width=2,
           stroke_fill=(0, 0, 0))
    img.save(out, quality=92)


def thumb_review(out: Path) -> None:
    src = Image.open(RUNS / "verdict_hero3" / "0049.png").convert("RGB")
    img = src.resize((W, H), Image.LANCZOS)
    shade = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(shade)
    for y in range(H):
        sd.line([(0, y), (W, y)], fill=int(max(0.0, 1 - y / (H * 0.45)) * 200))
    img = Image.composite(Image.new("RGB", (W, H), MF.PANEL), img, shade)
    d = ImageDraw.Draw(img)
    # the ring round the foot (verdict.json's mark for the decision frame)
    import json
    v = json.loads((RUNS / "verdict_hero3" / "verdict.json").read_text(encoding="utf-8"))
    mx, my = v["marks"][v["decision_png"] - 1]
    R = H * 0.10
    d.ellipse((mx * W - R, my * H - R, mx * W + R, my * H + R), outline=(232, 48, 40), width=9)
    y = _title(d, W * 0.5, H * 0.015, int(H * 0.13))
    q = ImageFont.truetype(MF.BRUSH, int(H * 0.15))
    s = "判定は？"
    d.text(((W - d.textlength(s, font=q)) / 2, y + H * 0.02), s, font=q, fill=MF.ACCENT, stroke_width=6,
           stroke_fill=MF.PANEL)
    big = ImageFont.truetype(BOLD, int(H * 0.07))
    s = "CRAB SUMO · VIDEO REVIEW"
    d.text(((W - d.textlength(s, font=big)) / 2, H * 0.86), s, font=big, fill=MF.INK, stroke_width=4,
           stroke_fill=(0, 0, 0))
    img.save(out, quality=92)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    thumb_versus(out / "thumbnail_A_versus.jpg")
    thumb_review(out / "thumbnail_B_review.jpg")
    print("thumbnails ->", out)
