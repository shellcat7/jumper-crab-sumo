"""verdict_shot.py's frames to the replay judgement's clip (10/23): the slow motion up to the decision,
then the picture held there with a red ring drawn round the loser's foot and the call written as the
broadcast writes it -- 判定リプレイ at the top, the reason under it in Japanese and English. The ring
and the words stay clear of the bottom tenth (players put their controls there).

    python verdict_clip.py VERDICT_DIR CARD.json OUT.mp4 [HOLD_S]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import compose_video as C


def main(vdir: str, card_path: str, out: str, hold: float = 2.8) -> None:
    import imageio_ffmpeg

    v = json.loads((Path(vdir) / "verdict.json").read_text(encoding="utf-8"))
    card = json.loads(Path(card_path).read_text(encoding="utf-8"))
    loser = card["east"] if v["foot"].startswith("robot/") else card["west"]
    files = [Path(vdir) / f"{i:04d}.png" for i in range(1, v["decision_png"] + 1)]
    W, H = Image.open(files[0]).size
    fps = 30
    writer = imageio_ffmpeg.write_frames(out, (W, H), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                         output_params=["-crf", "16"], macro_block_size=8)
    writer.send(None)
    mx, my = v["marks"][v["decision_png"] - 1]
    still = Image.open(files[-1]).convert("RGB")
    n_hold = int(hold * fps)
    for j in range(len(files) + n_hold):
        img = Image.open(files[j]).convert("RGB") if j < len(files) else still.copy()
        ink = C.Ink((W, H))
        # the mark, as the replay's: a red dot, 判定リプレイ, VIDEO REVIEW under it
        f = C.font(C.HEAVY, H * 0.045)
        ink.text((W / 2, H * 0.075), "判定リプレイ", f, C.YELLOW, 1.0, anchor="ms", weight=0.02)
        ink.text((W / 2, H * 0.115), "VIDEO REVIEW", C.font(C.LATIN, H * 0.026), C.WHITE, 1.0, anchor="ms",
                 weight=0.0, edge=0.08)
        d = ImageDraw.Draw(ink.face)
        r = H * 0.011
        tw = d.textlength("判定リプレイ", font=f)
        cx, cy = W / 2 - tw / 2 - r * 3, H * 0.075 - f.size * 0.35
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 40, 40, 255))
        if j >= len(files):
            u = (j - len(files)) / fps
            a = C.ease(u / 0.35)
            # the ring round the foot, drawn in
            R = H * 0.085
            px, py = mx * W, my * H
            box = [px - R, py - R, px + R, py + R]
            d.arc(box, -90, -90 + 360 * a, fill=(232, 48, 40, 255), width=max(4, int(H * 0.007)))
            # the call
            k = C.ease((u - 0.3) / 0.5)
            if k > 0:
                ink.text((W / 2, H * 0.205), f"{loser['shikona']}の足が俵の外に", C.font(C.HEAVY, H * 0.062), C.WHITE, k,
                         anchor="ms", weight=0.02)
                en = f"{loser.get('romaji', '')}'s foot is down outside the straw"
                ink.text((W / 2, H * 0.255), en, C.font(C.LATIN, H * 0.034), C.YELLOW, k, anchor="ms", weight=0.0,
                         edge=0.08)
        writer.send(np.asarray(ink.onto(img).convert("RGB")))
    writer.close()
    print(f"[verdict] {len(files)} + {n_hold} frames ({(len(files) + n_hold) / fps:.1f} s) -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 2.8)
