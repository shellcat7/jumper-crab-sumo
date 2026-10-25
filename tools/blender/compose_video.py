"""Blender's frames to an mp4 with sumo-broadcast graphics (10/19).

The user asked for the introduction the way a television sumo broadcast shows it -- the
two shikona set huge down the screen over the live picture, rank, birthplace and stable
in a column beside each, the record in another -- and for the result to be designed, not
dropped on (no panels, no boxes). The layout and the colours follow that convention:
heavy Mincho (HG明朝E), white names, yellow rank and record, a black edge and a soft
shadow, nothing behind them. No broadcaster's logo, name or marks are used.

    python compose_video.py DUMP.npz FRAMES_DIR OUT.mp4 --card card.json [--intro 8] [--single]

The card (JSON; the record is shown in kanji, 八勝七敗):

    {"title": "十月場所", "bout": "結びの一番",
     "east": {"shikona": "鋏ノ富士", "rank": "関脇", "origin": "雲上", "heya": "鋏浦", "record": [8, 7]},
     "west": {...}, "h2h": [86, 86]}      # h2h from the east's side; [] for a first meeting

Timeline: during the --intro seconds (Blender held the robots still while its camera
closed in) the names; then the bout with the two names small in the top corners; at the
decision (the dump's caption 勝負あり　西の勝ち　引き落とし) the result; with --single a
half-speed replay of the decision (from another angle if rendered) and the result again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HEAVY = "C:/Windows/Fonts/HGRME.TTC"           # HG明朝E: names, ranks, records, kimarite
LATIN = "C:/Windows/Fonts/seguisb.ttf"          # Segoe UI Semibold: the English under them (10/23)

#: kimarite as the gyoji calls them -> (romaji, English), for viewers abroad (10/23, the user:
#: the broadcast graphics need English too)
KIMARITE_EN = {
    "押し出し": ("Oshidashi", "push-out"), "寄り切り": ("Yorikiri", "force-out"), "引き落とし": ("Hikiotoshi", "pull-down"),
    "足取り": ("Ashitori", "the leg pick"), "とったり": ("Tottari", "arm-bar throw"), "突き出し": ("Tsukidashi", "thrust-out"),
    "突き倒し": ("Tsukitaoshi", "thrust-down"), "押し倒し": ("Oshitaoshi", "push-down"), "寄り倒し": ("Yoritaoshi", "force-out-down"),
    "送り出し": ("Okuridashi", "push-out from behind"), "はたき込み": ("Hatakikomi", "slap-down"),
    "突き落とし": ("Tsukiotoshi", "thrust-down"), "上手投げ": ("Uwatenage", "overarm throw"),
    "下手投げ": ("Shitatenage", "underarm throw"), "うっちゃり": ("Utchari", "backward pivot throw"),
    "勇み足": ("Isamiashi", "stepped out"), "腰砕け": ("Koshikudake", "fell backward"),
}

WHITE = (255, 255, 255)
YELLOW = (250, 222, 64)
SHADOW = (0, 0, 0)

_FONTS: dict = {}


def font(path, size):
    key = (path, int(size))
    if key not in _FONTS:
        try:
            _FONTS[key] = ImageFont.truetype(path, int(size))
        except OSError:
            _FONTS[key] = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", int(size))
    return _FONTS[key]


def ease(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def window(t, t0, t1, fade=0.4):
    """0..1: a graphic shown from t0 to t1, fading in and out over `fade` s."""
    if t < t0 or t > t1:
        return 0.0
    return min(ease((t - t0) / fade), ease((t1 - t) / fade))


_DIGITS = "〇一二三四五六七八九"


def kanji(n: int) -> str:
    """0..9999 in kanji, as a broadcast writes a record (十一, 八十六, 千三百五). 10/23: the
    head-to-heads of 1024 rings passed a thousand (1305) and the hundreds digit overflowed."""
    n = int(n)
    if n < 10:
        return _DIGITS[n]
    out = ""
    k, n = divmod(n, 1000)
    if k:
        out += ("" if k == 1 else _DIGITS[k]) + "千"
    h, n = divmod(n, 100)
    if h:
        out += ("" if h == 1 else _DIGITS[h]) + "百"
    t, o = divmod(n, 10)
    if t:
        out += ("" if t == 1 else _DIGITS[t]) + "十"
    if o:
        out += _DIGITS[o]
    return out


class Ink:
    """Text drawn as a broadcast draws it over a picture: the letters thickened, a black
    edge round them and a soft shadow under, onto three layers composited in order."""

    def __init__(self, size):
        self.shadow = Image.new("RGBA", size, (0, 0, 0, 0))
        self.edge = Image.new("RGBA", size, (0, 0, 0, 0))
        self.face = Image.new("RGBA", size, (0, 0, 0, 0))

    def text(self, xy, s, f, fill, alpha, anchor="la", weight=0.035, edge=0.10, edge_fill=SHADOW):
        if alpha <= 0:
            return
        a = int(255 * alpha)
        w = max(1, int(f.size * weight))          # thickening, in the face's own colour
        e = w + max(2, int(f.size * edge))        # the edge outside it (black; white round a black mark)
        for layer, dx, fill_, sw in ((self.shadow, f.size * 0.05, (*SHADOW, int(a * 0.8)), e),
                                     (self.edge, 0, (*edge_fill, a), e),
                                     (self.face, 0, (*fill, a), w)):
            ImageDraw.Draw(layer).text((xy[0] + dx, xy[1] + dx), s, font=f, fill=fill_, anchor=anchor,
                                       stroke_width=sw, stroke_fill=fill_)

    def column(self, x, y, s, f, fill, alpha, gap=1.0, **kw):
        """Characters down a column centred on x (a vertical line of text)."""
        step = f.size * gap
        for i, ch in enumerate(s):
            if ch in "　 ":
                continue
            self.text((x, y + i * step), ch, f, fill, alpha, anchor="mt", **kw)
        return y + len(s) * step

    def onto(self, frame: Image.Image) -> Image.Image:
        out = frame.convert("RGBA")
        out.alpha_composite(self.shadow.filter(ImageFilter.GaussianBlur(radius=max(2, frame.height // 180))))
        out.alpha_composite(self.edge)
        out.alpha_composite(self.face)
        return out


def names_overlay(ink, W, H, card, alpha):
    """The two wrestlers as the broadcast introduces them: the shikona huge down the
    screen either side of the ring; outside it the rank, then birthplace and stable; inside
    it the record. East on the left (our main camera looks from the shomen side)."""
    big = font(HEAVY, H * 0.165)
    mid = font(HEAVY, H * 0.072)
    small = font(HEAVY, H * 0.060)
    for side, (x_out, x_name, x_in) in (("east", (0.20, 0.31, 0.415)), ("west", (0.80, 0.69, 0.585))):
        w = card[side]
        n = len(w["shikona"])
        gap = min(1.0, 0.86 * H / max(1, n) / big.size)
        top = H * 0.07
        ink.column(W * x_name, top, w["shikona"], big, WHITE, alpha, gap=gap)
        y = ink.column(W * x_out, top + H * 0.005, w.get("rank", ""), mid, YELLOW, alpha, gap=1.02)
        y += H * 0.035
        for word in (w.get("origin", ""), w.get("heya", "")):
            if word:
                y = ink.column(W * x_out, y, word, small, YELLOW, alpha, gap=1.02) + H * 0.025
        rec = w.get("record")
        if rec:
            ink.column(W * x_in, H * 0.33, f"{kanji(rec[0])}勝{kanji(rec[1])}敗", small, YELLOW, alpha, gap=1.02)
        if w.get("romaji"):
            xe = W * x_name
            ink.text((xe, H * 0.700), w["romaji"].upper(), font(LATIN, H * 0.040), WHITE, alpha, anchor="ms",
                     weight=0.0, edge=0.08)
            sub = " · ".join(x for x in (w.get("rank_en", ""), f"{rec[0]}–{rec[1]}" if rec else "") if x)
            ink.text((xe, H * 0.740), sub, font(LATIN, H * 0.030), YELLOW, alpha, anchor="ms", weight=0.0, edge=0.08)


def h2h_line(ink, W, H, card, alpha):
    """A line along the bottom: their record against each other, or a first meeting."""
    h = card.get("h2h")
    f = font(HEAVY, H * 0.055)
    if not h:
        ink.text((W / 2, H * 0.835), "初顔合わせ", f, YELLOW, alpha, anchor="ms")
        return
    e, w = card["east"]["shikona"], card["west"]["shikona"]
    s = f"対戦成績　{e}　{kanji(h[0])}勝　{w}　{kanji(h[1])}勝"
    ink.text((W / 2, H * 0.835), s, f, WHITE, alpha, anchor="ms", weight=0.02)
    re_, rw = card["east"].get("romaji"), card["west"].get("romaji")
    if re_ and rw:
        en = f"Head to head:  {re_} {h[0]:,}  –  {h[1]:,} {rw}"
        ink.text((W / 2, H * 0.885), en, font(LATIN, H * 0.032), WHITE, alpha, anchor="ms", weight=0.0, edge=0.08)


def title_line(ink, W, H, card, alpha):
    f = font(HEAVY, H * 0.048)
    s = "　".join(x for x in (card.get("title", ""), card.get("bout", "")) if x)
    ink.text((W * 0.04, H * 0.06), s, f, YELLOW, alpha, weight=0.02)
    en = card.get("title_en", "")
    if en:
        ink.text((W * 0.04, H * 0.06 + f.size * 1.25), en, font(LATIN, H * 0.030), WHITE, alpha, weight=0.0, edge=0.08)


def corner_names(ink, W, H, card, alpha):
    """During the bout: the two names small in the top corners, east left, west right."""
    f = font(HEAVY, H * 0.050)
    tag = font(HEAVY, H * 0.034)
    ink.text((W * 0.035, H * 0.055), "東", tag, YELLOW, alpha, weight=0.02)
    ink.text((W * 0.035 + tag.size * 1.25, H * 0.045), card["east"]["shikona"], f, WHITE, alpha, weight=0.02)
    ink.text((W * 0.965, H * 0.055), "西", tag, YELLOW, alpha, anchor="ra", weight=0.02)
    ink.text((W * 0.965 - tag.size * 1.6, H * 0.045), card["west"]["shikona"], f, WHITE, alpha, anchor="ra",
             weight=0.02)
    small = font(LATIN, H * 0.022)
    if card["east"].get("romaji"):
        ink.text((W * 0.035, H * 0.118), f"EAST  {card['east']['romaji'].upper()}", small, WHITE, alpha, weight=0.0, edge=0.08)
    if card["west"].get("romaji"):
        ink.text((W * 0.965, H * 0.118), f"{card['west']['romaji'].upper()}  WEST", small, WHITE, alpha, anchor="ra",
                 weight=0.0, edge=0.08)


def result(ink, W, H, card, side, kimarite, alpha):
    """The result over a dark fade along the bottom: 決まり手 small in yellow, the
    kimarite large, and under it the winner (white circle) and the loser (black circle)."""
    win = card["east" if side == "東" else "west"]["shikona"]
    lose = card["west" if side == "東" else "east"]["shikona"]
    loser_side = "西" if side == "東" else "東"
    ink.text((W / 2, H * 0.605), "決まり手", font(HEAVY, H * 0.040), YELLOW, alpha, anchor="ms", weight=0.02)
    ink.text((W / 2, H * 0.740), kimarite, font(HEAVY, H * 0.120), WHITE, alpha, anchor="ms", weight=0.03)
    ro, en = KIMARITE_EN.get(kimarite, ("", ""))
    wr = card["east" if side == "東" else "west"].get("romaji", "")
    if ro:
        s = f"{ro.upper()} — {en}" + (f"   ·   {wr} wins" if wr else "")
        ink.text((W / 2, H * 0.895), s, font(LATIN, H * 0.032), YELLOW, alpha, anchor="ms", weight=0.0, edge=0.08)
    f = font(HEAVY, H * 0.052)
    d = ImageDraw.Draw(ink.face)
    ink.text((W / 2 - H * 0.04, H * 0.840), f"○ {win}（{side}）", f, WHITE, alpha, anchor="rs", weight=0.02)
    # the loser: a black circle edged in white, then the name a little greyer
    x = W / 2 + H * 0.04
    ink.text((x, H * 0.840), "●", f, (12, 12, 12), alpha, anchor="ls", weight=0.0, edge=0.06, edge_fill=WHITE)
    x += d.textlength("● ", font=f)
    ink.text((x, H * 0.840), f"{lose}（{loser_side}）", f, (215, 215, 215), alpha, anchor="ls", weight=0.02)


def dark_fade(frame: Image.Image, alpha: float, top=0.50) -> Image.Image:
    """A soft dark band rising from the bottom, for the result to sit on."""
    if alpha <= 0:
        return frame
    W, H = frame.size
    y0 = int(H * top)
    ramp = np.linspace(0.0, 0.78, H - y0)[:, None] ** 1.3 * alpha
    arr = np.asarray(frame.convert("RGB")).astype(np.float32)
    arr[y0:] *= (1.0 - ramp[:, :, None])
    return Image.fromarray(arr.astype(np.uint8))


def replay_mark(ink, W, H, alpha):
    f = font(HEAVY, H * 0.045)
    ink.text((W / 2, H * 0.075), "リプレイ", f, YELLOW, alpha, anchor="ms", weight=0.02)
    ink.text((W / 2, H * 0.115), "REPLAY", font(LATIN, H * 0.026), WHITE, alpha, anchor="ms", weight=0.0, edge=0.08)
    d = ImageDraw.Draw(ink.face)
    r = H * 0.011
    tw = d.textlength("リプレイ", font=f)
    cx, cy = W / 2 - tw / 2 - r * 3, H * 0.075 - f.size * 0.35
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 40, 40, int(255 * alpha)))


def compose(frame: Image.Image, t: float, intro: float, card: dict, cap: str, cap_since: float,
            decision: tuple | None, decision_t: float, replay: bool = False) -> Image.Image:
    del cap, cap_since          # the gyoji's calls are not written on the screen
    W, H = frame.size
    ink = Ink((W, H))
    if t < intro:
        title_line(ink, W, H, card, window(t, 0.3, 2.4))
        names_overlay(ink, W, H, card, window(t, 2.2, intro - 0.2, fade=0.6))
        h2h_line(ink, W, H, card, window(t, 3.4, intro - 0.2, fade=0.6))
        return ink.onto(frame).convert("RGB")
    k = 0.0
    if decision is not None and t >= decision_t:
        k = ease((t - decision_t) / 0.6)
    frame = dark_fade(frame, k)
    if replay:
        replay_mark(ink, W, H, 1.0)
    else:
        corner_names(ink, W, H, card, min(1.0, ease((t - intro) / 0.6)) * (1 - k))
    if k > 0:
        side, kim = decision
        result(ink, W, H, card, side, kim, k)
    return ink.onto(frame).convert("RGB")


def main() -> None:
    import imageio_ffmpeg

    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--card", required=True)
    ap.add_argument("--intro", type=float, default=0.0)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--replay-frames", default="", help="the replay rendered from another angle (Blender --camera)")
    ap.add_argument("--replay-first", type=int, default=0, help="the dump frame of that folder's first PNG")
    ap.add_argument("--single", action="store_true",
                    help="one bout: cut 2.5 s after the decision, then a half-speed replay of its last 4 s")
    a = ap.parse_args()
    D = np.load(a.dump, allow_pickle=False)
    card = json.loads(Path(a.card).read_text(encoding="utf-8"))
    caps = [str(c) for c in D["caption"]]
    files = sorted(Path(a.frames).glob("*.png"))
    if not files:
        raise SystemExit(f"no PNGs in {a.frames}")
    W, H = Image.open(files[0]).size
    writer = imageio_ffmpeg.write_frames(a.out, (W, H), fps=a.fps, codec="libx264", pix_fmt_out="yuv420p",
                                         output_params=["-crf", "16"], macro_block_size=8)
    writer.send(None)
    n_intro = int(round(a.intro * a.fps))
    # the frame where the (first) decision is called, for --single
    dec_i = next((n_intro + k for k, c in enumerate(caps) if c.startswith("勝負あり")), None)
    order = list(range(len(files)))
    replay_from = None
    if a.single and dec_i is not None:
        end = min(len(files), dec_i + int(2.5 * a.fps))
        # the next bout's 見合って (the wrestlers back on their marks): cut before it
        nxt = next((n_intro + k for k, c in enumerate(caps) if n_intro + k > dec_i and c == "見合って"), None)
        if nxt is not None:
            end = min(end, nxt - 3)
        r0 = max(n_intro, dec_i - int(4.0 * a.fps))
        order = list(range(end))
        replay_from = len(order)
        order += [j for j in range(r0, min(len(files), dec_i + int(0.5 * a.fps))) for _ in (0, 1)]   # half speed
        order += [order[-1]] * int(2.0 * a.fps)                                                       # hold
    last, since, decision, decision_t = "", 0.0, None, 0.0
    rep_dec_t = None
    rfiles = sorted(Path(a.replay_frames).glob("*.png")) if a.replay_frames else []
    timeline = []
    for n, i in enumerate(order):
        t = n / a.fps
        replay = replay_from is not None and n >= replay_from
        k = i - n_intro
        cap = caps[min(max(k, 0), len(caps) - 1)] if k >= 0 else ""
        if not replay:
            if cap != last:
                last, since = cap, t
                if cap == "見合って" and not a.single:
                    decision = None          # the next bout: the kimarite card goes
                                             # (one bout alone keeps it for the replay)
            if decision is None and cap.startswith("勝負あり"):
                parts = cap.replace("　", " ").split()
                side = "東" if "東" in parts[1] else "西"
                decision, decision_t = (side, parts[2] if len(parts) > 2 else ""), t
            dt = decision_t
        else:
            # the replay shows the card again when it reaches the decision
            if rep_dec_t is None and i >= dec_i:
                rep_dec_t = t
            dt = rep_dec_t if rep_dec_t is not None else 1e9
        src = files[i]
        if replay and rfiles:
            r = (i - n_intro) - a.replay_first        # the same moment from the other camera
            if 0 <= r < len(rfiles):
                src = rfiles[r]
        img = compose(Image.open(src).convert("RGB"), t, a.intro, card, cap, since, decision, dt, replay)
        writer.send(np.asarray(img))
        timeline.append([int(k), int(replay)])
    writer.close()
    # the timeline for the sound (sound_design.py): each output frame's dump frame (-1 in the
    # intro) and whether it is the replay
    Path(a.out).with_suffix(".timeline.json").write_text(
        json.dumps({"fps": a.fps, "frames": timeline}), encoding="utf-8")
    print(f"[compose] {len(order)} frames ({len(order) / a.fps:.1f} s, {W}x{H}) -> {a.out}")


if __name__ == "__main__":
    main()
