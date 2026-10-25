"""Cut the submission video (10/21): title cards and clips, with their sound, faded in and out.

    python -I tools/media/make_final.py final.json out.mp4

final.json:
    {"size": [1920, 1080], "fps": 30,
     "items": [
        {"card": ["鋏の大相撲", "十月場所"], "seconds": 3, "sound": "hyoshigi"},
        {"clip": "media/raw/....mp4", "start": 0.0, "seconds": 12.0, "text": "学習 0 回"},
        {"clip": "media/raw/....mp4", "start": 3.0, "seconds": 4.0, "speed": 0.5, "mute": true}
     ]}

Every item becomes a segment of the same size, rate and sound format (1920x1080, 30 fps,
48 kHz stereo AAC), faded in and out (`fade`, default 0.25 s, 0 for a hard cut); the segments
are joined without re-encoding. A clip keeps its own sound unless `mute` (or it has none):
then the hall's murmur, synthesized like everything else (tools/blender/sound_design.py: no
recording, no music, no voice), quiet under it. A card is drawn with the brush face of the
broadcast graphics; its `sound` is "hyoshigi" (the clappers), "murmur", "cheer" (applause, 10/23;
a clip without sound takes it too) or none. `text` lays a caption across the bottom of a clip (the
process footage: "学習 100 回"); `zoom` crops a clip from the top centre and scales it back up.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")
BRUSH = "C:/Windows/Fonts/HGRGY.TTC"
GOTHIC = "C:/Windows/Fonts/BIZ-UDGothicB.ttc"
SR = 48000
sys.path.insert(0, str(ROOT / "tools" / "blender"))
import sound_design as sd  # noqa: E402  (the synthesized hall)


def run(cmd: list[str]) -> None:
    subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", *cmd], check=True)


def has_audio(path: str) -> bool:
    out = subprocess.run([str(FFPROBE), "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return bool(out.strip())


def write_wav(path: Path, x: np.ndarray) -> None:
    x = np.clip(x, -1, 1)
    st = np.stack([x, x], 1) if x.ndim == 1 else x
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


def bed(seconds: float, kind: str | None) -> np.ndarray:
    """A card's or a muted clip's sound at 48 kHz, stereo in the hall's reverberation (10/23: the
    synthesized voices and hands of sound_design, not the noise band that sounded like rain): the
    murmur, the clappers on it, the cheer (applause and voices settling), or silence."""
    sd.SR = SR
    n = int(seconds * SR)
    if kind is None:
        return np.zeros((n, 2))
    x = np.zeros(n)                         # 10/23 21:45: no voices (the user: frightening); the music carries it
    if kind == "hyoshigi":
        x = x + sd.hyoshigi(n, [0.25, 0.75] + list(1.4 + np.cumsum(np.geomspace(0.25, 0.06, 10))), 0.5)
    if kind == "cheer":                     # the champion (10/23): the hall applauding, slowly settling
        lv = np.linspace(1.0, 0.55, n)
        x = x + sd.applause(n, lv) * 0.30
    return sd.hall(x, wet=0.4)


def _cjk(ch: str) -> bool:
    """Set in the Japanese face: kana, kanji, full-width forms, and the circled digits Segoe UI lacks."""
    return ord(ch) >= 0x2460


def mixed(d: ImageDraw.ImageDraw, cx: float, base: float, text: str, size: int, fill, stroke: int = 0,
          stroke_fill=(0, 0, 0)) -> None:
    """One line centred on cx, its baseline at base: the Latin in Segoe UI Semibold, the Japanese in
    BIZ UD Gothic (10/23, the user chose palette B; the English had been set in the Japanese face's
    Latin, which read like a typewriter)."""
    fonts = {False: ImageFont.truetype(LATIN, size), True: ImageFont.truetype(GOTHIC, size)}
    runs = []
    for ch in text:
        k = _cjk(ch)
        if runs and runs[-1][0] == k:
            runs[-1][1] += ch
        else:
            runs.append([k, ch])
    widths = [d.textlength(r, font=fonts[k]) for k, r in runs]
    x = cx - sum(widths) / 2
    for (k, r), wd in zip(runs, widths):
        d.text((x, base), r, font=fonts[k], fill=fill, anchor="ls", stroke_width=stroke, stroke_fill=stroke_fill)
        x += wd


def card_png(lines: list[str], size: tuple[int, int], out: Path) -> None:
    """White brush lettering on a sumi-ink ground, a vermilion rule under the title, the lines under it
    in white-grey (10/23, palette B: the user found the gold on dark brown hard on the eyes)."""
    w, h = size
    img = Image.new("RGB", size, PANEL)
    glow = Image.new("L", size, 0)
    ImageDraw.Draw(glow).ellipse((w * 0.15, h * 0.2, w * 0.85, h * 0.8), fill=70)
    glow = glow.filter(ImageFilter.GaussianBlur(h * 0.18))
    img = Image.composite(Image.new("RGB", size, (30, 36, 52)), img, glow)
    d = ImageDraw.Draw(img)
    big = ImageFont.truetype(BRUSH, int(h * 0.16))
    tw = d.textlength(lines[0], font=big)
    if tw > w * 0.9:
        big = ImageFont.truetype(BRUSH, int(big.size * w * 0.9 / tw))
        tw = d.textlength(lines[0], font=big)
    small = int(h * 0.045)
    gaps = [int(big.size * 1.35) + int(h * 0.045)] + [int(small * 1.45)] * (len(lines) - 1)
    y = (h - sum(gaps)) / 2
    d.text(((w - tw) / 2, y), lines[0], font=big, fill=INK)
    bb = d.textbbox(((w - tw) / 2, y), lines[0], font=big)
    rw, ry = w * 0.06, bb[3] + h * 0.028
    d.rectangle(((w - rw) / 2, ry, (w + rw) / 2, ry + max(4, h * 0.008)), fill=ACCENT)
    y += gaps[0]
    for line, g in zip(lines[1:], gaps[1:]):
        note = line.startswith("#") or "Apache" in line      # the end card's tags and credit: quieter
        sz = int(small * 0.78) if note else small
        mixed(d, w / 2, y + small, line, sz, MUTED if note else (200, 204, 214))
        y += g
    img.save(out)


def caption_png(text: str, size: tuple[int, int], out: Path) -> None:
    """A caption across the bottom of a clip, on a translucent band: the first line large, the next ones
    smaller (English first for the judges abroad, Japanese under it); the Latin in Segoe UI."""
    w, h = size
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lines = text.split("\n")
    sizes =[int(h * 0.046)] + [int(h * 0.032)] * (len(lines) - 1)
    gaps = [int(sz * 1.3) for sz in sizes]
    bottom = h * 0.885                     # 10/23: above the bottom tenth, where players put their controls
    y = bottom - sum(gaps)
    d.rectangle((0, y - h * 0.015, w, bottom + h * 0.01), fill=(10, 8, 14, 150))
    for line, sz, g in zip(lines, sizes, gaps):
        mixed(d, w / 2, y + sz * 0.95, line, sz, INK, stroke=2)
        y += g
    img.save(out)


LATIN = "C:/Windows/Fonts/seguisb.ttf"
MINCHO = "C:/Windows/Fonts/HGRME.TTC"
SIDE_COLOUR = {"east": (58, 91, 191), "west": (47, 138, 95)}      # the mawashi colours, a little muted
#: palette B (10/23, the user chose it over the gold): white-grey ink, vermilion (the ring's and the gyoji's
#: tassels' red) for the accents, on sumi ink
INK, ACCENT, MUTED, PANEL = (248, 246, 240), (232, 88, 60), (150, 154, 166), (12, 14, 20)


def _portrait(path: str | None, box: tuple[int, int], mirror: bool = False) -> Image.Image | None:
    """A portrait filling `box` (cropped to it), or None."""
    if not path:
        return None
    src = Path(path) if Path(path).is_absolute() else ROOT / path
    if not src.exists():
        return None
    im = Image.open(src).convert("RGB")
    bw, bh = box
    k = max(bw / im.width, bh / im.height)
    im = im.resize((int(im.width * k + 1), int(im.height * k + 1)))
    l, t = (im.width - bw) // 2, (im.height - bh) // 2
    im = im.crop((l, t, l + bw, t + bh))
    return im.transpose(Image.FLIP_LEFT_RIGHT) if mirror else im


def _fade(im: Image.Image, side: str, frac: float = 0.35) -> Image.Image:
    """Melts a portrait's inner edge into the panel colour."""
    w, h = im.size
    m = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(m)
    n = int(w * frac)
    for i in range(n):
        a = int(255 * (i / n) ** 1.5)
        x = w - n + i if side == "right" else n - 1 - i
        d.line([(x, 0), (x, h)], fill=255 - a)
    bg = Image.new("RGB", (w, h), PANEL)
    return Image.composite(im, bg, m)


def profile_png(p: dict, size: tuple[int, int], out: Path) -> None:
    """A wrestler's introduction (10/21 night, the user: who are these two?): the portrait at the left,
    then the shikona, rank, record, win rate and favourite kimarite as bars -- English first, the
    Japanese under it, calm colours."""
    w, h = size
    img = Image.new("RGB", size, PANEL)
    pw = int(w * 0.44)
    por = _portrait(p.get("portrait"), (pw, h))
    if por is not None:
        img.paste(_fade(por, "right"), (0, 0))
    d = ImageDraw.Draw(img)
    x = int(w * 0.48)
    y = int(h * 0.06)
    side = p.get("side", "east")
    tag = ImageFont.truetype(LATIN, int(h * 0.030))
    label = ("EAST" if side == "east" else "WEST") + "  "
    tw = d.textlength(label, font=tag)
    jtag = ImageFont.truetype(GOTHIC, int(h * 0.030))
    jw = d.textlength("東" if side == "east" else "西", font=jtag)
    d.rounded_rectangle((x, y, x + tw + jw + 28, y + int(h * 0.05)), radius=8, fill=SIDE_COLOUR[side])
    d.text((x + 14, y + int(h * 0.008)), label, font=tag, fill=INK)
    d.text((x + 14 + tw, y + int(h * 0.006)), "東" if side == "east" else "西", font=jtag, fill=INK)
    y += int(h * 0.075)
    big = ImageFont.truetype(MINCHO, int(h * 0.15))
    d.text((x - 6, y), p["shikona"], font=big, fill=INK, stroke_width=3, stroke_fill=(0, 0, 0))
    y += int(h * 0.185)
    rom = ImageFont.truetype(LATIN, int(h * 0.042))
    d.text((x, y), f"{p['romaji'].upper()}   ·   {p['rank'][0]}", font=rom, fill=ACCENT)
    rw = d.textlength(f"{p['romaji'].upper()}   ·   {p['rank'][0]} ", font=rom)
    d.text((x + rw + 6, y + int(h * 0.006)), p["rank"][1], font=ImageFont.truetype(GOTHIC, int(h * 0.036)), fill=ACCENT)
    y += int(h * 0.08)
    small = ImageFont.truetype(LATIN, int(h * 0.031))
    jsmall = ImageFont.truetype(GOTHIC, int(h * 0.024))
    for en, ja in p.get("facts", []):
        d.text((x, y), en, font=small, fill=INK)
        d.text((x, y + int(h * 0.040)), ja, font=jsmall, fill=MUTED)
        y += int(h * 0.078)
    y += int(h * 0.005)
    d.text((x, y), "SIGNATURE MOVES", font=small, fill=ACCENT)
    d.text((x + d.textlength("SIGNATURE MOVES  ", font=small), y + int(h * 0.005)), "得意技", font=jsmall, fill=MUTED)
    y += int(h * 0.058)
    bar_w = int(w * 0.30)
    lab = ImageFont.truetype(LATIN, int(h * 0.028))
    for k, m in enumerate(p.get("moves", [])[:3]):
        share = float(m["share"])
        txt = f"{m['kimarite']}  ({m['en']})"
        d.text((x, y), txt, font=lab, fill=INK)
        d.text((x + d.textlength(txt + "  ", font=lab), y + int(h * 0.004)), m["ja"], font=jsmall, fill=MUTED)
        by = y + int(h * 0.044)
        d.rounded_rectangle((x, by, x + bar_w, by + int(h * 0.018)), radius=5, fill=(42, 45, 53))
        d.rounded_rectangle((x, by, x + max(10, int(bar_w * share)), by + int(h * 0.018)), radius=5,
                            fill=ACCENT if k == 0 else (142, 140, 150))
        d.text((x + bar_w + 14, by - int(h * 0.010)), f"{share * 100:.0f} %", font=lab, fill=INK)
        y += int(h * 0.076)
    img.save(out)


def versus_png(v: dict, size: tuple[int, int], out: Path) -> None:
    """The two side by side before the final: WHO IS THE STRONGEST? and their head-to-head."""
    w, h = size
    img = Image.new("RGB", size, PANEL)
    half = w // 2
    for k, side in enumerate(("east", "west")):
        por = _portrait(v[side].get("portrait"), (half, h), mirror=(side == "west"))
        if por is not None:
            img.paste(_fade(por, "right" if side == "east" else "left", 0.30), (k * half, 0))
    shade = Image.new("L", size, 0)
    sd = ImageDraw.Draw(shade)
    for yy in range(h):                                  # dark bands top and bottom for the words
        a = max(0, int(200 * (1 - yy / (h * 0.28)))) if yy < h * 0.28 else max(0, int(200 * ((yy - h * 0.66) / (h * 0.34))))
        sd.line([(0, yy), (w, yy)], fill=a)
    img = Image.composite(Image.new("RGB", size, PANEL), img, shade)
    d = ImageDraw.Draw(img)
    t1 = ImageFont.truetype(LATIN, int(h * 0.075))
    s = "WHO IS THE STRONGEST?"
    d.text(((w - d.textlength(s, font=t1)) / 2, h * 0.05), s, font=t1, fill=ACCENT, stroke_width=3, stroke_fill=PANEL)
    t2 = ImageFont.truetype(GOTHIC, int(h * 0.045))
    s = "最強はどっちだ"
    d.text(((w - d.textlength(s, font=t2)) / 2, h * 0.155), s, font=t2, fill=INK)
    vs = ImageFont.truetype(LATIN, int(h * 0.16))
    d.text(((w - d.textlength("VS", font=vs)) / 2, h * 0.40), "VS", font=vs, fill=INK, stroke_width=4, stroke_fill=(0, 0, 0))
    name = ImageFont.truetype(MINCHO, int(h * 0.085))
    rom = ImageFont.truetype(LATIN, int(h * 0.032))
    for k, side in enumerate(("east", "west")):
        cx = half * k + half / 2
        e = v[side]
        d.text((cx - d.textlength(e["shikona"], font=name) / 2, h * 0.70), e["shikona"], font=name, fill=INK,
               stroke_width=3, stroke_fill=(0, 0, 0))
        r = f"{e['romaji'].upper()}  ·  {e['rank']}"
        d.text((cx - d.textlength(r, font=rom) / 2, h * 0.815), r, font=rom, fill=INK, stroke_width=3,
               stroke_fill=(0, 0, 0))                # vermilion sank into the red robots (10/23)
    if v.get("h2h"):
        a, b = v["h2h"]
        hs = f"Head to head  {a:,} – {b:,}"
        hf = ImageFont.truetype(LATIN, int(h * 0.034))
        d.text(((w - d.textlength(hs, font=hf)) / 2, h * 0.895), hs, font=hf, fill=INK)
        js = v.get("h2h_note", "直接対決（1024 土俵 × 20 秒）")
        jf = ImageFont.truetype(GOTHIC, int(h * 0.024))
        d.text(((w - d.textlength(js, font=jf)) / 2, h * 0.945), js, font=jf, fill=MUTED)
    img.save(out)


def segment(i: int, item: dict, size, fps: int, tmp: Path) -> Path:
    w, h = size
    out = tmp / f"seg{i:03d}.mp4"
    secs = float(item["seconds"])
    fade = float(item.get("fade", 0.25))
    vf_fade = (f",fade=t=in:st=0:d={fade},fade=t=out:st={max(0.0, secs - fade):.3f}:d={fade}" if fade > 0 else "")
    af_fade = (f",afade=t=in:st=0:d={fade},afade=t=out:st={max(0.0, secs - fade):.3f}:d={fade}" if fade > 0 else "")
    norm_v = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p"
    norm_a = f"aformat=sample_rates={SR}:channel_layouts=stereo"
    enc = ["-c:v", "libx264", "-crf", "16", "-preset", os.environ.get("MAKE_FINAL_PRESET", "veryfast"), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           "-ar", str(SR), "-t", f"{secs:.3f}", str(out)]
    if "card" in item or "image" in item or "profile" in item or "versus" in item:
        png, wav = tmp / f"card{i}.png", tmp / f"card{i}.wav"
        if "card" in item:
            card_png(item["card"], size, png)
        elif "profile" in item or "versus" in item:   # the finalists' introductions: a slow push-in
            (profile_png if "profile" in item else versus_png)(item.get("profile") or item["versus"], size, png)
            n = int(secs * fps)
            z = float(item.get("zoom", 0.04))
            norm_v = (f"scale={w * 2}:{h * 2},zoompan=z='1+{z}*on/{n}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                      f":d=1:s={w}x{h}:fps={fps},setsar=1,format=yuv420p")
            item = {**item, "sound": item.get("sound", "murmur")}
        else:                               # a still (a chart), the murmur under it
            src = str(ROOT / item["image"]) if not Path(item["image"]).is_absolute() else item["image"]
            Image.open(src).convert("RGB").resize(size).save(png)
            item = {**item, "sound": item.get("sound", "murmur")}
        write_wav(wav, bed(secs, item.get("sound")))
        run(["-loop", "1", "-framerate", str(fps), "-t", f"{secs:.3f}", "-i", str(png), "-i", str(wav),
             "-filter_complex", f"[0:v]{norm_v}{vf_fade}[v];[1:a]{norm_a}{af_fade}[a]", "-map", "[v]", "-map", "[a]", *enc])
        return out
    clip = str(ROOT / item["clip"]) if not Path(item["clip"]).is_absolute() else item["clip"]
    speed = float(item.get("speed", 1.0))
    src_secs = secs * speed
    inputs = ["-ss", str(item.get("start", 0.0)), "-t", f"{src_secs:.3f}", "-i", clip]
    # `zoom` (10/25): the MuJoCo footage's own name-and-score strip along its bottom pushed out of the
    # frame by a crop from the top centre, scaled back up (the opaque bar that hid it before made the
    # picture look squashed, the user)
    z = float(item.get("zoom", 1.0))
    crop = f"crop=iw/{z}:ih/{z}:(iw-iw/{z})/2:0," if z > 1.0 else ""
    v = f"[0:v]setpts=PTS/{speed},{crop}{norm_v}"
    if item.get("text"):
        cap = tmp / f"cap{i}.png"
        caption_png(item["text"], size, cap)
        inputs += ["-i", str(cap)]
        v = f"{v}[vb];[vb][1:v]overlay=0:0,format=yuv420p"
    v += f"{vf_fade}[v]"
    nin = 2 if item.get("text") else 1
    if has_audio(clip) and not item.get("mute"):
        a = f"[0:a]{norm_a}" + (f",atempo={speed}" if speed >= 0.5 else ",atempo=0.5") + f"{af_fade}[a]"
    else:
        wav = tmp / f"bed{i}.wav"
        write_wav(wav, bed(secs, item.get("sound", "murmur")))
        inputs += ["-i", str(wav)]
        a = f"[{nin}:a]{norm_a}{af_fade}[a]"
    run([*inputs, "-filter_complex", f"{v};{a}", "-map", "[v]", "-map", "[a]", *enc])
    return out


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2])
    size = tuple(spec.get("size", [1920, 1080]))
    fps = int(spec.get("fps", 30))
    tmp = Path(tempfile.mkdtemp())
    segs = [segment(i, it, size, fps, tmp) for i, it in enumerate(spec["items"])]
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{s.as_posix()}'\n" for s in segs), encoding="utf-8")
    run(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out)])
    dur = sum(float(it["seconds"]) for it in spec["items"])
    print(f"[final] {len(segs)} segments, {dur:.1f} s -> {out}")


if __name__ == "__main__":
    main()
