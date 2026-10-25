"""Fill media/final/final_v2.json's (or SPEC's) placeholders with the newest files and print the spec path.

    python -I tools/media/fill_v2.py OUT.json [SPEC]

HERO3 -> the newest *_blender_sumo-arena_hero3.mp4; HERO2 -> the newest *_blender_sumo-arena_hero2.mp4,
or else hero2's MuJoCo replay (*_scene_sumo-hero2s33.mp4, no 8 s intro: its times move 8 s earlier);
HL_<role>_<name> -> the newest readable *_scene_sumo-hl-<role>-<name>.mp4. v3 (10/23): HERO3 / HERO2
take the composing with English (*-en.mp4) when there is one; TROPHY -> *_blender_sumo-trophy_hero3.mp4;
CLAW -> *_scene_sumo-claw-waza70-*.mp4, its start `tail` seconds before its end; VERDICT ->
*_blender_sumo-verdict_hero3.mp4. An item whose file is
missing is dropped (the draft still builds), and said so.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "media" / "raw"
FFPROBE = os.environ.get("FFPROBE", "ffprobe")


def readable(p: Path) -> bool:
    out = subprocess.run([str(FFPROBE), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out) > 1.0
    except ValueError:
        return False


def newest(pattern: str) -> Path | None:
    for p in sorted(RAW.glob(pattern), reverse=True):
        if readable(p):
            return p
    return None


def duration(p: Path) -> float:
    return float(subprocess.run([str(FFPROBE), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(p)], capture_output=True, text=True).stdout.strip())


def main() -> None:
    src = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "media/final/final_v2.json"
    spec = json.loads(src.read_text(encoding="utf-8"))
    hero3 = newest("*_blender_sumo-arena_hero3-en.mp4") or newest("*_blender_sumo-arena_hero3.mp4")
    hero2 = newest("*_blender_sumo-arena_hero2-en.mp4") or newest("*_blender_sumo-arena_hero2.mp4")
    extra = {"TROPHY": newest("*_blender_sumo-trophy_hero3.mp4"), "CLAW": newest("*_scene_sumo-claw-waza70-*.mp4"),
             "VERDICT": newest("*_blender_sumo-verdict_hero3.mp4")}
    hero2_shift = 0.0
    if hero2 is None:
        hero2 = newest("*_scene_sumo-hero2s33.mp4")
        hero2_shift = -8.0
    items = []
    for it in spec["items"]:
        c = it.get("clip")
        if c == "HERO3":
            if hero3 is None:
                print("dropped (no hero3):", it.get("text", "")[:30]); continue
            it = {**it, "clip": str(hero3.relative_to(ROOT)).replace("\\", "/")}
        elif c == "HERO2":
            if hero2 is None:
                print("dropped (no hero2)"); continue
            it = {**it, "clip": str(hero2.relative_to(ROOT)).replace("\\", "/"), "start": it["start"] + hero2_shift}
        elif c in extra:
            p = extra[c]
            if p is None:
                print("dropped (no clip):", c); continue
            it = {**it, "clip": str(p.relative_to(ROOT)).replace("\\", "/")}
            if "tail" in it:
                it["start"] = max(0.0, round(duration(p) - it.pop("tail"), 2))
        elif c and c.startswith("HL_"):
            role, name = c[3:].rsplit("_", 2)[0], "_".join(c[3:].rsplit("_", 2)[1:])
            p = newest(f"*_scene_sumo-hl-{role}-{name}.mp4")
            if p is None:
                print("dropped (no clip):", c); continue
            it = {**it, "clip": str(p.relative_to(ROOT)).replace("\\", "/")}
        items.append(it)
    spec["items"] = items
    Path(sys.argv[1]).write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(items)} items -> {sys.argv[1]} (hero3 {hero3 and hero3.name}, hero2 {hero2 and hero2.name})")


if __name__ == "__main__":
    main()
