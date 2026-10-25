"""How much a bout is worth filming (10/21 night): the first bout of each replay, scored from its log.

    python -I tools/eval/drama.py DIR [TOP]      (DIR holds bout_<name>.log / bout_<name>.csv pairs)

The user, on the first hero bout: the finish "just walks diagonally" -- not sumo enough. A bout the
crowd rises for has the two locked together and straining in the middle, someone holding at the
straw, the pressure turning over, and a finish other than the plain push-out. From the gyoji's calls
(env 0: はっけよい, then 勝負あり with the winner and the kimarite) and JUMPER_SUMO_LOG (both bases
every 5 ms, `dist` between them), between the charge and the decision:

    locked   seconds the two are chest to chest (bases < 0.27 m apart)
    middle   of those, seconds with both well inside (< 0.35 m from the centre)
    held     seconds the loser stood at the straw (> 0.55 m out) before going over
    comeback the winner had been at the straw himself (> 0.55 m) -- the crowd's favourite
    turns    times the pressure changed sides (who stands nearer the straw, 0.08 m of play)

Printed best first; DIR/drama.json keeps the ranking.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

RING = 0.8
EDGE = 0.55


def first_bout(log: Path):
    text = log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    lines = [ln for ln in text.splitlines() if "gyoji (env 0)" in ln]
    if not lines:
        return None
    calls = lines[-1].split(" / ")
    t0 = t1 = None
    who = kim = ""
    for c in calls:
        m = re.search(r"([0-9.]+)s (.*)", c)
        if not m:
            continue
        t, what = float(m.group(1)), m.group(2).strip()
        if what.startswith("はっけよい") and t0 is None:
            t0 = t
        elif what.startswith("勝負あり") and t0 is not None:
            t1 = t
            parts = what.split("　")
            who = "east" if len(parts) > 1 and parts[1].startswith("東") else "west"
            kim = parts[2] if len(parts) > 2 else ""
            break
    if t0 is None or t1 is None:
        return None
    return t0, t1, who, kim


def score(log: Path, table: Path) -> dict | None:
    b = first_bout(log)
    if b is None or not table.exists():
        return None
    t0, t1, who, kim = b
    rows = []
    for r in csv.DictReader(open(table, encoding="utf-8")):
        t = float(r["t"])
        if t0 <= t <= t1 and r["reset"] == "0":
            re_ = math.hypot(float(r["robot_x"]), float(r["robot_y"]))
            rw = math.hypot(float(r["opp_x"]), float(r["opp_y"]))
            rows.append((t, float(r["dist"]), re_, rw))
    if len(rows) < 10:
        return None
    dt = (rows[-1][0] - rows[0][0]) / max(1, len(rows) - 1)
    locked = sum(dt for _, d, _, _ in rows if d < 0.27)
    middle = sum(dt for _, d, a, b in rows if d < 0.27 and a < 0.35 and b < 0.35)
    win_r = [a if who == "east" else b for _, _, a, b in rows]
    lose_r = [b if who == "east" else a for _, _, a, b in rows]
    held = sum(dt for x in lose_r[:-int(0.3 / dt) or None] if x > EDGE)
    comeback = max(win_r[:-int(0.3 / dt) or None], default=0) > EDGE
    turns, side = 0, 0
    for _, _, a, b in rows:
        lead = b - a                       # > 0: west nearer the straw
        s = 1 if lead > 0.08 else (-1 if lead < -0.08 else 0)
        if s and side and s != side:
            turns += 1
        if s:
            side = s
    dur = t1 - t0
    pts = (min(dur, 12) / 12 + min(middle, 4) / 4 + 1.5 * min(held, 2) / 2 + 2.0 * comeback
           + 0.5 * min(turns, 3) + (0.7 if kim and kim != "押し出し" else 0.0))
    return dict(name=log.stem.removeprefix("bout_"), score=round(pts, 2), t0=t0, t1=t1, seconds=round(dur, 2), winner=who,
                kimarite=kim, locked=round(locked, 2), middle=round(middle, 2), held=round(held, 2),
                comeback=comeback, turns=turns)


def main() -> None:
    d = Path(sys.argv[1])
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    res = [s for log in sorted(d.glob("bout_*.log")) if (s := score(log, log.with_suffix(".csv")))]
    res.sort(key=lambda s: -s["score"])
    print("| bout | score | s | winner | kimarite | locked | middle | held | comeback | turns |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in res[:top]:
        print(f"| {s['name']} | {s['score']} | {s['seconds']} | {s['winner']} | {s['kimarite']} | {s['locked']} | "
              f"{s['middle']} | {s['held']} | {'yes' if s['comeback'] else ''} | {s['turns']} |")
    (d / "drama.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
