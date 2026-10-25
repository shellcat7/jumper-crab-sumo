"""A wrestler's tale of the tape (10/21 night): record and favourite kimarite from the head-to-head files.

    python -I tools/eval/profile.py NAME env/rules15-2026-10-21.csv ...   -> JSON on stdout

The user: the two in the final must not appear from nowhere -- introduce them, their best moves
and their best wins, then "who is the strongest?" (an American sports-show touch). Each row of
rules_colab.sh's CSV has the kimarite of every decided bout per side ("hikiotoshi E692 W236; ...");
NAME's wins are summed over the rows it is in, as east or west.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter

#: kimarite in English for viewers abroad (the romaji stays: sumo's own words)
KIMARITE_EN = {
    "oshidashi": "push-out", "yorikiri": "force-out", "hikiotoshi": "pull-down", "ashitori": "leg pick",
    "tottari": "arm-bar throw", "oshitaoshi": "push-down", "tsukidashi": "thrust-out", "tsukitaoshi": "thrust-down",
    "okuridashi": "push-out from behind", "hatakikomi": "slap-down", "utchari": "backward pivot throw",
    "yoritaoshi": "force-out-down", "tsukiotoshi": "thrust-down",
}
KIMARITE_JA = {
    "oshidashi": "押し出し", "yorikiri": "寄り切り", "hikiotoshi": "引き落とし", "ashitori": "足取り", "tottari": "とったり",
    "oshitaoshi": "押し倒し", "tsukidashi": "突き出し", "tsukitaoshi": "突き倒し", "okuridashi": "送り出し",
    "hatakikomi": "はたき込み", "utchari": "うっちゃり", "yoritaoshi": "寄り倒し", "tsukiotoshi": "突き落とし",
}


def profile(name: str, paths: list[str]) -> dict:
    wins = losses = 0
    kim: Counter = Counter()
    for p in paths:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if name not in (r["east"], r["west"]):
                continue
            me = "E" if r["east"] == name else "W"
            wins += int(r["east_wins"] if me == "E" else r["west_wins"])
            losses += int(r["west_wins"] if me == "E" else r["east_wins"])
            for part in (r.get("kimarite") or "").split(";"):
                bits = part.split()
                if len(bits) < 3 or bits[0] == "dotai":
                    continue
                for b in bits[1:]:
                    if b.startswith(me):
                        kim[bits[0]] += int(b[1:])
    total = sum(kim.values()) or 1
    top = [{"kimarite": k, "en": KIMARITE_EN.get(k, k), "ja": KIMARITE_JA.get(k, k), "share": round(v / total, 3)}
           for k, v in kim.most_common(4)]
    return {"name": name, "wins": wins, "losses": losses, "win_rate": round(wins / max(1, wins + losses), 3),
            "kimarite": top}


if __name__ == "__main__":
    print(json.dumps(profile(sys.argv[1], sys.argv[2:]), ensure_ascii=False, indent=1))
