"""Win rate and how the wins came, per east wrestler of rules CSVs (10/22: the generation bred to
use its claws -- is it as strong, and does it win by them?).

    python -I tools/eval/kimarite_share.py env/rules18-2026-10-22.csv [more.csv ...]

For each row: east's wins / decided bouts, and of east's wins the share by the claws' kimarite
(waza.CLAW_KIMARITE: ashitori, tottari, tsukidashi, tsukitaoshi, hatakikomi, uwatenage, shitatenage),
by the push-out (oshidashi) and by the pull-down (hikiotoshi). Markdown for the devlog.
"""

from __future__ import annotations

import csv
import sys

CLAW = {"ashitori", "tottari", "tsukidashi", "tsukitaoshi", "hatakikomi", "uwatenage", "shitatenage"}


def wins_by(kimarite: str, side: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in kimarite.split(";"):
        bits = part.split()
        if len(bits) < 3 or bits[0] == "dotai":
            continue
        for b in bits[1:]:
            if b.startswith(side):
                out[bits[0]] = out.get(bits[0], 0) + int(b[1:])
    return out


def main() -> None:
    print("| east | west | east wins | win rate | claws | push-out | pull-down | claw kimarite |")
    print("|---|---|---|---|---|---|---|---|")
    for p in sys.argv[1:]:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            ew, ww = int(r["east_wins"]), int(r["west_wins"])
            k = wins_by(r.get("kimarite") or "", "E")
            tot = sum(k.values()) or 1
            claw = sum(v for n, v in k.items() if n in CLAW)
            which = ", ".join(f"{n} {v}" for n, v in sorted(k.items(), key=lambda x: -x[1]) if n in CLAW and v)
            print(f"| {r['east']} | {r['west']} | {ew}-{ww} | {ew / max(1, ew + ww):.3f} | {claw / tot:.1%} | "
                  f"{k.get('oshidashi', 0) / tot:.1%} | {k.get('hikiotoshi', 0) / tot:.1%} | {which} |")


if __name__ == "__main__":
    main()
