"""Standings from a sumo round robin (tools/eval/sumo_banzuke.sh's CSV).

Each wrestler's record is the sum over its pairings of bouts won and lost (bouts
are decided ones; a ring that never decides adds nothing). The order is by the
share of decided bouts won, then by wins; the ranks are handed out down that
order -- yokozuna, ozeki, sekiwake, komusubi, then maegashira 1, 2, ... --
and each wrestler's favourite kimarite (the ones its wins came by) is listed.

    python -I tools/eval/banzuke_table.py env/banzuke-2026-10-10.csv [names.csv] > table.md

names.csv (optional): name,shikona -- a ring name to print next to each policy.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict

KANJI = {
    "oshidashi": "押し出し", "oshitaoshi": "押し倒し", "yorikiri": "寄り切り",
    "yoritaoshi": "寄り倒し", "tsukidashi": "突き出し", "tsukitaoshi": "突き倒し",
    "okuridashi": "送り出し", "okuritaoshi": "送り倒し", "hikiotoshi": "引き落とし",
    "hatakikomi": "はたき込み", "uwatenage": "上手投げ",
    "shitatenage": "下手投げ", "ashitori": "足取り", "tottari": "とったり",
    "isamiashi": "勇み足", "koshikudake": "腰砕け", "dotai": "同体",
}
RANKS = ("横綱", "大関", "関脇", "小結")


def parse_kimarite(text: str) -> list[tuple[str, int, int]]:
    out = []
    for part in (text or "").split(";"):
        bits = part.split()
        if len(bits) < 3:
            continue
        e = int(bits[1][1:]) if bits[1].startswith("E") else 0
        w = int(bits[2][1:]) if bits[2].startswith("W") else 0
        out.append((bits[0], e, w))
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
    shikona = {}
    if len(sys.argv) > 2:
        shikona = {r["name"]: r["shikona"] for r in csv.DictReader(open(sys.argv[2], encoding="utf-8"))}
    won, lost = Counter(), Counter()
    techniques: dict[str, Counter] = defaultdict(Counter)
    h2h: dict[tuple[str, str], tuple[int, int]] = {}
    names: list[str] = []
    for r in rows:
        e, w = r["east"], r["west"]
        for n in (e, w):
            if n not in names:
                names.append(n)
        ew, ww = int(r["east_wins"]), int(r["west_wins"])
        won[e] += ew; lost[e] += ww; won[w] += ww; lost[w] += ew
        h2h[(e, w)] = (ew, ww)
        h2h[(w, e)] = (ww, ew)
        for k, ek, wk in parse_kimarite(r["kimarite"]):
            techniques[e][k] += ek
            techniques[w][k] += wk

    def share(n: str) -> float:
        total = won[n] + lost[n]
        return won[n] / total if total else 0.0

    order = sorted(names, key=lambda n: (-share(n), -won[n]))
    print("| 番付 | 力士 | 勝 | 敗 | 勝率 | 主な決まり手 |")
    print("|---|---|---:|---:|---:|---|")
    for i, n in enumerate(order):
        rank = RANKS[i] if i < len(RANKS) else f"前頭{i - len(RANKS) + 1}"
        favourite = "、".join(f"{KANJI.get(k, k)} {c}" for k, c in techniques[n].most_common(3) if c)
        label = f"{shikona[n]}（{n}）" if n in shikona else n
        print(f"| {rank} | {label} | {won[n]} | {lost[n]} | {share(n):.3f} | {favourite or '—'} |")
    print()
    print("対戦成績（行の力士から見た勝-敗）")
    print()
    print("| | " + " | ".join(order) + " |")
    print("|---|" + "---|" * len(order))
    for a in order:
        cells = []
        for b in order:
            cells.append("—" if a == b else ("{}-{}".format(*h2h[(a, b)]) if (a, b) in h2h else ""))
        print(f"| {a} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
