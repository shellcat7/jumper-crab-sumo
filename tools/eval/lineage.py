"""One strength scale for every wrestler from the head-to-head files (10/21), and its chart.

    python -I tools/eval/lineage.py OUT.png env/rules6-2026-10-20.csv env/rules9-2026-10-20.csv ...

Each file's rows are bouts between two wrestlers (east, west, east_wins, west_wins: rules_colab
.sh's columns). The ratings are a Bradley-Terry fit to all of them at once (the MM iteration),
on the Elo scale (400 points = 10:1 odds), waza24 at 1500; a wrestler met in one file only is
placed through whoever it met. The chart (1920x1080, the broadcast graphics' colours) draws
each generation at its rating with an arrow from the wrestler it was trained from (LINEAGE),
for the process part of the video; the table is printed (markdown) for the devlog.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict

import numpy as np

#: Who each wrestler was trained from (the devlogs), and how ("self-play", "exploiter", ...).
LINEAGE = {
    "sogo_waza21": ("sogo_waza18", "自己対戦"), "sogo_waza23": ("sogo_waza21", "自己対戦"),
    "sogo_waza24": ("sogo_waza20", "新しい脚"), "sogo_waza25": ("sogo_waza24", "自己対戦"),
    "sogo_waza26": ("sogo_waza23", "自己対戦"), "sogo_waza28": ("sogo_waza23", "リーグ"),
    "sogo_waza27": ("sogo_waza24", "リーグ"), "sogo_waza29": ("sogo_waza27", ""),
    "sogo_waza30": ("sogo_waza25", "リーグ"), "sogo_waza31": ("sogo_waza25", "自己対戦"),
    "sogo_waza32": ("sogo_waza29", "特化"), "sogo_waza33": ("sogo_waza29", ""),
    "sogo_waza34": ("sogo_waza25", "多彩さ"), "sogo_waza35": ("sogo_waza32", "多彩さ"),
    "sogo_waza36": ("sogo_waza32", "逆転"), "sogo_waza37": ("sogo_waza30", "特化"),
    # 10/21: the pusher legs (sumo_cont700, names ending in c) -- the same tacticians on other legs
    "sogo_waza38c": ("sogo_waza32", "押しの脚"), "sogo_waza39c": ("sogo_waza33", "押しの脚"),
    "sogo_waza40c": ("sogo_waza39c", ""), "sogo_waza41c": ("sogo_waza36", "押しの脚"),
    "sogo_waza42c": ("sogo_waza38c", "多彩さ"), "sogo_waza43c": ("sogo_waza40c", ""),
    "sogo_waza44c": ("sogo_waza42c", "多彩さ"), "sogo_waza45c": ("sogo_waza41c", ""),
    "sogo_waza46c": ("sogo_waza39c", "リーグ"), "sogo_waza47c": ("sogo_waza39c", "多彩さ"),
}
#: rules12/13 (10/21 night) named the cont700 wrestlers like the rikishi3 ones; the devlog says
#: which rows were on cont700 (file stem, east, west) -> the names on the pusher legs
CONT700_ROWS = {
    ("rules12-2026-10-21", "sogo_waza38", "league_sumo"): ("sogo_waza38c", "league_sumo"),
    ("rules12-2026-10-21", "sogo_waza38", "sogo_waza32"): ("sogo_waza38c", "sogo_waza32c"),
    ("rules13-2026-10-21", "sogo_waza33", "league_sumo"): ("sogo_waza33c", "league_sumo"),
    ("rules13-2026-10-21", "sogo_waza33", "sogo_waza32"): ("sogo_waza33c", "sogo_waza32c"),
    ("rules13-2026-10-21", "sogo_waza39", "league_sumo"): ("sogo_waza39c", "league_sumo"),
    ("rules13-2026-10-21", "sogo_waza39", "sogo_waza33"): ("sogo_waza39c", "sogo_waza33c"),
    ("rules13-2026-10-21", "sogo_waza39", "sogo_waza38"): ("sogo_waza39c", "sogo_waza38c"),
}
FIELD = {"league_sumo": "league_sumo", "tactics_mawari": "回ノ里", "sogo": "総ノ山"}


def load(paths: list[str]) -> dict[tuple[str, str], list[int]]:
    games: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for p in paths:
        stem = p.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".csv")
        for r in csv.DictReader(open(p, encoding="utf-8")):
            a, b = CONT700_ROWS.get((stem, r["east"], r["west"]), (r["east"], r["west"]))
            ew, ww = int(r["east_wins"]), int(r["west_wins"])
            if a > b:
                a, b, ew, ww = b, a, ww, ew
            games[(a, b)][0] += ew
            games[(a, b)][1] += ww
    return games


def bradley_terry(games, iters=2000, anchor="sogo_waza24"):
    names = sorted({n for k in games for n in k})
    p = {n: 1.0 for n in names}
    wins = defaultdict(float)
    for (a, b), (wa, wb) in games.items():
        wins[a] += wa
        wins[b] += wb
    for _ in range(iters):
        new = {}
        for n in names:
            den = 0.0
            for (a, b), (wa, wb) in games.items():
                if n in (a, b):
                    o = b if n == a else a
                    den += (wa + wb) / (p[n] + p[o])
            new[n] = (wins[n] + 0.5) / (den + 1.0 / (p[n] + 1.0))      # a half-win prior against a 1.0
        g = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        p = {n: v / g for n, v in new.items()}
    base = p.get(anchor, 1.0)
    return {n: 1500 + 400 * math.log10(v / base) for n, v in p.items()}


#: the tags in English (10/21 22:45, the user: the chart must read for viewers who do not read Japanese)
TAG_EN = {"自己対戦": "self-play", "新しい脚": "new legs", "リーグ": "league", "特化": "exploiter",
          "多彩さ": "variety", "逆転": "comeback", "押しの脚": "pusher legs"}
FIELD_EN = {"league_sumo": ("old league champion", "旧リーグ王者"), "tactics_mawari": ("baseline: circler", "回ノ里"),
            "sogo": ("baseline: all-rounder", "総ノ山")}


def chart(elo: dict[str, float], out: str) -> None:
    """One story for a five-second shot (10/23, the user: the chart looked machine-made and hard to read;
    then: a curve reads as a learning curve better than steps, and the colours -- a light, easy ground):
    the strongest so far as one smooth curve (monotone through each new best, a 10 % wash under it),
    every other run a grey dot, the two finalists in their mawashi colours with their names, the first
    runs and the pusher legs named; solid hairline grid, no arrows, no number on every dot. Colours:
    the dataviz reference palette's first three light slots (#eb6834 / #2a78d6 / #1baf7a), which its
    validator passes on all pairs on #fcfcfb (the aqua under 3:1, so it is always labelled)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    fams = []
    for f in ("C:/Windows/Fonts/seguisb.ttf", "C:/Windows/Fonts/BIZ-UDGothicB.ttc", "C:/Windows/Fonts/YuGothB.ttc"):
        try:
            font_manager.fontManager.addfont(f)
            fams.append(font_manager.FontProperties(fname=f).get_name())
        except Exception:
            continue
    plt.rcParams["font.family"] = fams       # Latin first, Japanese glyphs from the next face
    # the old tacticians only re-measured on the pusher legs (25c, 32c, ...) are left out: one dot per run
    num = {n: int(n.replace("sogo_waza", "").rstrip("abc")) for n in elo
           if n.startswith("sogo_waza") and n.replace("sogo_waza", "").rstrip("abc").isdigit()}   # not 47L etc.
    num = {n: g for n, g in num.items() if (n.endswith("c") and n in LINEAGE) or (not n.endswith("c") and g < 38)}
    order = sorted(num, key=lambda n: num[n])
    gen = {n: i for i, n in enumerate(order)}
    surface, ink, ink2, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#7a7974", "#e6e5e0"
    best_c, east_c, west_c, dot_c = "#eb6834", "#2a78d6", "#1baf7a", "#bdbcb6"
    finalists = {"sogo_waza47c": (east_c, "Hasami-no-Umi", "鋏ノ海"), "sogo_waza39c": (west_c, "Hasami-no-Sato", "鋏ノ里")}
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    xs = [gen[n] for n in order]
    ys = [elo[n] for n in order]
    # the pusher legs: a faint band from the first run on them, named once at its top
    first_c = min((gen[n] for n in order if n.endswith("c")), default=None)
    if first_c is not None:
        ax.axvspan(first_c - 0.5, len(order) - 0.4, color="#000000", alpha=0.035, lw=0, zorder=0)
        ax.text(first_c - 0.2, 0.03, "Pusher legs from here  押しの脚", transform=ax.get_xaxis_transform(),
                color=ink2, fontsize=19, va="bottom")
    # every run, quiet
    ax.scatter(xs, ys, s=95, color=dot_c, zorder=2, edgecolors=surface, linewidths=2)
    # the strongest so far: a step line, a dot where a new best was set
    run, bx, by = -1e9, [], []
    for x, y in zip(xs, ys):
        if y > run:
            run = y
            bx.append(x)
            by.append(y)
    from scipy.interpolate import PchipInterpolator
    kx, ky = bx + [len(order) - 0.4], by + [by[-1]]
    cx = np.linspace(kx[0], kx[-1], 400)
    cy = PchipInterpolator(kx, ky)(cx)                     # monotone: never above a best not yet set
    ax.fill_between(cx, cy, min(ys) - 70, color=best_c, alpha=0.10, lw=0, zorder=1)
    ax.plot(cx, cy, color=best_c, lw=4.5, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.scatter(bx, by, s=150, color=best_c, zorder=4, edgecolors=surface, linewidths=2.5)
    # the finalists, in their mawashi colours, named
    for n, (c, en, ja) in finalists.items():
        if n not in gen:
            continue
        x, y = gen[n], elo[n]
        ax.scatter([x], [y], s=520, color=c, zorder=5, edgecolors=surface, linewidths=3)
        below = n == "sogo_waza47c"
        ax.annotate(f"{en}  {ja}", (x, y), xytext=(0, -46 if below else 30), textcoords="offset points",
                    ha="right" if below else "center", va="top" if below else "bottom", color=ink, fontsize=23, zorder=6)
        ax.annotate(f"finalist · waza{num[n]}", (x, y), xytext=(0, -80 if below else 64), textcoords="offset points",
                    ha="right" if below else "center", va="top" if below else "bottom", color=ink2, fontsize=16, zorder=6)
    first = order[0]
    ax.annotate("First runs  最初の学習", (gen[first], elo[first]), xytext=(4, -22), textcoords="offset points",
                ha="left", va="top", color=ink2, fontsize=19, zorder=6)
    # the legend: two keys under the subtitle (identity is never colour alone)
    handles = [Line2D([], [], color=best_c, lw=4.5, marker="o", ms=11, mec=surface, mew=2,
                      label="Strongest so far  最強の推移"),
               Line2D([], [], color=dot_c, lw=0, marker="o", ms=10, mec=surface, mew=2,
                      label="One training run  学習 1 回")]
    lg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.86), ncol=2, frameon=False,
                    fontsize=19, labelcolor=ink2, handlelength=2.2, columnspacing=2.5)
    fig.text(0.06, 0.935, "How the robots got stronger", color=ink, fontsize=40, ha="left", va="center")
    fig.text(0.06, 0.885, "強さの系譜 — 学習を重ねるごとに強くなった", color=ink2, fontsize=22, ha="left", va="center")
    fig.text(0.06, 0.035, "+100 rating ≈ wins 64 % of bouts against the other  ·  MuJoCo simulation, 1,024 rings × 20 s per pairing",
             color=muted, fontsize=14, ha="left")
    ax.set_xticks([])
    ax.set_xlabel("Training runs, in order  →   学習の順", color=ink2, fontsize=19, labelpad=14)
    ax.set_ylabel("Strength rating  強さ", color=ink2, fontsize=19, labelpad=12)
    ax.tick_params(axis="y", colors=muted, labelsize=17, length=0, pad=10)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(100))
    ax.grid(axis="y", color=grid, lw=1.0, ls="-")
    ax.set_axisbelow(True)
    for k, sp in ax.spines.items():
        sp.set_visible(k == "bottom")
        sp.set_color(grid)
    ax.set_xlim(-0.8, len(order) - 0.4)
    lo, hi = min(ys), max(ys)
    ax.set_ylim(lo - 70, hi + 90)
    fig.subplots_adjust(left=0.11, right=0.95, top=0.78, bottom=0.12)
    fig.savefig(out, facecolor=surface)


def main() -> None:
    out, paths = sys.argv[1], sys.argv[2:]
    games = load(paths)
    elo = bradley_terry(games)
    n_games = defaultdict(int)
    for (a, b), (wa, wb) in games.items():
        n_games[a] += wa + wb
        n_games[b] += wa + wb
    print("| wrestler | Elo | bouts |\n|---|---|---|")
    for n in sorted(elo, key=elo.get, reverse=True):
        print(f"| {n} | {elo[n]:.0f} | {n_games[n]} |")
    chart(elo, out)
    print(f"chart -> {out}")


if __name__ == "__main__":
    main()
