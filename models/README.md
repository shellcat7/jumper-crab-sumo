# Models

PyTorch checkpoints from the toolkit's trainer (rsl_rl). Check them with
`sha256sum -c SHA256SUMS`. Elo is from `docs/RESULTS.md`.

| File | What it is | Trained with | Elo |
|---|---|---|---:|
| `legs_sumo_cont700.pt` | **Legs.** The official posture/walking policy, continued in the sumo ring to push. Follows a velocity + posture command, outputs 20 joint targets. Used by every wrestler below. | `jumper.sumo` | — |
| `tactician_waza47_hasami-no-umi.pt` | **鋏ノ海 Hasami-no-Umi**, finalist (east). From waza39, with the variety bonus (30). | `jumper.sumo.tactics`, trend + claws + memory | 1667 |
| `tactician_waza39_hasami-no-sato.pt` | **鋏ノ里 Hasami-no-Sato**, finalist (west), winner of the filmed final. A pusher: 70 % of its wins are push-outs. | `jumper.sumo.tactics`, trend + claws + memory | 1690 |
| `tactician_waza50.pt` | From waza46, league-heavy, with the variety bonus (30). Won the later top-7 round robin. | same | 1673 |
| `tactician_waza70_claws.pt` | From waza63, claw bonus 150 and comeback 40, about 1,500 iterations. 13–15 % of its wins by claw techniques (mostly tsukidashi). Beats waza50 1,422–1,156. | same | 1684 |
| `gyoji15_referee.pt` | **The referee's watching policy**: where to stand and look (4 commands to its legs). | `jumper.sumo.gyoji_policy` | — |
| `gyoji_walker_gwalk2.pt` | **The referee's legs**: walking on five legs, the gunbai in one claw. | `jumper.gyoji_walk` | — |

Every tactician needs the options it was trained with (`JUMPER_TACTICS_TREND=1`,
`JUMPER_TACTICS_WAZA=1`, `JUMPER_TACTICS_MEMORY=1`, `JUMPER_WAZA_LEAN=0.3`): they change the
observation width. See `docs/REPRODUCE.md`.

**Intended use:** replaying and studying the bouts in simulation. These policies have not been run
on a real Jumper. The legs were trained in simulation only, so the hardware's safety limits have
not been checked against them.

License: Apache-2.0, as the rest of the repository.
