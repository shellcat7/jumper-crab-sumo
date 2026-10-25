# Method

How two Jumper robots were taught sumo. The code is the patch in `jumper_patch/`; the paths below
are under `jumper/tasks/jumper/`.

## 1. The ring and the rules (`sumo/env_cfg.py`, `sumo/mdp.py`, `sumo/referee.py`)

- **Scale.** A real dohyō (4.55 m across) scaled to the crab at k ≈ 0.35: the ring's radius is
  0.8 m, with straw bales on the circle, shikiri lines, and tokudawara (the bales set out at the
  four sides).
- **Losing**, as for a person, but on six feet:
  - *Stepped out*: a foot touches the ground beyond the straw, at the moment it touches
    (`mdp.stepped_out`).
  - *Touched down*: any part other than a foot touches the clay (`mdp.touched_down`).
  - *Fallen*: tipped past `FALL_ANGLE`.

  The same tests end the training episodes and drive the referee's call.
- **Shikiri and tachiai.** Both wrestlers settle behind their lines with their claws raised. The
  bout starts only once **both** have put both claws down behind their lines (`BoutState.hands`,
  `line_x`).
- **Matta.** A wrestler whose base drifts more than `MATTA_DRIFT` before the tachiai is a false
  start, and so is a shikiri that runs too long without the hands down. The shikiri then restarts.
- **The ceremony** (chiri-chōzu, replays only, `sumo/arms.py`): the pincers clap twice, then both
  claws rise into a V with the pincers wide open, to show "no weapons". The V was chosen with
  forward kinematics so that no arm passes within 1.6 cm of a leg.

## 2. Two levels per wrestler

### Legs (frozen)
The official posture/walking policy follows a velocity and posture command and outputs 20 joint
targets. For sumo it was **retrained to push** (`sumo_cont700`): the same architecture and inputs,
continued in the sumo ring. This mattered more than anything else (see RESULTS).

### Tactician (learned, `sumo/tactics/`)
A PPO policy that does not move joints. At every control step (200 Hz) it chooses:

| Output | Range | Meaning |
|---|---|---|
| vx, vy, yaw rate | [-0.30, 0.45] m/s, ±0.30 m/s, ±0.75 rad/s | where to go |
| pitch, roll, height | [-0.10, 0.20] rad, ±0.15 rad, [0.080, 0.130] m | how to stand |
| claw L, claw R | [-1, 1] | open and reach, hold, or let go |
| strike L, strike R | [-1, 1] | strike with that arm |
| kind, power | [-1, 1] | slap down (hataki), thrust or open-claw slap (tsuki / harite), or upward (kachiage); and the strike's speed |
| stance | [-1, 1] | how deep the shikiri crouch is |

The six commands are smoothed with a time constant of 0.08 s so that the legs see a command as
smooth as a stick.

**Observation** (28 base inputs):
- its own body-frame velocity and gravity vector;
- the opponent's position, heading, velocity and tilt in the robot's frame;
- the way to the centre and how near each robot is to the straw;
- whether the bout has started, and for how long.

Options:
- `JUMPER_TACTICS_TREND=1`: a moving average of the opponent's motion.
- `JUMPER_TACTICS_WAZA=1`: 16 claw inputs (claw tips at the opponent, grips both ways, strikes).
- `JUMPER_TACTICS_MEMORY=1`: the last second of the bout.

The critic also sees whether the two are touching.

**Claws** (`tactics/waza.py`).
- A grip is a capped virtual spring (at most 12 N, about what one arm servo holds at the claw's
  reach) between the claw tip and the point it closed on: the opponent's shell, a leg segment
  (which makes the leg pick, ashitori, possible) or an arm segment (the arm pull, tottari). The
  grip breaks when it is stretched too far.
- The spring is applied as equal and opposite external forces, so a held leg really is pulled off
  the clay.
- Strikes are arm trajectories laid over the legs' joint targets. Their effect on the opponent is
  the contact physics.

### Rewards (`tactics/mdp.py`, `tactics/waza.py`)

| Term | |
|---|---|
| bout | +30 for a win, −30 for a loss |
| push | the opponent's outward progress while in contact |
| edge | a penalty for standing near the straw |
| engage | a small pull to stay in contact after the tachiai |
| smoothness | a penalty on jerky commands |
| claws (by style) | grips held, strikes landed, an opponent turned while held, a bonus for the style's kimarite |
| `JUMPER_TACTICS_VARIETY=b` | a win also pays b × (1 − that kimarite's share of recent wins): rare techniques pay more |
| `JUMPER_TACTICS_COMEBACK=b` | a win also pays b × how deep the winner had been driven towards the straw |
| `JUMPER_WAZA_CLAW_BONUS=b` | a win by a claw technique (ashitori, tottari, tsukidashi, tsukitaoshi, hatakikomi, uwatenage, shitatenage) also pays b |

### PPO (`tactics/rl_cfg.py`)
- Networks: actor and critic MLPs (256, 128, 64), initial std 0.5. With the claws, the std is
  clamped to [0.05, 1.5]: unclamped, the noise on the unused claw outputs grew to 3.3.
- γ = 0.999 (a 5 s horizon at 200 Hz: the prize is paid once, when the bout ends), λ = 0.95^¼.
- Learning rate 5e-4 with an adaptive KL target of 0.01, entropy 0.005.
- Per iteration: 48 steps per environment, 5 epochs × 4 minibatches.
- 2,560–4,096 parallel environments on MuJoCo Warp (GPU). A run is 1,000–4,000 iterations.

## 3. League training (`tactics/league.py`)
Every bout, each environment draws its opponent from a **roster**. The roster mixes the official
walkers (walking at the robot), scripted styles (the pusher, the circler, the reader) and frozen
tacticians from every earlier generation, each on its own legs and with its own options. The
roster format is `name|kind|checkpoint|legs|trend|waza|memory`. A new run usually starts from the
best tactician so far and meets frozen copies of itself.

## 4. The referee (gyoji) (`sumo/gyoji.py`, `sumo/referee.py`, `sumo/gyoji_policy/`)
- **Body.** A third Jumper in the ring, recoloured purple. It is a real body, so the wrestlers can
  bump into it. It walks on five legs with its own walking policy (`gyoji_walk`, model
  `gyoji_walker_gwalk2`) and holds the gunbai in its right claw.
- **Watching policy** (`gyoji_policy`, model `gyoji15`). It is trained with the wrestlers drawn
  from the league, and is paid for:
  - a side view of the pair from the watching distance, from the far side of the front when it can;
  - never being touched or within reach, and not standing out on the straw;
  - crouching to see the feet when the bout reaches the straw;
  - facing the winner at the decision;
  - having seen the loser clearly at the moment of the call (`saw_decision`).
- **The calls.** 見合って (face each other), 手をついて (hands down), 待ったなし (no more waiting),
  はっけよい (go), のこった (still in) while they move, then 勝負あり (decided) with the winner and
  the **kimarite**.
- **Kimarite** are named from what the simulator knows at the decision: contact between the two,
  whether their bodies were pressed together, who is behind whom, which way the loser was moving,
  and the claw grips.
  - oshidashi / oshitaoshi: out or down, bodies apart.
  - yorikiri / yoritaoshi: driven out or down, bodies pressed together.
  - tsukidashi / tsukitaoshi: thrust apart.
  - okuridashi: from behind.
  - hikiotoshi: the loser went forward.
  - ashitori / tottari: by a held leg or arm.
  - isamiashi / koshikudake: the loser stepped out or fell by itself.
  - dōtai: both at once, so the bout is re-fought (torinaoshi).

## 5. Evaluation (`tools/viewer/play_sumo.py`, `tools/eval/`)
- **Round robins.** Each pairing is fought in 1,024 parallel rings for 20 s (the first bout only,
  on MuJoCo Warp), with the referee judging. The results are kept as CSV (east, west, wins, dōtai,
  kimarite counts, checkpoint SHA-256).
- **One strength scale** (`lineage.py`). A Bradley–Terry model is fitted by MM iteration to every
  result at once and expressed on the Elo scale (400 points = 10:1 odds), anchored at
  waza24 = 1500. In all, there are 370,000+ bouts.
- **Choosing the bout to film** (`drama.py`). Hundreds of seeds are played without pictures
  (`JUMPER_NO_FRAMES=1`), and each first bout is scored from the referee's log and both bases'
  tracks: seconds locked chest to chest, of those seconds in the middle, the loser held at the
  straw, a comeback by the winner, pressure turnovers, and a kimarite other than the push-out. The
  filmed replay was checked to give the same call as the search.

## 6. From simulation to film (`tools/viewer/blender_dump.py`, `tools/blender/`, `tools/media/`)
1. **Dump.** The replay records every moving geom's pose per frame, plus the referee's captions, to
   an `.npz` file.
2. **Render.** Blender 5.2 (EEVEE) rebuilds the scene: the robots from the toolkit's meshes, a
   procedural arena and hanging roof, and a crowd of 1,000+ seated Jumpers built from the same
   meshes. The crowd is decoration, posed per frame and outside the physics.
3. **Graphics.** `compose_video.py` draws the broadcast graphics: shikona, rank, record, head to
   head and kimarite in Japanese, with English under them, inside the bottom 10 % safe area.
4. **Extra shots.** `verdict_shot.py` renders the video review: subframes between the dump's frames
   give a 3× slow motion, and the foot is ringed. `trophy_shot.py` renders the champion with an
   original cup.
5. **Sound.** Everything is synthesized in NumPy:
   - the hall: 140 clappers, each at its own pace, in the hall's reverberation;
   - the clappers (hyoshigi) and the pincers' clacks;
   - the score (`score.py`): the yagura drum (skin and rim), odaiko, shime, atarigane, shamisen
     (Karplus–Strong with sawari buzz), shinobue and koto, cued to the cut list and levelled to
     −16 LUFS.
