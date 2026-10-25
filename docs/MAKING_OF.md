# Making of: what failed, and what we changed

Two and a half weeks, October 9–25, 2026. The challenge asks entrants to share the ideas, the
debugging and the funny setbacks, so here they are, roughly in order.

## Timeline

| Dates | Milestone |
|---|---|
| Oct 9 | Toolkit set up. The official gesture task trained on a cloud GPU and replayed on a laptop. |
| Oct 10 | A ring and two Jumpers. First push training. The real rules: hands down for the tachiai, the foot over the straw. |
| Oct 11–13 | Claws (grips, strikes). Leagues of earlier generations. Wrestler styles. |
| Oct 14 | A learned referee (gyoji) and a five-legged walk for it. |
| Oct 19–20 | Round robins and a rating scale. The Blender arena, crowd and roof. The first broadcast-style bout. The hall's sound. |
| Oct 21 | The legs turn out to be the bottleneck. The pusher legs. The search for a dramatic final. |
| Oct 22–23 | The final picked. Long claw training. Drafts reviewed. |
| Oct 23–25 | Sound, captions, the video review, the chart and the colours. The final cut. |

## Setbacks and fixes

**Day 1: they just bumped.** Two official walkers, told to walk at each other in 256 rings for
30 s each, produced 0 wins and 0 losses: they leaned on each other and nobody went out. Training
the walking policy to push gave 86–0 against the official walker, then 1,197–2. *Fix: a sumo
reward (win or loss, outward push, edge penalty).*

**Every kimarite was yorikiri.** The first referee counted a claw touching the opponent as a belt
grip. A pushing crab always has a claw on the other, so the first 86 decisions were all force-outs.
*Fix: yori and oshi are told apart by whether the bodies are pressed together, not by claw
contact.*

**False starts and floating feet.** The wrestlers looked as if they put their hands down before the
call, a back leg lifted in the crouch, and the referee bobbed. *Fixes:*
- The claws are raised to 11.5 cm in the shikiri.
- The middle legs step forward under the shifted centre of mass.
- The tachiai waits until both pairs of claws are down behind the lines.
- A drift before then is called *matta*.

**The claw wrestler that never used its claws.** At first the policy had to close a claw at
exactly the right moment, and no run found that by chance. *Fix: a claw that is open and within
reach closes and holds by itself, and the policy only decides to reach or let go.* The claw
outputs then sat idle while their exploration noise grew to std 3.3. *Fix: the std is clamped to
[0.05, 1.5].*

**Lean v claws.** Rewarding a strong forward lean made claw users that were weak. A weak lean made
strong wrestlers that never used the claws. We kept the lean at 0.3 and paid for claw wins
directly. That only worked with long runs: about 350 iterations gave 2–5 % claw wins, and about
1,500 iterations gave 13–15 %.

**It was the legs.** For a week, the old league champion beat every new tactician. A swap test put
the same brain on different legs: waza32 on legs retrained to push beat waza32 on the agile legs
1,561–375. Every tactician since runs on the pusher legs.

**The referee.** A scripted referee stood in the way and was knocked over. The learned one (gyoji)
is paid for:
- a side view of the bout;
- staying out of reach;
- crouching to see the feet at the straw;
- having seen the loser clearly when it calls.

It walks on five legs because one claw holds the gunbai.

**Rendering.**
- The GPU driver crashed every few dozen frames in long Blender renders. *Fix: every renderer
  resumes from the frames already written.*
- One night the PNG frames filled the disk. *Fix: a free-space check, and finished frames are
  moved to cloud storage.*
- The broadcast graphics crashed on the head-to-head record: the kanji numerals stopped at 999,
  and the record was 1,305. *Fix: numerals up to 9,999.*

**Searching for drama without pictures.** The crowd is posed every frame, so a replay without
the crowd is far cheaper. The crowd is decoration outside the physics, so the bout is the same. We
checked this: the searched and the filmed runs of the same seed gave identical calls.

**The sound, three times.**
1. The crowd as band-limited noise sounded like rain, or wind.
2. Synthetic voices (a glottal pulse through vowel formants, 40 talkers) sounded like a horror game.
3. No voices at all: the hall is applause (140 clappers, each at its own pace), the wooden
   clappers, the pincers' clacks, and a synthesized score built on the yagura drum that a sumo
   hall is known for.

**Captions under the player's controls.** The English lines sat at the bottom 5 % of the frame,
where phone players draw their controls. *Fix: all graphics inside the bottom 10 % safe area.*
Hiding the simulator's own score strip with a black bar made the picture look squashed. *Fix: a
slight crop and zoom pushes the strip out of the frame.*

**Courtesy.** The champion first hopped for joy with the cup. Sumo values restraint in victory, so
we removed it. A bow tilted about the front feet lifted the back legs off the clay and read as a
hop again. The champion now stands still and only the camera moves.

## How we worked
One person directed and reviewed every draft. An AI coding assistant wrote and ran most of the
code, the training jobs and the renders, and kept a lab notebook. Training ran on cloud GPUs
(several runs at a time). Replays, evaluation and rendering ran on a laptop.
