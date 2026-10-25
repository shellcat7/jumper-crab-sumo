# Reproduce

Tested on Windows 11 with Python 3.11 (versions in `requirements-lock-windows-py311.txt`) and on
Google Colab for training. MuJoCo replays on the CPU are deterministic for a given seed on the
same versions; other versions or platforms may play a different bout.

## 1. The patched toolkit

```bash
git clone https://github.com/KingKongRobotics/jumper.git
cd jumper
git checkout 61d065219fca767f3142c8f10aff59eae5a5a004
git apply ../jumper-crab-sumo/jumper_patch/jumper-sumo.patch
```
Set up the toolkit's Python environment as its own README says. The commands below run from the
patched `jumper/` checkout with `M=../jumper-crab-sumo/models` and `T=../jumper-crab-sumo/tools`.
Check the models first with `sha256sum -c` in `models/`.

## 2. Replay the final

```bash
JUMPER_SEED=118 JUMPER_SUMO_GYOJI=1 JUMPER_SUMO_AUDIENCE=1 JUMPER_SUMO_CEREMONY=5 \
JUMPER_SUMO_GYOJI_POLICY=$M/gyoji15_referee.pt JUMPER_SUMO_GYOJI_WALKER=$M/gyoji_walker_gwalk2.pt \
JUMPER_GYOJI_SPEED=0.45,0.35,1.0 JUMPER_GYOJI_SMOOTH_S=0.05 JUMPER_GYOJI_LEAD_S=0.3 \
JUMPER_GYOJI_SIDE_POWER=6 JUMPER_GYOJI_CROUCH_ENGAGED=1 \
JUMPER_TACTICS_TREND=1 JUMPER_TACTICS_WAZA=1 JUMPER_TACTICS_MEMORY=1 JUMPER_WAZA_LEAN=0.3 \
JUMPER_TACTICS_LOW=$M/legs_sumo_cont700.pt \
JUMPER_SUMO_OPP_TACTICIAN=$M/tactician_waza39_hasami-no-sato.pt JUMPER_SUMO_OPP_TACTICIAN_LOW=$M/legs_sumo_cont700.pt \
JUMPER_SUMO_OPP_TREND=1 JUMPER_SUMO_OPP_WAZA=1 JUMPER_SUMO_OPP_WAZA_LEAN=0.3 JUMPER_SUMO_OPP_MEMORY=1 \
JUMPER_SUMO_NAMES="鋏ノ海,鋏ノ里" JUMPER_VIDEO=final.mp4 \
JUMPER_CAM_LOOKAT=0,0.05,0.03 JUMPER_CAM_FOLLOW=bout JUMPER_CAM_DISTANCE=1.4 JUMPER_CAM_ELEVATION=-24 JUMPER_CAM_AZIMUTH=90 \
python $T/viewer/render_offscreen.py --task jumper.sumo.tactics \
  --checkpoint $M/tactician_waza47_hasami-no-umi.pt --backend native --device cpu --cpu_threads 8 --headless --steps 5700
```
- The referee's calls are printed. The bout should end `25.76s 勝負あり 西の勝ち 足取り`
  (decided, west wins, ashitori).
- `JUMPER_NO_FRAMES=1` instead of `JUMPER_VIDEO` plays without pictures, which is fast; this is how
  seeds were searched. `JUMPER_SUMO_LOG=bout.csv` writes both bases' tracks for `tools/eval/drama.py`.
- `JUMPER_BLENDER_DUMP=dump.npz` writes the bout for Blender.
- Other pairings: swap the checkpoints, e.g. `tactician_waza70_claws.pt` v `tactician_waza50.pt`,
  seed 302 (tsukidashi at 6.98 s, no ceremony, `--steps 1800`).

## 3. Render, cut and score

Blender 5.2 and ffmpeg are needed (set `FFMPEG` and `FFPROBE` if they are not on the PATH). The
graphics use Windows fonts: HG明朝E, BIZ UD Gothic and Segoe UI.

```bash
blender -b -P $T/blender/sumo_scene.py -- --dump dump.npz --venue arena --out frames/ --intro 8 \
  --engine eevee --samples 20 --res 1920x1080
python $T/blender/compose_video.py dump.npz frames/ final.mp4 --card spec/card_final.json --intro 8 --single
python -I $T/blender/sound_design.py dump.npz final.mp4
blender -b -P $T/blender/verdict_shot.py -- dump.npz verdict/ 3 16     # the video review's frames
python $T/blender/verdict_clip.py verdict/ spec/card_final.json verdict.mp4
python -I $T/media/fill_v2.py filled.json spec/final_video.json         # resolves the clip placeholders
python -I $T/media/make_final.py filled.json nomusic.mp4                # cards, captions, cuts
python -I $T/media/score.py filled.json nomusic.mp4 video.mp4           # the music and the final mix
```

## 4. Train

Training uses the toolkit's trainer and the task `jumper.sumo.tactics`. Start from an earlier
tactician and play against a league (`JUMPER_SUMO_LEAGUE`, a roster file with lines
`name|kind|checkpoint|legs|trend|waza|memory`):
```bash
JUMPER_SUMO_LEAGUE=roster.txt JUMPER_TACTICS_TREND=1 JUMPER_TACTICS_WAZA=1 JUMPER_TACTICS_MEMORY=1 \
JUMPER_TACTICS_LOW=$M/legs_sumo_cont700.pt JUMPER_WAZA_LEAN=0.3 \
JUMPER_TACTICS_VARIETY=30 JUMPER_TACTICS_COMEBACK=40 JUMPER_WAZA_CLAW_BONUS=150 \
python scripts/train.py --task jumper.sumo.tactics --backend warp --device cuda:0 --headless \
  --max-iterations 1500 --checkpoint $M/tactician_waza39_hasami-no-sato.pt
```
- The referee's own policy trains with `--task jumper.sumo.gyoji_policy`, and its legs with
  `jumper.gyoji_walk`.
- Round robins: `tools/viewer/play_sumo.py` with `JUMPER_SUMO_REFEREE=1`. Ratings:
  `python -I tools/eval/lineage.py chart.png results/*.csv`.
