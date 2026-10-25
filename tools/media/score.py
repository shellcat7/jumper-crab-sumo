"""The submission's music and its final mix (10/23, the user on draft v3: the sound was odd -- rain or
wind -- and wants real sound effects and music; on v4: the music monotonous, "more exciting", and
大相撲らしく -- like grand sumo). Everything is synthesized here, as the hall's sound is
(docs/capture-guide.md: music only if it is our own, with clear rights).

The sound of a sumo hall is the yagura-daiko -- one big drum beaten with two thin sticks from the
tower, its skin and its rim, in quick bouncing rhythms -- and the hyoshigi, the wooden clappers.
The music is built on those: the yagura drum's patterns (denser as the section rises), the clappers
closing the phrases and marking the cuts, a big odaiko for the moments on the picture, hands on two
and four (as a sumo jinku is clapped), a shamisen driving sixteenths over a chord sequence, a
shinobue (the festival flute) with the hook, the koto only where it is calm. Miyako-bushi on D for
the drama, the yo scale for the celebration.

    python -I tools/media/score.py FILLED_SPEC.json VIDEO.mp4 OUT.mp4

The spec (make_final's, filled) gives every segment's length; each item's `cue` says what the music
does under it, and `hits` (seconds in the clip) where a big drum lands on the picture (the tachiai, a
decision). The video's own sound (the hall's hands, the knocks, the clappers) is kept under the
music; the whole is levelled to -16 LUFS (EBU R128; YouTube and X play near it).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
SR = 48000
rng = np.random.default_rng(20261023)

D4 = 293.66
MIYAKO = [0, 1, 5, 7, 8]          # D Eb G A Bb: the drama
YO = [0, 2, 5, 7, 9]              # D E G A B: the celebration


def note(scale, degree, octave=0):
    """The frequency of a scale degree (0.. wraps into octaves) above D4."""
    o, d = divmod(degree, len(scale))
    return D4 * 2 ** ((scale[d] + 12 * (o + octave)) / 12)


def _bp(x, lo, hi, order=2):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    m = np.ones_like(f)
    if lo:
        m *= 1 / (1 + (lo / np.maximum(f, 1e-3)) ** (2 * order))
    if hi:
        m *= 1 / (1 + (f / hi) ** (2 * order))
    return np.fft.irfft(X * m, len(x))


def _sd():
    sys.path.insert(0, str(ROOT / "tools" / "blender"))
    import sound_design as sd
    sd.SR = SR
    return sd


# ── instruments: each returns a mono signal at SR, peak about 1 ─────────────────────────────

def odaiko(f0=58.0, vel=1.0, dur=2.2):
    """The big drum: a membrane's modes with the pitch settling after the strike, the skin's slap
    and the stick's click on top, a sub-bass push under."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f0 * (1 + 0.22 * np.exp(-t / 0.035))
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = sum(a * np.sin(ph * k) * np.exp(-t / tau)
               for k, a, tau in ((1, 1.0, 0.65), (1.59, 0.45, 0.32), (2.14, 0.25, 0.2), (2.65, 0.12, 0.12)))
    skin = _bp(rng.standard_normal(n), 60, 900) * np.exp(-t / 0.018) * 0.9
    click = _bp(rng.standard_normal(n), 1500, 6000) * np.exp(-t / 0.003) * 0.25
    sub = np.sin(2 * np.pi * 36 * t) * np.exp(-t / 0.3) * 0.6
    y = body + (skin + click) * (0.6 + 0.4 * vel) + sub
    return y / np.max(np.abs(y)) * vel


@lru_cache(maxsize=None)
def _yagura(strong, var):
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    r = np.random.default_rng(100 + var)
    f0 = 92.0 if strong else 98.0
    f = f0 * (1 + 0.12 * np.exp(-t / 0.02))
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = sum(a * np.sin(ph * k) * np.exp(-t / tau)
               for k, a, tau in ((1, 1.0, 0.32 if strong else 0.18), (1.59, 0.5, 0.15), (2.14, 0.3, 0.09), (2.92, 0.15, 0.05)))
    stick = _bp(r.standard_normal(n), 1200, 7000) * np.exp(-t / 0.0035) * (0.9 if strong else 0.6)
    skin = _bp(r.standard_normal(n), 120, 1500) * np.exp(-t / 0.01) * 0.5
    y = body + stick + skin
    return y / np.max(np.abs(y))


def yagura(vel=1.0, strong=True, var=0):
    """The yagura-daiko: the tower's big drum under two thin sticks -- brighter and shorter than the
    odaiko's mallet, the stick's crack on the skin."""
    return _yagura(bool(strong), var % 4) * vel


@lru_cache(maxsize=None)
def _fuchi(var):
    n = int(0.1 * SR)
    t = np.arange(n) / SR
    r = np.random.default_rng(200 + var)
    y = (np.sin(2 * np.pi * 1450 * t) * np.exp(-t / 0.014) + 0.5 * np.sin(2 * np.pi * 2600 * t) * np.exp(-t / 0.008)
         + _bp(r.standard_normal(n), 1500, 8000) * np.exp(-t / 0.0018) * 0.7)
    return y / np.max(np.abs(y))


def fuchi(vel=1.0, var=0):
    """The stick on the drum's wooden rim (fuchi): the dry 'ka' between the skin strokes."""
    return _fuchi(var % 4) * vel


def shime(vel=1.0):
    """The small tight drum: a high, short 'ten'."""
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    body = sum(a * np.sin(2 * np.pi * 410 * k * t) * np.exp(-t / tau)
               for k, a, tau in ((1, 1.0, 0.07), (1.59, 0.5, 0.045), (2.14, 0.3, 0.03)))
    crack = _bp(rng.standard_normal(n), 1800, 8000) * np.exp(-t / 0.004) * 0.8
    y = body + crack
    return y / np.max(np.abs(y)) * vel


def kane(vel=1.0):
    """The atarigane, a small hand gong: bright inharmonic partials, a hard strike."""
    n = int(0.8 * SR)
    t = np.arange(n) / SR
    y = sum(a * np.sin(2 * np.pi * 1180 * k * t + rng.uniform(0, 6.3)) * np.exp(-t / tau)
            for k, a, tau in ((1, 1.0, 0.32), (2.32, 0.7, 0.2), (4.25, 0.45, 0.12), (6.63, 0.25, 0.07)))
    y += _bp(rng.standard_normal(n), 3000, 12000) * np.exp(-t / 0.002) * 0.5
    return y / np.max(np.abs(y)) * vel


def hyoshigi(vel=1.0):
    """The wooden clappers (sound_design's): the sumo hall's own sign."""
    y = _sd().hyoshigi(int(0.4 * SR), [0.0], 1.0)
    return y / np.max(np.abs(y)) * vel


def koto(f, vel=1.0, dur=2.6, bright=0.55, t60=None):
    """A plucked silk string (Karplus-Strong at 4x the rate, so the pitch is true, then down-sampled):
    a noise burst through the string's loop, its highs dying first; the bridge's ring under it."""
    up = 4
    sr = SR * up
    N = int(round(sr / f - 0.5))
    n = int(dur * sr)
    exc = _bp(rng.standard_normal(N), 0, (1200 + 7000 * bright) / up) if N > 8 else rng.standard_normal(N)
    exc -= 0.6 * np.roll(exc, int(N * 0.14))            # plucked near the bridge
    y = np.zeros(n + N + 1)
    y[:N] = exc
    g = 10 ** (-3 * (N / sr) / (t60 or (2.4 if f < 500 else 1.6)))
    for s in range(N + 1, len(y), N):
        e = min(s + N, len(y))
        y[s:e] += g * 0.5 * (y[s - N:e - N] + y[s - N - 1:e - N - 1])
    y = resample_poly(y[:n], 1, up)
    t = np.arange(len(y)) / SR
    y += 0.12 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.4) * np.max(np.abs(y))
    y[:int(0.002 * SR)] *= np.linspace(0, 1, int(0.002 * SR))
    return y / (np.max(np.abs(y)) + 1e-9) * vel


@lru_cache(maxsize=None)
def _shamisen(f, dur):
    y = koto(f, 1.0, dur, bright=1.0, t60=0.55)
    y = np.tanh(2.5 * y) / np.tanh(2.5)                  # the sawari's buzz
    t = np.arange(len(y)) / SR
    y += _bp(rng.standard_normal(len(y)), 2000, 8000) * np.exp(-t / 0.0025) * 0.5   # the bachi on the skin
    return y / np.max(np.abs(y))


def shamisen(f, vel=1.0, dur=0.5):
    """A shamisen's struck string: bright, short, buzzing, the plectrum's snap on top."""
    return _shamisen(round(float(f), 2), dur) * vel


def shinobue(f, dur, vel=1.0):
    """The high bamboo festival flute: a quick vibrato, a little breath, a lift into each note."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.008 * np.sin(2 * np.pi * 6.2 * t) * np.clip((t - 0.15) / 0.3, 0, 1)
    scoop = 2 ** (-0.4 / 12 * np.exp(-t / 0.05))
    ph = 2 * np.pi * np.cumsum(f * vib * scoop) / SR
    tone = np.sin(ph) + 0.18 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)
    breath = _bp(rng.standard_normal(n), f * 0.9, f * 3) * 0.15 + _bp(rng.standard_normal(n), 4000, 10000) * 0.03
    env = np.minimum(1, t / 0.04) * np.minimum(1, (dur - t) / 0.08).clip(0, 1)
    y = (tone + breath) * env
    return y / np.max(np.abs(y)) * vel


@lru_cache(maxsize=None)
def _claps(seed):
    bank = _sd()._clap_bank()
    r = np.random.default_rng(seed)
    y = np.zeros(int(0.08 * SR))
    for _ in range(8):
        c = bank[r.integers(len(bank))] * r.uniform(0.5, 1.0)
        k = int(r.uniform(0, 0.016) * SR)
        y[k:k + len(c)] += c[:len(y) - k]
    return y / np.max(np.abs(y))


def claps(vel=1.0, seed=0):
    """Hands clapping together on the beat, as a sumo jinku is clapped: eight people, a few ms apart."""
    return _claps(seed % 6) * vel


def drone(dur, vel=1.0):
    """A low held fifth (D2, A2) breathing slowly: the tension under the final's ceremony."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = sum(a * np.sin(2 * np.pi * f * t + 0.3 * np.sin(2 * np.pi * 0.17 * t)) for f, a in ((73.4, 1.0), (110.0, 0.6),
                                                                                            (146.8, 0.25)))
    y *= (0.75 + 0.25 * np.sin(2 * np.pi * 0.25 * t)) * np.minimum(1, t / 1.5) * np.minimum(1, (dur - t) / 1.5).clip(0, 1)
    return y / np.max(np.abs(y)) * vel


# ── the score ───────────────────────────────────────────────────────────────────────────────

class Score:
    """Two buses: the music (ducked where the picture speaks) and the accents (the big strikes on the
    picture, never ducked); each with a send to the hall's reverberation."""

    def __init__(self, seconds):
        self.n = int(seconds * SR) + SR * 4
        self.dry = np.zeros((2, self.n, 2))
        self.send = np.zeros((2, self.n))

    def put(self, sig, t, gain=1.0, pan=0.0, wet=0.25, acc=False):
        k = int(t * SR)
        if k >= self.n or k + len(sig) <= 0:
            return
        s = sig[max(0, -k):max(0, min(len(sig), self.n - k))] * gain
        k = max(0, k)
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        b = int(acc)
        self.dry[b, k:k + len(s), 0] += s * l * 1.41
        self.dry[b, k:k + len(s), 1] += s * r * 1.41
        self.send[b, k:k + len(s)] += s * wet

    def mix(self):
        sd = _sd()
        return [self.dry[b] + sd.hall(self.send[b], rt60=2.2, wet=1.0, seed=31 + b) for b in (0, 1)]


def _steps(pattern):
    """A 16-step pattern: (step, symbol, velocity 0..1) for each struck step. Digits are skin strokes
    (9 strongest); 'k' is the rim."""
    return [(i, c, int(c) / 9 if c.isdigit() else 0.7) for i, c in enumerate(pattern) if c != "."]


#: the drums by intensity, 16 steps a bar. yagura: the tower drum's skin (digits) and rim (k) in its
#: bouncing figures; odaiko under the bar's turns; shime ticking; hands on two and four; the gong.
DRUMS = {
    1: {"yagura": "9.k59.k.9.k59k5.", "shime": "5.3.5.3.5.3.5.3."},
    2: {"yagura": "9k598k5.9k59k959", "odaiko": "9.......7..5....", "shime": "6353635363536353",
        "claps": "....9.......9...", "shamisen": "7464746474647464"},
    3: {"yagura": "9k599k599k59k999", "odaiko": "9..5..7.9..5..7.", "shime": "7464746474647464",
        "claps": "....9.......9...", "kane": "7.4.7.4.7.4.7.4.", "shamisen": "8565856585658565"},
}
#: the chords, as scale degrees of their roots (in miyako-bushi on D: D, B-flat, G, A -- i VI iv v)
PROG = [0, -1, -3, -2]
#: the shamisen's sixteenths over a chord, scale degrees above its root
RIFF = [0, 2, 1, 2, 0, 2, 3, 2, 0, 2, 1, 2, 4, 3, 2, 1]
#: the flute's hook, four bars: (degree, beats)
HOOK = [[(3, 1), (4, 0.5), (3, 0.5), (2, 1), (0, 1)], [(1, 0.5), (0, 0.5), (1, 1), (2, 2)],
        [(3, 1), (4, 1), (5, 1), (4, 0.5), (3, 0.5)], [(2, 1.5), (3, 0.5), (0, 2)]]


def band(sc, t0, t1, bpm, level, scale, intensity, hook=False, hook_oct=1, bar0=0, seed=0):
    """The ensemble from t0 to t1 at an intensity of 1-3 (DRUMS), the shamisen on the chords of
    PROG, the flute's hook on top; every fourth bar closes with a yagura run and the clappers.
    Returns the number of bars played."""
    r = np.random.default_rng(seed)
    beat = 60 / bpm
    st = beat / 4
    pat = DRUMS[intensity]
    nbar = int((t1 - t0) / (4 * beat) + 0.999)
    for b in range(nbar):
        tb = t0 + b * 4 * beat
        root = PROG[(bar0 + b) % len(PROG)]
        fill = (bar0 + b) % 4 == 3
        for name, p in pat.items():
            for i, c, v in _steps(p):
                t = tb + i * st + r.normal(0, 0.002)
                if t >= t1 or (fill and i >= 12 and name in ("yagura", "shime")):
                    continue
                if name == "yagura":
                    if c == "k":
                        sc.put(fuchi(0.8, var=i), t, level * 0.28, pan=0.25, wet=0.12)
                    else:
                        sc.put(yagura(0.5 + 0.5 * v, strong=v > 0.8, var=i), t, level * 0.75 * v, pan=-0.1, wet=0.25)
                elif name == "odaiko":
                    sc.put(odaiko(vel=0.6 + 0.4 * v, dur=1.4), t, level * 0.75 * v, wet=0.3)
                elif name == "shime":
                    sc.put(shime(0.4 + 0.6 * v), t, level * 0.25 * v, pan=0.4, wet=0.12)
                elif name == "claps":
                    sc.put(claps(v, seed=b * 16 + i), t, level * 0.45 * v, wet=0.3)
                elif name == "kane":
                    sc.put(kane(0.5 + 0.5 * v), t, level * 0.11 * v, pan=0.55, wet=0.2)
                elif name == "shamisen":
                    sc.put(shamisen(note(scale, root + RIFF[i], 0), 0.5 + 0.5 * v), t, level * 0.22 * v, pan=0.3,
                           wet=0.15)
        if fill:                                     # the yagura's run, the clappers on the turn
            for i in range(12, 16):
                for half in (0, 0.5):
                    t = tb + (i + half) * st
                    if t < t1:
                        sc.put(yagura(0.75 + 0.06 * (i - 12), strong=False, var=i), t, level * 0.6, pan=-0.1, wet=0.25)
            t = tb + 16 * st
            if t < t1 + 0.05:
                sc.put(hyoshigi(0.9), t - 0.01, level * 0.35, pan=0.2, wet=0.35)
        if hook:
            t = tb
            for d, ln in HOOK[(bar0 + b) % len(HOOK)]:
                if t + 0.05 < t1:
                    sc.put(shinobue(note(scale, d, hook_oct), min(ln * beat, t1 - t) + 0.05, 0.8), t, level * 0.33,
                           pan=0.1, wet=0.35)
                t += ln * beat
    return nbar


def roll(sc, t0, t1, level, scale=None):
    """The drum roll into the final: the yagura in a quickening run, the odaiko swelling under it,
    the shamisen's tremolo climbing the scale, the clappers' own quickening run on top."""
    t, k = t0, 0
    while t < t1:
        u = (t - t0) / (t1 - t0)
        sc.put(yagura(0.5 + 0.4 * u, strong=k % 4 == 0, var=k), t, level * (0.3 + 0.5 * u), pan=-0.1, wet=0.25)
        if k % 4 == 0:
            sc.put(odaiko(vel=0.35 + 0.5 * u, dur=1.0), t, level * (0.3 + 0.6 * u), wet=0.35)
        if scale is not None and k % 2 == 0:
            sc.put(shamisen(note(scale, int(u * 6), 0), 0.8), t, level * 0.2 * (0.4 + 0.6 * u), pan=0.3, wet=0.15)
        t += 0.13 - 0.07 * u
        k += 1
    for j, dt in enumerate(np.cumsum(np.geomspace(0.22, 0.05, 9))):
        if t1 - 1.2 + dt < t1:
            sc.put(hyoshigi(0.8), t1 - 1.2 + dt, level * 0.3, pan=0.2, wet=0.3)


def gliss(sc, t, scale, start, steps, gap, level, octave=0, pan=-0.3):
    for i in range(steps):
        sc.put(koto(note(scale, start + i, octave), 0.75), t + i * gap, level, pan=pan, wet=0.35)


def big(sc, t, level=1.0):
    """A big strike for a moment on the picture (a tachiai, a decision)."""
    sc.put(odaiko(vel=1.0), t, level, wet=0.45, acc=True)
    sc.put(yagura(0.9), t + 0.004, level * 0.4, pan=0.15, wet=0.3, acc=True)


def triumph(sc, t, scale, level=1.0):
    """After a decision: the clappers, the shamisen's strum, the flute ringing high, the hands."""
    sc.put(hyoshigi(1.0), t + 0.05, level * 0.45, pan=0.2, wet=0.35, acc=True)
    for j, d in enumerate((0, 2, 4, 7)):
        sc.put(shamisen(note(scale, d, 0), 0.9), t + 0.15 + j * 0.03, level * 0.3, pan=0.3, wet=0.25, acc=True)
    sc.put(shinobue(note(scale, 5, 1), 1.4, 0.9), t + 0.2, level * 0.38, pan=0.1, wet=0.4, acc=True)
    for i in range(6):
        sc.put(claps(0.9, seed=i), t + 0.6 + i * 0.24, level * 0.32, wet=0.35, acc=True)


def heartbeat(sc, t0, t1, i0, i1, level, accel=1.0):
    """The tension: the odaiko like a heartbeat (lub-dub), its interval going from i0 to i1."""
    t = t0
    while t < t1 - 0.1:
        u = ((t - t0) / max(1e-6, t1 - t0)) ** accel
        sc.put(odaiko(vel=0.5 + 0.3 * u, dur=1.2), t, level * (0.45 + 0.4 * u), wet=0.35)
        sc.put(odaiko(vel=0.35 + 0.3 * u, dur=1.0), t + 0.16, level * (0.3 + 0.3 * u), wet=0.35)
        t += i0 + (i1 - i0) * u


def _run(items, starts, idx, cues):
    """The end time of the run of items from idx whose cue is in `cues`."""
    k, end = idx, starts[idx]
    while k < len(items) and items[k].get("cue") in cues:
        end = starts[k] + float(items[k]["seconds"])
        k += 1
    return k, end


def compose(spec):
    items = spec["items"]
    starts, t = [], 0.0
    for it in items:
        starts.append(t)
        t += float(it["seconds"])
    total = t
    sc = Score(total)
    duck = np.ones(int(total * SR) + SR * 4)        # the music's level over the video's own sound
    bar = 0                                          # the chord sequence runs on across sections
    for idx, (it, t0) in enumerate(zip(items, starts)):
        t1 = t0 + float(it["seconds"])
        cue = it.get("cue", "")
        prev = items[idx - 1].get("cue", "") if idx else ""
        hits = [t0 + (h - float(it.get("start", 0.0))) / float(it.get("speed", 1.0)) for h in it.get("hits", [])]
        if cue == "cold":
            for h in hits:
                big(sc, h, 1.0)
            k, i = t0 + 1.2, 0
            while k < t1:                            # the yagura's run building into the title
                u = (k - t0 - 1.2) / (t1 - t0 - 1.2)
                sc.put(yagura(0.4 + 0.5 * u, strong=False, var=i), k, 0.2 + 0.4 * u, pan=-0.1, wet=0.25)
                k += 0.14 - 0.06 * u
                i += 1
        elif cue == "title":
            big(sc, t0, 0.95)
            sc.put(hyoshigi(1.0), t0 + 0.02, 0.4, pan=0.2, wet=0.4)
            sc.put(shinobue(note(MIYAKO, 3, 1), 1.6, 0.9), t0 + 0.5, 0.34, pan=0.1, wet=0.4)
            for j, dt in enumerate((2.05, 2.3, 2.55, 2.68, 2.81)):
                sc.put(yagura(0.9, strong=j in (0, 1, 4), var=j), t0 + dt, 0.7, pan=-0.1, wet=0.25)
        elif cue == "demo":
            if prev != "demo":
                _, end = _run(items, starts, idx, ("demo",))
                bar += band(sc, t0, end, 124, 0.6, MIYAKO, 1, bar0=bar, seed=idx)
            for h in hits:
                big(sc, h, 0.85)
            duck[int(t0 * SR):int(t1 * SR)] = 0.85
        elif cue in ("intro", "h2h"):
            if prev not in ("intro", "h2h"):
                k, end = _run(items, starts, idx, ("intro", "h2h"))
                h2h0 = next((starts[j] for j in range(idx, k) if items[j].get("cue") == "h2h"), end)
                bar += band(sc, t0, h2h0, 124, 0.85, MIYAKO, 2, hook=True, bar0=bar, seed=idx)
                if h2h0 < end:
                    bar += band(sc, h2h0, end, 124, 0.95, MIYAKO, 3, hook=True, bar0=bar, seed=idx + 50)
            if it.get("profile"):
                sc.put(odaiko(vel=0.9), t0, 0.6, wet=0.35)
                sc.put(hyoshigi(0.9), t0 + 0.01, 0.3, pan=0.2, wet=0.35)
        elif cue == "hl":
            # the highlight cut (10/24): the ensemble at full tilt under every "hl" item in a row, the
            # moments on the picture struck
            if prev != "hl":
                _, end = _run(items, starts, idx, ("hl",))
                bar += band(sc, t0, end, 124, 0.9, MIYAKO, 3, hook=True, bar0=bar, seed=idx)
            for h in hits:
                big(sc, h, 1.0)
                sc.put(hyoshigi(1.0), h + 0.03, 0.4, pan=0.2, wet=0.3, acc=True)
            duck[int(t0 * SR):int(t1 * SR)] = 0.9
        elif cue == "versus":
            roll(sc, t0, t1 - 0.05, 0.95, MIYAKO)
            big(sc, t1, 1.0)
        elif cue == "final_intro":
            sc.put(drone(t1 - t0 + 0.6, 0.8), t0, 0.22, wet=0.3)
            heartbeat(sc, t0 + 1.0, t1, 1.5, 1.2, 0.45)
            duck[int(t0 * SR):int(t1 * SR)] = 0.9
        elif cue == "ceremony":
            sc.put(drone(t1 - t0 + 0.6, 0.8), t0, 0.2, wet=0.3)
            k, i = t0 + 0.3, 0
            while k < t1 - 0.3:
                sc.put(koto(note(MIYAKO, [0, 2, 3, 4][i % 4], 0), 0.6, dur=2.5), k, 0.2, pan=-0.3, wet=0.4)
                k += 1.1
                i += 1
            duck[int(t0 * SR):int(t1 * SR)] = 0.8
        elif cue == "bout":
            # the tension builds to the decision: the heartbeat quickening, the yagura ticking in under
            # it; the decision a big strike, the clappers and the triumph
            tach = hits[0] if hits else t0
            dec = hits[-1] if len(hits) > 1 else t1
            for h in hits:
                big(sc, h, 1.0)
            heartbeat(sc, tach + 0.8, dec - 0.1, 0.95, 0.32, 0.5, accel=1.6)
            k, i = tach + 2.0, 0
            while k < dec - 0.1:
                u = (k - tach) / max(1e-6, dec - tach)
                if i % 2:
                    sc.put(fuchi(0.6, var=i), k, 0.08 + 0.25 * u ** 1.5, pan=0.25, wet=0.12)
                else:
                    sc.put(shime(0.3 + 0.5 * u), k, 0.08 + 0.3 * u ** 1.5, pan=0.35, wet=0.12)
                k += 0.24 - 0.12 * u
                i += 1
            sc.put(drone(dec - tach, 0.8), tach, 0.16, wet=0.25)
            if len(hits) > 1:
                triumph(sc, dec, YO, 1.0)
        elif cue == "replay":
            k, i = t0 + 0.2, 0
            while k < t1 - 0.4:
                sc.put(koto(note(MIYAKO, [0, 2, 4, 5, 4, 2][i % 6], -1), 0.6, dur=3.0), k, 0.3, pan=-0.3, wet=0.45)
                k += 0.75
                i += 1
            sc.put(drone(t1 - t0 + 0.5, 0.6), t0, 0.16, wet=0.3)
            heartbeat(sc, t0 + 0.5, t1, 1.1, 0.7, 0.32)
            duck[int(t0 * SR):int(t1 * SR)] = 0.8
        elif cue == "verdict":
            for h in hits:
                sc.put(hyoshigi(1.0), h, 0.5, pan=0.2, wet=0.3, acc=True)
                big(sc, h + 0.02, 0.9)
            sc.put(drone(t1 - t0, 0.6), t0, 0.14, wet=0.3)
        elif cue == "trophy":
            # the champion: dignified, not a party (10/23, the user: sumo's courtesy) -- the clappers,
            # the big drum's slow strokes, the flute's long line over a held koto chord
            sc.put(hyoshigi(1.0), t0, 0.45, pan=0.2, wet=0.4)
            for j, dt in enumerate((0.0, 1.2, 2.4)):
                sc.put(odaiko(vel=0.85), t0 + dt, 0.6, wet=0.45)
            gliss(sc, t0 + 0.05, YO, 0, 6, 0.08, 0.3)
            k = t0 + 0.6
            for d, ln in ((5, 1.0), (4, 0.6), (5, 1.4)):
                sc.put(shinobue(note(YO, d, 1), ln + 0.1, 0.8), k, 0.32, pan=0.1, wet=0.45)
                k += ln
        elif cue == "bts":
            if prev != "bts":
                k, end = _run(items, starts, idx, ("bts",))
                chart0 = next((starts[j] for j in range(idx, k) if items[j].get("image")), end)
                b = band(sc, t0, chart0, 104, 0.6, YO, 1, hook=True, bar0=0, seed=idx)
                band(sc, chart0, end, 104, 0.8, YO, 2, hook=True, bar0=b, seed=idx + 60)
            for h in hits:
                big(sc, h, 0.8)
            duck[int(t0 * SR):int(t1 * SR)] = 0.9
        elif cue == "end":
            for j, (dt, v) in enumerate(((0.0, 1.0), (0.48, 0.75), (0.97, 0.9), (1.21, 0.9), (1.45, 1.0))):
                sc.put(odaiko(vel=v), t0 + dt, 0.85, wet=0.45)
                sc.put(yagura(0.8, var=j), t0 + dt, 0.4, pan=-0.1, wet=0.25)
            for j, d in enumerate((0, 2, 4, 7)):
                sc.put(shamisen(note(MIYAKO, d, 0), 0.9), t0 + 1.45 + j * 0.03, 0.3, pan=0.3, wet=0.3)
            sc.put(shinobue(note(MIYAKO, 5, 1), 1.8, 0.9), t0 + 1.5, 0.33, wet=0.45)
            for j, dt in enumerate([0.0, 0.5] + list(1.0 + np.cumsum(np.geomspace(0.2, 0.05, 8)))):
                sc.put(hyoshigi(0.8), t0 + 1.9 + dt * 0.6, 0.25, pan=0.2, wet=0.35)
    bed, acc = sc.mix()
    # a little drive on the music for punch (a soft clip after levelling)
    pk = np.max(np.abs(bed)) + 1e-9
    bed = np.tanh(1.6 * bed / pk) / np.tanh(1.6) * pk
    return bed[:len(duck)], acc[:len(duck)], duck, total


def read_audio(video: Path, wav: Path) -> np.ndarray:
    subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "2", "-ar", str(SR),
                    "-c:a", "pcm_s16le", str(wav)], check=True)
    with wave.open(str(wav)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2) / 32768.0


def write_wav(path: Path, st: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(st, -1, 1) * 32767).astype(np.int16).tobytes())


def main(spec_path: str, video: str, out: str) -> None:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    bed, acc, duck, total = compose(spec)
    tmp = Path(out).with_suffix(".tmp.wav")
    own = read_audio(Path(video), tmp)
    n = len(own)
    fit = lambda a: np.pad(a, ((0, max(0, n - len(a))), (0, 0)))[:n]       # noqa: E731
    bed, acc = fit(bed), fit(acc)
    duck = np.pad(duck, (0, max(0, n - len(duck))), constant_values=1.0)[:n]
    # the ducking smoothed (no steps), the music a little under the hall where the picture speaks
    k = np.hanning(int(0.6 * SR))
    duck = np.convolve(duck, k / k.sum(), "same")
    g = 0.55 / (np.max(np.abs(bed + acc)) + 1e-9)
    mix = own * 0.8 + (bed * duck[:, None] + acc) * g
    f = int(1.2 * SR)
    mix[-f:] *= np.linspace(1, 0, f)[:, None]
    mix /= np.max(np.abs(mix)) + 1e-9
    write_wav(tmp, mix * 0.9)
    wav = Path(out).with_suffix(".wav")
    # level to -16 LUFS integrated, true peak -1.5 dB
    subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-i", str(tmp), "-af",
                    "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", str(SR), str(wav)], check=True)
    subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-i", video, "-i", str(wav), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out], check=True)
    tmp.unlink()
    print(f"[score] {total:.1f} s of music under {video} -> {out} ({wav.name})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
