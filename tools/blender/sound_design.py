"""The hall's sound for a composed bout (10/20): everything synthesized here -- no recording,
no music, no voice of anyone's -- and laid on the video's timeline (compose_video.py's
OUT.timeline.json) from what the dump says happened:

- the hall: its hands only (10/23: a noise band sounded like rain or wind, synthetic voices like a
  horror game) -- people clapping, each at a pace of their own, in the hall's reverberation
- the shikiri: chatter; a hush at 待ったなし
- the tachiai: the crowd's roar rising, and a thud when the two meet
- the bout: the roar swelling as a wrestler nears the straw
- the decision: a burst of cheering and applause, fading
- the intro: wooden clappers (hyoshigi), two strikes and a quickening run
- the replay: the hall low, the moment's thud again, softer

    python -I sound_design.py DUMP.npz VIDEO.mp4        (writes VIDEO.wav, muxes VIDEO.mp4 in place)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

SR = 44100
RING_R = 0.8
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
rng = np.random.default_rng(20261020)


def _butter_bp(x, lo, hi, order=2):
    """A band-pass by FFT masking (smooth edges): enough for noise shaping."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    m = np.ones_like(f)
    if lo:
        m *= 1 / (1 + (lo / np.maximum(f, 1e-3)) ** (2 * order))
    if hi:
        m *= 1 / (1 + (f / hi) ** (2 * order))
    return np.fft.irfft(X * m, n)


def _smooth(env, n):
    """A per-frame envelope to a per-sample one, smoothed."""
    x = np.interp(np.arange(n), np.linspace(0, n - 1, len(env)), env)
    k = int(0.15 * SR)
    ker = np.hanning(k)
    ker /= ker.sum()
    return np.convolve(x, ker, mode="same")


#: the five vowels' first three formants (Hz), a voice between a man's and a woman's
VOWELS = np.array([(780, 1250, 2550), (330, 2250, 2950), (370, 1300, 2450), (480, 1880, 2580), (520, 880, 2480)], float)


def voices(n, count, f0=(95, 240), syll=(0.09, 0.22), vowels=(0, 1, 2, 3, 4), talk=0.65, lift=None, seed=1):
    """Many voices, none of them anyone's (10/23, the user: the hall sounded like rain or wind -- it was
    band-passed noise): each a glottal buzz at a pitch of its own through the formants of a vowel that
    changes syllable by syllable, in phrases and pauses. The formants are laid on in the spectrum of
    overlapping 46 ms blocks (a periodic Hann at half overlap sums to one). `lift` (0..1 per sample)
    raises every voice's pitch -- excitement; `talk` is the share of time each one is speaking."""
    r = np.random.default_rng(seed)
    H = int(0.023 * SR)
    B = 2 * H
    nb = n // H + 2
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(B) / B)
    fr = np.fft.rfftfreq(B, 1 / SR)
    tilt = 1 / np.sqrt(1 + (fr / 2200) ** 2) / np.sqrt(1 + (90 / np.maximum(fr, 1)) ** 4)
    tb = (np.arange(nb) + 1) * H / SR
    lift_b = np.interp(tb * SR, np.arange(n), lift) if lift is not None else np.zeros(nb)
    idx = np.arange(nb)[:, None] * H + np.arange(B)[None, :]
    acc = np.zeros((nb + 1, H))
    for _ in range(count):
        ns = int(n / SR / syll[0]) + 4
        lens = r.uniform(*syll, ns)
        ends = np.cumsum(lens) - r.uniform(0, syll[1])
        si = np.clip(np.searchsorted(ends, tb), 0, ns - 1)
        pos = np.clip((tb - (ends[si] - lens[si])) / lens[si], 0, 1)
        vow = np.asarray(vowels)[r.integers(0, len(vowels), ns)][si]
        F = VOWELS[vow] * r.uniform(0.9, 1.12)
        F = np.stack([np.convolve(np.pad(F[:, i], 2, mode="edge"), np.ones(5) / 5, "valid") for i in range(3)], 1)
        # phrases and pauses
        on = np.zeros(nb)
        t, speaking = r.uniform(-2.0, 0.0), r.random() < talk
        while t < tb[-1]:
            d = r.uniform(0.8, 3.2) if speaking else r.uniform(0.2, 1.4) * (1 - talk) / 0.35
            if speaking:
                on[(tb >= t) & (tb < t + d)] = 1.0
            t += d
            speaking = not speaking
        on = np.convolve(on, np.hanning(7) / np.hanning(7).sum(), "same")
        amp = on * (0.3 + 0.7 * np.sin(np.pi * pos) ** 0.7) * r.uniform(0.3, 1.0)
        drift = np.convolve(r.standard_normal(nb), np.hanning(60), "same")
        drift /= np.max(np.abs(drift)) + 1e-9
        f0b = r.uniform(*f0) * (1 + 0.06 * drift + 0.04 * np.sin(np.pi * pos)) * (1 + 0.45 * lift_b)
        f0s = np.interp(np.arange((nb + 1) * H), tb * SR, f0b * (1 + 0.008 * r.standard_normal(nb)))
        ph = np.cumsum(f0s / SR)
        src = (2 * (ph % 1.0) - 1) + 0.06 * r.standard_normal(len(ph))
        X = np.fft.rfft(src[idx] * win, axis=1)
        E = sum(g / np.sqrt(1 + ((fr[None, :] - F[:, i, None]) / (bw / 2)) ** 2)
                for i, (bw, g) in enumerate(((90, 1.0), (120, 0.6), (170, 0.3))))
        Y = np.fft.irfft(X * (E * tilt[None, :]) * amp[:, None], B, axis=1)
        acc[:-1] += Y[:, :H]
        acc[1:] += Y[:, H:]
    return acc.ravel()[:n]


def _norm(x):
    return x / (np.max(np.abs(x)) + 1e-9)


def murmur(n, chatter):
    """The hall talking: forty voices, far off (the highs gone), as loud as `chatter` (0..1)."""
    return _norm(_butter_bp(voices(n, 40, seed=11), 120, 3600)) * chatter


def roar(n, level):
    """The crowd's 'ooh': voices holding o and a, their pitch rising with the bout's excitement."""
    x = voices(n, 36, f0=(105, 250), syll=(0.5, 1.6), vowels=(4, 0), talk=0.97, lift=level, seed=12)
    return _norm(_butter_bp(x, 120, 4000)) * level


def cheer(n, level):
    """The cheer at a decision: higher, open voices (a, e, o), rising."""
    x = voices(n, 40, f0=(170, 380), syll=(0.3, 0.9), vowels=(0, 3, 4), talk=0.97, lift=level, seed=13)
    return _norm(_butter_bp(x, 150, 5000)) * level


def _clap_bank(m=24):
    """Single hand claps: a short burst through the ring of cupped hands (a resonance 0.9-2.4 kHz)."""
    from scipy.signal import lfilter
    r = np.random.default_rng(5)
    bank = []
    for _ in range(m):
        d = int(0.045 * SR)
        t = np.arange(d) / SR
        x = r.standard_normal(d) * np.exp(-t / r.uniform(0.0025, 0.006))
        w = 2 * np.pi * r.uniform(900, 2400) / SR
        rad = np.exp(-w / (2 * r.uniform(1.2, 3.5)))
        y = lfilter([1 - rad], [1, -2 * rad * np.cos(w), rad * rad], x) + 0.25 * x
        bank.append(_norm(y))
    return bank


def applause(n, level):
    """Hands: a hundred and forty people, each clapping at a pace of their own (that regularity is what
    tells applause from rain), joining as `level` rises past a threshold of their own."""
    r = np.random.default_rng(6)
    bank = _clap_bank()
    out = np.zeros(n + int(0.05 * SR))
    for p in range(140):
        rate, th, g, c = r.uniform(3.2, 5.8), r.uniform(0.04, 0.9), r.uniform(0.25, 1.0), bank[p % len(bank)]
        t = r.uniform(0, 1 / rate)
        while t < n / SR:
            k = int(t * SR)
            lv = level[min(k, n - 1)]
            if lv > th:
                out[k:k + len(c)] += c * g * min(1.0, (lv - th) * 4)
                t += (1 + 0.07 * r.standard_normal()) / (rate * (0.85 + 0.3 * lv))
            else:
                t += 0.05
    return _norm(out[:n]) * np.clip(level * 1.2, 0, 1)


def hall(x, rt60=1.9, wet=0.4, seed=7):
    """The hall's reverberation, stereo: two noise tails (the highs dying sooner), one for each ear,
    after a 12 ms pre-delay; dry and wet at equal energy before the mix."""
    from scipy.signal import fftconvolve
    r = np.random.default_rng(seed)
    L = int(rt60 * 1.1 * SR)
    t = np.arange(L) / SR
    chans = []
    for _ in range(2):
        ir = sum(_butter_bp(r.standard_normal(L), lo, hi) * np.exp(-6.9 * t / (rt60 * k))
                 for lo, hi, k in ((0, 600, 1.0), (600, 3000, 0.8), (3000, 0, 0.45)))
        ir[:int(0.012 * SR)] = 0
        ir /= np.sqrt(np.sum(ir ** 2))
        chans.append(fftconvolve(x, ir)[:len(x)])
    return (1 - wet) * np.stack([x, x], 1) + wet * np.stack(chans, 1)


def thud(n, at, gain=1.0):
    """Two bodies meeting: a low knock with a pitch drop and a dull click."""
    out = np.zeros(n)
    d = int(0.25 * SR)
    t = np.arange(d) / SR
    f = 95 * np.exp(-t * 8) + 45
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 18)
    click = _butter_bp(rng.standard_normal(d), 300, 2500) * np.exp(-t * 120) * 0.4
    s = (body + click) * gain
    for a in at:
        p = int(a * SR)
        e = min(n, p + d)
        if p < n:
            out[p:e] += s[:e - p]
    return out


def hyoshigi(n, at, gain=1.0):
    """Wooden clappers: a bright, short, ringing knock."""
    out = np.zeros(n)
    d = int(0.35 * SR)
    t = np.arange(d) / SR
    s = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * k)
            for f, a, k in ((1720, 1.0, 22), (2960, 0.6, 30), (4410, 0.35, 45), (870, 0.3, 28)))
    s += _butter_bp(rng.standard_normal(d), 1500, 8000) * np.exp(-t * 300) * 0.8
    s /= np.max(np.abs(s))
    for a in at:
        p = int(a * SR)
        e = min(n, p + d)
        if p < n:
            out[p:e] += s[:e - p] * gain
    return out


def clack(n, at, gain=1.0):
    """A crab's pincers shutting: a hard, dry click (chitin on chitin) with a short ring."""
    out = np.zeros(n)
    d = int(0.08 * SR)
    t = np.arange(d) / SR
    s = _butter_bp(rng.standard_normal(d), 1800, 7000) * np.exp(-t * 400)
    s += 0.5 * np.sin(2 * np.pi * 2650 * t) * np.exp(-t * 90) + 0.3 * np.sin(2 * np.pi * 4100 * t) * np.exp(-t * 140)
    s /= np.max(np.abs(s))
    for a in at:
        p = int(a * SR)
        e = min(n, p + d)
        if p < n:
            out[p:e] += s[:e - p] * gain
    return out


def pincer_shuts(D, names, caps):
    """The dump frames where the pincers shut before the first はっけよい (the ceremony's
    claps, upstream 9b019df): each claw's tip-to-palm distance falling back to closed after
    it had opened; the four claws' shuts within 0.15 s counted as one."""
    hk = next((k for k, c in enumerate(caps) if c == "はっけよい"), len(caps))
    ev = []
    for e in ("robot", "opponent"):
        for s in ("LF", "RF"):
            try:
                i = names.index(f"{e}/{s}_finger_tip_link_visual")
                j = names.index(f"{e}/{s}_palm_link_visual")
            except ValueError:
                continue
            d = np.linalg.norm(D["dyn_pos"][:hk, i] - D["dyn_pos"][:hk, j], axis=1)
            lo, hi = float(d.min()), float(d.max())
            if hi - lo < 0.01:                       # never opened by a centimetre
                continue
            opened = False
            for k, v in enumerate(d):
                if v > lo + 0.6 * (hi - lo):
                    opened = True
                elif opened and v < lo + 0.25 * (hi - lo):
                    ev.append(k)
                    opened = False
    ev.sort()
    groups = []
    for k in ev:
        if groups and k - groups[-1][-1] <= 4:
            groups[-1].append(k)
        else:
            groups.append([k])
    return [g[0] for g in groups]


def main(dump: str, video: str) -> None:
    D = np.load(dump, allow_pickle=False)
    tl = json.loads(Path(video).with_suffix(".timeline.json").read_text(encoding="utf-8"))
    fps, frames = tl["fps"], tl["frames"]
    N = len(frames)
    n = int(N / fps * SR)
    caps = [str(c) for c in D["caption"]]
    names = [str(x) for x in D["dyn_name"]]
    ids = [names.index(f"{e}/base_link_visual") for e in ("robot", "opponent")]
    pos = D["dyn_pos"][:, ids, :2]
    dist = np.linalg.norm(pos[:, 0] - pos[:, 1], axis=1)
    edge = np.linalg.norm(pos, axis=2).max(1) / RING_R
    # how hard the bout is moving: the pair's midpoint speed (a push or a turn), smoothed
    mid = pos.mean(1)
    sp = np.r_[0.0, np.linalg.norm(np.diff(mid, axis=0), axis=1) * 30.0]
    sp = np.convolve(sp, np.ones(9) / 9, mode="same")
    surge = np.clip(sp / 0.25, 0, 1)
    dec_k = next((k for k, c in enumerate(caps) if c.startswith("勝負あり")), None)
    hakkeyoi = [k for k, c in enumerate(caps) if c == "はっけよい"]
    # the first meeting after each はっけよい: the tachiai -- where the closing stops within 0.45 m
    # of each other (10/21: the hero bout's bases never came nearer than 0.31 m, the claws meeting
    # first at 0.37, and the 0.30 m test found no tachiai)
    meets = []
    for h in hakkeyoi:
        m = next((k for k in range(h, len(dist) - 3) if dist[k] < 0.45 and dist[k + 3] > dist[k] - 0.005), None)
        if m is not None and (not meets or m > meets[-1] + 30):
            meets.append(m)
    chatter = np.zeros(N)
    excite = np.zeros(N)
    clap = np.zeros(N)
    thuds, thuds_soft = [], []
    for j, (k, rep) in enumerate(frames):
        t = j / fps
        if k < 0:                                     # the intro: the hall waiting
            chatter[j], excite[j] = 0.45, 0.0
            continue
        kk = min(k, len(caps) - 1)
        cap = caps[kk]
        started = any(h <= kk for h in hakkeyoi) and not (cap in ("見合って", "手をついて", "待ったなし"))
        if rep:
            chatter[j], excite[j] = 0.18, 0.18
            continue
        if cap in ("見合って", "手をついて"):
            chatter[j] = 0.40
        elif cap == "待ったなし":
            chatter[j] = 0.12                         # the hush
        elif dec_k is not None and kk >= dec_k:
            since = (kk - dec_k) / 30.0
            excite[j] = max(0.0, 1.0 - since / 4.0) * 0.9 + 0.1
            clap[j] = max(0.0, 1.0 - since / 5.0)
            chatter[j] = 0.35
        elif started:
            near = float(np.clip((edge[kk] - 0.55) / 0.45, 0, 1))
            excite[j] = 0.30 + 0.35 * near + 0.35 * float(surge[kk])
            chatter[j] = 0.25
        else:
            chatter[j] = 0.35
    for j, (k, rep) in enumerate(frames):
        if k in meets and (j == 0 or frames[j - 1][0] != k):
            (thuds_soft if rep else thuds).append(j / fps)
    # the ceremony: the pincers' clacks, and the hall answering the V of the claws (between the
    # second shut and the third) with a murmur of delight and a scatter of hands
    shuts = pincer_shuts(D, names, caps)
    clacks = [j / fps for j, (k, rep) in enumerate(frames)
              if not rep and k in shuts and (j == 0 or frames[j - 1][0] != k)]
    if len(shuts) >= 3:
        for j, (k, rep) in enumerate(frames):
            if not rep and shuts[1] + 10 <= k <= shuts[2] + 30:
                u = (k - shuts[1] - 10) / max(1, shuts[2] + 30 - shuts[1] - 10)
                bump = float(np.sin(np.pi * u))
                excite[j] = max(excite[j], 0.25 * bump)
                clap[j] = max(clap[j], 0.30 * bump)
    # 10/23 21:45 (the user on v4: the synthetic voices' cheering was wind, or a horror game -- frightening):
    # no voices at all. The hall is its hands (the applause at the ceremony's V and at the decision);
    # the excitement of the bout is the music's (tools/media/score.py). murmur / roar / cheer stay
    # here, unused.
    del chatter, excite
    crowd = 0.6 * applause(n, _smooth(clap, n))
    th = thud(n, thuds, 0.9) + thud(n, thuds_soft, 0.35) + clack(n, clacks, 0.5)
    hy = hyoshigi(n, [0.35, 0.95] + list(1.9 + np.cumsum(np.geomspace(0.32, 0.06, 14))), 0.55)
    # 10/23: the crowd in the hall's reverberation, the knocks drier and centred
    st = hall(crowd, wet=0.45) + hall(0.55 * th + 0.45 * hy, wet=0.18, seed=8)
    st /= np.max(np.abs(st)) + 1e-9
    st *= 0.85
    # fades in and out
    f = int(0.4 * SR)
    st[:f] *= np.linspace(0, 1, f)[:, None]
    st[-f:] *= np.linspace(1, 0, f)[:, None]
    wav = Path(video).with_suffix(".wav")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())
    tmp = Path(video).with_name(Path(video).stem + "_av.mp4")
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-i", video, "-i", str(wav), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(tmp)], check=True)
    tmp.replace(video)
    print(f"[sound] {N / fps:.1f} s: clacks at {['%.1f' % x for x in clacks]}, tachiai at {['%.1f' % x for x in thuds]}, decision "
          f"{'none' if dec_k is None else dec_k} -> {video} (with sound), {wav.name}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
