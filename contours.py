"""
Pitch-contour based singing extraction.

Finds every pitched voice in the vocal stem as a set of pitch contours
(multi-pitch, so a singer and a talker at the same time give separate
contours), classifies each contour as singing or talking, and builds a
time-frequency mask that keeps the full harmonic series of the sung contours.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.ndimage import maximum_filter1d, uniform_filter1d

FMIN, FMAX = 75.0, 1000.0
CENTS_STEP = 10.0
N_HARM = 15
HARM_DECAY = 0.85
SALIENCE_MAX_FREQ = 5000.0


def hz_to_cents(f):
    return 1200.0 * np.log2(np.asarray(f) / 55.0)


def cents_to_hz(c):
    return 55.0 * 2.0 ** (np.asarray(c) / 1200.0)


def salience_matrix(freqs: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Sparse (n_candidates x n_bins) matrix: harmonic summation with linear
    interpolation between FFT bins."""
    cents = np.arange(hz_to_cents(FMIN), hz_to_cents(FMAX), CENTS_STEP)
    f0s = cents_to_hz(cents)
    df = freqs[1] - freqs[0]
    rows, cols, vals = [], [], []
    for i, f0 in enumerate(f0s):
        for h in range(1, N_HARM + 1):
            f = h * f0
            if f > SALIENCE_MAX_FREQ:
                break
            pos = f / df
            b = int(np.floor(pos))
            frac = pos - b
            w = HARM_DECAY ** (h - 1)
            rows += [i, i]
            cols += [b, b + 1]
            vals += [w * (1 - frac), w * frac]
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(len(f0s), len(freqs)))
    return W, cents


def track_contours(sal: np.ndarray, cents: np.ndarray, fps: float,
                   rel_thr: float = 0.3, abs_pct: float = 70.0,
                   max_jump: float = 60.0, max_gap: int = 3,
                   max_peaks: int = 4) -> list[dict]:
    """Peak-pick salience per frame and link peaks into contours."""
    n_c, n_t = sal.shape
    frame_max = sal.max(0)
    active = frame_max[frame_max > 0]
    g_thr = np.percentile(active, abs_pct) * 0.25 if active.size else 0.0

    # local maxima along pitch axis
    is_peak = np.zeros_like(sal, dtype=bool)
    is_peak[1:-1] = (sal[1:-1] > sal[:-2]) & (sal[1:-1] >= sal[2:])
    is_peak &= sal >= np.maximum(rel_thr * frame_max[None, :], g_thr)

    contours: list[dict] = []
    open_: list[dict] = []
    for t in range(n_t):
        idx = np.nonzero(is_peak[:, t])[0]
        if idx.size > max_peaks:
            idx = idx[np.argsort(sal[idx, t])[::-1][:max_peaks]]
        idx = idx[np.argsort(sal[idx, t])[::-1]]
        used = set()
        for i in idx:
            # parabolic refinement of the peak position
            c = cents[i]
            if 0 < i < n_c - 1:
                a, b, d = sal[i - 1, t], sal[i, t], sal[i + 1, t]
                den = a - 2 * b + d
                if den < 0:
                    c += 0.5 * (a - d) / den * CENTS_STEP
            best, best_d = None, max_jump
            for k, ct in enumerate(open_):
                if k in used:
                    continue
                gap = t - ct["t"][-1]
                dd = abs(ct["c"][-1] - c)
                if dd < best_d * gap:  # allow larger jumps across small gaps
                    best, best_d = k, dd / gap
            if best is None:
                open_.append({"t": [t], "c": [c], "s": [sal[i, t]]})
                used.add(len(open_) - 1)
            else:
                ct = open_[best]
                ct["t"].append(t)
                ct["c"].append(c)
                ct["s"].append(sal[i, t])
                used.add(best)
        still = []
        for ct in open_:
            if t - ct["t"][-1] > max_gap:
                contours.append(ct)
            else:
                still.append(ct)
        open_ = still
    contours += open_
    out = []
    for ct in contours:
        if len(ct["t"]) < 3:
            continue
        out.append({k: np.asarray(v, dtype=float) for k, v in ct.items()})
    return out


def contour_features(ct: dict, fps: float) -> dict:
    t, c = ct["t"], ct["c"]
    dur = (t[-1] - t[0] + 1) / fps
    # remove vibrato (~5-7 Hz) with a ~180 ms moving average, then measure glide
    k = max(3, int(0.18 * fps) | 1)
    ci = np.interp(np.arange(t[0], t[-1] + 1), t, c)
    sm = uniform_filter1d(ci, k, mode="nearest")
    slope = np.abs(np.gradient(sm)) * fps          # cents / second
    steady = float((slope < 350.0).mean())
    # longest steady stretch (a held note)
    run, best = 0, 0
    for v in slope < 350.0:
        run = run + 1 if v else 0
        best = max(best, run)
    held = best / fps
    return dict(dur=dur, steady=steady, held=held, sal=float(np.mean(ct["s"])),
                rng=float(np.percentile(c, 95) - np.percentile(c, 5)))


def singing_score(f: dict, min_held: float, ref_sal: float) -> float:
    """Log-odds that a contour is sung.

    The main cue is a *held* note: a stretch of steady pitch (vibrato allowed)
    longer than a spoken syllable. Total duration and loudness help a little.
    """
    x = (f["held"] - min_held) / 0.04
    x += 0.5 * (f["dur"] - 0.3) / 0.15
    x += 1.5 * np.log(max(f["sal"], 1e-9) / ref_sal)
    return float(x)


def link_to_sung_notes(contours: list[dict], fps: float, max_gap: float = 0.08,
                       max_cents: float = 250.0, passes: int = 2) -> None:
    """A short contour that starts right where a sung note ends (or ends where
    one starts) at a nearby pitch is the slide/ornament between notes of the
    same singer: lift its score to that of the note it joins."""
    gap = max_gap * fps
    for _ in range(passes):
        sung = [c for c in contours if c["score"] > 0.5]
        if not sung:
            return
        ends = np.array([[c["t"][0], c["t"][-1], c["c"][0], c["c"][-1], c["score"]] for c in sung])
        for ct in contours:
            if ct["score"] > 0.5:
                continue
            t0, t1, c0, c1 = ct["t"][0], ct["t"][-1], ct["c"][0], ct["c"][-1]
            after = (t0 - ends[:, 1] >= -2) & (t0 - ends[:, 1] <= gap) & (np.abs(c0 - ends[:, 3]) < max_cents)
            before = (ends[:, 0] - t1 >= -2) & (ends[:, 0] - t1 <= gap) & (np.abs(c1 - ends[:, 2]) < max_cents)
            hit = after | before
            if hit.any():
                ct["score"] = max(ct["score"], 0.8 * float(ends[hit, 4].max()))


def drop_subharmonic_ghosts(contours: list[dict], tol: float = 40.0) -> list[dict]:
    """Harmonic summation also produces contours at f0/2, f0/3 (and 2*f0).
    When two contours run parallel at an integer pitch ratio, keep only the
    more salient one; the other is a ghost of it."""
    by_t: dict[int, list[int]] = {}
    for i, ct in enumerate(contours):
        for t in ct["t"].astype(int):
            by_t.setdefault(t, []).append(i)
    drop = set()
    order = sorted(range(len(contours)), key=lambda i: -contours[i]["sal_mean"])
    ratios = 1200 * np.log2([2, 3, 4])
    for i in order:
        if i in drop:
            continue
        a = contours[i]
        cand = {j for t in a["t"].astype(int) for j in by_t.get(t, []) if j != i and j not in drop}
        for j in cand:
            b = contours[j]
            common, ia, ib = np.intersect1d(a["t"], b["t"], return_indices=True)
            if len(common) < 0.5 * min(len(a["t"]), len(b["t"])):
                continue
            diff = np.median(np.abs(a["c"][ia] - b["c"][ib]))
            if np.min(np.abs(ratios - diff)) < tol:
                drop.add(j)  # j is less salient than i (processed in order)
    return [ct for k, ct in enumerate(contours) if k not in drop]


def build_mask(n_bins: int, n_t: int, freqs: np.ndarray, fps: float,
               sung: list[dict], width_cents: float, hf_cutoff: float,
               consonant_gain: float, phrase_pad: float,
               hf_gain: float = 0.5) -> np.ndarray:
    """Soft TF mask keeping the full harmonic series of the sung contours."""
    M = np.zeros((n_bins, n_t), dtype=np.float32)
    nyq = freqs[-1]
    df = freqs[1] - freqs[0]
    activity = np.zeros(n_t, dtype=np.float32)
    for ct in sung:
        t0, t1 = int(ct["t"][0]), int(ct["t"][-1])
        tt = np.arange(t0, t1 + 1)
        f0 = cents_to_hz(np.interp(tt, ct["t"], ct["c"]))
        w = float(ct.get("w", 1.0))
        activity[t0:t1 + 1] = np.maximum(activity[t0:t1 + 1], w)
        for j, t in enumerate(tt):
            hs = f0[j] * np.arange(1, int(min(hf_cutoff, nyq) // f0[j]) + 1)
            if hs.size == 0:
                continue
            sig = np.maximum(hs * (2 ** (width_cents / 1200) - 1), 1.2 * df)
            lo = np.maximum(((hs - 3 * sig) / df).astype(int), 0)
            hi = np.minimum(((hs + 3 * sig) / df).astype(int) + 2, n_bins)
            col = M[:, t]
            for h, s, a, b in zip(hs, sig, lo, hi):
                g = w * np.exp(-0.5 * ((freqs[a:b] - h) / s) ** 2)
                np.maximum(col[a:b], g, out=col[a:b])
    # phrase activity: inside a sung phrase keep some non-harmonic content
    # (consonants, breath) and all of the high band (air, sibilants)
    pad = max(1, int(phrase_pad * fps))
    phrase = maximum_filter1d(activity, 2 * pad + 1)
    phrase = uniform_filter1d(phrase, max(1, pad))
    hf = freqs >= hf_cutoff
    M[~hf] = np.maximum(M[~hf], consonant_gain * phrase[None, :])
    M[hf] = hf_gain * phrase[None, :]
    # light smoothing over time to avoid "musical noise"
    M = uniform_filter1d(M, 3, axis=1)
    return M


def extract_singing(vocals: np.ndarray, sr: int, *, min_held: float = 0.22,
                    threshold: float = 0.0, width_cents: float = 35.0,
                    hf_cutoff: float = 9000.0, consonant_gain: float = 0.0,
                    hf_gain: float = 0.5,
                    phrase_pad: float = 0.12, link: bool = True, log=print,
                    debug: dict | None = None
                    ) -> np.ndarray:
    import librosa

    n_fft = 8192 if sr > 32000 else 4096
    hop = int(round(sr / 100))  # 10 ms
    fps = sr / hop
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    specs = [librosa.stft(ch, n_fft=n_fft, hop_length=hop) for ch in vocals]
    mono_mag = np.abs(np.mean(specs, axis=0)).astype(np.float32)

    W, cents = salience_matrix(freqs)
    comp = np.sqrt(mono_mag)  # compress so loud partials don't dominate
    # whiten spectrum a little (remove slow spectral envelope)
    sal = np.asarray(W @ comp)
    log(f"tracking pitch contours ({sal.shape[1]} frames)...")
    contours = track_contours(sal, cents, fps)

    for ct in contours:
        ct["sal_mean"] = float(np.mean(ct["s"]))
    n_all = len(contours)
    contours = drop_subharmonic_ghosts(contours)
    log(f"{n_all} pitch contours, {n_all - len(contours)} octave ghosts dropped")
    feats = [contour_features(ct, fps) for ct in contours]
    sals = np.array([f["sal"] for f in feats]) if feats else np.array([1.0])
    ref_sal = float(np.percentile(sals, 75))
    for ct, f in zip(contours, feats):
        ct.update(f)
        ct["score"] = singing_score(f, min_held, ref_sal) - threshold
    if link:
        link_to_sung_notes(contours, fps)
    sung = []
    for ct in contours:
        sc = ct["score"]
        if sc > -1.0:
            ct["w"] = 1 / (1 + np.exp(-2.0 * sc))
            sung.append(ct)
    n_sung = sum(1 for ct in sung if ct["w"] > 0.5)
    log(f"{n_sung} of {len(contours)} contours classified as singing")

    M = build_mask(len(freqs), sal.shape[1], freqs, fps, sung, width_cents,
                   hf_cutoff, consonant_gain, phrase_pad, hf_gain)
    if debug is not None:
        debug.update(contours=contours, fps=fps, mask=M, freqs=freqs, sal=sal, cents=cents)

    out = np.empty_like(vocals)
    for ch, S in enumerate(specs):
        out[ch] = librosa.istft(S * M[:, :S.shape[1]], hop_length=hop,
                                length=vocals.shape[1])
    return out.astype(np.float32)
