#!/usr/bin/env python3
"""
songfixer - remove background talking from a recorded song while keeping the
sung vocals and the instruments as intact as possible.

How it works
------------
1. Source separation (Demucs, htdemucs_ft by default) splits the recording
   into a "vocals" stem and the instruments. Talking voices end up in the
   vocals stem together with the singer, so the instruments are not touched
   by anything that follows.

2. Inside the vocals stem every pitched voice is tracked as pitch lines
   (several at once, so singer and talker overlapping are kept apart).
   Sung notes are *held*: a steady pitch (vibrato allowed) for longer than a
   spoken syllable. Speech pitch lines are short and slide constantly. Each
   line is scored, and the full harmonic series of the sung lines is kept
   (up to 9 kHz, so the voice keeps its brightness); everything between the
   harmonics - where the talking is - is removed. See contours.py.

3. Optional (--gate): an audio classifier mutes the vocal stem where it
   hears talking with no singing at all.

4. The output is   original - strength * (vocals - singing)
   i.e. only what was judged to be talking is subtracted from the untouched
   original recording. Instruments pass through bit-identical.

Usage
-----
    python songfixer.py recording.wav                  # -> recording_fixed.wav
    python songfixer.py recording.mp3 --preset strong  # gentle | balanced | strong
    python songfixer.py recording.wav --save-stems     # also write what was removed
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def log(msg: str) -> None:
    print(f"[songfixer] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def load_audio(path: Path) -> tuple[np.ndarray, int, str]:
    """Return audio as (channels, samples) float32, sample rate, subtype."""
    try:
        info = sf.info(str(path))
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return data.T.copy(), sr, info.subtype
    except Exception:
        import warnings

        import librosa  # falls back to audioread/ffmpeg for other formats

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data, sr = librosa.load(str(path), sr=None, mono=False)
        data = np.atleast_2d(data).astype(np.float32)
        return data, sr, "PCM_24"


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    import librosa

    return librosa.resample(x, orig_sr=sr_from, target_sr=sr_to, res_type="soxr_hq")


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[-1] >= n:
        return x[..., :n]
    pad = [(0, 0)] * (x.ndim - 1) + [(0, n - x.shape[-1])]
    return np.pad(x, pad)


def write(path: Path, x: np.ndarray, sr: int, subtype: str) -> None:
    if subtype not in sf.available_subtypes("WAV"):
        subtype = "PCM_24"
    if subtype.startswith("PCM") or subtype in ("ULAW", "ALAW"):
        x = np.clip(x, -1.0, 1.0)
    sf.write(str(path), x.T, sr, subtype=subtype)
    log(f"wrote {path}")


# --------------------------------------------------------------------------
# Step 1: source separation
# --------------------------------------------------------------------------
def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def separate_vocals(mix: np.ndarray, sr: int, model_name: str, device: str,
                    shifts: int) -> np.ndarray:
    """Return the vocals stem, same shape and sample rate as `mix`."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    log(f"loading separation model '{model_name}' (first run downloads it)...")
    model = get_model(model_name)
    model.eval()
    msr = model.samplerate

    x = resample(mix, sr, msr)
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    elif x.shape[0] > 2:
        x = x[:2]

    wav = torch.from_numpy(np.ascontiguousarray(x)).float()
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std

    def run(dev: str) -> torch.Tensor:
        log(f"separating vocals on {dev} ...")
        with torch.no_grad():
            return apply_model(model, wav[None], device=dev, shifts=shifts,
                               split=True, overlap=0.25, progress=True)[0]

    try:
        sources = run(device)
    except Exception as e:  # MPS occasionally lacks an op
        if device == "cpu":
            raise
        log(f"device '{device}' failed ({e.__class__.__name__}: {e}); retrying on cpu")
        sources = run("cpu")

    sources = sources * std + mean
    vocals = sources[model.sources.index("vocals")].cpu().numpy()

    vocals = fit_length(resample(vocals, msr, sr), mix.shape[1])
    if mix.shape[0] == 1:
        vocals = vocals.mean(0, keepdims=True)
    elif mix.shape[0] > 2:  # multichannel: only the first two were separated
        full = np.zeros_like(mix)
        full[:2] = vocals
        vocals = full
    return vocals.astype(np.float32)


def cached_separation(path: Path, mix: np.ndarray, sr: int, args) -> np.ndarray:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    cfile = Path(args.cache_dir) / f"{h.hexdigest()[:16]}_{args.model}_s{args.shifts}_vocals.npy"
    if cfile.exists() and not args.no_cache:
        log(f"using cached vocal separation ({cfile.name})")
        return np.load(cfile)
    vocals = separate_vocals(mix, sr, args.model, pick_device(args.device), args.shifts)
    if not args.no_cache:
        cfile.parent.mkdir(parents=True, exist_ok=True)
        np.save(cfile, vocals)
    return vocals


# --------------------------------------------------------------------------
# Step 2: singing vs. talking -> see contours.py (pitch-contour tracking)
# --------------------------------------------------------------------------
PRESETS = {
    # threshold: how sure a pitch line must be to count as singing
    # air: how much of the >9 kHz band is kept inside sung phrases
    # consonants: how much non-harmonic sound is kept inside sung phrases
    # link: keep slides that join sung notes
    "gentle":   dict(threshold=0.0, air=0.7, consonants=0.05, link=True),
    "balanced": dict(threshold=1.0, air=0.4, consonants=0.0, link=True),
    "strong":   dict(threshold=3.0, air=0.2, consonants=0.0, link=False),
}

# --------------------------------------------------------------------------
# Step 3 (optional): mute talk-only stretches using an audio classifier
# --------------------------------------------------------------------------
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SPEECH_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 70]          # Speech, Conversation, Narration, Babble...
SINGING_IDS = [27, 28, 30, 32, 33, 34, 35, 36, 37]  # Singing, Choir, Chant, Rapping, Humming...


def speech_gate(kept: np.ndarray, sr: int, device: str, win: float, hop: float,
                speech_thr: float, sing_thr: float, floor_db: float) -> np.ndarray:
    """Per-sample gain (0..1) that mutes windows classified as speech without singing.

    A sample is kept if ANY analysis window covering it contains singing, so
    the gate errs on the side of keeping the singer.
    """
    import librosa
    import torch
    from scipy.ndimage import uniform_filter1d
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    log("classifying speech vs. singing in the kept vocals (AudioSet AST)...")
    fe = ASTFeatureExtractor.from_pretrained(AST_MODEL)
    model = ASTForAudioClassification.from_pretrained(AST_MODEL).eval()
    try:
        model.to(device)
    except Exception:
        device = "cpu"
        model.to(device)

    mono = librosa.resample(kept.mean(0), orig_sr=sr, target_sr=16000)
    dur = len(mono) / 16000
    starts = np.arange(0.0, max(dur - win, 0.0) + 1e-6, hop)
    segs = [mono[int(s * 16000):int((s + win) * 16000)] for s in starts]
    probs = []
    with torch.no_grad():
        for i in range(0, len(segs), 16):
            feats = fe(segs[i:i + 16], sampling_rate=16000, return_tensors="pt")
            logits = model(input_values=feats["input_values"].to(device)).logits
            probs.append(torch.sigmoid(logits).cpu().numpy())
    p = np.concatenate(probs)
    speech = p[:, SPEECH_IDS].max(1)
    sing = p[:, SINGING_IDS].max(1)

    def sig(x):
        return 1 / (1 + np.exp(-x))

    talk_only = sig((speech - speech_thr) / 0.05) * sig((sing_thr - sing) / 0.015)
    floor = 10 ** (floor_db / 20)
    win_gain = 1 - talk_only * (1 - floor)

    # per 10 ms grid: interpolate between window centres; a short max-filter
    # keeps the gate open a little around detected singing, then crossfade.
    from scipy.ndimage import maximum_filter1d

    grid = np.arange(0, dur, 0.01)
    if len(starts) == 0:
        return np.ones(kept.shape[1], dtype=np.float32)
    gain = np.interp(grid, starts + win / 2, win_gain)
    gain = maximum_filter1d(gain, int(hop / 0.01) + 1)
    gain = uniform_filter1d(gain, 15)  # ~150 ms crossfades

    n_muted = float((gain < 0.5).mean())
    log(f"talk-only (no singing) in {n_muted * 100:.0f}% of the track -> muted there")
    t = np.arange(kept.shape[1]) / sr
    return np.interp(t, grid, gain).astype(np.float32)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Remove background talking from a song recording while "
                    "keeping the sung vocals and instruments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("input", type=Path, help="input audio (wav, flac, mp3, ...)")
    ap.add_argument("-o", "--output", type=Path,
                    help="output wav (default: <input>_fixed.wav next to the input)")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="0..1, how much of the detected talking to remove")
    ap.add_argument("--preset", choices=list(PRESETS), default="balanced",
                    help="how hard to go after the talking")
    ap.add_argument("--threshold", type=float,
                    help="override preset: singing-score threshold (higher = stricter)")
    ap.add_argument("--air", type=float,
                    help="override preset: 0..1 of the >9 kHz band kept while singing")
    ap.add_argument("--consonants", type=float,
                    help="override preset: 0..1 of non-harmonic sound kept while singing")
    ap.add_argument("--no-link", action="store_true",
                    help="don't keep the slides between sung notes")
    ap.add_argument("--min-held", type=float, default=0.22,
                    help="seconds a pitch must be held steady to count as a sung note")
    ap.add_argument("--width", type=float, default=35.0,
                    help="width (cents) kept around each sung harmonic")
    ap.add_argument("--gate", action="store_true",
                    help="also run an audio classifier that mutes talk-only stretches "
                         "(can mistake cleaned-up singing for talk; off by default)")
    ap.add_argument("--gate-speech", type=float, default=0.3,
                    help="speech probability above which a stretch counts as talking")
    ap.add_argument("--gate-singing", type=float, default=0.06,
                    help="singing probability below which a stretch has no singing")
    ap.add_argument("--gate-floor", type=float, default=-30.0,
                    help="dB applied to the vocal stem in talk-only stretches")
    ap.add_argument("--gate-window", type=float, default=2.0,
                    help="classifier window in seconds")
    ap.add_argument("--model", default="htdemucs_ft",
                    help="Demucs model (htdemucs_ft = best, htdemucs = ~4x faster)")
    ap.add_argument("--shifts", type=int, default=1,
                    help="Demucs random shifts (higher = slightly cleaner, slower)")
    ap.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    ap.add_argument("--save-stems", action="store_true",
                    help="also write the raw vocal stem, kept singing, removed talk "
                         "and instruments next to the output")
    ap.add_argument("--cache-dir", default=str(Path(__file__).resolve().parent / ".cache"),
                    help="where separated vocals are cached so re-runs with other "
                         "settings are fast")
    ap.add_argument("--no-cache", action="store_true", help="disable the cache")
    args = ap.parse_args(argv)

    if not args.input.exists():
        ap.error(f"input not found: {args.input}")
    out_path = args.output or args.input.with_name(args.input.stem + "_fixed.wav")
    strength = float(np.clip(args.strength, 0.0, 1.0))

    t0 = time.time()
    mix, sr, subtype = load_audio(args.input)
    log(f"loaded {args.input.name}: {mix.shape[1] / sr:.1f}s, {sr} Hz, {mix.shape[0]} ch")

    vocals = cached_separation(args.input, mix, sr, args)

    import contours

    cfg = dict(PRESETS[args.preset])
    for k in ("threshold", "air", "consonants"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    if args.no_link:
        cfg["link"] = False
    log(f"splitting singing from talking (preset {args.preset}: {cfg})...")
    singing = contours.extract_singing(
        vocals, sr, min_held=args.min_held, threshold=cfg["threshold"],
        width_cents=args.width, consonant_gain=cfg["consonants"],
        hf_gain=cfg["air"], link=cfg["link"], log=log)
    if args.gate:
        g = speech_gate(singing, sr, pick_device(args.device), args.gate_window,
                        args.gate_window / 4, args.gate_speech, args.gate_singing,
                        args.gate_floor)
        singing = singing * g[None, :]

    removed = strength * (vocals - singing)
    fixed = mix - removed

    def db(x):
        return 10 * np.log10(np.mean(x ** 2) + 1e-12)

    log(f"vocal stem {db(vocals):.1f} dB, removed talk {db(removed):.1f} dB "
        f"({db(removed) - db(vocals):+.1f} dB relative to vocal stem)")

    peak = float(np.abs(fixed).max())
    if peak > 1.0:
        log(f"output peak {peak:.2f} > 1.0, scaling down to avoid clipping")
        fixed /= peak * 1.001

    write(out_path, fixed, sr, subtype)
    if args.save_stems:
        base = out_path.with_suffix("")
        write(Path(f"{base}_vocals_raw.wav"), vocals, sr, "FLOAT")
        write(Path(f"{base}_vocals_kept.wav"), vocals - removed, sr, "FLOAT")
        write(Path(f"{base}_talk_removed.wav"), removed, sr, "FLOAT")
        write(Path(f"{base}_instruments.wav"), mix - vocals, sr, "FLOAT")
    log(f"done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
