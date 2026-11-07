# songfixer

Removes background talking from a recorded song while keeping the singing and instruments.

## Run it

```bash
.venv/bin/python songfixer.py "Standard recording 6.mp3"
```

This writes `Standard recording 6_fixed.wav` next to the input. WAV, FLAC and MP3 inputs all work. The first run downloads the separation model. Separated vocals are cached in `.cache/`, so re-running with other settings takes about 8 s.

Pick how aggressive it is with `--preset`:

| preset | sound |
|---|---|
| `gentle` | keeps the most of the singer; some talk remains |
| `balanced` (default) | middle ground |
| `strong` | removes the most talk; singing gets gaps and sounds thinner |

Add `--save-stems` to also get these files:

| file | contents |
|---|---|
| `*_talk_removed.wav` | what was taken out. It should be talking only. |
| `*_vocals_kept.wav` | the vocals that remain |
| `*_vocals_raw.wav` | the full vocal stem: singing and talking |
| `*_instruments.wav` | everything except the vocal stem |

## How it works

1. **Demucs** (`htdemucs_ft`) splits the recording into vocals and instruments. Talking voices land in the vocals stem, so the instruments are never touched.
2. **Pitch-line tracking** ([contours.py](contours.py)). Every pitched voice in the vocal stem is traced as pitch lines. Overlapping voices get separate lines. Fake "ghost" lines at ½ or ⅓ of a real pitch are dropped.
3. **Singing or talking.** A sung note is held: a steady pitch, vibrato allowed, for longer than about 0.22 s. Spoken syllables are shorter and slide in pitch. Short slides that join two sung notes count as singing.
4. **Harmonic mask.** The full harmonic series of each sung line is kept up to 9 kHz, which preserves the voice's brightness. Sound between those harmonics is removed, and that is where the talking is.
5. The output is `original − (vocal stem − kept singing)`, so the instruments pass through unchanged.

## Fine-tuning

These options override the preset:

| problem | try |
|---|---|
| Talk is still audible | `--preset strong`, or `--threshold 2` |
| Singing has gaps or drops out | `--preset gentle`, or `--min-held 0.15` |
| Singing sounds dull | `--air 0.8` |
| Singing sounds robotic | `--consonants 0.1` (lets a little talk back in), `--width 50` |
| Talk-only stretches remain | `--gate` (can also mute real singing; check the result) |
| Just turn it down | `--strength 0.7` |

## Limitations

- Talk at exactly the same pitch as the singer, at the same moment, cannot be separated.
- Sung consonants are not harmonic, so they get softened. `--consonants` brings them back, together with some talk.
- Talk that Demucs puts into the instrument stems is not removed.
- Held spoken vowels such as "sooo…" can pass as singing.
