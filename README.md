<div align="center">

# 🗣️ AudioToText — WhisperX

**Turn audio and video into accurate transcripts, subtitles, and translations — free, in your browser, on Google Colab.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ulysis022219/AudioToText-WhisperX/blob/main/AudioToText_WhisperX.ipynb)
[![Validate notebook](https://github.com/ulysis022219/AudioToText-WhisperX/actions/workflows/validate.yml/badge.svg)](https://github.com/ulysis022219/AudioToText-WhisperX/actions/workflows/validate.yml)
![Python 3.13](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)
![WhisperX 3.8.6](https://img.shields.io/badge/WhisperX-3.8.6-6E40C9)

</div>

---

## ✨ Features

- **Accurate local transcription** with [WhisperX](https://github.com/m-bain/whisperX) (`large-v3` by default) on Colab's free T4 GPU — audio never leaves the runtime.
- **Word-level timestamps** via forced alignment, with subtitle splitting that is safe for Japanese and Chinese.
- **Batch processing** — point it at a Drive folder and it transcribes every audio/video file inside.
- **Crash-safe** — results are checkpointed to Google Drive after every file, and Step 4 recovers them after a disconnect.
- **Subtitle-ready output** in `srt`, `vtt`, `txt`, `tsv`, and `json` (UTF-8 BOM + CRLF for broad player support).
- **Optional DeepL translation** into 30+ languages, translating whole sentences instead of subtitle fragments.
- **Optional OpenAI backend** for when you don't want to use a GPU.

## 🚀 Quick start

1. Click **Open in Colab** above.
2. Select **Runtime → Change runtime type → T4 GPU → Save**.
3. Run **Step 1**. It installs pinned dependencies, then restarts the runtime once — this is expected.
4. Run **Step 2** to connect Google Drive, then drop your files into `My Drive/for process`.
5. In **Step 3**, paste a file path (or the whole folder path) into `audio_file` and run it.
6. Run **Step 4** to save the results to `My Drive/audio_transcription`.

## 🧭 Workflow

| Step | What it does | Required |
|:---:|---|:---:|
| **1** | Installs and verifies the pinned Python 3.13 / PyTorch cu128 stack, then restarts | ✅ |
| **2** | Mounts Google Drive and lists the files in `MyDrive/for process` | ✅ |
| **3** | Transcribes (or translates to English) and checkpoints each finished file | ✅ |
| **4** | Writes the output files to Drive, recovering from the checkpoint if needed | ✅ |
| **5** | Translates the transcript with DeepL and saves the translated subtitles | — |

## ⚙️ Settings

### Step 3 — Transcribe

| Option | Default | Description |
|---|---|---|
| `transcription_backend` | Local WhisperX | Local GPU (private) or OpenAI API (uploads audio) |
| `task_label` | Transcribe | `Transcribe`, or `Translate to English` with Whisper |
| `audio_file` | — | A file path or a folder path. Several paths can be given, one per line, by editing the code |
| `language` | Japanese | Pick the spoken language, or `Auto-Detect` (it only listens to the first 30 s) |
| `use_model` | `large-v3` | `large-v2` is an alternative that some find steadier on long audio. `kotoba-whisper-v2.0` is a Japanese-only [distilled large-v3](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster), about 6× faster (transcribe only). `anime-whisper` is [Kotoba fine-tuned on voice-acted Japanese](https://huggingface.co/litagin/anime-whisper) (whispers, breaths, emotive speech); converted once (~5 min) and cached in Drive under `_models/`. Both Kotoba-based models run on 15-second chunks, as Kotoba does |
| `quality_mode` | Balanced | `Balanced` (beam 3) or `High accuracy` (beam 5, slower) |
| `quiet_speech` | off | More sensitive voice detection and 15-second chunks, for whispering/ASMR |
| `reduce_repetition` | off | Blocks exact loops (a word hallucinated dozens of times) and shortens any run of 8+ repeats to four (るるるる…). Can also trim genuine repetition |
| `prompt` | — | Short terminology hints, e.g. names Whisper keeps mishearing |
| `compute_type` | `float16` | Switched to `int8` automatically on CPU |
| `batch_size` | `8` | Halved automatically on CUDA out-of-memory, down to `1` |
| `word_timestamps` | on | Forced alignment for word-level timing |
| `fail_if_alignment_fails` | off | Stop instead of saving output marked as degraded |
| `max_chars_per_line` | `20` | Subtitle line length for Japanese/Chinese |

> [!TIP]
> **Japanese ASMR / voice drama:** use `anime-whisper` with `quiet_speech` and `reduce_repetition` on, then `cleanup_repetitions` in Step 4. Try `large-v3` on the same settings if a file has long stretches with no text.

> [!TIP]
> Keep `prompt` short. Whisper can repeat hint text over silence. If you see hallucinated hint text, clear the field and rerun.

### Step 4 — Save

| Option | Default | Description |
|---|---|---|
| `output_dir` | `MyDrive/audio_transcription` | Where the files are written |
| `output_formats` | `srt` | Any comma-separated mix of `txt, vtt, srt, tsv, json` |
| `cleanup_repetitions` | off | Collapses repetition loops (e.g. a phrase hallucinated over silence) |
| `save_raw_json` | on | Also writes an uncleaned `.raw.json` with word-level data |
| `overwrite_existing` | off | When off, a new run gets a `-1`, `-2`, … suffix instead of replacing old files |

### Step 5 — DeepL (optional)

| Option | Default | Description |
|---|---|---|
| `deepl_target_language` | English (American) | Any DeepL target language |
| `deepl_formality` | default | `formal` / `informal` where the target supports it |
| `share_context` | on | Sends neighbouring lines as context for better coherence |
| `deepl_instructions` | subtitle style hint | Free-text style hint (DeepL custom instructions) for EN/DE/ES/FR/IT/JA/KO/ZH targets; skipped automatically if your plan rejects it. Clear it to turn off |

Step 5 rejoins Step 3's short display fragments into whole sentences before translating them (a sentence shorter than 1.5 s is joined to the next, so crammed moments don't flash by), uses DeepL's quality-optimized model where available, and splits the translation back into subtitles of at most two 42-character lines, keeping sentences together. Brief cues stay on screen for at least 1 s (≈15 chars/s) when the next cue allows it, and walls of repeats (`slurp-slurp-slurp-…`) are shortened. Progress is checkpointed after every batch, so an interrupted run resumes where it stopped.

## 🔐 Secrets

Add these in Colab's **🔑 Secrets** panel and turn on **Notebook access**. Never paste keys into a cell.

| Secret | Needed for |
|---|---|
| `HF_TOKEN` | Optional. Only needed if an alignment model is gated |
| `OPENAI_API_KEY` | The OpenAI backend |
| `DEEPL_API_KEY` | Step 5 |

## 🛡️ Privacy & security

| Where your data goes | When |
|---|---|
| **Colab runtime only** | Local WhisperX (default). Only model weights are downloaded |
| **OpenAI** | OpenAI backend: the audio (or ≤24 MiB MP3 chunks of it) is uploaded |
| **DeepL** | Step 5: transcript text only, never audio |
| **Google Drive** | Checkpoints and outputs under `MyDrive/audio_transcription` |

- Every dependency version is pinned exactly and verified after install.
- PyTorch's weights-only loading is enforced globally. The one legacy checkpoint WhisperX bundles (its VAD model) is verified against a pinned SHA-256 before it is loaded.
- Alignment models come from TorchAudio or from an allowlist of Hugging Face repos pinned to immutable safetensors revisions. Unsafe pickle fallbacks are disabled.
- Output files are written atomically and are never overwritten unless you ask.

## 📁 Outputs & recovery

Step 3 atomically updates `MyDrive/audio_transcription/_last_results.json` after every file. If the runtime disconnects, run Step 4 on its own and it will reload that checkpoint. Files with the same name from different folders get a short hash suffix so they never collide.

For each input, the JSON output also records the alignment status, the alignment model revision, and `quality_notes`, which flag segments with low confidence, likely non-speech, repetition loops, stock phrases Whisper invents over silence (`ご視聴ありがとうございました`, `Thanks for watching`), cues stretched over silence or too fast to read, and stretches of a minute or more with no text (possibly missed whispers). Step 3 prints them all. Voice detection cuts audio exactly where its score crosses a threshold, which can clip soft, whispered starts and endings, so each speech chunk is widened by up to 0.4 s into the silence around it. Timing is also repaired after alignment: the aligner hands the silence after a word (a breath, a kiss) to that word, so each word is capped at 1.5 s (or 0.5 s per character) and a cue ends when its speech does; text crammed into a sliver right after such a silence is started earlier, at about 8 characters per second; and cues too brief to read are held on screen longer (up to 8 characters per second, minimum 1 second) when the silence after them allows.

## 🩺 Troubleshooting

| Problem | Fix |
|---|---|
| Runtime restarted after Step 1 | Expected. Continue with Step 2 |
| Environment verification failed | Delete `/tmp/deps_installed_v4` and rerun Step 1 |
| CUDA out of memory | Handled automatically. If it still fails at `batch_size=1`, restart the runtime and use `int8` or a smaller model |
| `cudaErrorInvalidDevice` | Restart the runtime (or switch the GPU type), then rerun from Step 1 |
| Transcript is in the wrong language | Set `language` to the spoken language (the default is Japanese) |
| A name or term is misheard | Add the correct spelling to Step 3 `prompt` |
| VAD security check failed | Rerun Step 1 to reinstall WhisperX 3.8.6. Do not bypass the checksum |
| Alignment failed | Check `alignment_error` in the JSON output. Enable `fail_if_alignment_fails` for strict runs |
| Secret not found | Check the exact secret name and that Notebook access is on |

## 🧪 Development

The notebook is self-contained, so the Colab badge works without cloning. Its pure functions are tested directly from the notebook source, without GPU or network access:

```bash
python -m unittest discover -s tests -v
```

CI runs the suite on Python 3.11 and 3.13 for every push and pull request.

## 🙏 Credits

[WhisperX](https://github.com/m-bain/whisperX) · [OpenAI Whisper](https://github.com/openai/whisper) · [DeepL API](https://www.deepl.com/pro-api) · based on [Carleslc/AudioToText](https://github.com/Carleslc/AudioToText)
