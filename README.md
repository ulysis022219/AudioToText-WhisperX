# 🗣️ AudioToText — WhisperX (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ulysis022219/AudioToText-WhisperX/blob/main/AudioToText_WhisperX.ipynb)

Transcribe or translate audio/video in Google Colab using [WhisperX](https://github.com/m-bain/whisperX) — faster Whisper with word-level timestamps. Tuned for long-form and Japanese/CJK audio, with subtitle-friendly output (`srt`, `vtt`, `txt`, `tsv`, `json`) and optional DeepL translation.

Based on [Carleslc/AudioToText](https://github.com/Carleslc/AudioToText), reworked for WhisperX with CJK-safe segment splitting, hallucination/repetition cleanup, VRAM cleanup between files, and a Google Drive workflow.

---

## ✨ Features

- **WhisperX** transcription with word-level timestamps and forced alignment.
- **CJK-safe subtitle splitting** — splits by character count and joins without spaces (Japanese/Chinese have no word spaces).
- **Hallucination cleanup** — collapses repeated words/phrases and no-space CJK repeats (e.g. `ぐりぐりぐり…` → `ぐり`), drops duplicate segments.
- **Google Drive workflow** — files persist across sessions, no re-uploading.
- **Optional OpenAI API** path (`whisper-1`) with automatic chunking for files over 25 MB.
- **Optional DeepL** translation of the finished transcript.
- **Correct subtitle encoding** — UTF-8 BOM + CRLF so `.srt`/`.vtt` open cleanly everywhere.

---

## 🚀 Quick start

1. Open `AudioToText_WhisperX.ipynb` in Google Colab.
2. **Runtime → Change runtime type → GPU** (T4 is fine; CPU works but is very slow).
3. Add your API keys as Colab Secrets — see [🔑 API keys](#-api-keys-colab-secrets) below.
4. Run the steps in order.

### Steps

| Step | What it does |
|------|--------------|
| **1** | Installs dependencies (`uv`, PyTorch cu124, WhisperX, …). **Runtime auto-restarts** — expected, just continue. |
| **2** | Mounts Google Drive. Drop audio/video into `MyDrive/for process`. |
| **2.5** | *(Optional)* Record from your microphone → `recording.wav`. |
| **3** | Transcribe / translate. Set `audio_file` to your file path. |
| **4** | Save results to the `audio_transcription` folder (`srt` by default). |
| **5** | *(Optional)* Translate the transcript with DeepL. |

> Step 1 restarts the runtime once. This is normal — dependencies need a clean process. Continue to Step 2 after it comes back.

---

## 📁 Adding audio (Step 2)

Run the **Mount Google Drive** cell. It mounts your Drive and ensures the folder exists:

```
/content/drive/MyDrive/for process
```

Drop your audio/video files into that folder (via the Colab **Files** sidebar on the left, or at [drive.google.com](https://drive.google.com)). The cell lists what's currently there so you can copy a path.

In **Step 3**, set `audio_file` to the full path, e.g.:

```
/content/drive/MyDrive/for process/yourfile.wav
```

Process multiple files by comma-separating paths:

```
/content/drive/MyDrive/for process/a.wav, /content/drive/MyDrive/for process/b.wav
```

Almost any audio/video format is [supported](https://gist.github.com/Carleslc/1d6b922c8bf4a7e9627a6970d178b3a6).

---

## 🔑 API keys (Colab Secrets)

Keys are read from **Colab Secrets**, not hard-coded in the notebook. Secrets are stored per-account, encrypted, and never saved into the notebook file — safe to share the `.ipynb`.

### Add a secret

1. In Colab, click the **🔑 key icon** in the left sidebar (**Secrets**).
2. **+ Add new secret**.
3. Enter the **Name** and **Value**.
4. Toggle **Notebook access** ON for that secret.

| Secret name | Needed for | Where to get it |
|-------------|-----------|-----------------|
| `HF_TOKEN` | **Required** for WhisperX alignment / diarization models | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `DEEPL_API_KEY` | Optional — Step 5 DeepL translation | [deepl.com/pro-api](https://www.deepl.com/pro-api) |

### How the notebook reads them

**`HF_TOKEN`** — Step 3 pulls it from Secrets and exports it to the environment:

```python
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
```

Just add the `HF_TOKEN` secret (with notebook access ON) and it works — nothing else to do.

**`DEEPL_API_KEY`** — Step 5 reads it straight from Secrets (falling back to the `DEEPL_API_KEY` environment variable if set):

```python
deepl_api_key = os.environ.get("DEEPL_API_KEY", "")
if not deepl_api_key:
    from google.colab import userdata
    deepl_api_key = userdata.get('DEEPL_API_KEY') or ""
```

Just add the `DEEPL_API_KEY` secret (with notebook access ON) — no extra step needed.

**OpenAI API key** — optional. It is **not** a secret; it's a form field. To use OpenAI's hosted `whisper-1` instead of local WhisperX, paste your key into the `api_key` field in **Step 3**. Leave it blank to run WhisperX locally on the GPU (the default, and free).

> **Never paste keys directly into shared notebook cells.** Use Secrets. If a key ever lands in the notebook, rotate it.

---

## ⚙️ Key parameters (Step 3)

| Parameter | Meaning |
|-----------|---------|
| `task` | `Transcribe` or `Translate to English`. |
| `audio_file` | Path(s) to input. Comma-separate for batch. |
| `use_model` | `tiny` … `large-v3`. `large-v3` = best quality. |
| `language` | Force a language or `Auto-Detect`. |
| `prompt` | Hotwords / style hint to steer decoding. |
| `compute_type` | `float16` (GPU) or `int8` (CPU / low VRAM). |
| `batch_size` | Higher = faster, more VRAM. Lower if you hit OOM. |
| `word_timestamps` | Enables forced alignment (needed for tight subtitles). |
| `max_words_per_segment` | **Characters** per subtitle line for CJK (not words). |
| `api_key` | OpenAI key → use hosted `whisper-1` instead of local. |

---

## 💾 Output (Step 4)

Results are written to the `audio_transcription/` folder. Change `output_formats` to any of `txt,vtt,srt,tsv,json` (comma-separated). `.srt`/`.vtt` are post-processed to UTF-8 BOM + CRLF for maximum player compatibility.

---

## 🩹 Troubleshooting

- **CUDA out of memory** — lower `batch_size`, or use a smaller `use_model`. The notebook frees VRAM between files, but very long audio + large batch can still OOM.
- **Runtime restarts after Step 1** — expected, once. Continue to Step 2.
- **`HF_TOKEN` errors** — confirm the secret exists and **Notebook access** is ON.
- **DeepL does nothing / auth error** — check the `DEEPL_API_KEY` secret exists with Notebook access ON, or the source and target language are the same.
- **CPU warning** — enable GPU: **Runtime → Change runtime type → GPU**.
- **Japanese alignment one-time conversion** — the first Japanese run converts the alignment model to safetensors (CVE-2025-32434 fix). This is cached; later runs skip it.

---

## 🙏 Credits

- [WhisperX](https://github.com/m-bain/whisperX) — Max Bain et al.
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Carleslc/AudioToText](https://github.com/Carleslc/AudioToText) — original notebook this is based on.
- [DeepL API](https://www.deepl.com/pro-api)
