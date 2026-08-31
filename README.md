# 🗣️ AudioToText — WhisperX (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ulysis022219/AudioToText-WhisperX/blob/main/AudioToText_WhisperX.ipynb)
[![Validate notebook](https://github.com/ulysis022219/AudioToText-WhisperX/actions/workflows/validate.yml/badge.svg)](https://github.com/ulysis022219/AudioToText-WhisperX/actions/workflows/validate.yml)

Transcribe audio/video in Google Colab with [WhisperX](https://github.com/m-bain/whisperX), word-level alignment, CJK-safe subtitle splitting, persistent Google Drive outputs, optional OpenAI transcription, and optional DeepL translation.

## Quick start

1. Open the notebook with the badge above.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Run Step 1. It verifies exact package versions and intentionally restarts once.
4. Run Step 2 and add files to `/content/drive/MyDrive/for process`.
5. Run Step 3, then Step 4 to save outputs.

Steps 2.5 (microphone recording) and 5 (DeepL) are optional.

## Workflow

| Step | Purpose |
|---|---|
| **1** | Install and verify pinned Python 3.13 / PyTorch cu128 dependencies; restart cleanly. |
| **2** | Mount Drive and create `MyDrive/for process`. |
| **2.5** | Optionally record audio directly to the Drive input folder. |
| **3** | Transcribe locally with WhisperX or explicitly upload audio to OpenAI. Checkpoint every completed file. |
| **4** | Recover checkpoints if needed and save persistent outputs to Drive. |
| **5** | Optionally send transcript text to DeepL and save translated outputs. |

## Secrets

Add secrets with the **🔑 Secrets** panel in Colab and enable **Notebook access**.

| Secret | When needed |
|---|---|
| `HF_TOKEN` | Optional for public alignment downloads; required only if a safe allowlisted model is gated. |
| `OPENAI_API_KEY` | Required only for the OpenAI backend. Never entered into a notebook form. |
| `DEEPL_API_KEY` | Required only when Step 5 is run. |

Do not paste credentials into cells or save a notebook containing a secret. Rotate any credential that was exposed.

## Privacy and storage

- **Local WhisperX:** audio is processed inside the Colab runtime. ASR and alignment model files are downloaded from Hugging Face; WhisperX's bundled VAD checkpoint is verified locally before loading.
- **OpenAI backend:** selected audio/chunks are uploaded to OpenAI for transcription.
- **DeepL:** transcript text, but not audio, is sent to DeepL when Step 5 is run.
- **Google Drive:** raw Step 3 checkpoints and generated outputs are stored under `MyDrive/audio_transcription` by default.

Review the applicable provider policies before processing sensitive material.

## Step 3 options

| Option | Meaning |
|---|---|
| `transcription_backend` | Local WhisperX (default) or OpenAI API. |
| `task_label` | Transcribe (default) or translate speech to English. |
| `audio_file` | One full input path per line; commas in filenames are supported. |
| `language` | Force a language or auto-detect it. |
| `use_model` | Local Whisper model; `large-v3` favors accuracy. |
| `prompt` | Optional free-text vocabulary/style hint; blank by default to avoid bias. |
| `prompt_names` / `prompt_terms` / `prompt_acronyms` / `subject_area` | Structured hints merged into the prompt to favor names, terminology, abbreviations, and the subject domain. |
| `quality_mode` | `Balanced` (beam 3) or `High accuracy` (beam 5). `High accuracy` is slower. |
| `context_conditioning` | Feed previous text into each pass for continuity; may repeat on noisy audio, so off by default. |
| `compute_type` | `float16` for GPU, automatically changed to `int8` on CPU. |
| `batch_size` | Local WhisperX batch size. Defaults to `8` for a Colab T4; on CUDA OOM it automatically retries at half the size down to `1`. |
| `word_timestamps` | Request local WhisperX forced alignment for transcription. |
| `fail_if_alignment_fails` | Stop instead of producing explicitly marked degraded output. |
| `max_chars_per_line` | CJK character limit; Latin-language segments are not destructively rejoined. |

WhisperX 3.8.6's bundled Pyannote VAD uses a legacy checkpoint, so Step 3 verifies its pinned SHA-256 before loading it in a narrowly scoped exception. Forced weights-only loading remains enabled everywhere else. Alignment status, model revision, and any error are preserved in JSON. Torchaudio models are pinned by the exact Torch stack; Hugging Face alignment uses an immutable safetensors allowlist. Languages without a safe artifact degrade explicitly instead of loading legacy pickle weights. Translation-to-English does not run word alignment.

For English output, keep `language` on the known source language, use `quality_mode="High accuracy"` and `context_conditioning` for continuity, add specialist hints under the terminology fields, and review the per-file low-confidence segment flags. Local WhisperX segments also record `avg_logprob`, `no_speech_prob`, and `compression_ratio` quality signals; flagged segments are collected in the checkpoint as `quality_notes` and summarized at the end of Step 3 so ASR uncertainty is visible before translation.

## Outputs and recovery

Step 3 atomically updates:

```text
/content/drive/MyDrive/audio_transcription/_last_results.json
```

Step 4 automatically validates and loads that checkpoint if in-memory `results` were lost. Final files default to the same Drive folder.

Supported formats are `txt`, `vtt`, `srt`, `tsv`, and `json`. Step 4:

- prevents same-basename and previous-run overwrites by default;
- writes atomically;
- normalizes only newly generated SRT/VTT files to UTF-8 BOM + CRLF;
- preserves untouched word-level data in `.raw.json` by default;
- leaves repetition cleanup off by default because repeated speech may be intentional.

## OpenAI behavior

Audio over the upload safety threshold is converted by FFmpeg to verified sub-24-MiB MP3 chunks in an isolated temporary directory. Chunk timestamps retain their real source offsets, including silence, and temporary files are removed automatically.

## DeepL behavior

Step 5 sends a list of segment strings through the DeepL API and optionally supplies neighboring text as API context. It does not embed transcript text as XML. Each returned batch is cardinality-checked and checkpointed after translation so partial work remains inspectable if a later request fails.

## Troubleshooting

- **Runtime restarts after Step 1:** expected once; reconnect and continue to Step 2.
- **Environment verification fails:** remove `/tmp/deps_installed_v3` and rerun Step 1.
- **CUDA out of memory:** Step 3 automatically halves `batch_size` and retries. If it still fails at `1`, restart the runtime, then use `compute_type="int8"` or a smaller model.
- **CPU warning:** enable a GPU runtime; CPU execution is much slower.
- **VAD security check failed:** rerun Step 1 to reinstall the pinned WhisperX 3.8.6 package; do not bypass the checksum.
- **Alignment failed:** inspect `alignment_error` in JSON, confirm network/model access, or enable `fail_if_alignment_fails` for strict behavior.
- **Secret not found:** confirm its exact name and enable Notebook access.
- **Output already exists:** the notebook adds a suffix unless `overwrite_existing` is enabled.

## Development validation

The notebook remains self-contained so the Colab badge works without cloning the repository. Pure utility behavior is tested directly from the notebook source:

```bash
python -m unittest discover -s tests -v
```

CI runs the checks on Python 3.11 and 3.13. Tests cover notebook syntax and metadata, dependency/security invariants, checkpointing, CJK versus Latin splitting, OpenAI source offsets and chunk limits, output validation/collisions, raw JSON preservation, subtitle normalization scope, and DeepL response checks.

## Credits

- [WhisperX](https://github.com/m-bain/whisperX)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Carleslc/AudioToText](https://github.com/Carleslc/AudioToText), the original notebook basis
- [DeepL API](https://www.deepl.com/pro-api)
