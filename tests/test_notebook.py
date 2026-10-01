import ast
import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import types
import unittest
import zlib
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "AudioToText_WhisperX.ipynb"
NOTEBOOK = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
CELLS = {cell.get("metadata", {}).get("id"): "".join(cell.get("source", []))
         for cell in NOTEBOOK["cells"]}
INSTALL = CELLS["SJl7HJOeo0-P"]
TRANSCRIBE = CELLS["opNkn_Lgpat4"]
OUTPUT = CELLS["wNsrB45_lCIl"]
DEEPL = CELLS["28f7EIP-rez0"]


def load_functions(source, names, extra_globals=None):
    tree = ast.parse(source)
    selected = [node for node in tree.body
                if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names)
                or (isinstance(node, ast.Assign) and len(node.targets) == 1  # constants
                    and getattr(node.targets[0], "id", None) in names)]
    namespace = {
        "Path": Path,
        "copy": copy,
        "hashlib": __import__("hashlib"),
        "json": json,
        "math": __import__("math"),
        "os": os,
        "re": __import__("re"),
        "tempfile": tempfile,
    }
    namespace.update(extra_globals or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<notebook>", "exec"), namespace)
    return namespace


class NotebookStructureTests(unittest.TestCase):
    def test_notebook_json_and_every_code_cell_parse(self):
        self.assertEqual(NOTEBOOK["nbformat"], 4)
        self.assertEqual(NOTEBOOK["nbformat_minor"], 5)
        self.assertEqual(NOTEBOOK["metadata"]["language_info"]["version"], "3.13")
        for index, cell in enumerate(NOTEBOOK["cells"]):
            if cell["cell_type"] == "code":
                with self.subTest(cell=index):
                    ast.parse("".join(cell["source"]))

    def test_notebook_text_has_no_mojibake(self):
        raw = NOTEBOOK_PATH.read_text(encoding="utf-8")
        for marker in ("Ã", "â€", "â€”", "ðŸ", "ï¸", "�"):
            self.assertNotIn(marker, raw)
        self.assertIn("🗣️ AudioToText — WhisperX", raw)

    def test_install_is_exact_and_removed_bloat_stays_removed(self):
        for requirement in (
            '"torch": "2.8.0+cu128"', '"whisperx": "3.8.6"',
            '"openai": "3.6.0"', '"deepl": "1.32.0"',
            '"numpy": "2.5.2"', '"ctranslate2": "4.8.2"',
            '"torchcodec": "0.7.0+cu128"',
        ):
            self.assertIn(requirement, INSTALL)
        for removed in ("cohere", "tensorflow-probability", "requests==", "ffmpeg-python", "pydub",
                        "openai-whisper"):
            self.assertNotIn(removed, INSTALL)
        self.assertIn('["ffmpeg", "-version"]', INSTALL)
        self.assertIn("verify_environment()", INSTALL)
        # A single uv resolve installs exactly the verified pins.
        self.assertIn('f"{name}=={pinned}" for name, pinned in EXPECTED.items()', INSTALL)
        self.assertEqual(INSTALL.count("uv + ["), 1)
        for source in (TRANSCRIBE, OUTPUT, DEEPL):
            self.assertNotRegex(source, r"(from|import) whisper(\.| |$)")

    def test_credentials_and_model_loading_are_safe(self):
        self.assertIn('userdata.get("OPENAI_API_KEY")', TRANSCRIBE)
        self.assertNotIn("api_key = '' #@param", TRANSCRIBE)
        self.assertNotIn("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", TRANSCRIBE)
        self.assertIn('TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"', TRANSCRIBE)
        self.assertIn("WHISPERX_VAD_SHA256", TRANSCRIBE)
        self.assertIn("load_whisperx_with_verified_vad", TRANSCRIBE)
        self.assertIn("SAFE_ALIGN_MODELS_HF", TRANSCRIBE)
        self.assertIn("2785e99ab97df77a32b5bd0ece5c9fa188a02f19", TRANSCRIBE)
        self.assertIn('allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"]', TRANSCRIBE)
        self.assertIn('glob("*.safetensors")', TRANSCRIBE)
        self.assertIn('unsafe pickle fallback is disabled', TRANSCRIBE)
        self.assertIn('"weights_only": True', TRANSCRIBE)

    def test_privacy_and_defaults_are_explicit(self):
        self.assertIn('transcription_backend = "Local WhisperX (Colab GPU)"', TRANSCRIBE)
        self.assertIn('task_label = "Transcribe"', TRANSCRIBE)
        self.assertIn('prompt = ""', TRANSCRIBE)
        self.assertIn('batch_size = 8', TRANSCRIBE)
        self.assertIn("audio will be uploaded to OpenAI", TRANSCRIBE)
        self.assertIn("Sending transcript text to DeepL", DEEPL)
        self.assertIn("/content/drive/MyDrive/audio_transcription", OUTPUT)

    def test_false_legacy_paths_are_gone(self):
        self.assertNotIn("split_on_silence", TRANSCRIBE)
        self.assertNotIn("api_audio_chunk_path", TRANSCRIBE)
        self.assertNotIn("tag_handling", DEEPL)
        self.assertNotIn("<br/>", DEEPL)
        self.assertNotIn("glob.glob", OUTPUT)

    def test_english_translation_quality_controls_exist(self):
        self.assertIn('quality_mode = "Balanced"', TRANSCRIBE)
        self.assertIn('"High accuracy"', TRANSCRIBE)
        self.assertIn('quality["beam_size"]', TRANSCRIBE)
        # Options the batched WhisperX decoder ignores must not pose as settings.
        self.assertNotIn("context_conditioning", TRANSCRIBE)
        self.assertNotIn("best_of", TRANSCRIBE)
        self.assertIn("effective_prompt", TRANSCRIBE)
        self.assertIn("collect_quality_notes", TRANSCRIBE)
        self.assertIn("quality_notes", TRANSCRIBE)
        self.assertIn("Review", TRANSCRIBE)
        self.assertNotIn("use_asmr_profile", TRANSCRIBE + OUTPUT)
        self.assertIn('language = "Japanese" #@param', TRANSCRIBE)
        self.assertLess(TRANSCRIBE.index('use_model = "kotoba-tech/kotoba-whisper-v2.0-faster"'),
                        TRANSCRIBE.index("lang_code = None if"))
        # asr_options are fixed at load time; a changed option must force a reload.
        self.assertIn("json.dumps([asr_options, vad_options], sort_keys=True)", TRANSCRIBE)
        self.assertIn('"no_repeat_ngram_size": 10 if reduce_repetition else 0', TRANSCRIBE)
        self.assertIn('chunk_size=vad_options["chunk_size"]', TRANSCRIBE)
        # Kotoba-based models learned from short clips; their authors run them on 15 s chunks.
        self.assertIn('"chunk_size": 15 if quiet_speech or kotoba_based else 30', TRANSCRIBE)
        self.assertIn('segment["end"] = min(segment["end"], round(audio_seconds, 3))', TRANSCRIBE)
        self.assertIn('if re.search(r"\\w", segment.get("text", ""))]', TRANSCRIBE)  # drop "…"-only segments
        self.assertIn('"anime-whisper"]', TRANSCRIBE)
        self.assertLess(TRANSCRIBE.index('use_model = ensure_anime_whisper()'),
                        TRANSCRIBE.index("model_name = use_model"))
        self.assertIn('allow_patterns=["*.json", "*.txt", "*.safetensors"]', TRANSCRIBE)
        self.assertIn("asr_options=asr_options", TRANSCRIBE)
        self.assertLess(
            TRANSCRIBE.index('raw_quality_notes = collect_quality_notes(raw_asr_segments)'),
            TRANSCRIBE.index('result = whisperx.align'))
        self.assertIn('result["quality_notes"] = raw_quality_notes', TRANSCRIBE)
        self.assertNotIn('"hotwords": prompt or None', TRANSCRIBE)
        self.assertNotIn('api_options["prompt"] = prompt', TRANSCRIBE)


class TranscriptionUtilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_functions(
            TRANSCRIBE,
            {"verify_file_sha256", "atomic_json_dump", "join_word_texts",
             "split_long_segments", "merge_api_segments", "prepare_api_chunks",
             "collect_quality_notes", "compression_ratio", "STOCK_HALLUCINATIONS", "pad_vad_chunks",
             "split_vad_chunks_at_pauses",
             "is_cuda_out_of_memory", "transcribe_with_batch_backoff"},
            {"CJK_LANGUAGE_CODES": {"ja", "zh"}, "free_vram": lambda: None, "zlib": zlib},
        )

    def test_cuda_oom_retries_with_smaller_batches(self):
        attempted_batches = []

        class FakeModel:
            def transcribe(self, _audio, batch_size, **_kwargs):
                attempted_batches.append(batch_size)
                if batch_size > 8:
                    raise RuntimeError("CUDA failed with error out of memory")
                return {"segments": [{"text": "ok"}], "language": "ja"}

        free_vram = mock.Mock()
        with mock.patch("builtins.print"), mock.patch.dict(self.ns, {"free_vram": free_vram}):
            result, used_batch_size = self.ns["transcribe_with_batch_backoff"](
                FakeModel(), "audio", 32, language="ja", task="transcribe")
        self.assertEqual(attempted_batches, [32, 16, 8])
        self.assertEqual(free_vram.call_count, 2)
        self.assertEqual(used_batch_size, 8)
        self.assertEqual(result["language"], "ja")

    def test_non_oom_runtime_error_is_not_retried(self):
        class FakeModel:
            def transcribe(self, _audio, batch_size, **_kwargs):
                raise RuntimeError("invalid device ordinal")

        with self.assertRaisesRegex(RuntimeError, "invalid device ordinal"):
            self.ns["transcribe_with_batch_backoff"](
                FakeModel(), "audio", 8, language="ja", task="transcribe")

    def test_cuda_oom_at_batch_one_has_actionable_message(self):
        class FakeModel:
            def transcribe(self, _audio, batch_size, **_kwargs):
                raise RuntimeError("CUDA out of memory")

        with self.assertRaisesRegex(RuntimeError, "smaller model or compute_type='int8'"):
            self.ns["transcribe_with_batch_backoff"](
                FakeModel(), "audio", 1, language="ja", task="transcribe")

    def test_vad_checkpoint_hash_must_match(self):
        verify = self.ns["verify_file_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vad.bin"
            path.write_bytes(b"trusted checkpoint")
            expected = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            verify(path, expected)
            with self.assertRaises(RuntimeError):
                verify(path, "0" * 64)

    def test_verified_vad_load_restores_weights_only_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "whisperx" / "vads"
            package.mkdir(parents=True)
            module_file = package / "pyannote.py"
            module_file.write_text("", encoding="utf-8")
            vad_file = package.parent / "assets" / "pytorch_model.bin"
            vad_file.parent.mkdir()
            vad_file.write_bytes(b"trusted")

            events = []
            vad_model = types.SimpleNamespace()

            def fake_vad(*_args, **kwargs):
                events.append((os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD"), kwargs["vad_onset"]))
                return vad_model
            fake_vad.merge_chunks = lambda _scores, chunk_size, _onset, _offset: [
                {"start": 1.0, "end": 1.0 + chunk_size, "segments": [(1.0, 10.0), (13.0, 1.0 + chunk_size)]}]
            fake_pyannote = types.SimpleNamespace(__file__=str(module_file), Pyannote=fake_vad)
            fake_vads = types.ModuleType("whisperx.vads")
            fake_vads.pyannote = fake_pyannote
            fake_whisperx = types.SimpleNamespace(
                load_model=lambda *_args, **kwargs: kwargs["vad_options"] is vad and kwargs["vad_model"])
            fake_torch = types.SimpleNamespace(device=lambda value: value)
            namespace = load_functions(
                TRANSCRIBE, {"load_whisperx_with_verified_vad", "pad_vad_chunks", "split_vad_chunks_at_pauses"}, {
                    "WHISPERX_VAD_SHA256": __import__("hashlib").sha256(b"trusted").hexdigest(),
                    "verify_file_sha256": self.ns["verify_file_sha256"],
                    "whisperx": fake_whisperx,
                    "torch": fake_torch,
                    "logging": __import__("logging"),
                    "contextlib": __import__("contextlib"),
                    "io": __import__("io"),
                })
            with mock.patch.dict(sys.modules, {"whisperx.vads": fake_vads}):
                with mock.patch.dict(os.environ, {"TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1"}):
                    vad = {"chunk_size": 15, "vad_onset": 0.3, "vad_offset": 0.2}
                    self.assertIs(namespace["load_whisperx_with_verified_vad"](
                        "large-v3", "cuda", vad), vad_model)
                    self.assertEqual(os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"], "1")
            # The same thresholds reach the VAD model and transcribe()'s chunk merging.
            self.assertEqual(events, [(None, 0.3)])
            # Chunks are merged 1 s shorter, split at long pauses, then padded, so they still fit the 30 s window.
            self.assertEqual(vad_model.merge_chunks("scores", 30, onset=0.3, offset=0.2),
                             [{"start": 0.6, "end": 10.4, "segments": [(1.0, 10.0)]},
                              {"start": 12.6, "end": 30.4, "segments": [(13.0, 30.0)]}])

    def test_vad_chunks_split_at_long_pauses_only(self):
        split = self.ns["split_vad_chunks_at_pauses"]([
            {"start": 0.0, "end": 9.0, "segments": [(0.0, 2.0), (3.0, 5.0), (8.0, 9.0)]},
            {"start": 12.0, "end": 14.0, "segments": [(12.0, 14.0)]}])
        self.assertEqual([(chunk["start"], chunk["end"]) for chunk in split], [(0.0, 5.0), (8.0, 9.0), (12.0, 14.0)])
        # A click inside a pause neither blocks the split nor becomes a chunk of its own.
        split = self.ns["split_vad_chunks_at_pauses"]([
            {"start": 0.0, "end": 9.0, "segments": [(0.0, 2.0), (3.5, 3.55), (6.0, 7.0), (8.0, 8.05)]}])
        self.assertEqual([chunk["segments"] for chunk in split], [[(0.0, 2.0)], [(6.0, 7.0)]])

    def test_vad_chunks_are_padded_into_silence_only(self):
        chunks = [{"start": 0.1, "end": 5.0}, {"start": 5.3, "end": 9.0},
                  {"start": 9.0, "end": 12.0}, {"start": 20.0, "end": 25.0}]
        padded = self.ns["pad_vad_chunks"](chunks)
        # Short gaps are shared, touching chunks (a long turn split mid-speech) stay put.
        self.assertEqual([(round(chunk["start"], 3), round(chunk["end"], 3)) for chunk in padded],
                         [(0.0, 5.15), (5.15, 9.0), (9.0, 12.4), (19.6, 25.4)])

    def test_atomic_checkpoint_round_trip_and_replace(self):
        dump = self.ns["atomic_json_dump"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "checkpoint.json"
            dump({"schema_version": 1, "results": {"a": 1}}, path)
            dump({"schema_version": 1, "results": {"b": 2}}, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["results"], {"b": 2})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_cjk_split_preserves_timestamps(self):
        split = self.ns["split_long_segments"]
        segment = {
            "start": 0, "end": 4, "text": "日本語字幕",
            "words": [
                {"start": 0, "end": 1, "word": "日本"},
                {"start": 1, "end": 2, "word": "語"},
                {"start": 2, "end": 3, "word": "字幕"},
            ],
        }
        result = split([segment], max_chars=3, language_code="ja")
        self.assertEqual([item["text"] for item in result], ["日本語", "字幕"])
        self.assertEqual((result[1]["start"], result[1]["end"]), (2, 3))

    def test_latin_segments_are_not_rejoined_without_spaces(self):
        split = self.ns["split_long_segments"]
        segment = {"text": "hello world", "words": [
            {"start": 0, "end": 1, "word": "hello"},
            {"start": 1, "end": 2, "word": "world"},
        ]}
        self.assertIs(split([segment], max_chars=3, language_code="en")[0], segment)
        self.assertEqual(segment["text"], "hello world")

    def test_quality_notes_flag_low_confidence_segments(self):
        collect = self.ns["collect_quality_notes"]
        notes = collect([
            {"start": 0, "end": 1, "text": "clean", "avg_logprob": -0.2,
             "no_speech_prob": 0.1, "compression_ratio": 1.1},
            {"start": 1, "end": 2, "text": "???", "avg_logprob": -1.8,
             "no_speech_prob": 0.3, "compression_ratio": 1.0},
        ])
        self.assertEqual(len(notes), 1)
        self.assertIn("confidence", notes[0]["reasons"])
        self.assertEqual(notes[0]["text"], "???")

    def test_api_segments_use_source_offsets_not_prior_text_end(self):
        merge = self.ns["merge_api_segments"]
        result = merge([
            {"offset_seconds": 0, "response": {"language": "en", "segments": [
                {"start": 1, "end": 4, "text": "first"}]}},
            {"offset_seconds": 60, "response": {"language": "en", "segments": [
                {"start": 2, "end": 5, "text": "second"}]}},
        ])
        self.assertEqual(result["segments"][1]["start"], 62)
        self.assertEqual(result["segments"][1]["end"], 65)

    def test_api_chunk_export_enforces_limit_and_offsets(self):
        class Result:
            def __init__(self, stdout=""):
                self.stdout = stdout

        def fake_run(args, **_kwargs):
            if args[0] == "ffprobe":
                return Result("120\n")
            output_path = Path(args[-1])
            seconds = float(args[args.index("-t") + 1])
            output_path.write_bytes(b"x" * int(seconds * 1000))
            return Result()

        prepare = load_functions(
            TRANSCRIBE, {"prepare_api_chunks"},
            {"subprocess": types.SimpleNamespace(run=fake_run)})["prepare_api_chunks"]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.aiff"
            source.write_bytes(b"source")
            chunks = prepare(str(source), directory, max_bytes=50_000)
            self.assertGreater(len(chunks), 1)
            self.assertEqual(chunks[0]["offset_seconds"], 0)
            self.assertTrue(all(Path(item["path"]).stat().st_size < 50_000
                                for item in chunks))
            self.assertEqual(chunks, sorted(chunks, key=lambda item: item["offset_seconds"]))


class OutputUtilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_functions(
            OUTPUT,
            {"validate_output_formats", "load_results_checkpoint",
             "clean_repeated_words", "clean_segments", "output_name_map",
             "reserve_output_name", "normalize_subtitle"},
            {"ALLOWED_OUTPUT_FORMATS": {"txt", "vtt", "srt", "tsv", "json"}},
        )

    def test_output_formats_are_validated_and_deduplicated(self):
        validate = self.ns["validate_output_formats"]
        self.assertEqual(validate("srt, json,srt"), ["srt", "json"])
        with self.assertRaises(ValueError):
            validate("srt,exe")

    def test_checkpoint_validation(self):
        load = self.ns["load_results_checkpoint"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            path.write_text(json.dumps({"schema_version": 1, "results": {"x": {}}}), encoding="utf-8")
            self.assertIn("x", load(path)["results"])
            path.write_text(json.dumps({"schema_version": 2, "results": {}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load(path)

    def test_duplicate_basenames_get_stable_distinct_names(self):
        names = self.ns["output_name_map"](["/a/foo.wav", "/b/foo.mp3", "/c/bar.wav"])
        self.assertNotEqual(names["/a/foo.wav"], names["/b/foo.mp3"])
        self.assertEqual(names["/c/bar.wav"], "bar")

    def test_existing_outputs_are_exclusively_reserved(self):
        reserve = self.ns["reserve_output_name"]
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "clip.srt").write_text("old", encoding="utf-8")
            name = reserve("clip", ["srt", "raw.json"], directory)
            self.assertEqual(name, "clip-1")
            self.assertTrue(Path(directory, "clip-1.srt").exists())
            self.assertTrue(Path(directory, "clip-1.raw.json").exists())
            self.assertEqual(reserve("clip", ["srt"], directory, overwrite=True), "clip")

    def test_subtitle_normalization_only_touches_requested_file(self):
        normalize = self.ns["normalize_subtitle"]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new.srt"
            old = Path(directory) / "old.srt"
            target.write_bytes(b"one\ntwo\n")
            old.write_bytes(b"leave\nme\n")
            normalize(target)
            self.assertEqual(target.read_bytes(), b"\xef\xbb\xbfone\r\ntwo\r\n")
            self.assertEqual(old.read_bytes(), b"leave\nme\n")

    def test_cross_segment_character_stretch_is_removed_from_clean_outputs(self):
        segments = [
            {"start": 0, "end": 1, "text": "おいしぃぃぃぃぃぃ"},
            {"start": 1, "end": 2, "text": "ぃ" * 20},
            {"start": 2, "end": 3, "text": "ぃ" * 20 + "�"},
            {"start": 3, "end": 4, "text": "次の台詞"},
        ]
        cleaned = self.ns["clean_segments"](segments)
        self.assertEqual([segment["text"] for segment in cleaned], ["おいしぃぃぃ", "次の台詞"])
        self.assertEqual(segments[1]["text"], "ぃ" * 20)

    def test_short_drawl_and_replacement_character_are_preserved(self):
        segments = [
            {"start": 0, "end": 1, "text": "ありがとお"},
            {"start": 1, "end": 2, "text": "お" * 4},
            {"start": 2, "end": 3, "text": "さ�みつ"},
        ]
        cleaned = self.ns["clean_segments"](segments)
        self.assertEqual(
            [segment["text"] for segment in cleaned], ["ありがとお", "おおお", "さ�みつ"])

    def test_raw_json_is_written_before_words_are_removed(self):
        json_branch = OUTPUT.index('if output_format == "json":')
        remove_words = OUTPUT.index('segment.pop("words", None)')
        self.assertLess(json_branch, remove_words)
        self.assertIn("save_raw_json = True", OUTPUT)
        self.assertIn('cleanup_repetitions = False #@param {type:"boolean"}', OUTPUT)

    def test_failed_write_with_overwrite_keeps_previous_outputs(self):
        # With overwrite on, "reservations" are the user's real previous files.
        cleanup = OUTPUT.index("for reservation in reservations:")
        self.assertIn("if not overwrite_existing:", OUTPUT[cleanup - 200:cleanup])


class DeepLTests(unittest.TestCase):
    def test_translation_uses_secrets_and_validated_batch_resume(self):
        self.assertIn('userdata.get("DEEPL_API_KEY")', DEEPL)
        self.assertNotIn('os.environ.get("DEEPL_API_KEY")', DEEPL)
        self.assertIn("text=texts", DEEPL)
        self.assertIn("source_units[max(0, start - 10):start + len(batch)]", DEEPL)
        self.assertIn("if share_context else None", DEEPL)
        self.assertIn("len(responses) != len(batch)", DEEPL)
        self.assertIn("valid_resume_prefix", DEEPL)
        self.assertIn("source_signature", DEEPL)
        self.assertIn("atomic_deepl_checkpoint", DEEPL)
        self.assertIn('"translation_status": "in_progress"', DEEPL)
        self.assertIn('translated["translation_status"] = "complete"', DEEPL)

    def test_steps_four_and_five_keep_lines_that_translate_alike(self):
        class Translator:
            def __init__(self, key):
                pass
            def get_target_languages(self):
                return [types.SimpleNamespace(name="English (American)", code="EN-US")]
            def get_source_languages(self):
                return [types.SimpleNamespace(code="JA")]
            def translate_text(self, text, **options):
                return [types.SimpleNamespace(text="I love you.") for _ in text]

        utils = types.SimpleNamespace(get_writer=FakeWhisperWriter, WriteTXT=object,
                                      TO_LANGUAGE_CODE={"japanese": "ja"})
        deepl = types.SimpleNamespace(Translator=Translator, DeepLException=RuntimeError,
                                      QuotaExceededException=KeyError, AuthorizationException=KeyError)
        colab = types.SimpleNamespace(userdata=types.SimpleNamespace(get=lambda name: "key"))
        # Two different Japanese lines that DeepL translates alike must both stay on screen.
        namespace = {"task": "transcribe", "results": {"/in/a.wav": {"language": "ja", "text": "", "segments": [
            {"id": 0, "start": 1.0, "end": 2.0, "text": "好きだよ"},
            {"id": 1, "start": 5.0, "end": 6.0, "text": "好き"}]}}}
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {
                "whisperx": types.ModuleType("whisperx"), "whisperx.utils": utils, "deepl": deepl,
                "google": types.ModuleType("google"), "google.colab": colab}), \
                contextlib.redirect_stdout(io.StringIO()):
            for cell in (OUTPUT, DEEPL):
                exec(cell.replace('"/content/drive/MyDrive/audio_transcription"', repr(directory))
                     .replace("cleanup_repetitions = False", "cleanup_repetitions = True"), namespace)
            srt = Path(directory, "a-en-us.srt").read_text(encoding="utf-8-sig")
        self.assertEqual(srt.count("I love you."), 2)

    def test_translation_outputs_are_marked_english(self):
        self.assertIn('result["source_language"] = result.get("language")', TRANSCRIBE)
        self.assertIn('result["language"] = "en"', TRANSCRIBE)

    def test_translation_uses_sentence_units(self):
        self.assertNotIn("translation_mode", DEEPL)
        self.assertNotIn("translate_with_retry", DEEPL)  # deepl client retries itself
        self.assertNotIn("glossary", DEEPL)
        self.assertIn('model_type="prefer_quality_optimized"', DEEPL)
        self.assertIn("build_translation_units", DEEPL)
        self.assertLess(DEEPL.index('raise RuntimeError("Run Step 4 once'),
                        DEEPL.index("load_results_checkpoint(CHECKPOINT_PATH)"))
        self.assertIn("normalize_translation_source", DEEPL)
        self.assertIn("split_display_lines", DEEPL)
        # U+FFFD must appear only as a Python escape in source, never as the raw char.
        self.assertIn("\\ufffd", DEEPL)
        self.assertNotIn("\ufffd", DEEPL)

    def test_translation_checkpoint_invalidates_on_scheme_change(self):
        # The DeepL checkpoint is keyed to the input and would otherwise be reused
        # verbatim even after the translation scheme changes. A version constant must
        # be folded into the signature and persisted/validated so stale output is not
        # silently kept on re-run.
        self.assertIn("TRANSLATION_VERSION", DEEPL)
        self.assertIn('"version": TRANSLATION_VERSION', DEEPL)
        self.assertIn('"translation_version": TRANSLATION_VERSION', DEEPL)
        self.assertIn(
            'saved_translation.get("translation_version") == TRANSLATION_VERSION', DEEPL)


class DeepLTranslationUtilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_functions(
            DEEPL,
            {"normalize_translation_source", "build_translation_units",
             "split_display_lines"},
        )

    def test_normalize_translation_source_strips_artifacts(self):
        norm = self.ns["normalize_translation_source"]
        # U+FFFD removed, leading/trailing Japanese comma & space trimmed.
        self.assertEqual(norm("\ufffd、どうだろう?。"), "どうだろう?")
        self.assertEqual(norm("\ufffd"), "")
        self.assertEqual(norm("　、なかなかない。　"), "なかなかない")

    def test_build_translation_units_merges_fragments_into_sentences(self):
        build = self.ns["build_translation_units"]
        segments = [
            {"id": 0, "start": 0.0, "end": 1.5, "text": "んーと、じゃあ"},
            {"id": 1, "start": 1.5, "end": 3.0, "text": "ちょっと、待って。"},
            {"id": 2, "start": 3.0, "end": 4.5, "text": "一緒に"},
            {"id": 3, "start": 4.5, "end": 6.0, "text": "してみよっか。"},
        ]
        units = build(segments)
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0]["text"], "んーと、じゃあちょっと、待って")
        self.assertEqual((units[0]["start"], units[0]["end"]), (0.0, 3.0))
        self.assertEqual(units[0]["id"], 0)
        self.assertEqual(units[1]["text"], "一緒にしてみよっか")
        self.assertEqual(units[1]["id"], 2)

    def test_build_translation_units_skips_replacement_only_fragments(self):
        build = self.ns["build_translation_units"]
        units = build([
            {"id": 0, "start": 0.0, "end": 0.2, "text": "\ufffd"},
            {"id": 1, "start": 0.2, "end": 1.0, "text": "うん。"},
        ])
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["text"], "うん")

    def test_build_translation_units_caps_long_runon_without_punctuation(self):
        build = self.ns["build_translation_units"]
        # 10 fragments, 6s each, no sentence-final mark anywhere: the old
        # max_chars fallback bundled these into ~150s cues.
        segments = [
            {"id": i, "start": i * 6.0, "end": i * 6.0 + 6.0, "text": "あああああ"}
            for i in range(10)
        ]
        units = build(segments)
        self.assertTrue(units)
        self.assertTrue(all(u["end"] - u["start"] <= 12.0 + 1e-6 for u in units))

    def test_build_translation_units_breaks_on_vad_gap(self):
        build = self.ns["build_translation_units"]
        units = build([
            {"id": 0, "start": 0.0, "end": 2.0, "text": "こんにちは"},
            {"id": 1, "start": 25.0, "end": 27.0, "text": "おはよう"},
        ])
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0]["id"], 0)
        self.assertEqual(units[1]["id"], 1)
        self.assertLessEqual(units[0]["end"], 2.0 + 1e-6)

    def test_build_translation_units_splits_on_midfragment_sentence_end(self):
        build = self.ns["build_translation_units"]
        units = build([
            {"id": 0, "start": 0.0, "end": 4.0, "text": "はい。それでね"},
            {"id": 1, "start": 4.0, "end": 6.0, "text": "つづき。"},
        ])
        self.assertEqual([u["text"] for u in units], ["はい", "それでねつづき"])
        # First unit ends mid-fragment (interpolated), not at the fragment end.
        self.assertLess(units[0]["end"], 2.0)
        self.assertEqual(units[0]["id"], 0)

    def test_split_display_lines_breaks_long_sentences(self):
        split = self.ns["split_display_lines"]
        long_text = ("This is a fairly long English sentence that should be broken "
                     "into readable subtitle lines.")
        lines = split(long_text, max_len=30)
        for line in lines.split("\n"):
            self.assertLessEqual(len(line), 30)
        self.assertEqual(" ".join(lines.split("\n")), long_text)


class TranscriptionEdgeCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_functions(
            TRANSCRIBE,
            {"resolve_audio_inputs", "join_word_texts", "split_long_segments",
             "merge_api_segments", "prepare_api_chunks",
             "format_timestamp", "collect_quality_notes", "compression_ratio", "redecode_loops",
             "STOCK_HALLUCINATIONS",
             "transcribe_with_batch_backoff",
             "is_cuda_out_of_memory", "find_timing_issues",
             "ease_short_cues", "collapse_repeats", "trim_smeared_words", "widen_crammed_cues"},
            {"CJK_LANGUAGE_CODES": {"ja", "zh"}, "zlib": zlib,
             "MEDIA_EXTENSIONS": {".wav", ".mp3", ".mp4", ".m4a"},
             "free_vram": lambda: None},
        )

    def test_inputs_expand_folders_strip_quotes_and_deduplicate(self):
        resolve = self.ns["resolve_audio_inputs"]
        with tempfile.TemporaryDirectory() as directory:
            for name in ("b.mp3", "a.WAV", ".hidden.wav", "notes.txt", "c,with comma.m4a"):
                Path(directory, name).write_bytes(b"x")
            Path(directory, "sub").mkdir()
            a_path = str(Path(directory, "a.WAV"))
            files = resolve(f'  "{directory}"  \n\n{a_path}\n\'{a_path}\'')
            self.assertEqual([Path(item).name for item in files],
                             ["a.WAV", "b.mp3", "c,with comma.m4a"])

    def test_inputs_reject_missing_blank_and_media_free_folders(self):
        resolve = self.ns["resolve_audio_inputs"]
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "readme.txt").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "No audio/video"):
                resolve(directory)
            with self.assertRaisesRegex(FileNotFoundError, "File not found"):
                resolve(str(Path(directory, "missing.wav")))
        with self.assertRaisesRegex(ValueError, "empty"):
            resolve(" \n  \n")

    def test_cjk_split_keeps_unaligned_digits_and_symbols(self):
        split = self.ns["split_long_segments"]
        segment = {"start": 0, "end": 3, "text": "3時に会う!", "words": [
            {"word": "3"},
            {"start": 0.5, "end": 1.0, "word": "時"},
            {"start": 1.0, "end": 1.5, "word": "に"},
            {"start": 1.5, "end": 2.0, "word": "会"},
            {"start": 2.0, "end": 2.5, "word": "う"},
            {"word": "!"},
        ]}
        result = split([segment], max_chars=2, language_code="ja")
        self.assertEqual("".join(item["text"] for item in result), "3時に会う!")
        self.assertEqual(result[0]["start"], 0.5)
        self.assertEqual(result[-1]["text"], "会う!")  # trailing untimed word merged back
        self.assertEqual(result[-1]["end"], 2.5)

    def _chars(self, text, gaps=None):
        words, clock = [], 0.0
        for index, char in enumerate(text):
            clock += (gaps or {}).get(index, 0.0)
            words.append({"word": char, "start": clock, "end": clock + 0.15})
            clock += 0.15
        return {"start": 0, "end": clock, "text": text, "words": words}

    def test_cjk_split_never_starts_a_line_with_small_kana_or_punctuation(self):
        split = self.ns["split_long_segments"]
        for text in ("お題はいらないわよちゃんと飲め干さないから探していこうかな",
                     "それってすごいことじゃないですかほんとにそう思う?",
                     "あああああああああああああああああああちゃんと"):
            lines = [item["text"] for item in split([self._chars(text)], 20, "ja")]
            self.assertEqual("".join(lines), text)
            self.assertTrue(all(line[0] not in "ゃゅょっー?、。" for line in lines), lines)

    def test_cjk_split_prefers_punctuation_then_word_starts(self):
        split = self.ns["split_long_segments"]
        lines = [item["text"] for item in split(
            [self._chars("ちょっと待って、一緒にやろうか。それでいいならお願いします。")], 20, "ja")]
        self.assertEqual(lines, ["ちょっと待って、一緒にやろうか。", "それでいいならお願いします。"])
        lines = [item["text"] for item in split(
            [self._chars("街か街の泳い込んじゃっててなんか大変だったんだよね")], 20, "ja")]
        self.assertEqual(lines, ["街か街の泳い込んじゃっててなんか", "大変だったんだよね"])

    def test_cjk_split_breaks_at_pauses_and_folds_tiny_tails(self):
        split = self.ns["split_long_segments"]
        result = split([self._chars("よしこんばんは今日は和服です", {2: 24.0})], 20, "ja")
        self.assertEqual([item["text"] for item in result], ["よし", "こんばんは今日は和服です"])
        self.assertLess(result[0]["end"], 1)
        self.assertGreater(result[1]["start"], 24)
        lone = {"start": 5, "end": 5.02, "text": "?", "words": [{"word": "?", "start": 5, "end": 5.02}]}
        result = split([self._chars("そうなの"), lone], 20, "ja")
        self.assertEqual([item["text"] for item in result], ["そうなの?"])
        text = "あいうえおかきくけこさしすせそたちつてとなに"
        self.assertEqual([item["text"] for item in split([self._chars(text)], 20, "ja")], [text])

    def test_cjk_split_never_cuts_inside_a_word(self):
        # From a real ASMR run: drawled words and chunk boundaries cut mid-word.
        split = self.ns["split_long_segments"]
        lines = lambda *segments: [item["text"] for item in split(list(segments), 20, "ja")]
        self.assertEqual(lines(self._chars("んしょ…それでは、今月もありがとうございましたー…")),
                         ["んしょ…それでは、今月もありがとうございましたー…"])
        self.assertEqual(lines(self._chars("お礼は、またバスタートーク枠に移動しますね。来月もよろしくー", {18: 0.66})),
                         ["お礼は、またバスタートーク枠に", "移動しますね。来月もよろしくー"])
        self.assertEqual(lines(self._chars("…って、なんかセクシーなお姉さんがいて…なんもわかんないまま…", {14: 1.4})),
                         ["…って、なんかセクシーなお姉さんがいて…", "なんもわかんないまま…"])
        first = self._chars("大好")
        second = self._chars("き…はぁ…びしょびしょになっちゃう…もう、")
        for word in second["words"]:
            word["start"] += first["end"] + 0.14
            word["end"] += first["end"] + 0.14
        result = split([first, second], 20, "ja")
        self.assertEqual([item["text"] for item in result], ["大好き…", "はぁ…びしょびしょになっちゃう…もう、"])
        self.assertEqual(result[0]["end"], second["words"][1]["end"])
        self.assertEqual(result[1]["start"], second["words"][2]["start"])
        for word in second["words"]:  # after a real pause it stays as said
            word["start"] += 3.0
            word["end"] += 3.0
        self.assertEqual(len(split([first, second], 20, "ja")), 2)
        self.assertEqual(split([first, second], 20, "ja")[0]["text"], "大好")

    def test_cjk_split_without_any_timed_word_is_untouched(self):
        split = self.ns["split_long_segments"]
        segment = {"start": 0, "end": 1, "text": "１２３", "words": [{"word": "１２３"}]}
        self.assertEqual(split([segment], max_chars=1, language_code="zh"), [segment])

    def test_api_merge_handles_null_segments_and_renumbers_ids(self):
        merge = self.ns["merge_api_segments"]
        result = merge([
            {"offset_seconds": 0, "response": {"language": "japanese", "segments": [
                {"id": 0, "start": 0, "end": 1, "text": "a"},
                {"id": 1, "start": 1, "end": None, "text": "b"}]}},
            {"offset_seconds": 30, "response": {"language": None, "segments": None}},
            {"offset_seconds": 60, "response": {"segments": [
                {"id": 0, "start": 0, "end": 2, "text": "c"}]}},
        ])
        self.assertEqual([item["id"] for item in result["segments"]], [0, 1, 2])
        self.assertEqual(result["language"], "japanese")
        self.assertEqual(result["duration"], 62)
        self.assertEqual(merge([])["segments"], [])

    def _prepare_with_fake_ffmpeg(self, duration_text, source_name, max_bytes=50_000):
        calls = []

        class Result:
            def __init__(self, stdout=""):
                self.stdout = stdout

        def fake_run(args, **_kwargs):
            if args[0] == "ffprobe":
                return Result(duration_text)
            calls.append((float(args[args.index("-ss") + 1]),
                          float(args[args.index("-t") + 1])))
            Path(args[-1]).write_bytes(b"x" * int(float(args[args.index("-t") + 1]) * 1000))
            return Result()

        prepare = load_functions(
            TRANSCRIBE, {"prepare_api_chunks"},
            {"subprocess": types.SimpleNamespace(run=fake_run)})["prepare_api_chunks"]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / source_name
            source.write_bytes(b"source")
            chunks = prepare(str(source), directory, max_bytes=max_bytes)
        return chunks, calls

    def test_api_chunks_have_no_float_drift_sliver(self):
        chunks, calls = self._prepare_with_fake_ffmpeg("100.30000000000001\n", "input.aiff")
        self.assertAlmostEqual(sum(length for _start, length in calls), 100.3, places=6)
        self.assertGreater(min(length for _start, length in calls), 1.0)
        self.assertEqual(len(chunks), len(calls))

    def test_api_chunks_reject_unreadable_duration(self):
        for duration in ("N/A\n", "0\n", "nan\n", ""):
            with self.subTest(duration=duration):
                with self.assertRaisesRegex(RuntimeError, "valid duration"):
                    self._prepare_with_fake_ffmpeg(duration, "input.aiff")

    def test_small_supported_files_upload_directly(self):
        for name in ("clip.flac", "clip.OGG", "clip.mp3"):
            with self.subTest(name=name):
                chunks, calls = self._prepare_with_fake_ffmpeg("5\n", name)
                self.assertEqual(calls, [])
                self.assertEqual(chunks[0]["offset_seconds"], 0.0)

    def test_timestamps_clamp_and_round(self):
        fmt = self.ns["format_timestamp"]
        self.assertEqual(fmt(-3), "00:00:00.000")
        self.assertEqual(fmt(3599.9996), "01:00:00.000")
        self.assertEqual(fmt("61.5"), "00:01:01.500")

    def test_odd_batch_sizes_back_off_to_one(self):
        attempted = []

        class FakeModel:
            def transcribe(self, _audio, batch_size, **_kwargs):
                attempted.append(batch_size)
                if batch_size > 1:
                    raise RuntimeError("CUBLAS_STATUS_ALLOC_FAILED out of memory")
                return {"segments": []}

        with mock.patch("builtins.print"):
            _result, used = self.ns["transcribe_with_batch_backoff"](FakeModel(), "a", 3)
        self.assertEqual((attempted, used), ([3, 1], 1))

    def test_quality_notes_catch_loops_without_reported_compression(self):
        notes = self.ns["collect_quality_notes"]([
            {"start": 0, "end": 30, "text": "ごし" * 80, "avg_logprob": -0.3},
            {"start": 30, "end": 40, "text": "よしよしいい子だね、大丈夫だよ", "avg_logprob": -0.3}])
        self.assertEqual([note["start"] for note in notes], [0])
        self.assertIn("repetition", notes[0]["reasons"])

    def test_looping_chunks_are_redecoded_at_a_higher_temperature(self):
        calls = []

        class Whisper:
            def transcribe(self, clip, **options):
                calls.append(options)
                if clip[0] == 4:
                    raise RuntimeError("CUDA failed")
                text = {0: "今日は一緒に寝ようね。おやすみ", 1: "おやすみ" * 30, 2: ""}[clip[0]]
                return iter([types.SimpleNamespace(text=text, avg_logprob=-0.4)] if text else []), None

        audio = [0] * 1000 + [1] * 1000 + [2] * 1000 + [3] * 1000 + [4] * 1000  # 1/16 s each at 16 kHz
        segments = [{"start": 0.0, "end": 0.0625, "text": "今日は" + "寝よう" * 60, "avg_logprob": -0.1},
                    {"start": 0.0625, "end": 0.125, "text": "おやすみ" * 40, "avg_logprob": -0.1},
                    {"start": 0.125, "end": 0.1875, "text": "よしよし" * 30, "avg_logprob": -0.1},
                    {"start": 0.1875, "end": 0.25, "text": "よしよし、いい子だね", "avg_logprob": -0.1},
                    {"start": 0.25, "end": 0.3125, "text": "ちゅっ" * 30, "avg_logprob": -0.1}]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            redone = self.ns["redecode_loops"](Whisper(), segments, audio, language="ja", task="transcribe")
        # Only the four loops are decoded again; one that still loops, comes back empty or fails stays as it was.
        self.assertEqual((redone, len(calls)), (1, 4))
        self.assertEqual([segment["text"] for segment in segments],
                         ["今日は一緒に寝ようね。おやすみ", "おやすみ" * 40, "よしよし" * 30, "よしよし、いい子だね",
                          "ちゅっ" * 30])
        self.assertIn("CUDA failed", output.getvalue())
        self.assertEqual((segments[0]["avg_logprob"], segments[0].get("redecoded")), (-0.4, True))
        self.assertEqual((calls[0]["temperature"], calls[0]["log_prob_threshold"], calls[0]["language"]),
                         ([0.2, 0.4, 0.6], None, "ja"))
        self.assertIn("re-decoded", self.ns["collect_quality_notes"](segments)[0]["reasons"])

    def test_timing_issues_are_flagged(self):
        notes = self.ns["find_timing_issues"]([
            {"start": 5.8, "end": 32.5, "text": "こんばんは"},
            {"start": 40, "end": 48, "text": "ちょっと待って、一緒にやろうか。"},
            {"start": 50, "end": 51, "text": "も"},
            {"start": 52, "end": 52.3, "text": "お姉さんの身体触りたいの?"},
            {"start": 130, "end": 132, "text": "よろしく"}])
        self.assertEqual([(note["start"], note["reasons"].split(" (")[1]) for note in notes], [
            (5.8, "timing may be off)"), (52, "too fast to read)"), (52.3, "missed whispers?)")])

    def test_short_cues_are_held_until_the_next_cue(self):
        segments = self.ns["ease_short_cues"]([
            {"start": 0, "end": 0.1, "text": "ちゅぱぁっ"},
            {"start": 0.5, "end": 0.8, "text": "お姉さんの身体触りたいの?"},
            {"start": 10, "end": 14, "text": "さっきも触ったじゃない"}])
        self.assertEqual([segment["end"] for segment in segments], [0.5, 2.125, 14])

    def test_word_stretched_through_silence_is_trimmed(self):
        # CTC backtracking gives the blank frames after a character to it.
        segments = self.ns["trim_smeared_words"]([
            {"start": 543.9, "end": 566.4, "text": "和服です",
             "words": [{"word": "和", "start": 543.9, "end": 544.1}, {"word": "服", "start": 544.1, "end": 544.3},
                       {"word": "で", "start": 544.3, "end": 544.5}, {"word": "す", "start": 544.5, "end": 566.4}]},
            {"start": 0, "end": 1, "text": "untimed", "words": [{"word": "1"}]}])
        self.assertEqual(segments[0]["words"][-1]["end"], 546.0)
        self.assertEqual(segments[0]["end"], 546.0)
        self.assertEqual(segments[1]["end"], 1)

    def test_crammed_cues_start_earlier_into_free_time(self):
        segments = self.ns["widen_crammed_cues"]([
            {"start": 543.9, "end": 546.0, "text": "和服です"},
            {"start": 566.4, "end": 566.9, "text": "ここは?すりすりすり…ここは?"},  # 15 chars in 0.5 s
            {"start": 567.0, "end": 567.2, "text": "ちゅ"}])
        self.assertEqual([segment["start"] for segment in segments], [543.9, 565.025, 567.0])
        overlapping = self.ns["widen_crammed_cues"]([
            {"start": 0, "end": 2, "text": "あ"}, {"start": 1.9, "end": 2.1, "text": "ここは?すりすり"}])
        self.assertEqual(overlapping[1]["start"], 1.9)  # never moved later

    def test_runaway_repeats_are_collapsed(self):
        segments = self.ns["collapse_repeats"]([
            {"text": "んじゅ" + "る" * 80 + "っ!"}, {"text": "ごしごしごし…こっちは"},
            {"text": "ちゅぱ" * 9}])
        self.assertEqual([segment["text"] for segment in segments],
                         ["んじゅるるるる…っ!", "ごしごしごし…こっちは", "ちゅぱちゅぱちゅぱちゅぱ…"])

    def test_stock_hallucinations_are_flagged(self):
        notes = self.ns["collect_quality_notes"]([
            {"start": 0, "end": 2, "text": "ご視聴ありがとうございました"},
            {"start": 2, "end": 4, "text": "Thank you for watching!"},
            {"start": 4, "end": 6, "text": "ありがとう"},
            {"start": 6, "end": 8, "text": "最後までご視聴いただきありがとうございました。"},
            {"start": 8, "end": 10, "text": "ご覧いただきありがとうございました"},
            {"start": 10, "end": 12, "text": "字幕由Amara.org社区提供"},
            {"start": 12, "end": 14, "text": "聞いてくれてありがとう。おやすみなさい"}])
        self.assertEqual([note["start"] for note in notes], [0, 2, 6, 8, 10])
        self.assertIn("stock phrase", notes[0]["reasons"])

    def test_quality_notes_tolerate_missing_signals(self):
        self.assertEqual(self.ns["collect_quality_notes"](
            [{"start": 0, "end": 1, "text": "x", "avg_logprob": None}]), [])


class FakeWhisperWriter:
    """Mirrors whisperx.utils.ResultWriter naming: basename minus the last extension."""

    def __init__(self, extension, output_dir):
        self.extension, self.output_dir = extension, output_dir

    def __call__(self, result, audio_path, options=None):
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        with open(os.path.join(self.output_dir, f"{stem}.{self.extension}"), "w",
                  encoding="utf-8") as handle:
            if self.extension == "txt":
                handle.write(result["text"] + "\n")
            else:
                for segment in result["segments"]:
                    handle.write(f"{segment['start']}\t{segment['text']}\n")


class OutputEdgeCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_functions(
            OUTPUT,
            {"validate_output_formats", "load_results_checkpoint", "output_name_map",
             "reserve_output_name", "normalize_subtitle", "write_result",
             "clean_repeated_words", "clean_segments"},
            {"ALLOWED_OUTPUT_FORMATS": {"txt", "vtt", "srt", "tsv", "json"},
             "cleanup_repetitions": False,
             "WriteText": lambda directory: FakeWhisperWriter("txt", directory),
             "get_writer": lambda extension, directory: FakeWhisperWriter(extension, directory)},
        )

    def test_format_validation_normalizes_case_and_rejects_empty(self):
        validate = self.ns["validate_output_formats"]
        self.assertEqual(validate(" SRT ,Txt,srt"), ["srt", "txt"])
        for bad in ("", " , ", "raw.json", "srt;txt"):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    validate(bad)

    def test_checkpoint_rejects_wrong_shapes_and_bad_json(self):
        load = self.ns["load_results_checkpoint"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            for payload in ('{"schema_version": 1, "results": []}', "[]", "{not json"):
                with self.subTest(payload=payload):
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaises((ValueError, AttributeError)):
                        load(path)

    def test_dotted_filenames_write_to_the_right_path(self):
        write = self.ns["write_result"]
        result = {"text": "a\nb", "segments": [
            {"start": 0, "end": 1, "text": "two\nlines", "words": [{"word": "x"}]},
            {"start": 1, "end": 2, "text": "tab\there"}]}
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(self.ns, {"output_dir": directory}):
                for extension in ("srt", "txt", "tsv"):
                    path = write(result, extension, "my.song.v2")
                    self.assertEqual(Path(path).name, f"my.song.v2.{extension}")
                    self.assertTrue(Path(path).is_file())
                tsv = Path(directory, "my.song.v2.tsv").read_text(encoding="utf-8")
                self.assertEqual(len(tsv.splitlines()), 2)  # newline flattened, one row each
            self.assertEqual(result["segments"][0]["words"], [{"word": "x"}])  # input untouched
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             ["my.song.v2.srt", "my.song.v2.tsv", "my.song.v2.txt"])

    def test_json_output_serializes_numpy_like_scalars(self):
        class Float32:
            def __float__(self):
                return 0.25

        write = self.ns["write_result"]
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(self.ns, {"output_dir": directory}):
                path = write({"text": "", "segments": [{"score": Float32()}]}, "json", "x.raw")
            self.assertEqual(json.loads(Path(path).read_text(encoding="utf-8"))["segments"][0]["score"], 0.25)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["x.raw.json"])

    def test_subtitle_normalization_is_idempotent_and_handles_bare_cr(self):
        normalize = self.ns["normalize_subtitle"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.vtt"
            path.write_bytes(b"\xef\xbb\xbfone\rtwo\r\nthree\n")
            normalize(path)
            once = path.read_bytes()
            normalize(path)
            self.assertEqual(once, path.read_bytes())
            self.assertEqual(once, b"\xef\xbb\xbfone\r\ntwo\r\nthree\r\n")
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["a.vtt"])

    def test_reserve_skips_names_blocked_by_any_single_format(self):
        reserve = self.ns["reserve_output_name"]
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "clip.raw.json").write_text("old", encoding="utf-8")
            Path(directory, "clip-1.srt").write_text("old", encoding="utf-8")
            self.assertEqual(reserve("clip", ["srt", "raw.json"], directory), "clip-2")
            self.assertFalse(Path(directory, "clip.srt").exists())  # rolled back

    def test_name_map_handles_dotted_and_identical_stems(self):
        names = self.ns["output_name_map"](["/a/x.tar.gz", "/b/x.tar.mp3", "/c/y.wav"])
        self.assertEqual(names["/c/y.wav"], "y")
        self.assertEqual(len(set(names.values())), 3)

    def test_repetition_cleanup_collapses_loops_but_keeps_normal_text(self):
        clean = self.ns["clean_repeated_words"]
        self.assertEqual(clean("no no no no no way"), "no way")
        # Short sounds keep three copies; loops of phrases or punctuation collapse to one.
        self.assertEqual(clean("あはははははは"), "あははは")
        self.assertEqual(clean("ちゅっちゅっちゅっちゅっちゅっ"), "ちゅっちゅっちゅっ")
        self.assertEqual(clean("hahahahaha"), "hahaha")
        self.assertEqual(clean("あ、あ、あ、あ、あ、"), "あ、")
        self.assertEqual(clean("…………"), "…")
        self.assertEqual(clean("ありがとうございます。" * 3), "ありがとうございます。")
        self.assertEqual(clean("I said no, no."), "I said no, no.")
        self.assertEqual(clean(""), "")


class DeepLEdgeCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_functions(
            DEEPL,
            {"normalize_translation_source", "build_translation_units",
             "split_display_lines", "valid_resume_prefix", "deepl_source_code",
             "collapse_translation_repeats", "subtitle_segments"},
            {},
        )

    def texts(self, segments, **options):
        return [unit["text"] for unit in self.ns["build_translation_units"](segments, **options)]

    def test_latin_ellipsis_decimals_and_closers_do_not_make_junk_units(self):
        self.assertEqual(self.texts([
            {"id": 0, "start": 0, "end": 5, "text": "Wait... it costs 3.5 dollars?! Really."}],
            min_seconds=0), ["Wait...", "it costs 3.5 dollars?!", "Really."])
        self.assertEqual(self.texts([
            {"id": 0, "start": 0, "end": 2, "text": "「はい。」そう!?うん"}], min_seconds=0),
            ["「はい。」", "そう!?", "うん"])

    def test_sentences_shorter_than_min_seconds_join_the_next_one(self):
        # A crammed 0.7 s fragment used to become four 0.1 s cues.
        self.assertEqual(self.texts([
            {"id": 0, "start": 0, "end": 0.7, "text": "ここは?すりすり…ここは?"},
            {"id": 1, "start": 0.7, "end": 3, "text": "全部敏感なんだね。どう?"}]),
            ["ここは?すりすり…ここは?全部敏感なんだね", "どう?"])

    def test_bad_timestamps_and_empty_input_are_tolerated(self):
        self.assertEqual(self.texts([]), [])
        self.assertEqual(self.texts([
            {"id": 0, "text": "no times"},
            {"id": 1, "start": "x", "end": 1, "text": "bad"},
            {"id": 2, "start": 5, "end": 4, "text": "逆"},
        ]), ["逆"])

    def test_units_never_exceed_char_cap(self):
        units = self.ns["build_translation_units"](
            [{"id": 0, "start": 0, "end": 1, "text": "あ" * 500}], max_seconds=1e9)
        self.assertTrue(all(len(unit["text"]) <= 120 for unit in units))
        self.assertEqual(sum(len(unit["text"]) for unit in units), 500)

    def test_display_lines_hard_wrap_unspaced_text(self):
        split = self.ns["split_display_lines"]
        lines = split("あ" * 100, max_len=42).split("\n")
        self.assertEqual([len(line) for line in lines], [42, 42, 16])
        self.assertEqual(split("x" * 42, max_len=42), "x" * 42)
        self.assertEqual(split("", max_len=42), "")

    def test_display_lines_are_balanced_without_orphans(self):
        split = self.ns["split_display_lines"]
        self.assertEqual(split("Everyone’s a cat… but there are cats here, too"),
                         "Everyone’s a cat… but\nthere are cats here, too")
        self.assertEqual(split("You don’t realize how serious this is, do you?"),
                         "You don’t realize how\nserious this is, do you?")

    def test_translated_walls_of_repeats_are_shortened(self):
        collapse = self.ns["collapse_translation_repeats"]
        self.assertEqual(collapse("Mmm, slurp-slurp-slurp-slurp-slurp-slurp"), "Mmm, slurp-slurp-slurp…")
        self.assertEqual(collapse("Lurururururururururu"), "Lurururur…")
        self.assertEqual(collapse("あああああああああ好き"), "ああああ…好き")
        self.assertEqual(collapse("No, no, no, no way"), "No, no, no, no way")

    def test_subtitle_cues_have_two_lines_and_stay_readable(self):
        cues = self.ns["subtitle_segments"]([
            {"id": 0, "start": 0, "end": 6, "text": " ".join(f"word{i}" for i in range(25))},
            {"id": 1, "start": 6, "end": 6.1, "text": "Mwah"},
            {"id": 2, "start": 6.3, "end": 6.4, "text": "Haa"}])
        self.assertTrue(all(cue["text"].count("\n") <= 1 for cue in cues))
        self.assertEqual([cue["start"] for cue in cues[:2]], [0, cues[0]["end"]])
        self.assertEqual(cues[1]["end"], 6)
        self.assertEqual([(cue["start"], cue["end"]) for cue in cues[2:]], [(6, 6.3), (6.3, 7.3)])
        cues = self.ns["subtitle_segments"]([{"id": 0, "start": 0, "end": 4, "text":
            "What about here? Rub, rub, rub... What about here? Here? Everything's so sensitive."}])
        self.assertEqual([cue["text"] for cue in cues],
                         ["What about here? Rub, rub,\nrub... What about here? Here?",
                          "Everything's so sensitive."])
        cues = self.ns["subtitle_segments"]([{"id": 0, "start": 0, "end": 4, "text":
            "I missed you so much\nCome here. Let me hold you."}])
        self.assertEqual(" ".join(cue["text"].replace("\n", " ") for cue in cues),
                         "I missed you so much Come here. Let me hold you.")

    def test_resume_prefix_rejects_mismatches(self):
        valid = self.ns["valid_resume_prefix"]
        source = [{"id": 0, "start": 0.0, "end": 1.0}, {"id": 1, "start": 1.0, "end": 2.0}]
        self.assertTrue(valid([], source))
        self.assertTrue(valid(source[:1], source))
        self.assertFalse(valid(source + source, source))
        self.assertFalse(valid([{"id": 0, "start": 0.0, "end": 1.5}], source))
        self.assertFalse(valid("nope", source))

    def test_source_codes_accept_whisper_codes_and_names(self):
        fake_utils = types.ModuleType("whisperx.utils")
        fake_utils.TO_LANGUAGE_CODE = {"japanese": "ja", "norwegian": "no"}
        with mock.patch.dict(sys.modules, {"whisperx": types.ModuleType("whisperx"),
                                           "whisperx.utils": fake_utils}):
            code = self.ns["deepl_source_code"]
            self.assertEqual(code("ja"), "JA")
            self.assertEqual(code("japanese"), "JA")
            self.assertEqual(code("norwegian"), "NB")
            self.assertEqual(code(None), "")


if __name__ == "__main__":
    unittest.main()
