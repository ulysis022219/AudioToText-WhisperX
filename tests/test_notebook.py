import ast
import copy
import json
import os
import sys
import tempfile
import types
import unittest
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
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in names]
    namespace = {
        "Path": Path,
        "copy": copy,
        "hashlib": __import__("hashlib"),
        "json": json,
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
            '"numpy": "2.5.2"', '"ctranslate2": "4.8.1"',
            '"torchcodec": "0.7.0+cu128"',
        ):
            self.assertIn(requirement, INSTALL)
        for removed in ("cohere", "tensorflow-probability", "requests==", "ffmpeg-python", "pydub"):
            self.assertNotIn(removed, INSTALL)
        self.assertIn('["ffmpeg", "-version"]', INSTALL)
        self.assertIn("verify_environment()", INSTALL)

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
        self.assertIn('quality["best_of"]', TRANSCRIBE)
        self.assertIn('"condition_on_previous_text": context_conditioning', TRANSCRIBE)
        self.assertIn("context_conditioning", TRANSCRIBE)
        self.assertIn("effective_prompt", TRANSCRIBE)
        self.assertIn("prompt_names", TRANSCRIBE)
        self.assertIn("prompt_terms", TRANSCRIBE)
        self.assertIn("prompt_acronyms", TRANSCRIBE)
        self.assertIn("subject_area", TRANSCRIBE)
        self.assertIn("collect_quality_notes", TRANSCRIBE)
        self.assertIn("quality_notes", TRANSCRIBE)
        self.assertIn("Review", TRANSCRIBE)
        self.assertNotIn("asmr_terms =", TRANSCRIBE)
        self.assertNotIn("asmr_acronyms =", TRANSCRIBE)
        self.assertIn("manual terminology hints only", TRANSCRIBE)
        self.assertLess(
            TRANSCRIBE.index('raw_quality_notes = collect_quality_notes(raw_asr_segments)'),
            TRANSCRIBE.index('result = whisperx.align'))
        self.assertLess(
            TRANSCRIBE.index('no_speech_ranges = find_no_speech_ranges(raw_asr_segments)',
                             TRANSCRIBE.index('raw_quality_notes =')),
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
             "collect_quality_notes", "find_no_speech_ranges", "mark_non_speech",
             "is_cuda_out_of_memory", "transcribe_with_batch_backoff"},
            {"CJK_LANGUAGE_CODES": {"ja", "zh"}, "NO_SPEECH_MARK_THRESHOLD": 0.6,
             "free_vram": lambda: None},
        )

    def test_no_speech_ranges_use_shared_threshold(self):
        ranges = self.ns["find_no_speech_ranges"]([
            {"start": 0, "end": 1, "no_speech_prob": 0.6},
            {"start": 1, "end": 2, "no_speech_prob": 0.61},
            {"start": 2, "end": 3},
        ])
        self.assertEqual(ranges, [(1.0, 2.0)])

    def test_aligned_non_speech_uses_raw_asr_ranges(self):
        segments = [
            {"start": 0, "end": 2, "text": "real speech", "words": [{"word": "real"}]},
            {"start": 10, "end": 12, "text": "ASMR,R18,KU100", "words": [{"word": "ASMR"}]},
            {"start": 12, "end": 14, "text": "boundary speech", "words": [{"word": "boundary"}]},
        ]
        marked = self.ns["mark_non_speech"](segments, [(9.5, 12.5)])
        self.assertEqual(marked[0]["text"], "real speech")
        self.assertEqual(marked[1]["text"], "[ASMR sounds]")
        self.assertEqual(marked[1]["words"], [{"word": "ASMR"}])
        self.assertEqual(marked[2]["text"], "boundary speech")

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
            fake_pyannote = types.SimpleNamespace(
                __file__=str(module_file),
                Pyannote=lambda *_args, **_kwargs: events.append(
                    os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD")) or "vad")
            fake_vads = types.ModuleType("whisperx.vads")
            fake_vads.pyannote = fake_pyannote
            fake_whisperx = types.SimpleNamespace(
                load_model=lambda *_args, **kwargs: kwargs["vad_model"])
            fake_torch = types.SimpleNamespace(device=lambda value: value)
            namespace = load_functions(
                TRANSCRIBE, {"load_whisperx_with_verified_vad"}, {
                    "WHISPERX_VAD_SHA256": __import__("hashlib").sha256(b"trusted").hexdigest(),
                    "verify_file_sha256": self.ns["verify_file_sha256"],
                    "whisperx": fake_whisperx,
                    "torch": fake_torch,
                })
            with mock.patch.dict(sys.modules, {"whisperx.vads": fake_vads}):
                with mock.patch.dict(os.environ, {"TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1"}):
                    self.assertEqual(namespace["load_whisperx_with_verified_vad"]("large-v3", "cuda"), "vad")
                    self.assertEqual(os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"], "1")
            self.assertEqual(events, [None])

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
            source = Path(directory) / "input.flac"
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
        self.assertEqual([segment["text"] for segment in cleaned], ["おいしぃ", "次の台詞"])
        self.assertEqual(segments[1]["text"], "ぃ" * 20)

    def test_single_character_speech_is_not_treated_as_stretching(self):
        segments = [
            {"start": 0, "end": 1, "text": "はい"},
            {"start": 1, "end": 2, "text": "い"},
        ]
        self.assertEqual(self.ns["clean_segments"](segments), segments)

    def test_raw_json_is_written_before_words_are_removed(self):
        json_branch = OUTPUT.index('if output_format == "json":')
        remove_words = OUTPUT.index('segment.pop("words", None)')
        self.assertLess(json_branch, remove_words)
        self.assertIn("save_raw_json = True", OUTPUT)
        self.assertIn("cleanup_repetitions = False", OUTPUT)


class DeepLTests(unittest.TestCase):
    def test_translation_uses_secrets_and_validated_batch_resume(self):
        self.assertIn('userdata.get("DEEPL_API_KEY")', DEEPL)
        self.assertNotIn('os.environ.get("DEEPL_API_KEY")', DEEPL)
        self.assertIn("text=texts", DEEPL)
        self.assertIn("context=context", DEEPL)
        self.assertIn("len(responses) != len(batch)", DEEPL)
        self.assertIn("valid_resume_prefix", DEEPL)
        self.assertIn("source_signature", DEEPL)
        self.assertIn("atomic_deepl_checkpoint", DEEPL)
        self.assertIn('"translation_status": "in_progress"', DEEPL)
        self.assertIn('translated["translation_status"] = "complete"', DEEPL)

    def test_translation_outputs_are_marked_english(self):
        self.assertIn('result["source_language"] = result.get("language")', TRANSCRIBE)
        self.assertIn('result["language"] = "en"', TRANSCRIBE)


if __name__ == "__main__":
    unittest.main()
