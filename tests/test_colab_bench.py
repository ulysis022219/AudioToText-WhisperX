import ast
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("colab_bench", ROOT / "tools" / "colab_bench.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)
CELLS = bench.cell_sources(bench.load_notebook(str(ROOT / bench.NOTEBOOK_NAME)))


class ColabBenchTests(unittest.TestCase):
    def test_every_preset_override_matches_a_notebook_param(self):
        step3 = CELLS[bench.STEP3_ID]
        for preset, configs in bench.PRESETS.items():
            for label, overrides in configs:
                with self.subTest(preset=preset, config=label):
                    source = bench.apply_overrides(
                        step3, {"audio_file": "/a.wav\n/b.wav", **overrides},
                        {"CHECKPOINT_PATH": "/tmp/bench.json"})
                    ast.parse(source)
                    self.assertIn("audio_file = '/a.wav\\n/b.wav' #@param", source)
                    self.assertIn("CHECKPOINT_PATH = '/tmp/bench.json'", source)
                    for name, value in overrides.items():
                        self.assertIn(f"{name} = {value!r} #@param", source)

    def test_step4_overrides_match_notebook_params(self):
        source = bench.apply_overrides(CELLS[bench.STEP4_ID], {
            "output_dir": "/tmp/out", "output_formats": "srt,json",
            "save_raw_json": False, "overwrite_existing": True,
        }, {"CHECKPOINT_PATH": "/tmp/bench.json"})
        ast.parse(source)
        self.assertIn("output_formats = 'srt,json' #@param", source)
        self.assertNotIn("_last_results.json", source)

    def test_unknown_param_is_rejected(self):
        with self.assertRaises(KeyError):
            bench.set_assignment(CELLS[bench.STEP3_ID], "no_such_param", 1)
        with self.assertRaises(KeyError):  # plain assignment, not a form field
            bench.set_assignment(CELLS[bench.STEP3_ID], "CHECKPOINT_PATH", "/x")

    def test_parse_config(self):
        self.assertEqual(bench.parse_config("q:quiet_speech=True,use_model=anime-whisper,batch_size=4"),
                         ("q", {"quiet_speech": True, "use_model": "anime-whisper", "batch_size": 4}))

    def test_cer_ignores_punctuation_width_and_spaces(self):
        self.assertEqual(bench.character_error_rate("こんにちは、世界！", "こんにちは 世界"), 0)
        self.assertAlmostEqual(bench.character_error_rate("ＡＢＣＤ", "abxd"), 0.25)
        self.assertIsNone(bench.character_error_rate("。", "x"))

    def test_score_result_counts_notes_runs_and_cue_problems(self):
        result = {
            "alignment_status": "complete",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "はい"},
                {"start": 1.0, "end": 2.0, "text": "はい"},
                {"start": 1.5, "end": 12.0, "text": "あ" * 25},
                {"start": 12.0, "end": 12.1, "text": "え"},
            ],
            "quality_notes": [
                {"reasons": "low confidence (-1.20); repetition risk (3.10)"},
                {"reasons": "1 chars over 9s (timing may be off)"},
            ],
        }
        metrics = bench.score_result(result, audio_seconds=60, reference="はいはい")
        self.assertEqual(metrics["notes"], {"lowconf": 1, "nonspeech": 0, "repeat": 1, "stretched": 1})
        self.assertEqual(metrics["same_line_run"], 2)
        self.assertEqual((metrics["long_lines"], metrics["long_cues"], metrics["short_cues"]), (1, 1, 1))
        self.assertEqual(metrics["overlaps"], 1)
        self.assertEqual(metrics["chars"], 30)
        self.assertEqual(metrics["chars_per_min"], 30.0)
        self.assertGreater(metrics["cer"], 1)

    def test_excerpt_name(self):
        self.assertEqual(bench.excerpt_name("/d/rika daten.mp3", 60, 180), "rika daten@60+180")
        self.assertEqual(bench.excerpt_name("/d/rika daten.mp3", 0, 0), "rika daten")


if __name__ == "__main__":
    unittest.main()
