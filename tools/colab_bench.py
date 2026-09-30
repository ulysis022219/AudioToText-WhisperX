"""Benchmark notebook Steps 3-4 on Colab across a matrix of settings.

Run Steps 1-2 of the notebook first, then paste this into one new cell:

    ref = "main"  # branch, tag or commit to test
    !curl -fsSL https://raw.githubusercontent.com/ulysis022219/AudioToText-WhisperX/{ref}/tools/colab_bench.py -o /tmp/colab_bench.py
    %run /tmp/colab_bench.py --ref {ref} --preset models --seconds 180

It executes the real Step 3 and Step 4 cell code from the chosen ref (not a copy),
with #@param values overridden per configuration. Each input is first cut to a
short excerpt so a full matrix fits in the free T4 quota. Results, SRT/JSON
output, per-run logs and report.md land in MyDrive/audio_transcription/_bench/.
The user's own _last_results.json checkpoint is never touched.

Accuracy (CER) is scored when a reference transcript exists for an excerpt:
<refs>/<excerpt name>.txt, e.g. "rika daten 07312026@60+180.txt" for
--start 60 --seconds 180, or "<file stem>.txt" with --seconds 0.
"""
import argparse
import ast
import contextlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
import unicodedata
import urllib.request
import zlib
from datetime import datetime
from pathlib import Path

REPO = "ulysis022219/AudioToText-WhisperX"
NOTEBOOK_NAME = "AudioToText_WhisperX.ipynb"
STEP3_ID = "opNkn_Lgpat4"
STEP4_ID = "wNsrB45_lCIl"
DRIVE_ROOT = "/content/drive/MyDrive"
MEDIA_EXTENSIONS = {
    ".aac", ".aif", ".aiff", ".amr", ".flac", ".m4a", ".mka", ".mkv", ".mov", ".mp3",
    ".mp4", ".mpeg", ".mpga", ".oga", ".ogg", ".opus", ".wav", ".webm", ".wma", ".wmv",
}

PRESETS = {
    "quick": [("large-v3", {})],
    "models": [
        ("large-v3", {}),
        ("kotoba", {"use_model": "kotoba-whisper-v2.0"}),
        ("anime", {"use_model": "anime-whisper"}),
    ],
    "options": [
        ("large-v3", {}),
        ("quiet", {"quiet_speech": True}),
        ("no-repeat", {"reduce_repetition": True}),
        ("high-acc", {"quality_mode": "High accuracy"}),
    ],
}
PRESETS["all"] = PRESETS["models"] + PRESETS["options"][1:]


def load_notebook(ref):
    """Read the notebook from a local path, or from GitHub at a branch/tag/commit."""
    if os.path.isfile(ref):
        return json.loads(Path(ref).read_text(encoding="utf-8"))
    url = f"https://raw.githubusercontent.com/{REPO}/{ref}/{NOTEBOOK_NAME}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def cell_sources(notebook):
    return {cell.get("metadata", {}).get("id"): "".join(cell.get("source", []))
            for cell in notebook["cells"]}


def set_assignment(source, name, value, require_param=True):
    """Replace the value of a top-level `name = ...` line, keeping any #@param tail."""
    pattern = re.compile(rf"^{re.escape(name)}[ \t]*=[ \t]*[^\n#]*?([ \t]*#@param[^\n]*)?$",
                         re.MULTILINE)
    matches = [match for match in pattern.finditer(source)
               if match.group(1) or not require_param]
    if not matches:
        kind = "#@param" if require_param else "assignment"
        raise KeyError(f"No {kind} named {name!r} in cell")
    match = matches[0]
    line = f"{name} = {value!r}{match.group(1) or ''}"
    return source[:match.start()] + line + source[match.end():]


def apply_overrides(source, params, constants=None):
    for name, value in params.items():
        source = set_assignment(source, name, value)
    for name, value in (constants or {}).items():
        source = set_assignment(source, name, value, require_param=False)
    return source


def parse_config(text):
    """'label:key=value,key=value' -> (label, {key: value}). Values are Python literals or strings."""
    label, _, body = text.partition(":")
    overrides = {}
    for item in filter(None, (part.strip() for part in body.split(","))):
        key, _, raw = item.partition("=")
        try:
            value = ast.literal_eval(raw.strip())
        except (ValueError, SyntaxError):
            value = raw.strip()
        overrides[key.strip()] = value
    return label.strip(), overrides


def excerpt_name(path, start, seconds):
    stem = Path(path).stem
    return stem if seconds <= 0 else f"{stem}@{start:g}+{seconds:g}"


def media_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def make_excerpts(inputs, out_dir, start, seconds):
    """Cut [start, start+seconds) of each input to 16 kHz mono FLAC. seconds<=0 keeps the file."""
    if seconds <= 0:
        return {path: path for path in inputs}
    os.makedirs(out_dir, exist_ok=True)
    excerpts = {}
    for path in inputs:
        destination = os.path.join(out_dir, excerpt_name(path, start, seconds) + ".flac")
        if not os.path.isfile(destination):
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(start),
                            "-t", str(seconds), "-i", path, "-ac", "1", "-ar", "16000",
                            destination], check=True)
        excerpts[path] = destination
    return excerpts


def normalize_for_cer(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(char for char in text
                   if not char.isspace() and unicodedata.category(char)[0] not in "PSC")


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, 1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1]


def character_error_rate(reference, hypothesis):
    reference, hypothesis = normalize_for_cer(reference), normalize_for_cer(hypothesis)
    if not reference:
        return None
    return edit_distance(reference, hypothesis) / len(reference)


NOTE_KINDS = {"low confidence": "lowconf", "possible non-speech": "nonspeech",
              "repetition risk": "repeat", "timing may be off": "stretched"}


def score_result(result, audio_seconds=None, reference=None, max_chars=20):
    segments = result.get("segments", [])
    texts = [str(segment.get("text", "")).strip() for segment in segments]
    joined = "".join(texts)
    notes = {kind: 0 for kind in NOTE_KINDS.values()}
    for note in result.get("quality_notes", []):
        for marker, kind in NOTE_KINDS.items():
            if marker in note.get("reasons", ""):
                notes[kind] += 1
    longest_run = run = 0
    for index, text in enumerate(texts):
        run = run + 1 if index and text and text == texts[index - 1] else 1
        longest_run = max(longest_run, run)
    durations = [float(s.get("end", 0)) - float(s.get("start", 0)) for s in segments]
    overlaps = sum(1 for a, b in zip(segments, segments[1:])
                   if float(b.get("start", 0)) < float(a.get("end", 0)) - 0.01)
    encoded = joined.encode("utf-8")
    metrics = {
        "segments": len(segments),
        "chars": len(normalize_for_cer(joined)),
        "notes": notes,
        "alignment": result.get("alignment_status"),
        "long_lines": sum(1 for text in texts if len(text) > max_chars),
        "long_cues": sum(1 for duration in durations if duration > 7.0),
        "short_cues": sum(1 for duration in durations if duration < 0.3),
        "overlaps": overlaps,
        "same_line_run": longest_run,
        "compression": round(len(encoded) / len(zlib.compress(encoded)), 2) if encoded else None,
        "cer": round(character_error_rate(reference, joined), 4) if reference else None,
    }
    if audio_seconds:
        metrics["chars_per_min"] = round(metrics["chars"] / audio_seconds * 60, 1)
    return metrics


def find_reference(refs_dir, name):
    path = Path(refs_dir) / f"{name}.txt"
    return path.read_text(encoding="utf-8-sig") if path.is_file() else None


def run_config(namespace, step3, step4, audio_paths, run_dir, label, overrides):
    config_dir = os.path.join(run_dir, label)
    os.makedirs(config_dir, exist_ok=True)
    checkpoint = os.path.join(config_dir, "_checkpoint.json")
    step3_source = apply_overrides(step3, {"audio_file": "\n".join(audio_paths), **overrides},
                                   {"CHECKPOINT_PATH": checkpoint})
    step4_source = apply_overrides(step4, {
        "output_dir": config_dir, "output_formats": "srt,json",
        "save_raw_json": False, "overwrite_existing": True,
    }, {"CHECKPOINT_PATH": checkpoint})
    namespace.pop("results", None)
    record = {"label": label, "overrides": overrides, "error": None}
    started = time.time()
    with open(os.path.join(config_dir, "run.log"), "w", encoding="utf-8") as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            exec(compile(step3_source, f"<step3:{label}>", "exec"), namespace)
            record["step3_seconds"] = round(time.time() - started, 1)
            exec(compile(step4_source, f"<step4:{label}>", "exec"), namespace)
        except BaseException as error:  # keep the matrix going; KeyboardInterrupt re-raised below
            traceback.print_exc()
            record["error"] = f"{type(error).__name__}: {error}"
            if isinstance(error, KeyboardInterrupt):
                raise
    record["seconds"] = round(time.time() - started, 1)
    record["results"] = namespace.get("results") or {}
    return record


def format_report(rows, configs):
    header = ("| clip | config | time s | x realtime | segs | chars/min | lowconf | nonspeech "
              "| repeat | stretched | long lines | long cues | same-line run | align | CER |")
    lines = [header, "|" + "---|" * (header.count("|") - 1)]
    for row in rows:
        m = row.get("metrics") or {}
        notes = m.get("notes", {})
        cells = [row["clip"], row["config"],
                 row.get("seconds", ""), row.get("realtime", ""),
                 m.get("segments", ""), m.get("chars_per_min", ""),
                 notes.get("lowconf", ""), notes.get("nonspeech", ""),
                 notes.get("repeat", ""), notes.get("stretched", ""),
                 m.get("long_lines", ""), m.get("long_cues", ""), m.get("same_line_run", ""),
                 m.get("alignment", "") or "",
                 "" if m.get("cer") is None else f"{m['cer']:.1%}"]
        if row.get("error"):
            cells[4] = f"ERROR: {row['error'][:80]}"
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    legend = ["", "Configs: " + "; ".join(
        f"`{label}` = {overrides or 'defaults'}" for label, overrides in configs)]
    return "\n".join(lines + legend)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default="main", help="Git ref or local notebook path to test")
    parser.add_argument("--clips", default=f"{DRIVE_ROOT}/for process",
                        help="Folder (or one file) of audio/video to test")
    parser.add_argument("--refs", default=None, help="Folder of reference .txt transcripts "
                        "(default: <clips>/refs)")
    parser.add_argument("--preset", default="quick", choices=sorted(PRESETS))
    parser.add_argument("--config", action="append", default=[],
                        help="Extra config 'label:key=value,...'; replaces the preset if given")
    parser.add_argument("--start", type=float, default=0.0, help="Excerpt start (s)")
    parser.add_argument("--seconds", type=float, default=180.0,
                        help="Excerpt length (s); 0 = whole file")
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N files")
    parser.add_argument("--out", default=f"{DRIVE_ROOT}/audio_transcription/_bench")
    args = parser.parse_args(argv)

    configs = [parse_config(text) for text in args.config] or PRESETS[args.preset]
    clips_path = Path(args.clips)
    if clips_path.is_dir():
        inputs = sorted(str(path) for path in clips_path.iterdir()
                        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS)
    else:
        inputs = [str(clips_path)]
    if args.limit:
        inputs = inputs[:args.limit]
    if not inputs:
        raise SystemExit(f"No media files in {args.clips}")
    refs_dir = args.refs or str((clips_path if clips_path.is_dir() else clips_path.parent) / "refs")

    notebook = load_notebook(args.ref)
    cells = cell_sources(notebook)
    ref_label = re.sub(r"[^\w.-]+", "_", Path(args.ref).stem if os.path.isfile(args.ref) else args.ref)
    run_dir = os.path.join(args.out, f"{datetime.now():%Y%m%d-%H%M%S}_{ref_label}")
    os.makedirs(run_dir, exist_ok=True)
    excerpts = make_excerpts(inputs, "/content/bench_excerpts", args.start, args.seconds)
    names = {path: excerpt_name(path, args.start, args.seconds) for path in inputs}
    durations = {path: media_duration(excerpts[path]) for path in inputs}
    print(f"🧪 ref={args.ref} | {len(inputs)} clip(s) | {len(configs)} config(s) | out={run_dir}")
    for path in inputs:
        has_ref = "ref ✅" if find_reference(refs_dir, names[path]) else "no ref"
        print(f"  • {names[path]} ({durations[path] or 0:.0f}s, {has_ref})")

    try:
        from IPython import get_ipython
        namespace = get_ipython().user_ns if get_ipython() else {}
    except ImportError:
        namespace = {}
    namespace.setdefault("__name__", "__main__")

    rows = []
    for label, overrides in configs:
        print(f"\n▶ {label} {overrides or ''}", flush=True)
        record = run_config(namespace, cells[STEP3_ID], cells[STEP4_ID],
                            [excerpts[path] for path in inputs], run_dir, label, overrides)
        audio_total = sum(durations.values()) if all(durations.values()) else None
        status = f"❌ {record['error']}" if record["error"] else "✅"
        print(f"  {status} in {record['seconds']}s (log: {run_dir}/{label}/run.log)")
        for path in inputs:
            result = record["results"].get(excerpts[path])
            row = {"clip": names[path], "config": label, "error": record["error"],
                   "seconds": record.get("step3_seconds", record["seconds"])}
            if audio_total and "step3_seconds" in record:
                row["realtime"] = round(audio_total / record["step3_seconds"], 1)
            if result:
                row["metrics"] = score_result(
                    result, durations[path], find_reference(refs_dir, names[path]),
                    max_chars=namespace.get("max_chars_per_line", 20))
                row["error"] = None
            rows.append(row)

    report = format_report(rows, configs)
    Path(run_dir, "report.md").write_text(report + "\n", encoding="utf-8")
    Path(run_dir, "report.json").write_text(json.dumps(
        {"ref": args.ref, "args": vars(args), "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("\n" + report)
    print(f"\n📄 {run_dir}/report.md")
    return rows


if __name__ == "__main__":
    main()
