"""Replay Task 1C in frame order. Frame numbers in all artifacts start at zero."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np


def load_detector(source):
    spec = importlib.util.spec_from_file_location("nv_lane_detector", source)
    detector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detector)
    return detector


def draw(detector, frame, result):
    canvas = detector.draw_overlay(frame, result)
    # Draw fitted features only when this implementation exposes its actual geometry.
    helpers = ("_lane_colour_masks", "_find_dashes", "_fit_divider", "_collect_edge_samples")
    if all(hasattr(detector, name) for name in helpers):
        white, yellow, dark = detector._lane_colour_masks(frame)
        dashes = detector._find_dashes(white, dark)
        divider = detector._fit_divider(dashes)
        if divider and dashes:
            for y in range(int(min(d[3].min() for d in dashes)), int(max(d[3].max() for d in dashes)) + 1, 3):
                x = round(divider.x_at(y))
                if 0 <= x < canvas.shape[1]:
                    cv2.circle(canvas, (x, y), 2, (255, 180, 0), -1)
        for points in detector._collect_edge_samples(yellow, divider):
            pixels = [(round(x), round(y)) for y, x in sorted(points)]
            if len(pixels) >= 2:
                cv2.polylines(canvas, [np.array(pixels, np.int32)], False, (255, 0, 255), 2)
    return canvas


def reference(path):
    if path is None:
        return {}
    with path.open(newline="") as f:
        rows = {}
        for row in csv.DictReader(f):
            key = (row["clip"], int(row["frame"]))
            if key in rows:
                raise ValueError(f"Duplicate reference row: {key}")
            rows[key] = (row["lane"], int(row["center_x"]))
        return rows


def replay(source, clip, out, writer, labels, baseline, args):
    detector = load_detector(source)  # reset temporal memory for each independent clip
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise OSError(f"Cannot open {clip}")
    previous = None
    first = {}
    counts = {}
    index = 0
    matched = set()
    def image(frame_number, value):
        frame, result = value
        name = f"{clip.stem}-{frame_number:04d}.jpg"
        if not cv2.imwrite(str(out / name), draw(detector, frame, result)):
            raise OSError(f"Failed to save {name}")
        return name

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        expected = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = detector.validate_result(detector.detect_lane(frame.copy()), index)
            x, lane = result["center_x"], result["lane"]
            key = (clip.name, index)
            flags = []
            if x == -1 or lane == "unknown":
                flags.append("missing_detection")
            if x < -1 or x >= frame.shape[1]:
                flags.append("out_of_bounds")
            if previous:
                prev = previous[1]
                if prev["center_x"] >= 0 and x >= 0 and abs(x - prev["center_x"]) > args.jump_px:
                    flags.append("center_jump")
                if lane != prev["lane"] and "unknown" not in (lane, prev["lane"]):
                    flags.append("lane_switch")
            if key in labels:
                matched.add(key)
                label, center = labels[key]
                if lane != label or (x == -1) != (center == -1) or abs(x - center) > args.label_tolerance:
                    flags.append("label_mismatch")
            if baseline:
                if key not in baseline:
                    raise ValueError(f"Baseline missing {key}")
                if baseline[key] != (lane, x):
                    flags.append("baseline_change")
            writer.writerow((clip.name, index, round(index/fps, 4) if fps > 0 else "", lane, x, ",".join(flags)))
            for flag in flags:
                counts[flag] = counts.get(flag, 0) + 1
            new = [flag for flag in flags if flag not in first]
            if index == 0 or new or any(item["frame"] == index-1 for item in first.values()):
                filename = image(index, (frame, result))
                if new and previous:
                    image(index-1, previous)
                for flag in new:
                    first[flag] = dict(frame=index, time_s=index/fps if fps > 0 else None, image=filename)
            previous = (frame, result)
            index += 1
    finally:
        cap.release()
    if not index or (expected > 0 and index != expected):
        raise ValueError(f"{clip.name}: decoded {index} frames, metadata reports {expected}")
    if {k for k in labels if k[0] == clip.name} != matched:
        raise ValueError(f"{clip.name}: reference label outside decoded frames")
    if baseline and len([k for k in baseline if k[0] == clip.name]) != index:
        raise ValueError(f"{clip.name}: baseline length differs")
    return dict(frames=index, fps=fps, sha256=hashlib.sha256(clip.read_bytes()).hexdigest(),
                counts=counts, first=first, labels_checked=len(matched))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="*", type=Path)
    parser.add_argument("--labels", type=Path, help="Reviewed CSV: clip,frame,lane,center_x")
    parser.add_argument("--baseline", type=Path, help="Earlier replay's predictions.csv")
    parser.add_argument("--label-tolerance", type=int, default=20, metavar="PIXELS")
    parser.add_argument("--jump-px", type=int, default=64, help="Review jumps above this size")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    source = root / "task1c/lane_detection.py"
    clips = args.clips or sorted((source.parent / "public").glob("*.mp4"))
    if not clips or len(set(c.name for c in clips)) != len(clips):
        parser.error("Choose one or more clips with distinct filenames")
    labels, baseline = reference(args.labels), reference(args.baseline)
    if args.baseline and not baseline:
        parser.error("Empty baseline")
    if args.labels and not any(k[0] in {c.name for c in clips} for k in labels):
        parser.error("No reviewed labels for the chosen clips")
    runs = root / "vehicle-test-runs/vision"
    runs.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(dir=runs, prefix="replay-"))
    print(f"Replay directory: {out}", flush=True)
    summary = {}
    with (out / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("clip", "frame", "time_s", "lane", "center_x", "flags"))
        for clip in clips:
            summary[clip.name] = replay(source, clip, out, writer, labels, baseline, args)
    report = dict(detector_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  jump_review_px=args.jump_px, label_tolerance_px=args.label_tolerance, clips=summary)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{sum(c['frames'] for c in summary.values())} frames across {len(clips)} clips")
    return int(any("label_mismatch" in c["first"] or "out_of_bounds" in c["first"] for c in summary.values()))


if __name__ == "__main__":
    raise SystemExit(main())
