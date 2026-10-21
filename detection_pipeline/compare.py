"""Compare metrics produced by the shared evaluator on exactly the same split."""
import argparse
import csv
from pathlib import Path

from .data import read_json


def compare(paths, output):
    results = [read_json(path) for path in paths]
    if not results or len({r["annotations_sha256"] for r in results}) != 1:
        raise ValueError("Compare results on the same annotation file only")
    rows = []
    for result in results:
        tuned = result.get("threshold_search_on_validation", {})
        selected = tuned.get("selected") or {}
        point = result["operating_point"]
        rows.append({"model": result["model"], "augmentation": result["config"].get("augmentation"),
                     **{k: result["coco"][k] for k in ("map_50_95", "ap50", "ap75")},
                     "precision_at_fixed_threshold": point["precision"],
                     "recall_at_fixed_threshold": point["recall"], "fixed_threshold": point["threshold"],
                     "target_precision_on_val": tuned.get("min_precision"),
                     "recall_at_target_precision_on_val": selected.get("recall"),
                     "selected_threshold_on_val": selected.get("threshold")})
    rows.sort(key=lambda r: r["map_50_95"] if r["map_50_95"] is not None else -1, reverse=True)
    with Path(output).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for row in compare(args.metrics, args.output):
        print(row)
