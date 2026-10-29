import argparse
import csv
import importlib
import json
from pathlib import Path
import unicodedata
from tqdm import tqdm

from src.detection.detector import StampDetector
from src.paths import CERAMIC_DATA_DIR, CONFIGS_DIR, REPORTS_DIR
from src.pipeline import CeramicPipeline
from src.preprocessing.crop import crop_stamp
from src.preprocessing.image import read_bgr
from src.retrieval.books import load_book_records, project_path, resolve_reference
from src.retrieval.encoder import StampEncoder
from src.retrieval.index import StampIndex
from src.utils import load_yaml
import numpy as np


def build_index_for_model(
    encoder: StampEncoder,
    books_dir: Path,
    meta_csv: Path,
    output_path: Path,
    batch_size: int = 16,
) -> StampIndex:
    records = load_book_records(meta_csv, books_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vectors = []
    batch = []
    for record in tqdm(records, desc="  Building index", unit="mark"):
        img = read_bgr(books_dir / record["image_path"])
        h, w = img.shape[:2]
        prepared = crop_stamp(
            image=img,
            box=(0.0, 0.0, float(w), float(h)),
            padding_ratio=0.0,
            target_size=encoder.input_size,
        )
        batch.append(prepared)
        if len(batch) == batch_size:
            vectors.append(encoder.embed_many(batch))
            batch = []
    if batch:
        vectors.append(encoder.embed_many(batch))

    embeddings = np.concatenate(vectors, axis=0)
    index = StampIndex(embeddings, records)
    index.save(output_path, books_dir)
    print(f"  ✓ Auto-built index: {output_path} ({len(index)} marks)")
    return index


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for stamp retrieval benchmark")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=CERAMIC_DATA_DIR / "annotations" / "test_retrieval_annotations.json",
        help="Path to retrieval annotations JSON",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "retrieval" / "retrieval_config.yaml",
        help="Retrieval configuration YAML",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model name to test. If omitted, tests all models from config.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K cutoff (default: 10)")
    parser.add_argument("--no-detect", action="store_true", help="Disable stamp detector (pure crop retrieval)")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild index even if .npz file already exists on disk",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPORTS_DIR / "retrieval_test_results.csv",
        help="Destination path for detailed evaluation CSV",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=REPORTS_DIR / "retrieval_summary.csv",
        help="Destination path for summary metrics CSV",
    )
    return parser.parse_args()


def resolve_image_path(images_dir: Path, raw_filename: str) -> Path | None:
    norm_name = unicodedata.normalize("NFC", raw_filename).strip()
    candidate = images_dir / norm_name
    if candidate.is_file():
        return candidate
    candidate_def = images_dir / "default" / norm_name
    if candidate_def.is_file():
        return candidate_def

    # Fallback: поиск по совпадению нормализованных имен файлов
    for child in images_dir.iterdir():
        if child.is_file() and unicodedata.normalize("NFC", child.name) == norm_name:
            return child
    return None


def evaluate_single_model(pipeline, records, images_dir, top_k: int, no_detect: bool):
    results = []
    for item in records:
        raw_query = str(item["query_file"])
        image_path = resolve_image_path(images_dir, raw_query)

        gt_filenames = {
            unicodedata.normalize("NFC", Path(m["filename"]).name)
            for m in item.get("reference_matches", [])
        }

        if image_path is None:
            print(f"[Warning] Image not found: {raw_query}")
            continue

        pred = pipeline.predict(
            image_path,
            top_k=top_k,
            detect=not no_detect,
            fallback_to_full_image=False,
        )
        detections = pred.get("detections", [])

        best_rank = None
        top1_filename = None
        top1_score = None

        if detections:
            best_det = sorted(detections, key=lambda d: d.get("score") or 0.0, reverse=True)[0]
            matches = best_det.get("matches", [])

            if matches:
                top1_filename = Path(matches[0]["image_path"]).name
                top1_score = matches[0]["score"]

            for m in matches:
                hit_name = unicodedata.normalize("NFC", Path(m["image_path"]).name)
                if hit_name in gt_filenames:
                    best_rank = m["rank"]
                    break

        results.append({
            "query_file": image_path.name,
            "detected": 1 if detections else 0,
            "gt_matches": "; ".join(sorted(gt_filenames)),
            "best_rank": best_rank if best_rank is not None else -1,
            "top1_pred": top1_filename or "",
            "top1_score": round(top1_score, 4) if top1_score is not None else "",
            "hit_top1": int(best_rank == 1),
            "hit_top3": int(best_rank is not None and best_rank <= 3),
            "hit_top5": int(best_rank is not None and best_rank <= 5),
            "hit_top10": int(best_rank is not None and best_rank <= 10),
            "mrr": round(1.0 / best_rank, 4) if best_rank is not None else 0.0,
            "note": item.get("note") or "",
        })
    return results


def print_summary_table(summary_rows):
    headers = ["Model", "Total", "Detected", "No_Det", "Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR"]
    col_widths = {h: len(h) for h in headers}
    for r in summary_rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(r.get(h, ""))))

    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    separator = "-+-".join("-" * col_widths[h] for h in headers)
    print("\nSummary Table:")
    print(header_line)
    print(separator)
    for r in summary_rows:
        print(" | ".join(str(r.get(h, "")).ljust(col_widths[h]) for h in headers))


def main():
    args = parse_args()

    if not args.annotations.is_file():
        raise FileNotFoundError(f"Annotations file not found: {args.annotations}")

    with open(args.annotations, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    if not records:
        print("No test records found.")
        return

    raw_config = load_yaml(args.config)

    # Determine list of models to evaluate
    if args.model:
        if "models" in raw_config and args.model not in raw_config["models"]:
            raise KeyError(f"Model '{args.model}' not found in config. Available: {list(raw_config['models'].keys())}")
        model_names = [args.model]
    elif "models" in raw_config:
        model_names = list(raw_config["models"].keys())
    else:
        model_names = [raw_config.get("active_model", "default")]

    print(f"Total test queries: {len(records)}")
    print(f"Models to evaluate: {', '.join(model_names)}")

    books_path, books_meta_csv = resolve_reference(
        books_dir=raw_config.get("books", {}).get("dir"),
        meta_csv=raw_config.get("books", {}).get("meta_csv"),
    )

    # Init shared detector once
    detector = None
    if not args.no_detect:
        det_cfg = load_yaml(CONFIGS_DIR / "detection" / "detection_config.yaml")
        module_path, class_name = det_cfg["processor"].rsplit(".", 1)
        module = importlib.import_module(module_path)
        processor = getattr(module, class_name)(
            imgsz=det_cfg.get("imgsz", 640),
            conf_threshold=det_cfg.get("conf_threshold", 0.7),
            iou_threshold=det_cfg.get("iou_threshold", 0.5),
        )
        detector = StampDetector(model_path=det_cfg["weights"], processor=processor)

    images_dir = CERAMIC_DATA_DIR / "images"
    all_detailed_results = []
    summary_rows = []

    for name in model_names:
        print(f"\nRunning model: {name}")

        if "models" in raw_config:
            model_cfg = raw_config["models"][name]
            encoder_dict = model_cfg["encoder"]
            index_dict = model_cfg["index"]
        else:
            encoder_dict = raw_config["encoder"]
            index_dict = raw_config["index"]

        weights_file = project_path(encoder_dict["weights"])
        index_file = project_path(index_dict["path"])

        if not weights_file.is_file():
            print(f"[Skip] Weights not found: {weights_file}")
            continue

        print(f"  Weights: {weights_file}")
        encoder = StampEncoder.from_mapping(encoder_dict, weights=weights_file)

        if args.rebuild_index or not index_file.is_file():
            if args.rebuild_index and index_file.is_file():
                print(f"  [Rebuild] Rebuilding index (--rebuild-index passed): {index_file}")
            else:
                print(f"  [Index missing] Index not found at: {index_file}")
            print(f"  --> Automatically building index for '{name}'...")
            index = build_index_for_model(encoder, books_path, books_meta_csv, index_file)
        else:
            print(f"  Index:   {index_file}")
            index = StampIndex.load(index_file, books_dir=books_path)

        pipeline = CeramicPipeline(
            encoder=encoder,
            index=index,
            detector=detector,
            top_k=args.top_k,
        )

        with tqdm(total=len(records), desc=f"  Evaluating", unit="query") as pbar:
            results = evaluate_single_model(
                pipeline=pipeline,
                records=records,
                images_dir=images_dir,
                top_k=args.top_k,
                no_detect=args.no_detect,
            )
            pbar.update(len(records))

        total = len(results)
        if total == 0:
            continue

        n_detected = sum(r["detected"] for r in results)
        n_missed_det = total - n_detected
        n_hit1 = sum(r["hit_top1"] for r in results)
        n_hit3 = sum(r["hit_top3"] for r in results)
        n_hit5 = sum(r["hit_top5"] for r in results)
        n_hit10 = sum(r["hit_top10"] for r in results)
        mean_mrr = sum(r["mrr"] for r in results) / total

        print(f"  Detected: {n_detected}/{total} ({n_missed_det} missed by detector)")
        print(f"  Hit@1:    {n_hit1 / total:.4f} ({n_hit1}/{total})")
        print(f"  Hit@3:    {n_hit3 / total:.4f} ({n_hit3}/{total})")
        print(f"  Hit@5:    {n_hit5 / total:.4f} ({n_hit5}/{total})")
        print(f"  Hit@10:   {n_hit10 / total:.4f} ({n_hit10}/{total})")
        print(f"  MRR:      {mean_mrr:.4f}")

        summary_rows.append({
            "Model": name,
            "Total": str(total),
            "Detected": f"{n_detected}/{total}",
            "No_Det": str(n_missed_det),
            "Hit@1": f"{n_hit1 / total:.4f}",
            "Hit@3": f"{n_hit3 / total:.4f}",
            "Hit@5": f"{n_hit5 / total:.4f}",
            "Hit@10": f"{n_hit10 / total:.4f}",
            "MRR": f"{mean_mrr:.4f}",
        })

        for r in results:
            r_copy = dict(r)
            r_copy["model"] = name
            all_detailed_results.append(r_copy)

    if summary_rows:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_headers = ["Model", "Total", "Detected", "No_Det", "Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR"]
        existing_summary = {}
        if args.summary_csv.is_file():
            try:
                with open(args.summary_csv, "r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if "Model" in row:
                            existing_summary[row["Model"]] = row
            except Exception:
                existing_summary = {}

        for row in summary_rows:
            existing_summary[row["Model"]] = row

        all_summary_rows = list(existing_summary.values())
        print_summary_table(all_summary_rows)

        with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_headers)
            writer.writeheader()
            writer.writerows(all_summary_rows)
        print(f"\nSummary saved: {args.summary_csv}")

    if all_detailed_results:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["model"] + [k for k in all_detailed_results[0].keys() if k != "model"]
        new_models = {r["model"] for r in all_detailed_results}
        existing_detailed = []
        if args.output_csv.is_file():
            try:
                with open(args.output_csv, "r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if row.get("model") not in new_models:
                            existing_detailed.append(row)
            except Exception:
                existing_detailed = []

        combined_detailed = existing_detailed + all_detailed_results
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_detailed)
        print(f"Detailed results saved: {args.output_csv}")


if __name__ == "__main__":
    main()
