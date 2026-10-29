import argparse
import inspect
import shutil
import zipfile
from pathlib import Path

import gdown

from src.paths import ROOT_DIR, DATA_DIR, WEIGHTS_DIR, CONFIGS_DIR
from src.utils import load_yaml

CATEGORIES = ("detection", "retrieval", "books", "dataset")
DEFAULT_CATEGORIES = ("detection", "retrieval", "books")

def get_active_priority() -> dict:
    priority = {
        "detection": ["rtdetr_v2_s.onnx", "yolo26s_v1.onnx", "yolo26s.onnx", "rtmdet.onnx"],
        "retrieval": ["student_encoder_katya.onnx", "stamp_encoder_v2.onnx", "stamp_encoder.onnx"],
    }
    try:
        det_cfg = load_yaml(CONFIGS_DIR / "detection" / "detection_config.yaml")
        if "weights" in det_cfg:
            w_name = Path(det_cfg["weights"]).name
            if w_name in priority["detection"]:
                priority["detection"].remove(w_name)
            priority["detection"].insert(0, w_name)
    except Exception:
        pass
    try:
        ret_cfg = load_yaml(CONFIGS_DIR / "retrieval" / "retrieval_config.yaml")
        act = ret_cfg.get("active_model", "student_encoder_katya")
        if "models" in ret_cfg and act in ret_cfg["models"]:
            w_name = Path(ret_cfg["models"][act]["encoder"]["weights"]).name
            if w_name in priority["retrieval"]:
                priority["retrieval"].remove(w_name)
            priority["retrieval"].insert(0, w_name)
    except Exception:
        pass
    return priority

PRIORITY_MODELS = get_active_priority()

CANONICAL_SOURCES = {
    "detection": "rtdetr_v2_s.onnx",
    "retrieval": "student_encoder_katya.onnx",
}

CANONICAL_TARGETS = {
    "detection": "rtdetr_v2_s.onnx",
    "retrieval": "student_encoder_katya.onnx",
}


def call_supported(func, **kwargs):
    """Call func, passing only the kwargs supported by the installed gdown version."""
    params = inspect.signature(func).parameters
    return func(**{k: v for k, v in kwargs.items() if k in params})


def copy_weight(src: Path, target_dir: Path, category: str = None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Always copy file under its downloaded filename
    dst = target_dir / src.name
    shutil.copy2(src, dst)
    print(f"Added: {src.name} -> {target_dir.name}")

    # 2. If this is the chosen best model, also save/alias as canonical name
    if category:
        source_name = CANONICAL_SOURCES.get(category)
        canonical_name = CANONICAL_TARGETS.get(category)
        if canonical_name and (src.name == source_name or src.name == canonical_name):
            dst_canonical = target_dir / canonical_name
            shutil.copy2(src, dst_canonical)
            if src.name != canonical_name:
                print(f"Created canonical alias: {canonical_name} (from {src.name}) -> {target_dir.name}")


def extract_archive(zip_file: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_file.name} to {extract_to}")
    try:
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(extract_to)
        print(f"Extracted {zip_file.name}")
    except Exception as e:
        print(f"Extraction error for {zip_file.name}: {e}")


def extract_smart(zip_file: Path, prefix: str, target: Path) -> None:
    """If the archive already contains the top-level folder, extract into DATA_DIR."""
    with zipfile.ZipFile(zip_file, "r") as zf:
        names = zf.namelist()
        first_entry = names[0] if names else ""
    destination = DATA_DIR if first_entry.startswith(prefix) else target
    extract_archive(zip_file, destination)


def has_dir(rel_path: str, word: str) -> bool:
    return any(word in part.lower() for part in Path(rel_path).parts[:-1])


def categorize(rel_path: str, has_detection_dir: bool) -> set:
    name = Path(rel_path).name.lower()
    cats = set()
    if name.endswith(".onnx"):
        in_det = has_dir(rel_path, "detection")
        in_ret = has_dir(rel_path, "retrieval")
        if in_det or (not has_detection_dir and not in_ret):
            cats.add("detection")
        if in_ret:
            cats.add("retrieval")
    if name == "books.zip":
        cats.add("books")
    if name == "ceramic_dataset.zip":
        cats.add("dataset")
    return cats


def list_folder(url: str):
    files = call_supported(
        gdown.download_folder,
        url=url,
        skip_download=True,
        quiet=True,
        remaining_ok=True,
        use_cookies=False,
    )
    if not files:
        raise SystemExit("Could not get the file list. Check the folder id and access.")
    return sorted(files, key=lambda f: f.path.lower())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CeramicNet artifact synchronization. By default downloads only runtime essentials."
    )
    parser.add_argument("--folder_id", type=str, required=True, help="Google Drive shared folder ID")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Download all artifacts: full training dataset (ceramic_dataset.zip) and all experimental weights",
    )
    parser.add_argument("--list", action="store_true", help="Only show files in the folder and exit")

    group = parser.add_argument_group("Selective download (combine freely)")
    group.add_argument("--detection", action="store_true", help="Detection model weights (*.onnx)")
    group.add_argument("--retrieval", action="store_true", help="Retrieval model weights (*.onnx)")
    group.add_argument("--books", action="store_true", help="Books catalog archive (books.zip)")
    group.add_argument("--dataset", action="store_true", help="Training dataset (ceramic_dataset.zip)")
    args = parser.parse_args()

    user_selected = [c for c in CATEGORIES if getattr(args, c)]
    if user_selected:
        selected = user_selected
    elif args.dev:
        selected = list(CATEGORIES)
    else:
        # Default mode: runtime essentials only (single active models, no heavy dataset)
        selected = list(DEFAULT_CATEGORIES)

    url = f"https://drive.google.com/drive/folders/{args.folder_id}"
    print(f"Reading folder: {args.folder_id}")
    files = list_folder(url)

    has_detection_dir = any(
        has_dir(f.path, "detection") for f in files if f.path.lower().endswith(".onnx")
    )
    plan = {c: [] for c in CATEGORIES}
    for f in files:
        for c in categorize(f.path, has_detection_dir):
            plan[c].append(f)
    for c in ("books", "dataset"):
        plan[c] = plan[c][:1]

    # In default mode (non-dev, non-selective): download ONLY the single active model for each category
    if not args.dev and not user_selected:
        for cat in ("detection", "retrieval"):
            candidates = plan[cat]
            chosen = None
            for preferred in PRIORITY_MODELS[cat]:
                for f in candidates:
                    if Path(f.path).name.lower() == preferred.lower():
                        chosen = f
                        break
                if chosen:
                    break
            if chosen:
                plan[cat] = [chosen]
            elif candidates:
                plan[cat] = [candidates[0]]

    if args.list:
        print(f"\nFiles in folder: {len(files)}")
        for f in files:
            tags = ", ".join(sorted(categorize(f.path, has_detection_dir))) or "-"
            print(f"  {f.path}    [{tags}]")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    temp_dir = ROOT_DIR / "_temp_sync"
    detection_target = WEIGHTS_DIR / "detection_weights"
    retrieval_target = WEIGHTS_DIR / "retrieval_weights"
    dataset_target = DATA_DIR / "ceramic_dataset"
    books_target = DATA_DIR / "books"

    books_exists = books_target.exists() and any(books_target.iterdir())
    dataset_exists = dataset_target.exists() and any(dataset_target.iterdir())

    mode_label = "all dev artifacts" if args.dev else ("selective" if user_selected else "runtime essentials")
    print(f"Mode: {mode_label} | Categories: {', '.join(selected)}")

    to_download = {}
    for c in selected:
        if not plan[c]:
            print(f"Not found in folder: {c}")
            continue
        if c == "books" and books_exists:
            print("Books already exist, skipping download and extraction")
            continue
        if c == "dataset" and dataset_exists:
            print("Ceramic dataset already exists, skipping download and extraction")
            continue
        for f in plan[c]:
            to_download[f.id] = f

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    for f in to_download.values():
        dest = temp_dir / f.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading: {f.path}")
        try:
            call_supported(gdown.download, id=f.id, output=str(dest), quiet=False, use_cookies=False)
        except Exception as e:
            print(f"Download error for {f.path}: {e}")

    def local(f) -> Path | None:
        p = temp_dir / f.path
        return p if p.is_file() else None

    if "detection" in selected:
        for f in plan["detection"]:
            if (p := local(f)):
                copy_weight(p, detection_target, category="detection")

    if "retrieval" in selected:
        for f in plan["retrieval"]:
            if (p := local(f)):
                copy_weight(p, retrieval_target, category="retrieval")

    if "books" in selected and not books_exists:
        for f in plan["books"]:
            if (p := local(f)):
                extract_smart(p, "books/", books_target)

    if "dataset" in selected and not dataset_exists:
        for f in plan["dataset"]:
            if (p := local(f)):
                extract_smart(p, "ceramic_dataset/", dataset_target)

    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Synchronization finished")


if __name__ == "__main__":
    main()
