import argparse
import inspect
import shutil
import zipfile
from pathlib import Path

import gdown

from src.paths import ROOT_DIR, DATA_DIR, WEIGHTS_DIR

CATEGORIES = ("detection", "retrieval", "index", "books", "dataset")

def call_supported(func, **kwargs):
    """Call func, passing only the kwargs supported by the installed gdown version."""
    params = inspect.signature(func).parameters
    return func(**{k: v for k, v in kwargs.items() if k in params})


def copy_weight(src: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    dst = target_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        print(f"Added weight: {src.name} -> {target_dir.name}")


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
    if name.endswith(".npz"):
        cats.add("index")
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
        description="artifact synchronization (no category flags = download everything)"
    )
    parser.add_argument("--folder_id", type=str, required=True)
    parser.add_argument("--list", action="store_true", help="only show files in the folder and exit")

    group = parser.add_argument_group("what to download (combine freely)")
    group.add_argument("--detection", action="store_true", help="detection weights (*.onnx)")
    group.add_argument("--retrieval", action="store_true", help="retrieval weights (*.onnx)")
    group.add_argument("--index", action="store_true", help="books index (*.npz)")
    group.add_argument("--books", action="store_true", help="books.zip")
    group.add_argument("--dataset", action="store_true", help="ceramic_dataset.zip")
    args = parser.parse_args()

    selected = [c for c in CATEGORIES if getattr(args, c)] or list(CATEGORIES)

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
    index_target = DATA_DIR / "books_index"
    index_target.mkdir(parents=True, exist_ok=True)

    books_exists = books_target.exists() and any(books_target.iterdir())
    dataset_exists = dataset_target.exists() and any(dataset_target.iterdir())

    print(f"Selected: {', '.join(selected)}")

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
                copy_weight(p, detection_target)

    if "retrieval" in selected:
        for f in plan["retrieval"]:
            if (p := local(f)):
                copy_weight(p, retrieval_target)

    if "index" in selected:
        for f in plan["index"]:
            if (p := local(f)):
                shutil.copy2(p, index_target / p.name)
                print(f"Added index: {p.name} -> {index_target.name}")

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