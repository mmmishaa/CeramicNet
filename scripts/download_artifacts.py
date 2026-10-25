import argparse
from pathlib import Path
import shutil
import zipfile
import gdown

from src.paths import ROOT_DIR, DATA_DIR, WEIGHTS_DIR


def copy_weights_by_pattern(source_root: Path, target_dir: Path, pattern: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in source_root.rglob(pattern):
        if path.is_file():
            dst = target_dir / path.name
            if not dst.exists():
                shutil.copy2(path, dst)
                print(f"Added weight: {path.name} -> {target_dir.name}")


def extract_archive(zip_file: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_file.name} to {extract_to}")
    try:
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(extract_to)
        print(f"Extracted {zip_file.name}")
    except Exception as e:
        print(f"Extraction error for {zip_file.name}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="artifact synchronization")
    parser.add_argument("--folder_id", type=str, required=True)
    args = parser.parse_args()

    temp_dir = ROOT_DIR / "_temp_sync"
    detection_target = WEIGHTS_DIR / "detection_weights"
    retrieval_target = WEIGHTS_DIR / "retrieval_weights"
    dataset_target = DATA_DIR / "ceramic_dataset"
    books_target = DATA_DIR / "books"
    index_target = DATA_DIR / "books_index"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    index_target.mkdir(parents=True, exist_ok=True)

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading folder: {args.folder_id}")
    url = f"https://drive.google.com/drive/folders/{args.folder_id}"

    try:
        gdown.download_folder(url=url, output=str(temp_dir), quiet=False, use_cookies=False)
    except Exception as e:
        print(f"Download error: {e}")

    det_source = next((p for p in temp_dir.rglob("*") if p.is_dir() and "detection" in p.name.lower()), None)
    if det_source:
        copy_weights_by_pattern(det_source, detection_target, "*.onnx")
    else:
        copy_weights_by_pattern(temp_dir, detection_target, "*.onnx")

    ret_source = next((p for p in temp_dir.rglob("*") if p.is_dir() and "retrieval" in p.name.lower()), None)
    if ret_source:
        copy_weights_by_pattern(ret_source, retrieval_target, "*.onnx")

    for npz_file in temp_dir.rglob("*.npz"):
        if npz_file.is_file():
            dst = index_target / npz_file.name
            shutil.copy2(npz_file, dst)
            print(f"Added index: {npz_file.name} -> {index_target.name}")

    meta_file = next(temp_dir.rglob("meta.csv"), None)
    if meta_file and meta_file.is_file():
        shutil.copy2(meta_file, DATA_DIR / "meta.csv")
        print("meta.csv copied to data directory")

    books_zip = next(temp_dir.rglob("books.zip"), None)
    books_exists = books_target.exists() and any(books_target.iterdir())
    if books_zip and not books_exists:
        with zipfile.ZipFile(books_zip, "r") as zf:
            first_entry = zf.namelist()[0] if zf.namelist() else ""
            destination = DATA_DIR if first_entry.startswith("books/") else books_target
        extract_archive(books_zip, destination)
    elif books_exists:
        print("Books already exist, skipping extraction")

    dataset_zip = next(temp_dir.rglob("ceramic_dataset.zip"), None)
    dataset_exists = dataset_target.exists() and any(dataset_target.iterdir())
    if dataset_zip and not dataset_exists:
        with zipfile.ZipFile(dataset_zip, "r") as zf:
            first_entry = zf.namelist()[0] if zf.namelist() else ""
            destination = DATA_DIR if first_entry.startswith("ceramic_dataset/") else dataset_target
        extract_archive(dataset_zip, destination)
    elif dataset_exists:
        print("Ceramic dataset already exists, skipping extraction")

    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Synchronization finished")


if __name__ == "__main__":
    main()