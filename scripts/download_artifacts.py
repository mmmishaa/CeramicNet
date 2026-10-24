import argparse
from pathlib import Path
import shutil
import zipfile
import gdown

from src.paths import ROOT_DIR, DATA_DIR, WEIGHTS_DIR

def main():
    parser = argparse.ArgumentParser(description="artifact synchronization")
    parser.add_argument("--folder_id", type=str, required=True)
    args = parser.parse_args()

    temp_dir = ROOT_DIR / "_temp_sync"
    
    detection_weights_target = WEIGHTS_DIR / "detection_weights"
    detection_weights_target.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset_exists = DATA_DIR.exists() and any(DATA_DIR.iterdir())

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading folder: {args.folder_id}")
    url = f"https://drive.google.com/drive/folders/{args.folder_id}"
    
    try:
        gdown.download_folder(url=url, output=str(temp_dir), quiet=False, use_cookies=False)
    except Exception as e:
        print(f"Download error: {e}")

    print("Synchronizing ONNX weights")
    for f in temp_dir.rglob("*.onnx"):
        if f.is_file():
            dst = detection_weights_target / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                print(f"Added ONNX weight: {f.name}")

    if not dataset_exists:
        zip_path = next(temp_dir.rglob("*.zip"), None)
        if zip_path:
            print(f"Extracting dataset from {zip_path.name}")
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(ROOT_DIR)
                print("Dataset extracted")
            except Exception as e:
                print(f"Extraction error: {e}")
        else:
            print("Dataset zip archive not found in downloaded artifacts")
    else:
        print("Dataset already exists, skipping extraction")

    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Synchronization finished")

if __name__ == "__main__":
    main()