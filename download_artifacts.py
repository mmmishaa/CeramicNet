import argparse
from pathlib import Path
import shutil
import zipfile
import gdown

def main():
    parser = argparse.ArgumentParser(description="artifact synchronization")
    parser.add_argument("--folder_id", type=str, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    temp_dir = repo_root / "_temp_sync"
    dataset_target = repo_root / "ceramic_dataset"
    weights_target = repo_root / "train_pipelines" / "detection_models" / "detection_model_weights"
    
    weights_target.mkdir(parents=True, exist_ok=True)
    dataset_exists = dataset_target.exists() and any(dataset_target.iterdir())

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading folder: {args.folder_id}")
    url = f"https://drive.google.com/drive/folders/{args.folder_id}"
    
    try:
        gdown.download_folder(url=url, output=str(temp_dir), quiet=False, use_cookies=False)
    except Exception as e:
        print(f"Download error: {e}")

    print("Synchronizing weights")
    weight_exts = {'.pt', '.pth', '.onnx', '.engine'}
    for f in temp_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in weight_exts:
            dst = weights_target / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                print(f"Added: {f.name}")

    if not dataset_exists:
        zip_path = next(temp_dir.rglob("ceramic_dataset.zip"), None)
        if zip_path:
            print("Extracting dataset")
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(repo_root)
                print("Dataset extracted")
            except Exception as e:
                print(f"Extraction error: {e}")
    else:
        print("Dataset already exists, skipping extraction")

    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Synchronization finished")

if __name__ == "__main__":
    main()