from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

CONFIGS_DIR = ROOT_DIR / "configs"
SCRIPTS_DIR = ROOT_DIR / "scripts"
REPORTS_DIR = ROOT_DIR / "reports"
WEIGHTS_DIR = ROOT_DIR / "weights"

DATA_DIR = ROOT_DIR / "data"
CERAMIC_DATA_DIR = DATA_DIR / "ceramic_dataset"
BOOKS_DIR = DATA_DIR / "books"