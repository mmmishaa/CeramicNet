import csv
from pathlib import Path
import yaml

from src.paths import CONFIGS_DIR, ROOT_DIR


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def load_retrieval_config(path: str | Path | None = None) -> dict:
    config_path = (
        project_path(path)
        if path
        else CONFIGS_DIR / "retrieval" / "retrieval_config.yaml"
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Конфиг не найден: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Конфиг пустой: {config_path}")
    return config


def resolve_reference(
    books_dir: str | Path | None = None,
    meta_csv: str | Path | None = None,
    config_path: str | Path | None = None,
) -> tuple[Path, Path]:
    config = load_retrieval_config(config_path)
    books_cfg = config.get("books", {})

    target_books = books_dir or books_cfg.get("dir")
    target_meta = meta_csv or books_cfg.get("meta_csv")

    if not target_books:
        raise ValueError("Путь к папке books не указан в аргументах и отсутствует в конфиге")
    if not target_meta:
        raise ValueError("Путь к meta_csv не указан в аргументах и отсутствует в конфиге")

    books = project_path(target_books)
    meta = project_path(target_meta)

    if not books.is_dir():
        raise FileNotFoundError(f"Папка эталона не найдена: {books}")
    if not meta.is_file():
        raise FileNotFoundError(f"Описание эталона не найдено: {meta}")

    return books, meta


def load_book_records(meta_csv: Path, books_dir: Path) -> list[dict]:
    records = []
    with meta_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "mark_id" not in reader.fieldnames:
            raise ValueError(f"В {meta_csv} нет колонки mark_id")
        if "image_path" not in reader.fieldnames:
            raise ValueError(f"В {meta_csv} нет колонки image_path")

        for row in reader:
            mark_id = (row.get("mark_id") or "").strip()
            image_path = (row.get("image_path") or "").replace("\\", "/").strip()
            if mark_id == "" or image_path == "":
                continue

            relative = Path(image_path)
            if relative.is_absolute():
                try:
                    relative = relative.relative_to(books_dir)
                except ValueError:
                    raise ValueError(f"Кадр вне папки эталона: {image_path}") from None

            records.append(
                {
                    "mark_id": mark_id,
                    "book": (row.get("book") or "").strip(),
                    "image_path": relative.as_posix(),
                    "page": (row.get("page") or "").strip(),
                    "catalog_no": (row.get("catalog_no") or "").strip(),
                    "factory": (row.get("factory") or "").strip(),
                }
            )

    if not records:
        raise ValueError(f"В {meta_csv} нет клейм")
    return records