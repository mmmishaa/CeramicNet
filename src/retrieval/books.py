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
) -> tuple[Path, Path]:
    if books_dir:
        books = project_path(books_dir)
        meta = project_path(meta_csv) if meta_csv else books.parent / "meta.csv"
    else:
        books, meta = discover_reference()
        if meta_csv:
            meta = project_path(meta_csv)

    if not books.is_dir():
        raise FileNotFoundError(f"Папка эталона не найдена: {books}")
    if not meta.is_file():
        raise FileNotFoundError(f"Описание эталона не найдено: {meta}")
    return books, meta


def discover_reference() -> tuple[Path, Path]:
    found = []
    for base in (ROOT_DIR, ROOT_DIR.parent):
        if not base.is_dir():
            continue
        candidates = [base, *[path for path in base.iterdir() if path.is_dir()]]
        for folder in candidates:
            books = folder / "artifacts" / "books"
            meta = folder / "artifacts" / "meta.csv"
            if books.is_dir() and meta.is_file():
                found.append((books, meta))

    unique = list(dict.fromkeys(found))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        listed = ", ".join(str(books) for books, _ in unique)
        raise FileNotFoundError(
            f"Найдено несколько эталонов, укажите --books явно: {listed}"
        )
    raise FileNotFoundError(
        "Эталон books не найден. Передайте --books и, если нужно, --meta."
    )


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
