import json
from pathlib import Path

import numpy as np


def require_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or isinstance(top_k, float):
        raise ValueError(
            f"top_k должен быть целым числом >= 1, получено: {top_k!r}"
        )
    if isinstance(top_k, np.integer):
        top_k = int(top_k)
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError(
            f"top_k должен быть целым числом >= 1, получено: {top_k!r}"
        )
    return top_k


class StampIndex:

    def __init__(self, embeddings: np.ndarray, records: list[dict]):
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(
                f"Эмбеддинги должны иметь форму (N, D), получено: {embeddings.shape}"
            )
        if len(records) != embeddings.shape[0]:
            raise ValueError(
                "Число записей эталона не совпадает с числом эмбеддингов: "
                f"{len(records)} и {embeddings.shape[0]}"
            )
        if embeddings.shape[0] == 0:
            raise ValueError("Индекс пуст")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms < 1e-12):
            raise ValueError("В индексе есть нулевой вектор")

        self.embeddings = embeddings / norms
        self.records = list(records)

    def __len__(self) -> int:
        return int(self.embeddings.shape[0])

    def search(
        self,
        vector: np.ndarray,
        top_k: int,
        exclude_mark_ids: set[str] | None = None,
        exclude_image_paths: set[str] | None = None,
    ) -> list[dict]:
        top_k = require_top_k(top_k)
        query = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                "Размерность запроса не совпадает с индексом: "
                f"{query.shape[0]} и {self.embeddings.shape[1]}"
            )

        norm = float(np.linalg.norm(query))
        if norm < 1e-12:
            raise ValueError("Вектор запроса нулевой")
        query = query / norm

        scores = self.embeddings @ query
        excluded_marks = {str(mark_id) for mark_id in (exclude_mark_ids or set())}
        excluded_paths = {str(path) for path in (exclude_image_paths or set())}
        if excluded_marks or excluded_paths:
            scores = scores.copy()
            for index, record in enumerate(self.records):
                image_path = str(record["image_path"])
                if str(record["mark_id"]) in excluded_marks or image_path in excluded_paths:
                    scores[index] = -np.inf

        best: dict[str, tuple[float, dict]] = {}
        for index, record in enumerate(self.records):
            score = float(scores[index])
            if not np.isfinite(score):
                continue
            mark_id = str(record["mark_id"])
            current = best.get(mark_id)
            if current is None or score > current[0]:
                best[mark_id] = (score, record)

        ranked = sorted(best.values(), key=lambda item: item[0], reverse=True)
        hits = []
        for rank, (score, record) in enumerate(ranked[:top_k], start=1):
            hits.append(
                {
                    "rank": rank,
                    "mark_id": str(record["mark_id"]),
                    "book": str(record.get("book", "")),
                    "score": score,
                    "image_path": str(record["image_path"]),
                    "page": str(record.get("page", "")),
                    "catalog_no": str(record.get("catalog_no", "")),
                    "factory": str(record.get("factory", "")),
                }
            )
        return hits

    def save(self, path: str | Path, books_dir: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.records, ensure_ascii=False)
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            records=np.array(payload),
            books_dir=np.array(str(books_dir)),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        books_dir: str | Path | None = None,
    ) -> "StampIndex":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Индекс не найден: {path}")

        data = np.load(path, allow_pickle=False)
        records = json.loads(data["records"].item())
        stored_books = str(data["books_dir"].item()) if "books_dir" in data.files else ""
        base = _books_base(books_dir, stored_books)

        resolved = []
        for record in records:
            item = dict(record)
            item["image_path"] = _resolve_image(base, item["image_path"])
            resolved.append(item)

        return cls(embeddings=data["embeddings"], records=resolved)


def _books_base(books_dir: str | Path | None, stored_books: str) -> Path | None:
    if books_dir:
        return Path(books_dir)
    if stored_books and Path(stored_books).is_dir():
        return Path(stored_books)
    return None


def _resolve_image(base: Path | None, image_path: str) -> str:
    path = Path(image_path)
    if path.is_absolute() or base is None:
        return str(path)
    return str(base / path)
