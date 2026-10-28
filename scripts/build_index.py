import argparse

import numpy as np
from tqdm import tqdm

from src.preprocessing.image import fit_square, read_bgr
from src.retrieval.books import (
    load_book_records,
    load_retrieval_config,
    project_path,
    resolve_reference,
)
from src.retrieval.encoder import StampEncoder
from src.retrieval.index import StampIndex


def parse_args():
    parser = argparse.ArgumentParser(
        description="Посчитать эмбеддинги эталона books и сохранить индекс"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Путь к configs/retrieval/retrieval_config.yaml",
    )
    parser.add_argument("--books", type=str, default=None, help="Папка books")
    parser.add_argument("--meta", type=str, default=None, help="CSV с mark_id и image_path")
    parser.add_argument("--encoder", type=str, default=None, help="ONNX энкодера")
    parser.add_argument("--output", type=str, default=None, help="Куда сохранить индекс .npz")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Взять только первые N клейм. Для проверки, что энкодер запускается.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size должен быть >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit должен быть >= 1")

    config = load_retrieval_config(args.config)
    books_cfg = config.get("books") or {}
    books_dir, meta_csv = resolve_reference(
        args.books or books_cfg.get("dir"),
        args.meta or books_cfg.get("meta_csv"),
    )

    records = load_book_records(meta_csv, books_dir)
    if args.limit is not None:
        records = records[: args.limit]

    missing = [
        record["image_path"]
        for record in records
        if not (books_dir / record["image_path"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"В эталоне не найдены {len(missing)} изображений. Пример: {missing[0]}"
        )

    encoder_cfg = dict(config["encoder"])
    if args.encoder:
        encoder_cfg["weights"] = str(project_path(args.encoder))
    else:
        encoder_cfg["weights"] = str(project_path(encoder_cfg["weights"]))
    encoder = StampEncoder.from_mapping(encoder_cfg)

    output = project_path(args.output or config["index"]["path"])
    vectors = []
    batch = []

    progress = tqdm(records, desc="Эмбеддинги books", unit="mark")
    for record in progress:
        image = read_bgr(books_dir / record["image_path"])
        batch.append(fit_square(image, encoder.input_size))
        if len(batch) == args.batch_size:
            vectors.append(encoder.embed_many(batch))
            batch = []
    if batch:
        vectors.append(encoder.embed_many(batch))

    embeddings = np.concatenate(vectors, axis=0)
    index = StampIndex(embeddings, records)
    index.save(output, books_dir)

    print(f"Клейм в индексе: {len(index)}")
    print(f"Размерность вектора: {embeddings.shape[1]}")
    print(f"Эталон: {books_dir}")
    print(f"Индекс: {output}")


if __name__ == "__main__":
    main()
