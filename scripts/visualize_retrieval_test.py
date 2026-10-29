import argparse
import json
from pathlib import Path
import unicodedata
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from src.paths import CERAMIC_DATA_DIR, DATA_DIR, REPORTS_DIR, ROOT_DIR
from src.preprocessing.image import read_bgr
from src.retrieval.preview import open_preview

def parse_args():
    parser = argparse.ArgumentParser(description="Просмотр экспертной разметки: Фото запроса vs Эталоны человека")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=CERAMIC_DATA_DIR / "annotations" / "test_retrieval_annotations.json",
        help="Путь к JSON с разметкой человека",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR / "ground_truth_previews",
        help="Папка для сохранения карточек разметки",
    )
    parser.add_argument("--query", type=str, default=None, help="Посмотреть конкретный файл (например, БП12-Г02-3659_5.png)")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить количество (например, 5)")
    parser.add_argument("--show", action="store_true", help="Автоматически открыть первую созданную карточку")
    return parser.parse_args()


def resolve_image(images_dir: Path, filename: str) -> Path | None:
    norm_name = unicodedata.normalize("NFC", filename).strip()
    candidate = images_dir / norm_name
    if candidate.is_file():
        return candidate
    candidate_def = images_dir / "default" / norm_name
    if candidate_def.is_file():
        return candidate_def
    for child in images_dir.iterdir():
        if child.is_file() and unicodedata.normalize("NFC", child.name) == norm_name:
            return child
    return None


def resolve_ref(rel_path: str) -> Path | None:
    norm = unicodedata.normalize("NFC", rel_path).replace("\\", "/")
    candidates = [
        DATA_DIR / norm,
        DATA_DIR / "books" / norm,
        ROOT_DIR / norm,
        DATA_DIR / "books" / Path(norm).name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def render_card(query_path: Path, matches: list[dict], note: str | None, output_path: Path):
    query_rgb = cv2.cvtColor(read_bgr(query_path), cv2.COLOR_BGR2RGB)

    n_refs = max(1, len(matches))
    total_cols = 1 + n_refs

    fig_w = max(10.0, total_cols * 3.5)
    fig = plt.figure(figsize=(fig_w, 4.2))
    gs = GridSpec(1, total_cols, figure=fig, width_ratios=[2.5] + [1.5] * n_refs)

    ax_q = fig.add_subplot(gs[0, 0])
    ax_q.imshow(query_rgb)
    ax_q.set_title(f"Фото: {query_path.name}", fontsize=10, weight="bold")
    ax_q.axis("off")

    for i, m in enumerate(matches):
        ax_r = fig.add_subplot(gs[0, 1 + i])
        ref_path = resolve_ref(m.get("path") or m.get("filename"))
        if ref_path and ref_path.is_file():
            ref_rgb = cv2.cvtColor(read_bgr(ref_path), cv2.COLOR_BGR2RGB)
            ax_r.imshow(ref_rgb)
        else:
            ax_r.text(0.5, 0.5, "Файл не найден", ha="center", va="center")

        m_name = Path(m.get("filename", "")).name
        ax_r.set_title(f"Эталон #{i+1} (Человек)\n{m_name}", fontsize=9, color="blue", weight="bold")
        ax_r.axis("off")

    note_str = f" | Заметка эксперта: {note}" if note else ""
    plt.suptitle(f"Разметка человека ({len(matches)} эталона){note_str}", fontsize=11, y=0.98)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    with open(args.annotations, "r", encoding="utf-8") as f:
        records = json.load(f).get("records", [])

    images_dir = CERAMIC_DATA_DIR / "images"

    if args.query:
        norm_q = unicodedata.normalize("NFC", args.query).strip()
        records = [r for r in records if unicodedata.normalize("NFC", r["query_file"]).strip() == norm_q]

    if args.limit:
        records = records[: args.limit]

    print(f"Формирование карточек разметки ({len(records)} шт.)...")
    saved = []

    for item in tqdm(records, unit="card"):
        q_path = resolve_image(images_dir, item["query_file"])
        if not q_path:
            continue

        out_file = args.output_dir / f"{q_path.stem}_gt.png"
        render_card(q_path, item.get("reference_matches", []), item.get("note"), out_file)
        saved.append(out_file)

    print(f"Готово. Карточки сохранены в: {args.output_dir}")
    if args.show and saved:
        open_preview(saved[0])


if __name__ == "__main__":
    main()