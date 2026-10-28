from pathlib import Path
import os
import platform
import subprocess
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.preprocessing.crop import crop_stamp
from src.preprocessing.image import read_bgr


def render_preview(result: dict, target_size: int, output_path: Path) -> list[Path]:
    image_bgr = read_bgr(result["image"])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    detections = result.get("detections", [])
    if not detections:
        detections = [{"box": None, "score": None, "matches": []}]

    saved_files: list[Path] = []

    for idx, detection in enumerate(detections):
        matches = detection.get("matches", [])
        n_matches = len(matches)

        matches_per_row = max(4, min(5, n_matches))
        match_rows = (n_matches + matches_per_row - 1) // matches_per_row if n_matches > 0 else 1
        total_rows = 1 + match_rows

        fig = plt.figure(figsize=(3.5 * matches_per_row, 3.8 * total_rows))
        gs = GridSpec(total_rows, matches_per_row, figure=fig)

        split = matches_per_row // 2
        ax_det = fig.add_subplot(gs[0, :split])
        ax_crop = fig.add_subplot(gs[0, split: 2 * split])

        photo_display = image_rgb.copy()
        box = detection.get("box")
        score = detection.get("score")

        if box is not None and score is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(photo_display, (x1, y1), (x2, y2), (255, 0, 0), 3)
            det_title = f"Детекция (conf: {score:.2f})"
        else:
            det_title = "Фото целиком"

        ax_det.imshow(photo_display)
        ax_det.set_title(det_title, fontsize=11, weight="bold")
        ax_det.axis("off")

        if box is None or score is None:
            h_img, w_img = image_bgr.shape[:2]
            crop_bgr = crop_stamp(
                image_bgr,
                (0.0, 0.0, float(w_img), float(h_img)),
                padding_ratio=0.0,
                target_size=target_size,
            )
        else:
            crop_bgr = crop_stamp(image_bgr, box, target_size=target_size)

        ax_crop.imshow(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        ax_crop.set_title(f"Кроп {target_size}x{target_size}", fontsize=11, weight="bold")
        ax_crop.axis("off")

        for m_idx, match in enumerate(matches):
            row_idx = 1 + (m_idx // matches_per_row)
            col_idx = m_idx % matches_per_row
            ax_m = fig.add_subplot(gs[row_idx, col_idx])

            raw_path = match.get("image_path", "")
            match_path = Path(raw_path)
            if not match_path.is_file():
                match_path = Path("data/books") / raw_path

            if match_path.is_file():
                ax_m.imshow(cv2.cvtColor(read_bgr(match_path), cv2.COLOR_BGR2RGB))
            else:
                ax_m.text(0.5, 0.5, "Нет фото", ha="center", va="center")

            rank = match.get("rank", m_idx + 1)
            score_val = match.get("score", 0.0)
            mark_id = match.get("mark_id", "")
            factory = match.get("factory") or "Не указана"

            ax_m.set_title(f"Top-{rank} ({score_val:.3f})\n{mark_id}\n{factory}", fontsize=9)
            ax_m.axis("off")

        plt.suptitle(Path(result["image"]).name, fontsize=13, y=0.99)
        plt.tight_layout()

        out_file = Path(output_path)
        if len(detections) > 1:
            out_file = out_file.with_name(f"{out_file.stem}_det{idx}{out_file.suffix}")

        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(out_file)

    return saved_files


def open_preview(path: Path) -> None:
    path = Path(path).resolve()
    if not path.is_file():
        return

    system = platform.system()
    if system == "Windows":
        os.startfile(str(path))
    elif system == "Darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])