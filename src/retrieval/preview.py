from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.crop import crop_stamp
from src.preprocessing.image import fit_square, read_bgr


def render_preview(result: dict, target_size: int, output_path: Path) -> Path:
    image_bgr = read_bgr(result["image"])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    detections = result["detections"]
    if not detections:
        detections = [{"box": None, "score": None, "matches": []}]

    for idx, detection in enumerate(detections):
        matches = detection.get("matches", [])
        num_cols = 2 + len(matches)

        fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4.5))
        if num_cols == 1:
            axes = [axes]

        photo_display = image_rgb.copy()
        box = detection["box"]
        score = detection["score"]

        if box is not None and score is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(photo_display, (x1, y1), (x2, y2), (255, 0, 0), 3)
            det_title = f"Детекция\nconf: {score:.2f}"
        else:
            det_title = "Фото целиком"

        axes[0].imshow(photo_display)
        axes[0].set_title(det_title, fontsize=11)
        axes[0].axis("off")

        if box is None or score is None:
            crop_bgr = fit_square(image_bgr, target_size)
        else:
            crop_bgr = crop_stamp(image_bgr, box, target_size=target_size)

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        axes[1].imshow(crop_rgb)
        axes[1].set_title(f"Кроп\n{target_size}x{target_size}", fontsize=11)
        axes[1].axis("off")

        for col_idx, match in enumerate(matches, start=2):
            match_path = Path(match["image_path"])
            if match_path.is_file():
                match_bgr = read_bgr(match_path)
                match_rgb = cv2.cvtColor(match_bgr, cv2.COLOR_BGR2RGB)
                axes[col_idx].imshow(match_rgb)
            else:
                axes[col_idx].text(0.5, 0.5, "Нет фото", ha="center", va="center")

            rank = match.get("rank", col_idx - 1)
            score_val = match.get("score", 0.0)
            mark_id = match.get("mark_id", "")
            factory = match.get("factory") or "Не указана"

            title_text = f"Top-{rank} ({score_val:.3f})\n{mark_id}\n{factory}"
            axes[col_idx].set_title(title_text, fontsize=9)
            axes[col_idx].axis("off")

        plt.suptitle(Path(result["image"]).name, fontsize=12)
        plt.tight_layout()

        out_file = Path(output_path)
        if len(detections) > 1:
            out_file = out_file.with_name(f"{out_file.stem}_det{idx}{out_file.suffix}")

        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return Path(output_path)


def open_preview(path: Path) -> None:
    import platform
    import subprocess

    path = Path(path)
    system = platform.system()
    if system == "Darwin":
        command = ["open", str(path)]
    elif system == "Windows":
        command = ["cmd", "/c", "start", "", str(path)]
    else:
        command = ["xdg-open", str(path)]
    subprocess.run(command, check=False)