import platform
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.preprocessing.crop import crop_stamp
from src.preprocessing.image import fit_square, read_bgr

PHOTO_W = 420
SLOT_H = 360
CROP_SIZE = 280
MATCH_W = 210
PAD = 18
GAP = 16
CAPTION_H = 96


def render_preview(result: dict, target_size: int, output_path: Path) -> Path:
    image = read_bgr(result["image"])
    rows = []
    if not result["detections"]:
        rows.append(_row(image, None, target_size))
    else:
        for detection in result["detections"]:
            rows.append(_row(image, detection, target_size))

    header = _header(result)
    width = max(header.width, *(row.width for row in rows))
    height = PAD + header.height + sum(row.height + GAP for row in rows) + PAD
    canvas = Image.new("RGB", (width, height), (244, 244, 244))
    canvas.paste(header, (PAD, PAD))

    top = PAD + header.height
    for row in rows:
        left = PAD
        canvas.paste(row, (left, top))
        top += row.height + GAP

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def open_preview(path: Path) -> None:
    path = Path(path)
    system = platform.system()
    if system == "Darwin":
        command = ["open", str(path)]
    elif system == "Windows":
        command = ["cmd", "/c", "start", "", str(path)]
    else:
        command = ["xdg-open", str(path)]
    subprocess.run(command, check=False)


def _row(image: np.ndarray, detection: dict | None, target_size: int) -> Image.Image:
    photo = _photo_panel(image, detection)
    if detection is None:
        crop = None
        matches = []
    else:
        crop = _crop_panel(image, detection, target_size)
        matches = [_match_panel(hit) for hit in detection["matches"]]

    panels = [photo]
    if crop is not None:
        panels.append(crop)
    panels.extend(matches)

    width = PAD + sum(panel.width + GAP for panel in panels)
    height = max(panel.height for panel in panels) + PAD
    sheet = Image.new("RGB", (width, height), (244, 244, 244))
    left = 0
    for panel in panels:
        sheet.paste(panel, (left, 0))
        left += panel.width + GAP
    return sheet


def _photo_panel(image: np.ndarray, detection: dict | None) -> Image.Image:
    rgb = Image.fromarray(image[:, :, ::-1])
    if detection is not None and detection["score"] is not None:
        draw = ImageDraw.Draw(rgb)
        x1, y1, x2, y2 = detection["box"]
        draw.rectangle((x1, y1, x2, y2), outline=(210, 35, 35), width=4)
        label = f"{detection['score']:.2f}"
        font = _font(22)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        tag_top = max(0, int(y1) - text_h - 8)
        draw.rectangle((int(x1), tag_top, int(x1) + text_w + 8, tag_top + text_h + 6), fill=(210, 35, 35))
        draw.text((int(x1) + 4, tag_top + 2), label, fill=(255, 255, 255), font=font)

    fitted = _fit(rgb, PHOTO_W, SLOT_H)
    if detection is None:
        caption = ["фото", "клеймо не найдено"]
    elif detection["score"] is None:
        caption = ["фото", "кроп целиком, без детектора"]
    else:
        caption = ["фото", f"детект {detection['score']:.3f}"]
    return _card(fitted, caption, max(fitted.width, 180))


def _crop_panel(image: np.ndarray, detection: dict, target_size: int) -> Image.Image:
    if detection["score"] is None:
        prepared = fit_square(image, target_size)
    else:
        prepared = crop_stamp(image, detection["box"], target_size=target_size)
    crop = Image.fromarray(prepared[:, :, ::-1]).resize((CROP_SIZE, CROP_SIZE), Image.Resampling.LANCZOS)
    return _card(crop, ["кроп", "вход энкодера"], CROP_SIZE)


def _match_panel(hit: dict) -> Image.Image:
    path = Path(hit["image_path"])
    if path.is_file():
        mark = Image.fromarray(read_bgr(path)[:, :, ::-1])
        mark = _fit(mark, MATCH_W, SLOT_H)
    else:
        mark = Image.new("RGB", (MATCH_W, SLOT_H), (220, 220, 220))
    catalog = hit["catalog_no"] or "—"
    factory = hit["factory"] or "фабрика не указана"
    caption = [
        f"{hit['rank']}. {hit['score']:.3f}",
        hit["mark_id"],
        f"№{catalog}  {hit['book']}",
        factory,
    ]
    return _card(mark, caption, max(mark.width, MATCH_W))


def _header(result: dict) -> Image.Image:
    name = Path(result["image"]).name
    count = len(result["detections"])
    text = f"{name}    top-{result['top_k']}    детекций: {count}"
    font = _font(22)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_w = int(probe.textlength(text, font=font))
    image = Image.new("RGB", (text_w + 4, 36), (244, 244, 244))
    ImageDraw.Draw(image).text((0, 4), text, fill=(20, 20, 20), font=font)
    return image


def _card(picture: Image.Image, lines: list[str], width: int) -> Image.Image:
    card_w = max(width, picture.width)
    card = Image.new("RGB", (card_w, SLOT_H + CAPTION_H), (255, 255, 255))
    left = (card_w - picture.width) // 2
    top = max(0, (SLOT_H - picture.height) // 2)
    card.paste(picture, (left, top))
    draw = ImageDraw.Draw(card)
    font = _font(16)
    wrapped = []
    for line in lines:
        wrapped.extend(_wrap(draw, line, font, card_w - 8))
    wrapped = _clip(wrapped, 4)
    text_top = SLOT_H + 8
    for line in wrapped:
        draw.text((6, text_top), line, fill=(20, 20, 20), font=font)
        text_top += 20
    return card


def _fit(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / image.width, max_h / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textlength(trial, font=font) <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _clip(lines: list[str], limit: int) -> list[str]:
    if len(lines) <= limit:
        return lines
    clipped = lines[:limit]
    clipped[-1] = clipped[-1].rstrip(" .") + "…"
    return clipped


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()
