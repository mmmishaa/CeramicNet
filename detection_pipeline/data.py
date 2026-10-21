"""Prepare immutable COCO train/val files without copying images.

python -m detection_pipeline.data --annotations annotations.json --images images --output splits
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def object_id(image):
    """Explicit group_id wins; otherwise strip the final _photo-number suffix."""
    if image.get("group_id") is not None:
        return str(image["group_id"])
    stem = unicodedata.normalize("NFC", Path(image["file_name"]).stem)
    if not re.search(r"_\d+$", stem):
        raise ValueError(f"Set group_id for image without _photo-number suffix: {stem}")
    return re.sub(r"_\d+$", "", stem)


def validate_coco(data):
    images = {im["id"]: im for im in data["images"]}
    categories = {c["id"] for c in data["categories"]}
    if len(images) != len(data["images"]) or not images:
        raise ValueError("Image IDs must be unique and dataset nonempty")
    if len(categories) != len(data["categories"]) or not categories:
        raise ValueError("Category IDs must be unique and nonempty")
    if len({a["id"] for a in data["annotations"]}) != len(data["annotations"]):
        raise ValueError("Annotation IDs must be unique")
    for im in images.values():
        if im["width"] <= 0 or im["height"] <= 0:
            raise ValueError(f"Invalid image dimensions: {im['id']}")
    for a in data["annotations"]:
        if a["image_id"] not in images or a["category_id"] not in categories:
            raise ValueError(f"Unknown image/category in annotation {a['id']}")
        if a.get("iscrowd", 0) or a.get("ignore", 0):
            raise ValueError("This benchmark supports ordinary bbox annotations, not crowd/ignore")
        im = images[a["image_id"]]
        x, y, w, h = a["bbox"]
        if not all(math.isfinite(v) for v in (x, y, w, h)) or not (
            x >= 0 and y >= 0 and w > 0 and h > 0
            and x + w <= im["width"] + 1e-5 and y + h <= im["height"] + 1e-5
        ):
            raise ValueError(f"Invalid/out-of-image bbox in annotation {a['id']}")


def prepare_split(annotations, images_root, output, seed=42, val_fraction=0.2):
    from PIL import Image

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    root, output = Path(images_root).resolve(), Path(output).resolve()
    data = deepcopy(read_json(annotations))
    validate_coco(data)
    # Match relocated files only if their basename is unambiguous.
    lookup = {}
    for path in root.rglob("*"):
        if path.is_file():
            lookup.setdefault(unicodedata.normalize("NFC", path.name), []).append(path)
    paths_changed = []
    for im in data["images"]:
        path = (root / im["file_name"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Image path outside images_root: {im['file_name']}")
        if not path.is_file():
            candidates = lookup.get(unicodedata.normalize("NFC", Path(im["file_name"]).name), [])
            if len(candidates) != 1:
                raise ValueError(f"Missing/ambiguous image: {im['file_name']}")
            path = candidates[0].resolve()
            paths_changed.append({"old": im["file_name"], "new": path.relative_to(root).as_posix()})
        im["file_name"] = path.relative_to(root).as_posix()
        with Image.open(path) as photo:
            if photo.size != (im["width"], im["height"]):
                raise ValueError(f"COCO dimensions disagree with actual image: {path}")
        im["group_id"] = object_id(im)
        im["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    # Contiguous zero-based IDs work for RT-DETR with remap_mscoco_category=False.
    # The torchvision adapter adds 1 for its reserved background label.
    original_categories = sorted(data["categories"], key=lambda c: c["id"])
    category_map = {c["id"]: i for i, c in enumerate(original_categories)}
    data["categories"] = [dict(c, id=category_map[c["id"]]) for c in original_categories]
    for a in data["annotations"]:
        a["category_id"] = category_map[a["category_id"]]
        a["area"] = a["bbox"][2] * a["bbox"][3]
        a["iscrowd"] = 0

    # Union object groups connected by byte-identical files, even with renamed images.
    parents = {im["group_id"]: im["group_id"] for im in data["images"]}

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    seen_hash = {}
    for im in sorted(data["images"], key=lambda x: x["id"]):
        group, digest = im["group_id"], im["sha256"]
        if digest in seen_hash:
            a, b = sorted((find(group), find(seen_hash[digest])))
            parents[b] = a
        seen_hash[digest] = group
    groups = sorted({find(im["group_id"]) for im in data["images"]})
    if len(groups) < 2:
        raise ValueError("At least two independent object groups are required")
    random.Random(seed).shuffle(groups)
    n_val = min(len(groups) - 1, max(1, math.ceil(len(groups) * val_fraction)))
    val_groups = set(groups[:n_val])
    ids = {split: [] for split in ("train", "val")}
    for im in data["images"]:
        ids["val" if find(im["group_id"]) in val_groups else "train"].append(im["id"])
    fingerprint = hashlib.sha256(json.dumps(
        {"data": data, "seed": seed, "val_fraction": val_fraction}, sort_keys=True, ensure_ascii=False
    ).encode()).hexdigest()
    if output.exists():
        raise FileExistsError(f"Split already exists: {output}. Reuse it, or choose a new version directory.")
    output.mkdir(parents=True)
    stats = {}
    for split, image_ids in ids.items():
        chosen = set(image_ids)
        subset = dict(data, images=[im for im in data["images"] if im["id"] in chosen],
                      annotations=[a for a in data["annotations"] if a["image_id"] in chosen])
        write_json(output / f"{split}.json", subset)
        positive = {a["image_id"] for a in subset["annotations"]}
        stats[split] = {"images": len(chosen), "boxes": len(subset["annotations"]),
                        "negative_images": len(chosen - positive),
                        "groups": len({find(im["group_id"]) for im in subset["images"]}),
                        "boxes_per_class": dict(Counter(a["category_id"] for a in subset["annotations"]))}
    manifest = {"source": str(Path(annotations).resolve()), "images_root": str(root),
                "seed": seed, "val_fraction_of_groups": val_fraction, "fingerprint": fingerprint,
                "grouping": "group_id or filename without final _photo-number; SHA256 union",
                "original_categories": original_categories, "category_id_map": category_map,
                "paths_changed": paths_changed, "image_ids": ids, "stats": stats,
                "test": None}
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare_split(args.annotations, args.images, args.output, args.seed)["stats"], indent=2))
