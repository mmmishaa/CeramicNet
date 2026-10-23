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


def add_negative_split(base_splits, images_root, negative_dirs, output,
                       seed=42, train_count=40, val_count=10, per_dir_count=None):
    """Add fixed counts of verified negatives, keeping the existing split intact.

    Paths must be inside images_root. Sampling uses NumPy RandomState(seed) over
    sorted relative paths. By default object groups stay in one split, including
    objects already present among positives. per_dir_count first samples that
    many files from each directory, then randomly splits the combined sample
    by image, without group constraints. Group overlaps are recorded explicitly.
    Exact copies are excluded before sampling in both modes.
    """
    import numpy as np
    from PIL import Image

    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0
           for n in (train_count, val_count)) or train_count + val_count == 0:
        raise ValueError("Negative counts must be nonnegative integers with a positive total")
    if per_dir_count is not None and (isinstance(per_dir_count, bool)
                                     or not isinstance(per_dir_count, int) or per_dir_count <= 0):
        raise ValueError("per_dir_count must be a positive integer")
    root, base, output = (Path(p).resolve() for p in (images_root, base_splits, output))
    if output.exists():
        raise FileExistsError(f"Split already exists: {output}. Choose a new version directory.")
    subsets = {s: deepcopy(read_json(base / f"{s}.json")) for s in ("train", "val")}
    group_split, hash_split, known_hashes, known_paths, known_ids = {}, {}, set(), set(), set()
    for split, subset in subsets.items():
        validate_coco(subset)
        annotated = {a["image_id"] for a in subset["annotations"]}
        if any(im["id"] not in annotated for im in subset["images"]):
            raise ValueError("Base split already contains negatives; use the original positive split")
        if subset["categories"] != subsets["train"]["categories"]:
            raise ValueError("Base train/val categories differ")
        for im in subset["images"]:
            group = object_id(im)
            if group in group_split and group_split[group] != split:
                raise ValueError(f"Object already crosses the base split: {group}")
            if im["id"] in known_ids:
                raise ValueError(f"Image ID already crosses the base split: {im['id']}")
            group_split[group] = split
            known_ids.add(im["id"])
            known_paths.add(unicodedata.normalize("NFC", im["file_name"]))
            path = (root / im["file_name"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Image outside images_root: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if im.get("sha256") is not None and im["sha256"] != digest:
                raise ValueError(f"Base image changed since splitting: {path}")
            if digest in hash_split and hash_split[digest] != split:
                raise ValueError(f"Exact copy already crosses the base split: {path}")
            hash_split[digest] = split
            known_hashes.add(digest)

    paths, folders = set(), set()
    for folder in negative_dirs:
        folder = Path(folder).resolve()
        if not folder.is_dir() or not folder.is_relative_to(root):
            raise ValueError(f"Negative directory must exist inside images_root: {folder}")
        folders.add(folder.relative_to(root).as_posix())
        paths.update(p.resolve() for p in folder.rglob("*")
                     if p.is_file() and p.suffix.lower() in
                     {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"})
    candidates, skipped = [], Counter()
    for path in sorted(paths, key=lambda p: unicodedata.normalize("NFC", p.relative_to(root).as_posix())):
        if not path.is_relative_to(root):
            raise ValueError(f"Negative image outside images_root: {path}")
        name = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if unicodedata.normalize("NFC", name) in known_paths or digest in known_hashes:
            skipped["base_image_or_exact_copy"] += 1
            continue
        record = {"file_name": name, "sha256": digest}
        record["group_id"] = object_id(record)
        with Image.open(path) as photo:
            photo.load()
            record.update(width=photo.width, height=photo.height)
        candidates.append(record)
        known_hashes.add(digest)

    candidate_fingerprint = hashlib.sha256(json.dumps(
        candidates, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if per_dir_count is not None:
        if len(folders) * per_dir_count != train_count + val_count:
            raise ValueError("per_dir_count * number of directories must equal train_count + val_count")
        sample, assigned_paths = [], set()
        for folder in sorted(folders):
            pool = [im for im in candidates
                    if Path(im["file_name"]).is_relative_to(Path(folder))]
            if len(pool) < per_dir_count:
                raise ValueError(f"Not enough eligible negatives in {folder}: need {per_dir_count}, "
                                 f"found {len(pool)}. No split was written.")
            chosen = [pool[int(i)] for i in np.random.RandomState(seed).permutation(len(pool))[:per_dir_count]]
            if any(im["file_name"] in assigned_paths for im in chosen):
                raise ValueError("Negative directories overlap; use separate source directories")
            sample.extend(chosen)
            assigned_paths.update(im["file_name"] for im in chosen)
        # This matches ShuffleSplit/train_test_split(random_state=seed) indices:
        # first n_test shuffled positions form val, the remainder form train.
        indices = np.random.RandomState(seed).permutation(len(sample))
        selected = {"val": [sample[int(i)] for i in indices[:val_count]],
                    "train": [sample[int(i)] for i in indices[val_count:]]}
        sampling = "sorted paths; sample per directory with RandomState(seed); concatenate; random image split with RandomState(seed)"
    else:
        # Fill val first, then train, keeping all photos of an object together.
        permutation = np.random.RandomState(seed).permutation(len(candidates))
        selected, used = {"train": [], "val": []}, set()
        for split, count in (("val", val_count), ("train", train_count)):
            for index in permutation:
                if len(selected[split]) == count:
                    break
                im = candidates[int(index)]
                if int(index) in used or group_split.get(im["group_id"], split) != split:
                    continue
                selected[split].append(im)
                used.add(int(index))
                group_split[im["group_id"]] = split
            if len(selected[split]) != count:
                raise ValueError(f"Not enough eligible negatives for {split}: need {count}, "
                                 f"found {len(selected[split])}. No split was written.")
        sampling = "sorted paths; NumPy RandomState permutation; val then train; object constraints"

    next_id, stats = max(known_ids) + 1, {}
    for split, subset in subsets.items():
        for im in selected[split]:
            im["id"] = next_id
            next_id += 1
        subset["images"].extend(selected[split])
        validate_coco(subset)
        stats[split] = {"images": len(subset["images"]), "boxes": len(subset["annotations"]),
                        "negative_images": len(selected[split]),
                        "groups": len({object_id(im) for im in subset["images"]}),
                        "boxes_per_class": dict(Counter(a["category_id"] for a in subset["annotations"]))}
    manifest = deepcopy(read_json(base / "manifest.json"))
    base_hashes = {f"{s}.json": hashlib.sha256((base / f"{s}.json").read_bytes()).hexdigest()
                   for s in ("train", "val", "manifest")}
    groups = {s: {object_id(im) for im in subset["images"]} for s, subset in subsets.items()}
    negative_groups = {s: {object_id(im) for im in chosen} for s, chosen in selected.items()}
    manifest.update(seed=seed, base_split={"name": base.name, "files_sha256": base_hashes},
                    image_ids={s: [im["id"] for im in d["images"]] for s, d in subsets.items()},
                    stats=stats,
                    negatives={"directories": sorted(folders), "random_state": seed,
                               "sampling": sampling, "per_directory_count": per_dir_count,
                               "split_by": "image" if per_dir_count is not None else "object",
                               "group_overlap": sorted(groups["train"] & groups["val"]),
                               "negative_group_overlap": sorted(negative_groups["train"] & negative_groups["val"]),
                               "candidate_count": len(candidates), "skipped": dict(skipped),
                               "candidate_sha256": candidate_fingerprint,
                               "counts": {"train": train_count, "val": val_count},
                               "selected": {s: [dict(im) for im in chosen] for s, chosen in selected.items()}})
    manifest["fingerprint"] = hashlib.sha256(json.dumps(
        {"data": subsets, "base": base_hashes, "negatives": manifest["negatives"]},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    output.mkdir(parents=True)
    for split, subset in subsets.items():
        write_json(output / f"{split}.json", subset)
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--annotations")
    source.add_argument("--base-splits", help="Existing positive split to extend without reshuffling")
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-dirs", nargs="+", help="Verified negatives, inside --images")
    parser.add_argument("--negative-train", type=int, default=40)
    parser.add_argument("--negative-val", type=int, default=10)
    parser.add_argument("--negative-per-dir", type=int,
                        help="Sample this many files per source, then randomly split combined images")
    args = parser.parse_args()
    if args.base_splits:
        if not args.negative_dirs:
            parser.error("--base-splits requires --negative-dirs")
        manifest = add_negative_split(args.base_splits, args.images, args.negative_dirs,
                                      args.output, args.seed, args.negative_train, args.negative_val,
                                      per_dir_count=args.negative_per_dir)
    else:
        if args.negative_dirs:
            parser.error("--negative-dirs requires --base-splits")
        manifest = prepare_split(args.annotations, args.images, args.output, args.seed)
    print(json.dumps(manifest["stats"], indent=2))
