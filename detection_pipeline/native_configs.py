"""Generate native configs after installing the corresponding official repo.

No framework is imported here. Generated configs must be smoke-tested in the
model owner's environment before long runs. Use a separate environment per stack.
"""
import argparse
import json
from pathlib import Path

from .data import read_json, write_json
from .experiments import DEFAULT_EXPERIMENTS, resolve_augmentation


def generate(splits, images, output, rtdetr_repo=None, mmdet_repo=None, profile=None,
             epochs=None, batch_size=None, experiments=None):
    settings = read_json(experiments or DEFAULT_EXPERIMENTS)
    epochs = settings["epochs"] if epochs is None else epochs
    batch_size = settings["batch_size"] if batch_size is None else batch_size
    seed = settings["seed"]
    override_profile = profile
    splits, images, output = map(lambda p: Path(p).resolve(), (splits, images, output))
    output.mkdir(parents=True, exist_ok=True)
    classes = tuple(c["name"] for c in sorted(read_json(splits / "train.json")["categories"], key=lambda c: c["id"]))
    generated = []
    if rtdetr_repo:
        profile, team_ops = resolve_augmentation("rtdetrv2_s", override_profile, experiments)
        # Pass the repo root, containing rtdetrv2_pytorch/.
        base = Path(rtdetr_repo).resolve() / "rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml"
        if not base.is_file():
            raise FileNotFoundError(base)
        steps_per_epoch = max(1, (len(read_json(splits / "train.json")["images"]) + batch_size - 1) // batch_size)
        config = {"__include__": [str(base)], "num_classes": len(classes), "remap_mscoco_category": False,
                  "num_top_queries": 100, "seed": seed, "epoches": epochs,
                  "lr_warmup_scheduler": {"warmup_duration": steps_per_epoch},
                  "lr_scheduler": {"milestones": [max(1, int(epochs * 0.8))]},
                  "output_dir": str(output / f"rtdetrv2_s_{profile}_run")}
        # JSON is valid YAML; the official loader accepts this .yml file.
        for split in ("train", "val"):
            dataset = {"type": "TeamCocoDetection", "img_folder": str(images),
                       "ann_file": str(splits / f"{split}.json"),
                       "augmentation_ops": team_ops if split == "train" else []}
            if split == "train" and profile == "native":
                dataset["transforms"] = {"policy": {"epoch": max(0, epochs - 5)}}
            else:
                ops = []  # Team augmentations were applied in StampDataset.load_raw().
                ops += [{"type": "Resize", "size": [640, 640]},
                        {"type": "ConvertPILImage", "dtype": "float32", "scale": True}]
                if split == "train":
                    ops += [{"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True}]
                dataset["transforms"] = {"type": "Compose", "ops": ops, "policy": {"name": "default"}}
            config[f"{split}_dataloader"] = {"dataset": dataset, "total_batch_size": batch_size,
                                                "num_workers": 2, "drop_last": False}
        # The official trainer requires labels/image_id/orig_size, whereas the
        # standalone get_collate_fn("rtdetr") returns class_labels and boxes.
        config["train_dataloader"]["collate_fn"] = {"type": "BatchImageCollateFunction", "scales": None}
        path = output / f"rtdetrv2_s_{profile}.yml"
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_json(path.with_suffix(".metadata.json"), {"model": "rtdetrv2_s", "augmentation": profile,
                   "augmentation_operations": team_ops, "seed": seed, "epochs": epochs, "batch_size": batch_size,
                   "pretraining": "COCO detector checkpoint supplied through --tuning"})
        generated.append(str(path))
    if mmdet_repo:
        for size in ("tiny", "s"):
            profile, team_ops = resolve_augmentation(f"rtmdet_{size}", override_profile, experiments)
            base = Path(mmdet_repo).resolve() / f"configs/rtmdet/rtmdet_{size}_8xb32-300e_coco.py"
            if not base.is_file():
                raise FileNotFoundError(base)
            config = {"_base_": str(base), "randomness": {"seed": seed},
                      "custom_imports": {"imports": ["detection_pipeline.integrations.mmdet"], "allow_failed_imports": False},
                      "model": {"bbox_head": {"num_classes": len(classes)},
                                "test_cfg": {"score_thr": 0.001, "max_per_img": 100}},
                      "train_cfg": {"max_epochs": epochs, "val_interval": 1, "dynamic_intervals": []},
                      "optim_wrapper": {"optimizer": {"lr": 0.004 * batch_size / 256}},
                      "auto_scale_lr": {"enable": False},
                      "param_scheduler": [{"type": "LinearLR", "start_factor": 0.1,
                                           "by_epoch": True, "begin": 0, "end": 1, "convert_to_iter_based": True},
                                          {"type": "CosineAnnealingLR", "T_max": max(1, epochs - 1),
                                           "eta_min_ratio": 0.05, "by_epoch": True, "begin": 1,
                                           "end": epochs, "convert_to_iter_based": True}],
                      "work_dir": str(output / f"rtmdet_{size}_{profile}_run"),
                      "default_hooks": {"checkpoint": {"interval": 1, "max_keep_ckpts": 2,
                                                        "save_best": "coco/bbox_mAP", "rule": "greater"}}}
            for split in ("train", "val", "test"):
                actual = "train" if split == "train" else "val"
                # Native configs require a test loader even without an independent test.
                dataset = {"type": "CocoDataset", "data_root": str(images) + "/",
                           "ann_file": str(splits / f"{actual}.json"), "data_prefix": {"img": ""},
                           "metainfo": {"classes": classes}, "filter_cfg": {"filter_empty_gt": False}}
                config[f"{split}_dataloader"] = {"batch_size": batch_size, "num_workers": 2, "dataset": dataset}
            for split in ("val", "test"):
                config[f"{split}_evaluator"] = {"ann_file": str(splits / "val.json"), "metric": "bbox"}
            if profile != "native":
                pipeline = [{"type": "LoadImageFromFile"}, {"type": "LoadAnnotations", "with_bbox": True}]
                if team_ops:
                    pipeline += [{"type": "TeamAugment", "operations": team_ops}]
                pipeline += [{"type": "Resize", "scale": (640, 640), "keep_ratio": True},
                             {"type": "Pad", "size": (640, 640), "pad_val": {"img": (114, 114, 114)}},
                             {"type": "PackDetInputs"}]
                config["train_dataloader"]["dataset"]["pipeline"] = pipeline
                config["custom_hooks"] = [{"type": "EMAHook", "ema_type": "ExpMomentumEMA",
                                           "momentum": 0.0002, "update_buffers": True, "priority": 49}]
            else:
                config["custom_hooks"] = [
                    {"type": "EMAHook", "ema_type": "ExpMomentumEMA", "momentum": 0.0002,
                     "update_buffers": True, "priority": 49},
                    {"type": "PipelineSwitchHook", "switch_epoch": max(0, epochs - 5),
                     "switch_pipeline": "__STAGE2__"}]
            path = output / f"rtmdet_{size}_{profile}.py"
            lines = [f"{key} = {value!r}" for key, value in config.items()]
            if profile == "native":
                # MMEngine substitutes this base-config reference during parsing.
                lines = [line.replace("'__STAGE2__'", "{{_base_.train_pipeline_stage2}}") for line in lines]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            write_json(path.with_suffix(".metadata.json"), {"model": f"rtmdet_{size}", "augmentation": profile,
                       "augmentation_operations": team_ops, "seed": seed, "epochs": epochs, "batch_size": batch_size,
                       "pretraining": "ImageNet backbone by default; record any detection checkpoint override"})
            generated.append(str(path))
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rtdetr-repo")
    parser.add_argument("--mmdet-repo")
    parser.add_argument("--profile", choices=["none", "light", "rotation", "native"],
                        help="Override each model's choice from experiments JSON")
    parser.add_argument("--experiments")
    args = parser.parse_args()
    if not (args.rtdetr_repo or args.mmdet_repo):
        parser.error("Pass --rtdetr-repo and/or --mmdet-repo")
    print("\n".join(generate(args.splits, args.images, args.output, args.rtdetr_repo, args.mmdet_repo, args.profile, experiments=args.experiments)))
