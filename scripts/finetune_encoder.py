import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.detection.detector import StampDetector
from src.detection.yolo_processing import YOLOProcessing
from src.paths import REPORTS_DIR, WEIGHTS_DIR
from src.preprocessing.crop import crop_stamp
from src.preprocessing.image import fit_square, read_bgr
from src.retrieval.books import load_book_records, project_path, resolve_reference
from src.retrieval.encoder import IMAGENET_MEAN, IMAGENET_STD
from src.retrieval.index import StampIndex

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
INPUT_SIZE = 224


def parse_args():
    parser = argparse.ArgumentParser(
        description="Дообучить DINOv2 на двух видах одного клейма и собрать новый индекс"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--unfreeze-blocks", type=int, default=2)
    parser.add_argument(
        "--output-onnx",
        type=str,
        default="weights/embedding_weights/encoder_finetuned.onnx",
    )
    parser.add_argument(
        "--output-index",
        type=str,
        default="weights/retrieval/books_index_finetuned.npz",
    )
    return parser.parse_args()


class TwoViewDataset(Dataset):

    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.fromarray(cv2.cvtColor(read_bgr(self.paths[index]), cv2.COLOR_BGR2RGB))
        return self.transform(image), self.transform(image)


def make_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                INPUT_SIZE,
                scale=(0.55, 1.0),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomRotation(180, fill=114),
            transforms.ColorJitter(0.4, 0.4, 0.25, 0.02),
            transforms.RandomGrayscale(p=0.15),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))],
                p=0.3,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def collect_stamp_crops(stamps_dir: Path, destination: Path) -> list[Path]:
    if not stamps_dir.is_dir():
        return []
    weights = WEIGHTS_DIR / "detection_weights" / "yolo26s_v2.onnx"
    if not weights.is_file():
        return []

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    detector = StampDetector(weights, YOLOProcessing())
    saved = []
    images = sorted(
        path
        for path in stamps_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )
    for path in tqdm(images, desc="Кропы stamps", unit="img"):
        image = read_bgr(path)
        boxes, scores = detector.predict(image)
        scores = np.asarray(scores).reshape(-1)
        if len(scores) == 0:
            continue
        best = int(np.argmax(scores))
        if float(scores[best]) < 0.75:
            continue
        crop = crop_stamp(image, boxes[best], target_size=INPUT_SIZE)
        out = destination / f"{path.stem}.png"
        ok, encoded = cv2.imencode(".png", crop)
        if not ok:
            continue
        encoded.tofile(out)
        saved.append(out)
    return saved


def load_backbone(device: torch.device, unfreeze_blocks: int):
    import timm

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=0,
        img_size=INPUT_SIZE,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    blocks = list(model.blocks)
    if unfreeze_blocks > len(blocks):
        raise ValueError(f"В сети {len(blocks)} блоков, запрошено {unfreeze_blocks}")
    for block in blocks[-unfreeze_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.norm.parameters():
        parameter.requires_grad = True
    model.to(device)
    return model


def info_nce(first: torch.Tensor, second: torch.Tensor, temperature: float) -> torch.Tensor:
    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    logits = first @ second.T / temperature
    labels = torch.arange(first.shape[0], device=first.device)
    forward = F.cross_entropy(logits, labels)
    backward = F.cross_entropy(logits.T, labels)
    return (forward + backward) / 2


def train(model, loader, device, epochs: int, lr: float, temperature: float) -> list[float]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.05)
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        progress = tqdm(loader, desc=f"Эпоха {epoch}/{epochs}", unit="batch")
        for view_a, view_b in progress:
            view_a = view_a.to(device)
            view_b = view_b.to(device)
            loss = info_nce(model(view_a), model(view_b), temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total += float(loss.item()) * view_a.shape[0]
            seen += view_a.shape[0]
            progress.set_postfix(loss=f"{loss.item():.4f}")
        epoch_loss = total / max(seen, 1)
        losses.append(epoch_loss)
        print(f"Эпоха {epoch}: loss {epoch_loss:.4f}")
    return losses


def embed_books(model, records: list[dict], books_dir: Path, device: torch.device) -> np.ndarray:
    model.eval()
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    vectors = []
    batch = []

    def flush():
        if not batch:
            return
        blob = torch.from_numpy(np.stack(batch)).to(device)
        blob = (blob - mean) / std
        with torch.no_grad():
            output = model(blob)
        vectors.append(output.float().cpu().numpy())
        batch.clear()

    for record in tqdm(records, desc="Индекс books", unit="mark"):
        image = read_bgr(books_dir / record["image_path"])
        square = fit_square(image, INPUT_SIZE)
        rgb = square[:, :, ::-1].astype(np.float32) / 255.0
        batch.append(np.transpose(rgb, (2, 0, 1)))
        if len(batch) == 64:
            flush()
    flush()
    return np.concatenate(vectors, axis=0)


def export_onnx(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.eval().cpu()
    torch.onnx.export(
        model_cpu,
        (torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE),),
        str(path),
        input_names=["input"],
        output_names=["embedding"],
        dynamo=False,
        external_data=False,
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.unfreeze_blocks < 1:
        raise ValueError("Нужны epochs >= 1, batch-size >= 2 и хотя бы один обучаемый блок")

    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Устройство: {device}")

    books_dir, meta_csv = resolve_reference(None, None)
    records = load_book_records(meta_csv, books_dir)
    book_paths = [books_dir / record["image_path"] for record in records]

    stamps_dir = books_dir.parent.parent / "stamps"
    crop_dir = WEIGHTS_DIR / "finetune_crops"
    stamp_paths = collect_stamp_crops(stamps_dir, crop_dir)
    train_paths = book_paths + stamp_paths
    print(f"Книжных клейм: {len(book_paths)}")
    print(f"Кропов с фото: {len(stamp_paths)}")
    print(f"Всего в обучении: {len(train_paths)}")

    dataset = TwoViewDataset(train_paths, make_transform())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    model = load_backbone(device, args.unfreeze_blocks)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Обучаемые параметры: {trainable} из {total}")

    losses = train(model, loader, device, args.epochs, args.lr, args.temperature)

    embeddings = embed_books(model, records, books_dir, device)
    index_path = project_path(args.output_index)
    StampIndex(embeddings, records).save(index_path, books_dir)
    print(f"Индекс: {index_path}")

    onnx_path = project_path(args.output_onnx)
    export_onnx(model, onnx_path)
    print(f"ONNX: {onnx_path}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORTS_DIR / "finetune_loss.txt"
    report.write_text(
        "\n".join(f"epoch {index}: {loss:.6f}" for index, loss in enumerate(losses, start=1)) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
