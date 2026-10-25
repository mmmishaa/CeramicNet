"""Дообучение DINOv2 чистым Barlow Twins: книги и реал, без учителя."""

from __future__ import annotations

import argparse
from itertools import cycle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from finetune.data_utils import BookPairDataset, RealPairDataset, load_books, load_real_crops, square_tensor
from finetune.losses import BarlowTwinsLoss
from finetune.metrics import evaluate_epoch
from finetune.model import DinoEncoder, Student
from src.paths import WEIGHTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Дообучение DINOv2 чистым Barlow Twins")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--unfreeze-blocks", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=0, help="Остановиться после N шагов. 0 — все эпохи.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("Barlow Twins требует batch-size >= 2")

    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # 4 воркера на Mac не блокируют MPS. Явный --num-workers 0 отключает их.
    num_workers = 4 if device.type == "mps" else 0
    if args.num_workers is not None:
        num_workers = args.num_workers
    task, logger = _init_clearml()
    print(f"Устройство: {device}")

    books = [sample for sample in load_books() if sample.path.is_file()]
    reals = load_real_crops()
    print(f"Книг: {len(books)}, реальных кропов: {len(reals)}")

    baseline = DinoEncoder(unfreeze_blocks=0, pretrained=True).to(device).eval()
    baseline_embeddings = _embed_clean(baseline, books, device)

    student = Student(unfreeze_blocks=args.unfreeze_blocks).to(device)
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.05)
    barlow = BarlowTwinsLoss()

    book_loader = DataLoader(
        BookPairDataset(books),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    real_loader = DataLoader(
        RealPairDataset(reals),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    # Реала меньше, чем книг: цикл не даёт оборвать эпоху на коротком потоке.
    real_cycle = cycle(real_loader) if len(real_loader) else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(len(book_loader), 1),
    )

    output_dir = WEIGHTS_DIR / "finetuned_v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_recall = float("-inf")
    best_encoder_state: dict[str, torch.Tensor] | None = None
    epoch_rows: list[dict[str, float]] = []

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        student.train()
        totals = {"book": 0.0, "real": 0.0, "total": 0.0, "steps": 0}
        progress = tqdm(book_loader, desc=f"Студент {epoch}/{args.epochs}", unit="batch")
        for view_a, view_b, _indexes in progress:
            view_a = view_a.to(device, non_blocking=True)
            view_b = view_b.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                emb_a = student(view_a)
                emb_b = student(view_b)
                if real_cycle is not None:
                    real_a, real_b, _ = next(real_cycle)
                    emb_real_a = student(real_a.to(device, non_blocking=True))
                    emb_real_b = student(real_b.to(device, non_blocking=True))
                else:
                    emb_real_a, emb_real_b = None, None

            # Ковариация Barlow в bfloat16 раздувает лосс. Считаем её в float32.
            loss_barlow_book = barlow(emb_a.float(), emb_b.float())
            loss_barlow_real = torch.zeros((), device=device)
            if emb_real_a is not None:
                loss_barlow_real = barlow(emb_real_a.float(), emb_real_b.float())
            total = loss_barlow_book + loss_barlow_real

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            learning_rate = optimizer.param_groups[0]["lr"]
            if global_step % 50 == 0 and device.type == "mps":
                allocated_gb = torch.mps.current_allocated_memory() / (1024**3)
                print(f"[MPS Memory] Выделено: {allocated_gb:.2f} GB / 36 GB")
            totals["book"] += loss_barlow_book.detach().item()
            totals["real"] += loss_barlow_real.detach().item()
            totals["total"] += total.detach().item()
            totals["steps"] += 1
            progress.set_postfix(
                book=f"{loss_barlow_book.item():.3f}",
                real=f"{loss_barlow_real.item():.3f}",
            )
            _scalar(logger, "train", "loss_barlow_book", loss_barlow_book.detach().item(), global_step)
            _scalar(logger, "train", "loss_barlow_real", loss_barlow_real.detach().item(), global_step)
            _scalar(logger, "train", "total_loss", total.detach().item(), global_step)
            _scalar(logger, "train", "lr", learning_rate, global_step)

            if args.log_every > 0 and global_step % args.log_every == 0:
                _log_pair_grid(logger, view_a, view_b, emb_a, emb_b, global_step)

            if args.max_steps and global_step >= args.max_steps:
                break

        metrics = evaluate_epoch(
            student,
            baseline_embeddings,
            books,
            reals,
            device,
            logger=logger,
            iteration=epoch,
        )
        steps = max(totals["steps"], 1)
        row = {
            "epoch": float(epoch),
            "loss_barlow_book": totals["book"] / steps,
            "loss_barlow_real": totals["real"] / steps,
            "total_loss": totals["total"] / steps,
            "lr": optimizer.param_groups[0]["lr"],
            **metrics,
        }
        epoch_rows.append(row)
        _save_plots(epoch_rows, output_dir)
        print(
            "Эпоха {epoch}: recall книг {book:.3f}, сохранение соседей {soft:.3f}, "
            "recall реала {real:.3f}, разнообразие внутри фабрики {div:.3f}".format(
                epoch=epoch,
                book=metrics["book_recall_at_1"],
                soft=metrics["soft_neighbor_preservation"],
                real=metrics["real_recall_at_1"],
                div=metrics["intra_factory_diversity"],
            )
        )

        payload = _checkpoint_payload(student, args.unfreeze_blocks, epoch, metrics)
        last_path = output_dir / "last_student.pt"
        torch.save(payload, last_path)
        print(f"Последние веса: {last_path}")
        recall = metrics["book_recall_at_1"]
        if recall > best_recall:
            best_recall = recall
            best_encoder_state = _encoder_state(student)
            best_path = output_dir / "best_student.pt"
            torch.save(payload, best_path)
            print(f"Лучшие веса (book recall@1 = {recall:.3f}): {best_path}")

        if args.max_steps and global_step >= args.max_steps:
            break

    onnx_path = output_dir / "student_encoder.onnx"
    export_encoder_onnx(best_encoder_state or _encoder_state(student), onnx_path)
    print(f"ONNX энкодера: {onnx_path}")
    print(f"Графики: {output_dir / 'loss.png'}, {output_dir / 'metrics.png'}")
    if task is not None:
        task.close()


def _checkpoint_payload(student: Student, unfreeze_blocks: int, epoch: int, metrics: dict[str, float]) -> dict:
    return {
        "student": {key: value.detach().cpu() for key, value in student.state_dict().items()},
        "unfreeze_blocks": unfreeze_blocks,
        "epoch": epoch,
        "metrics": metrics,
    }


def _encoder_state(student: Student) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("encoder."): value.detach().cpu().clone()
        for key, value in student.state_dict().items()
        if key.startswith("encoder.")
    }


def export_encoder_onnx(encoder_state: dict[str, torch.Tensor], path) -> None:
    """В ONNX уходит только энкодер: ч/б BT.601 уже внутри.

    dynamo=False и external_data=False обязательны для torch 2.13, иначе файл режется на части.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder = DinoEncoder(unfreeze_blocks=0, pretrained=False).eval()
    encoder.load_state_dict(encoder_state)
    dummy = torch.rand(1, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        encoder,
        (dummy,),
        str(path),
        input_names=["input"],
        output_names=["embedding"],
        dynamo=False,
        external_data=False,
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
    _assert_onnx_matches(encoder, path, dummy)


def _assert_onnx_matches(encoder: DinoEncoder, path, dummy: torch.Tensor) -> None:
    import onnxruntime as ort

    with torch.no_grad():
        reference = encoder(dummy).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    exported = session.run(None, {"input": dummy.numpy()})[0]
    delta = float(np.abs(reference - exported).max())
    if delta >= 1e-4:
        raise AssertionError(f"ONNX разошёлся с PyTorch: max |Δ| = {delta:.3e}")
    if exported.shape != (1, 384):
        raise AssertionError(f"Ожидался эмбеддинг (1, 384), получено {exported.shape}")
    print(f"ONNX совпал с PyTorch, max |Δ| = {delta:.3e}")


def _save_plots(rows: list[dict[str, float]], output_dir) -> None:
    epochs = [row["epoch"] for row in rows]
    _plot(
        epochs,
        [
            ("loss_barlow_book", "книги"),
            ("loss_barlow_real", "реал"),
            ("total_loss", "сумма"),
        ],
        rows,
        "Потери Barlow Twins",
        output_dir / "loss.png",
    )
    _plot(
        epochs,
        [
            ("book_recall_at_1", "recall книг"),
            ("soft_neighbor_preservation", "соседи"),
            ("real_recall_at_1", "recall реала"),
            ("intra_factory_diversity", "разнообразие"),
        ],
        rows,
        "Метрики эпох",
        output_dir / "metrics.png",
    )


def _plot(epochs, series: list[tuple[str, str]], rows: list[dict[str, float]], title: str, path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.2))
    for key, label in series:
        axis.plot(epochs, [row[key] for row in rows], marker="o", label=label)
    axis.set_title(title)
    axis.set_xlabel("эпоха")
    axis.legend()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _embed_clean(encoder: DinoEncoder, books, device: torch.device) -> torch.Tensor:
    encoder.eval()
    parts = []
    batch: list[torch.Tensor] = []
    for sample in tqdm(books, desc="Эмбеддинги baseline", unit="mark"):
        batch.append(square_tensor(sample.path))
        if len(batch) == 64:
            parts.append(_forward_encoder(encoder, torch.stack(batch).to(device)))
            batch = []
    if batch:
        parts.append(_forward_encoder(encoder, torch.stack(batch).to(device)))
    return torch.cat(parts, dim=0).detach()


def _forward_encoder(encoder: DinoEncoder, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return encoder(images).float().cpu()


def _log_pair_grid(logger, view_a, view_b, emb_a, emb_b, step: int) -> None:
    if logger is None:
        return
    count = min(4, view_a.shape[0])
    first = torch.nn.functional.normalize(emb_a[:count], dim=-1)
    second = torch.nn.functional.normalize(emb_b[:count], dim=-1)
    cosines = (first * second).sum(dim=-1).detach().cpu()
    tiles = []
    for index in range(count):
        left = _tensor_to_pil(view_a[index].detach().cpu())
        right = _tensor_to_pil(view_b[index].detach().cpu())
        tile = Image.new("RGB", (left.width + right.width, left.height + 24), (255, 255, 255))
        tile.paste(left, (0, 24))
        tile.paste(right, (left.width, 24))
        ImageDraw.Draw(tile).text((4, 4), f"cos {cosines[index]:.3f}", fill=(0, 0, 0))
        tiles.append(tile)
    width = sum(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    grid = Image.new("RGB", (width, height), (244, 244, 244))
    left = 0
    for tile in tiles:
        grid.paste(tile, (left, 0))
        left += tile.width
    logger.report_image("pairs", "book_views", iteration=step, image=grid)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def _init_clearml():
    try:
        from clearml import Task

        task = Task.init(
            project_name="CeramicNet",
            task_name="dino_barlow_v3",
            auto_connect_frameworks=False,
            auto_connect_arg_parser=False,
        )
        return task, task.get_logger()
    except Exception as error:
        print(f"ClearML недоступен, метрики останутся в консоли и в png: {error}")
        return None, None


def _scalar(logger, title: str, series: str, value: float, iteration: int) -> None:
    if logger is not None:
        logger.report_scalar(title, series, value, iteration)


if __name__ == "__main__":
    main()
