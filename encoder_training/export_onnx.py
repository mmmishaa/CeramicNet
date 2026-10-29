"""Чекпоинт -> ONNX с контрактом StampEncoder: вход RGB 0..1 NCHW 224, выход вектор; ч/б и нормировка внутри.

Запуск из корня репозитория:
    python -m encoder_training.export_onnx runs/encoder/s_u6_e20/last.pt --out weights/retrieval_weights/student_encoder.onnx
    python -m scripts.build_index

В retrieval_config.yaml: mean [0,0,0], std [1,1,1], enhance_contrast: true. Индекс эталонов пересобрать этим же энкодером.
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from encoder_training.model import load_checkpoint
from src.retrieval.books import project_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--out", default=None, help="По умолчанию student_encoder.onnx рядом с чекпоинтом")
    args = parser.parse_args()
    checkpoint = project_path(args.checkpoint)
    model = load_checkpoint(checkpoint).eval()
    out = project_path(args.out) if args.out else checkpoint.with_name("student_encoder.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(2, 3, 224, 224)
    torch.onnx.export(
        model, (dummy,), str(out), input_names=["input"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17, dynamo=False, external_data=False,
    )
    with torch.no_grad():
        reference = model(dummy).numpy()
    exported = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(None, {"input": dummy.numpy()})[0]
    delta = float(np.abs(reference - exported).max())
    assert delta < 1e-3, f"ONNX разошёлся с PyTorch: {delta:.2e}"
    print(f"{out} готов, выход {exported.shape}, max |Δ| = {delta:.1e}")
