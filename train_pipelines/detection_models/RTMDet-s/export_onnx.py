"""Export the trained one-class RTMDet-s without custom ONNX operators."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))


class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", model.data_preprocessor.mean[None].clone())
        self.register_buffer("std", model.data_preprocessor.std[None].clone())
        assert model.bbox_head.num_classes == 1
        assert not model.data_preprocessor._channel_conversion
        sizes = [(640 // s[0], 640 // s[1]) for s in model.bbox_head.prior_generator.strides]
        points = model.bbox_head.prior_generator.grid_priors(sizes, device="cpu")
        self.register_buffer("points", torch.cat(points)[None])

    def forward(self, images):
        feats = self.model.extract_feat((images - self.mean) / self.std)
        cls, dist = self.model.bbox_head(feats)
        scores = torch.cat([x.permute(0, 2, 3, 1).reshape(1, -1, 1) for x in cls], 1).sigmoid()
        dist = torch.cat([x.permute(0, 2, 3, 1).reshape(1, -1, 4) for x in dist], 1)
        boxes = torch.cat([self.points - dist[..., :2], self.points + dist[..., 2:]], -1)
        return torch.cat([boxes, scores], -1)


def export(config, checkpoint, output, repo, threads=4):
    from detection_pipeline.predict_native import prepare_imports
    prepare_imports("mmdet", repo)
    from mmengine.config import Config
    from mmdet.apis import init_detector

    torch.set_num_threads(threads)
    cfg = Config.fromfile(config)
    cfg.model.backbone.init_cfg = None
    model = init_detector(cfg, checkpoint, device="cpu").eval()
    wrapper = ExportModel(model).eval()
    sample = torch.full((1, 3, 640, 640), 114.0)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        expected = wrapper(sample).numpy()
        torch.onnx.export(wrapper, sample, str(output), opset_version=17,
                          input_names=["images"], output_names=["detections"],
                          do_constant_folding=True, dynamo=False)
    graph = onnx.load(output)
    onnx.checker.check_model(graph, full_check=True)
    if any(n.domain not in ("", "ai.onnx") for n in graph.graph.node):
        raise ValueError("Unexpected custom ONNX operators")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    session = ort.InferenceSession(str(output), sess_options=opts, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"images": sample.numpy()})[0]
    # Convolution implementations differ slightly between PyTorch and ORT.
    # Compare coordinates and probabilities in their respective units.
    np.testing.assert_allclose(actual[..., :4], expected[..., :4], rtol=1e-3, atol=0.1)
    np.testing.assert_allclose(actual[..., 4], expected[..., 4], rtol=1e-3, atol=1e-4)
    metadata = {"checkpoint": Path(checkpoint).name,
                "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                "onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "opset": 17, "input": "images: BGR float32 [1,3,640,640], range 0..255",
                "output": "detections: float32 [1,8400,5], xyxy + score; NMS external",
                "processor_module": "detection_pipeline.rtmdet_onnx_processing",
                "process_class": "RTMDetProcessing", "nms_iou": 0.65,
                "blank_image_max_abs_error": float(np.max(np.abs(expected - actual)))}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copy2(PROJECT / "detection_pipeline/rtmdet_onnx_processing.py", output.parent / "rtmdet_onnx_processing.py")
    print(json.dumps(metadata, indent=2))
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--repo", default=str(PROJECT / ".local/rtmdet/mmdetection"))
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    export(a.config, a.checkpoint, a.output, a.repo, a.threads)
