from pathlib import Path
import numpy as np
import onnxruntime as ort

class StampDetector:

    def __init__(
        self,
        model_path: Path,
        processor,
        num_threads: int = 4,
    ):
        self.model_path = model_path
        self.processor = processor

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def _forward(self, tensor: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input_name: tensor})

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor, metadata = self.processor.preprocess(image)
        raw_outputs = self._forward(tensor)
        boxes, scores = self.processor.postprocess(raw_outputs, metadata)
        return boxes, scores