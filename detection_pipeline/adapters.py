"""Unify POSTPROCESSED outputs. Boxes must already use original-image pixels."""


def xyxy_to_coco(boxes, labels, scores, label_to_category, image_size, score_floor=0.001):
    """Works with lists, numpy arrays or CPU/GPU tensors through .tolist()."""
    def as_list(value):
        return value.tolist() if hasattr(value, "tolist") else list(value)

    boxes, labels, scores = map(as_list, (boxes, labels, scores))
    if not len(boxes) == len(labels) == len(scores):
        raise ValueError("Prediction boxes, labels and scores have different lengths")
    width, height = image_size
    result = []
    for box, label, score in zip(boxes, labels, scores):
        if score < score_floor:
            continue
        label = int(label)
        if label not in label_to_category:
            raise ValueError(f"Unknown model label: {label}")
        x1, y1, x2, y2 = map(float, box)
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        result.append({"category_id": label_to_category[label], "bbox": [x1, y1, x2 - x1, y2 - y1],
                       "score": float(score)})
    return sorted(result, key=lambda d: d["score"], reverse=True)[:100]


def torchvision_prediction(output, dataset, image_size):
    return xyxy_to_coco(output["boxes"], output["labels"], output["scores"],
                        dataset.label_to_category, image_size)


def rtdetr_prediction(postprocessed_output, categories, image_size):
    # lyuwenyu/RT-DETR postprocessor expects orig_target_sizes=[width, height].
    # Other implementations (e.g. Hugging Face) may use [height, width].
    category_ids = [c["id"] for c in sorted(categories, key=lambda c: c["id"])]
    return xyxy_to_coco(postprocessed_output["boxes"], postprocessed_output["labels"],
                        postprocessed_output["scores"], dict(enumerate(category_ids)), image_size)


def mmdet_prediction(det_data_sample, categories, image_size):
    instances = det_data_sample.pred_instances
    category_ids = [c["id"] for c in sorted(categories, key=lambda c: c["id"])]
    return xyxy_to_coco(instances.bboxes, instances.labels, instances.scores,
                        dict(enumerate(category_ids)), image_size)


def yolo_prediction(result, class_to_category):
    # result.orig_shape is [height, width]; result.boxes.xyxy is already rescaled.
    if result.boxes is None:
        return []
    return xyxy_to_coco(result.boxes.xyxy, result.boxes.cls, result.boxes.conf,
                        class_to_category, result.orig_shape[::-1])
