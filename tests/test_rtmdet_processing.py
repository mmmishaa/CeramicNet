import unittest

import numpy as np

from src.detection.rtmdet_processing import RTMDetProcessing


class RTMDetProcessingTests(unittest.TestCase):
    def test_bgr_range_and_bottom_right_padding(self):
        image = np.full((3, 8, 3), [5, 20, 80], dtype=np.uint8)
        blob, metadata = RTMDetProcessing().preprocess(image)
        self.assertEqual(blob.shape, (1, 3, 640, 640))
        self.assertEqual(blob.dtype, np.float32)
        self.assertTrue(blob.flags.c_contiguous)
        np.testing.assert_array_equal(blob[0, :, 0, 0], [5, 20, 80])
        np.testing.assert_array_equal(blob[0, :, 239, 639], [5, 20, 80])
        np.testing.assert_array_equal(blob[0, :, 240, 0], [114, 114, 114])
        self.assertEqual(metadata["orig_shape"], (3, 8))

    def test_half_pixel_rounding_and_distinct_scale_factors(self):
        processor = RTMDetProcessing()
        _, metadata = processor.preprocess(np.zeros((1, 256, 3), dtype=np.uint8))
        self.assertEqual(metadata["scale_factor"], (2.5, 3.0))
        boxes, scores = processor.postprocess([np.array([[[0, 0, 640, 3, 0.9]]])], metadata)
        np.testing.assert_allclose(boxes, [[0, 0, 256, 1]])
        np.testing.assert_allclose(scores, [0.9])

    def test_nms_uses_padded_bounds_before_original_image_clipping(self):
        processor = RTMDetProcessing()
        _, metadata = processor.preprocess(np.zeros((640, 320, 3), dtype=np.uint8))
        output = np.array([[[0, 0, 640, 100, 0.9], [0, 0, 320, 100, 0.8]]], dtype=np.float32)
        boxes, scores = processor.postprocess([output], metadata)
        # IoU is 0.5 before final clipping, so both survive native NMS at 0.65.
        np.testing.assert_array_equal(boxes, [[0, 0, 320, 100], [0, 0, 320, 100]])
        np.testing.assert_allclose(scores, [0.9, 0.8])

    def test_overlapping_candidates_are_suppressed(self):
        processor = RTMDetProcessing()
        _, metadata = processor.preprocess(np.zeros((640, 640, 3), dtype=np.uint8))
        boxes, scores = processor.postprocess([np.array([[[10, 20, 50, 80, .8],
                                                         [10, 20, 50, 80, .9]]])], metadata)
        np.testing.assert_array_equal(boxes, [[10, 20, 50, 80]])
        np.testing.assert_allclose(scores, [.9])

    def test_empty_filtered_output_has_expected_shapes_and_dtype(self):
        processor = RTMDetProcessing(conf_threshold=0.5)
        _, metadata = processor.preprocess(np.zeros((640, 640, 3), dtype=np.uint8))
        output = np.array([[[1, 1, 5, 5, .5], [1, 1, 5, 5, np.inf],
                            [np.nan, 1, 5, 5, .9], [1, 1, 1, 5, .9]]])
        boxes, scores = processor.postprocess([output], metadata)
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(scores.shape, (0,))
        self.assertEqual(boxes.dtype, np.float32)
        self.assertEqual(scores.dtype, np.float32)

    def test_keeps_at_most_100_highest_scoring_detections(self):
        processor = RTMDetProcessing()
        _, metadata = processor.preprocess(np.zeros((640, 640, 3), dtype=np.uint8))
        rows = [[(i % 16)*10, (i // 16)*10, (i % 16)*10+5, (i // 16)*10+5, score]
                for i, score in enumerate(np.linspace(.99, .51, 101))]
        boxes, scores = processor.postprocess([np.array([rows])], metadata)
        self.assertEqual(len(boxes), 100)
        self.assertAlmostEqual(float(scores[0]), .99, places=6)
        self.assertTrue(np.all(scores[:-1] >= scores[1:]))

    def test_rejects_incompatible_export_and_options(self):
        for kwargs in [dict(imgsz=320), dict(conf_threshold=-.1), dict(iou_threshold=1.1),
                       dict(conf_threshold=float("nan"))]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RTMDetProcessing(**kwargs)
        with self.assertRaisesRegex(ValueError, "output"):
            RTMDetProcessing().postprocess([np.zeros((1, 5, 8400))], {})

    def test_rejects_empty_or_grayscale_input(self):
        for image in [np.zeros((0, 8, 3)), np.zeros((8, 8))]:
            with self.assertRaisesRegex(ValueError, "BGR"):
                RTMDetProcessing().preprocess(image)


if __name__ == "__main__":
    unittest.main()
