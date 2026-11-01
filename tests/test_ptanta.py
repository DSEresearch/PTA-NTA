"""Focused tests for PTA-NTA scores, patient splits and coordinate handling."""
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ptanta import (CLASSES, PROBS, assert_disjoint, cell_crop_box, matched_labels,
                    prediction_metrics, prototype_scores, validate_patch_frames)
from prepare_data import (attach_image_frames, audit_patch_images, cache_crops, canonicalize,
                          make_folds, remove_overlapping_patches)
from summarize import check_predictions, paired_interval


def predictions():
    p = np.array([[.9, .05, .05], [.05, .9, .05], [.5, .1, .4], [.1, .1, .8]])
    df = pd.DataFrame(p, columns=PROBS)
    df["true_class"] = [CLASSES[i] for i in [0, 1, 2, 2]]
    df["pred_class"] = matched_labels(p, p[:, 2])
    df["nta_score"] = p[:, 2]
    df["slide_id"] = ["s"] * 4
    df["patch_id"] = ["p0", "p0", "p1", "p2"]
    df["patch_x"] = [0, 0, 512, 2048]
    df["patch_y"] = 0
    df["patch_width"] = df["patch_height"] = 1024
    return df


class PTANTA(unittest.TestCase):
    def test_binary_metrics(self):
        result = prediction_metrics(predictions())
        self.assertAlmostEqual(result["nta_accuracy"], .75)
        self.assertAlmostEqual(result["nta_f1"], 2 / 3)
        self.assertAlmostEqual(result["nta_auc"], 1)
        self.assertAlmostEqual(result["nta_ap"], 1)

    def test_missing_class_is_undefined_without_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            result = prediction_metrics(predictions().iloc[:1])
        self.assertEqual(len(caught), 0)
        self.assertIsNone(result["nta_auc"])
        self.assertIsNone(result["pta_accuracy"])
        self.assertIsNone(result["nta_f1"])

    def test_bad_probabilities_rejected(self):
        df = predictions()
        df.loc[0, PROBS[0]] = 2
        with self.assertRaises(ValueError):
            prediction_metrics(df)

    def test_beta_zero_is_exact_identity(self):
        df = predictions()
        score, *_ = prototype_scores(df, np.eye(4), beta=0, min_anchors=1, mode="self")
        np.testing.assert_array_equal(score, df[PROBS[2]])

    def test_insufficient_anchors_fallback(self):
        df = predictions()
        score, _, available, _ = prototype_scores(df, np.eye(4), min_anchors=10, mode="self")
        np.testing.assert_array_equal(score, df[PROBS[2]])
        self.assertFalse(available.any())

    def test_zero_features_fallback(self):
        df = predictions()
        score, _, available, _ = prototype_scores(df, np.zeros((4, 2)), min_anchors=1, mode="self")
        np.testing.assert_array_equal(score, df[PROBS[2]])
        self.assertFalse(available.any())

    def test_guidance_does_not_use_reference_labels(self):
        df = predictions()
        z = np.ones((4, 2))
        before = prototype_scores(df, z, min_anchors=1, mode="self")[0]
        df["true_class"] = "Unknown"
        after = prototype_scores(df, z, min_anchors=1, mode="self")[0]
        np.testing.assert_array_equal(before, after)
        self.assertGreater(after[-1], df[PROBS[2]].iloc[-1])

    def test_cross_region_uses_original_footprint(self):
        _, _, available, count = prototype_scores(
            predictions(), np.ones((4, 2)), min_anchors=1, mode="cross_region")
        self.assertEqual(available.tolist(), [False, False, False, True])
        self.assertEqual(count.tolist(), [0, 0, 0, 1])

    def test_leave_cell_out(self):
        _, _, available, _ = prototype_scores(
            predictions(), np.ones((4, 2)), min_anchors=1, mode="leave_cell_out")
        self.assertFalse(available[1])
        self.assertTrue(available[2])

    def test_invalid_prototype_arguments(self):
        for kwargs in [{"beta": np.nan}, {"tau": 2}, {"min_anchors": 0}, {"gap": -1}]:
            with self.assertRaises(ValueError):
                prototype_scores(predictions(), np.ones((4, 2)), **kwargs)

    def test_positive_class_override(self):
        p = np.array([[.1, .8, .1]])
        self.assertEqual(matched_labels(p, np.array([.9]))[0], CLASSES[1])

    def test_patient_overlap_rejected(self):
        frame = pd.DataFrame({"patient_id": ["p"], "slide_id": ["s"], "patch_id": ["x"], "cell_id": ["c"]})
        with self.assertRaises(ValueError):
            assert_disjoint({"train": frame, "test": frame})

    def test_grouped_folds_cover_patients_once(self):
        rows = [{"patient_id": str(i), "slide_id": str(i), "patch_id": f"p{i}", "cell_id": f"{i}-{j}",
                 "cell_class": label} for i in range(6) for j, label in enumerate(CLASSES)]
        df = pd.DataFrame(rows)
        folds = make_folds(df, 3, 2026, .25, "constrained-patient")
        test_patients = [p for fold in folds for p in fold["test_patients"]]
        self.assertEqual(len(test_patients), 6)
        self.assertEqual(len(set(test_patients)), 6)
        for fold in folds:
            self.assertFalse(set(fold["train_patients"]) & set(fold["test_patients"]))
            self.assertFalse(set(fold["val_patients"]) & set(fold["test_patients"]))

    def test_patient_interval(self):
        self.assertIsNone(paired_interval([.1, .2], 50, 17)["ci_low"])
        result = paired_interval([0, 0, 0], 50, 17)
        self.assertEqual(result["mean_delta"], 0)
        self.assertEqual(result["ci_high"], 0)

    def test_native_image_crop_and_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "s_x0_y0_w1024_h1024.png"
            Image.new("RGB", (518, 518), "white").save(path)
            rows = [{"slide_id": "s", "patch_path": str(path), "cell_class": c,
                     "bbox_xmin": 100 + i * 20, "bbox_ymin": 100,
                     "bbox_xmax": 110 + i * 20, "bbox_ymax": 110} for i, c in enumerate(CLASSES)]
            frame, _ = canonicalize(pd.DataFrame(rows), pd.DataFrame({"slide_id": ["s"], "patient_id": ["p"]}))
            audit = audit_patch_images(frame, 1024, image_frame="native",
                                       native_image_sizes=[518], slide_footprint_size=1024)
            frame = attach_image_frames(frame, audit)
            frame["cache_index"] = np.arange(len(frame))
            validate_patch_frames(frame, require_images=True)
            self.assertEqual(cell_crop_box(frame.iloc[0], 40), (79, 79, 131, 131))
            cropped, _ = cache_crops(frame, root, 28, 40, 1, None)
            self.assertEqual(np.load(root / "crops.uint8.npy").shape, (3, 28, 28, 3))
            self.assertTrue(cropped.patch_width.eq(1024).all())
            wrong = frame.copy()
            wrong["image_to_slide_scale_x"] = 1
            with self.assertRaises(ValueError):
                validate_patch_frames(wrong, require_images=True)

    def test_duplicate_prediction_coverage_rejected(self):
        df = pd.DataFrame({"cell_id": ["c", "c"]})
        with self.assertRaises(ValueError):
            check_predictions(df, df)


if __name__ == "__main__":
    unittest.main()
