"""Shared, CPU-only contracts for the PTA-NTA training and cell-level evaluation."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

CLASSES = ["Non_tumor", "Tumor_CLDN18_positive", "Tumor_CLDN18_negative"]
PROBS = [f"prob_{c}" for c in CLASSES]
BBOX = ["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]
PATCH_RE = re.compile(r"^(?P<slide>.+)_x(?P<x>\d+)_y(?P<y>\d+)(?:_w\d+_h\d+)?$")
FRAME_COLUMNS = ["image_width", "image_height", "image_to_slide_scale_x", "image_to_slide_scale_y",
                 "annotation_frame", "spatial_frame"]


def read_csv(path):
    return pd.read_csv(path, dtype={k: str for k in ["slide_id", "patient_id", "cell_id", "patch_id"]}, low_memory=False)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)):
            return [clean(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, Path):
            return str(x)
        return x
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_columns(df, columns):
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Empty input table.")


def normalized_label(value):
    value = str(value).strip()
    aliases = {"Normal": CLASSES[0], "normal": CLASSES[0], "Non-tumor": CLASSES[0]}
    return aliases.get(value, value)


def assert_disjoint(parts):
    for name, left in parts.items():
        for other, right in parts.items():
            if name >= other:
                continue
            for col in ["patient_id", "slide_id", "patch_id", "cell_id"]:
                overlap = set(left[col]) & set(right[col])
                if overlap:
                    raise ValueError(f"{name}/{other} leakage in {col}: {sorted(overlap)[:5]}")


def validate_patch_frames(df, require_images=False):
    """patch_width/height are slide footprints; image_width/height bound local annotations."""
    spatial = ["patch_x", "patch_y", "patch_width", "patch_height"]
    require_columns(df, ["slide_id", "patch_id", *spatial])
    values = df[spatial].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :2] < 0).any() or (values[:, 2:] <= 0).any():
        raise ValueError("Invalid slide-coordinate patch footprints.")
    constant = ["slide_id", *spatial]
    if require_images or any(c in df for c in FRAME_COLUMNS):
        require_columns(df, [*FRAME_COLUMNS, *BBOX])
        numeric = df[[*FRAME_COLUMNS[:4], *BBOX]].to_numpy(float)
        if not np.isfinite(numeric).all() or (numeric[:, :4] <= 0).any():
            raise ValueError("Invalid image-frame dimensions, scales or annotations.")
        if (numeric[:, :2] != np.floor(numeric[:, :2])).any() or (numeric[:, 2:4] < 1).any():
            raise ValueError("Image dimensions must be integers; native mode does not reconstruct/upsample source detail.")
        if not df.annotation_frame.eq("image_pixels").all() or not df.spatial_frame.eq("slide_pixels").all():
            raise ValueError("Annotations must use image_pixels and footprints must use slide_pixels.")
        if (not np.allclose(df.image_width * df.image_to_slide_scale_x, df.patch_width, rtol=0, atol=1e-6)
                or not np.allclose(df.image_height * df.image_to_slide_scale_y, df.patch_height, rtol=0, atol=1e-6)):
            raise ValueError("Image-to-slide scale disagrees with the slide footprint; do not substitute JPEG dimensions.")
        if ((df.bbox_xmin < 0) | (df.bbox_ymin < 0) | (df.bbox_xmax > df.image_width) |
                (df.bbox_ymax > df.image_height) | (df.bbox_xmax <= df.bbox_xmin) |
                (df.bbox_ymax <= df.bbox_ymin)).any():
            raise ValueError("Cell boxes do not fit their native image coordinate frame.")
        constant += FRAME_COLUMNS
    if df.groupby("patch_id")[constant].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Inconsistent coordinate metadata within a patch.")


def cell_crop_box(row, padding):
    """Convert slide-pixel context to native pixels, leaving the annotated box unchanged."""
    if not np.isfinite(padding) or padding < 0:
        raise ValueError("Crop padding must be finite and nonnegative.")
    px, py = padding / row.image_to_slide_scale_x, padding / row.image_to_slide_scale_y
    return (max(0, int(np.floor(row.bbox_xmin - px))), max(0, int(np.floor(row.bbox_ymin - py))),
            min(int(row.image_width), int(np.ceil(row.bbox_xmax + px))),
            min(int(row.image_height), int(np.ceil(row.bbox_ymax + py))))


def load_study(directory):
    directory = Path(directory).resolve()
    meta = load_json(directory / "study.json")
    if sha256(directory / "cells.csv") != meta["cells_sha256"]:
        raise ValueError("cells.csv changed after study preparation. Create a new study directory.")
    df = read_csv(directory / "cells.csv")
    require_columns(df, ["cell_id", "patient_id", "slide_id", "patch_id", "cell_class", "cache_index"])
    if not df.cell_id.is_unique or set(df.cell_class) - set(CLASSES):
        raise ValueError("Duplicate IDs or invalid labels in study.")
    if df.patient_id.isna().any() or df.patient_id.str.strip().eq("").any():
        raise ValueError("Patient IDs must be supplied, not inferred from folder classes.")
    if meta.get("schema", 1) >= 2:
        validate_patch_frames(df, require_images=True)
    return df, meta


def load_split(directory, fold):
    df, meta = load_study(directory)
    split = load_json(Path(directory) / f"fold_{fold:02d}.json")
    if split["cells_sha256"] != meta["cells_sha256"]:
        raise ValueError("Split belongs to another cell table.")
    parts = {p: df[df.patient_id.isin(split[f"{p}_patients"])].copy() for p in ["train", "val", "test"]}
    if any(p.empty for p in parts.values()):
        raise ValueError("Every split must contain cells.")
    assert_disjoint(parts)
    if sum(map(len, parts.values())) != len(df):
        raise ValueError("Split does not cover the study exactly.")
    for p in ["train", "val"]:
        if set(parts[p].cell_class) != set(CLASSES):
            raise ValueError(f"{p} lacks a class. Redesign patient folds; never split cells instead.")
    return parts, meta, split


def prediction_metrics(df):
    """Fixed-label estimands; undefined full-class metrics stay missing, without sklearn warnings."""
    require_columns(df, ["true_class", "pred_class", *PROBS])
    y = pd.Categorical(df.true_class, categories=CLASSES).codes
    pred = pd.Categorical(df.pred_class, categories=CLASSES).codes
    p = df[PROBS].to_numpy(float)
    if (y < 0).any() or (pred < 0).any() or not np.isfinite(p).all():
        raise ValueError("Invalid prediction labels/probabilities.")
    if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-4):
        raise ValueError("Base probabilities must be a three-class distribution.")
    out = {"n_cells": len(df), "accuracy": float(np.mean(y == pred))}
    complete = len(np.unique(y)) == 3
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    out["macro_f1"] = float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)) if complete else None
    out["balanced_accuracy"] = float(np.mean(np.diag(cm) / cm.sum(1))) if complete else None
    out["base_probability_auc_ovr"] = float(roc_auc_score(y, p, labels=[0, 1, 2], multi_class="ovr")) if complete else None
    tumor = y != 0
    tasks = {
        "tumor": (tumor.astype(int), (pred != 0).astype(int), p[:, 1:].sum(1)),
        "pta": ((y[tumor] == 1).astype(int), (pred[tumor] == 1).astype(int), p[tumor, 1] / np.clip(p[tumor, 1:].sum(1), 1e-8, None)),
        "nta": ((y == 2).astype(int), (pred == 2).astype(int), df["nta_score"].to_numpy(float) if "nta_score" in df else p[:, 2]),
    }
    for name, (truth, hard, score) in tasks.items():
        both = len(np.unique(truth)) == 2
        # PTA has no observations for a patient with no reference tumor cells.
        c = confusion_matrix(truth, hard, labels=[0, 1]) if len(truth) else np.zeros((2, 2), dtype=np.int64)
        tn, fp, fn, tp = c.ravel()
        out.update({
            f"{name}_accuracy": float(np.mean(truth == hard)) if len(truth) else None,
            f"{name}_f1": float(f1_score(truth, hard, zero_division=0)) if both else None,
            f"{name}_balanced_accuracy": float((tp / (tp + fn) + tn / (tn + fp)) / 2) if both else None,
            f"{name}_macro_f1": float(f1_score(truth, hard, labels=[0, 1], average="macro", zero_division=0)) if both else None,
            f"{name}_auc": float(roc_auc_score(truth, score)) if both else None,
            f"{name}_ap": float(average_precision_score(truth, score)) if both else None,
            f"{name}_precision": float(tp / (tp + fp)) if tp + fp else None,
            f"{name}_recall": float(tp / (tp + fn)) if tp + fn else None,
            f"{name}_false_positive_rate": float(fp / (tn + fp)) if tn + fp else None,
            f"{name}_false_negative_rate": float(fn / (tp + fn)) if tp + fn else None,
            f"{name}_fp": int(fp), f"{name}_fn": int(fn),
        })
    return out


def matched_labels(p, nta_score, positive_threshold=0.5, negative_threshold=0.5):
    labels = np.zeros(len(p), dtype=int)
    labels[nta_score >= negative_threshold] = 2
    labels[p[:, 1] >= positive_threshold] = 1
    return np.asarray(CLASSES)[labels]


def prototype_scores(df, features, tau=0.8, beta=1.0, mode="cross_region", min_anchors=10, gap=0):
    """No reference labels are consulted. Cross-region excludes intersecting/nearby patch footprints."""
    if mode not in {"self", "leave_cell_out", "cross_region"} or not np.isfinite([beta, tau, gap]).all() or beta < 0 or not 0 <= tau <= 1 or min_anchors < 1 or gap < 0:
        raise ValueError("Invalid prototype settings.")
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or len(features) != len(df) or not np.isfinite(features).all():
        raise ValueError("Feature matrix must be finite and aligned with prediction rows.")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    z = features / np.maximum(norms, 1e-8)
    p = df[PROBS].to_numpy(float)
    similarity = np.zeros(len(df), dtype=float)
    available = np.zeros(len(df), dtype=bool)
    n_anchors = np.zeros(len(df), dtype=int)
    if mode == "cross_region":
        validate_patch_frames(df)
    for _, idx in df.groupby("slide_id", sort=False).indices.items():
        idx = np.asarray(idx)
        anchors = idx[p[idx, 1] >= tau]
        total = z[anchors].sum(0)
        if mode in {"self", "leave_cell_out"}:
            own = (p[idx, 1] >= tau) & (mode == "leave_cell_out")
            sums = total[None, :] - z[idx] * own[:, None]
            counts = len(anchors) - own.astype(int)
            length = np.linalg.norm(sums, axis=1)
            ok = (counts >= min_anchors) & (length > 1e-8)
            similarity[idx] = np.sum(z[idx] * sums, axis=1) / np.maximum(length, 1e-8)
            available[idx], n_anchors[idx] = ok, counts
        else:
            # Work per patch rather than per cell to avoid a quadratic cell-by-cell matrix.
            a = df.iloc[anchors]
            for patch, query in df.iloc[idx].groupby("patch_id", sort=False):
                qi = idx[df.iloc[idx].patch_id.to_numpy() == patch]
                q = query.iloc[0]
                separated = ((a.patch_x + a.patch_width + gap <= q.patch_x) |
                             (q.patch_x + q.patch_width + gap <= a.patch_x) |
                             (a.patch_y + a.patch_height + gap <= q.patch_y) |
                             (q.patch_y + q.patch_height + gap <= a.patch_y))
                kept = anchors[separated.to_numpy() & a.patch_id.ne(patch).to_numpy()]
                n_anchors[qi] = len(kept)
                vec = z[kept].sum(0)
                length = np.linalg.norm(vec)
                if len(kept) >= min_anchors and length > 1e-8:
                    available[qi] = True
                    similarity[qi] = z[qi] @ (vec / length)
    neg = p[:, 2].copy()
    if beta:
        clipped = np.clip(neg, 1e-7, 1 - 1e-7)
        logits = np.log(clipped / (1 - clipped)) + beta * similarity
        neg[available] = 1 / (1 + np.exp(-np.clip(logits[available], -60, 60)))
    return neg, similarity, available, n_anchors
