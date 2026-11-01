"""Audit cell labels, lock patient-level outer/inner splits, and cache RGB crops once."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from tqdm import tqdm

from ptanta import (BBOX, CLASSES, PATCH_RE, FRAME_COLUMNS, assert_disjoint, cell_crop_box,
                            fingerprint, normalized_label, read_csv, require_columns, sha256,
                            validate_patch_frames, write_json)


def canonicalize(df, patient_map, patch_root=None):
    require_columns(df, ["slide_id", "patch_path", "cell_class", *BBOX])
    require_columns(patient_map, ["slide_id", "patient_id"])
    patient_map = patient_map.copy().reset_index(drop=True)
    for key in ["slide_id", "patient_id"]:
        values = patient_map[key].astype("string").str.strip()
        missing = values.isna() | values.eq("").fillna(False)
        if missing.any():
            positions = patient_map.index[missing].tolist()
            examples = "\n".join(
                f"  data row {i + 1}: slide_id={patient_map.at[i, 'slide_id']!r}"
                for i in positions[:20]
            )
            more = f"\n  ... and {len(positions) - 20} more row(s)." if len(positions) > 20 else ""
            raise ValueError(
                f"Patient map has {len(positions)} missing/blank {key} value(s). "
                f"CSV data rows are numbered from 1, excluding the header:\n{examples}{more}\n"
                "Fill these entries in the CSV supplied with --patient-map using de-identified "
                "IDs from actual case records. Empty fields and CSV missing-value tokens "
                "such as NA/NaN are read as missing. Slides from the same patient must share "
                "one patient_id; IDs are not assigned automatically."
            )
        patient_map[key] = values
    if not patient_map.slide_id.is_unique:
        raise ValueError("Patient map must have exactly one row per slide.")
    df = df.copy().reset_index(drop=True)
    df["slide_id"] = df.slide_id.astype(str).str.strip()
    df["patient_id"] = df.slide_id.map(patient_map.set_index("slide_id").patient_id)
    if df.patient_id.isna().any():
        raise ValueError(f"Patient mapping missing for slides: {df.loc[df.patient_id.isna(), 'slide_id'].unique().tolist()}")
    index = {}
    if patch_root:
        for p in Path(patch_root).rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                index.setdefault(p.name, []).append(p)
    paths, patches, xs, ys = [], [], [], []
    for row in df.itertuples():
        original = str(row.patch_path).replace("\\", "/")
        name = original.rsplit("/", 1)[-1]
        path = Path(original)
        if patch_root:
            choices = index.get(name, [])
            if len(choices) != 1:
                raise ValueError(f"Expected one {name} under --patch-root; found {len(choices)}. Do not mix resized/raw copies.")
            path = choices[0]
        if not path.is_file():
            raise FileNotFoundError(f"Missing patch {path}; use --patch-root with images matching the annotation frame.")
        match = PATCH_RE.match(Path(name).stem)
        if not match or match.group("slide") != row.slide_id:
            raise ValueError(f"Filename/slide_id mismatch: {name}, slide_id={row.slide_id}. Repair upstream metadata.")
        paths.append(str(path.resolve()))
        patches.append(Path(name).stem)
        xs.append(int(match.group("x")))
        ys.append(int(match.group("y")))
    df["patch_path"], df["patch_id"], df["patch_x"], df["patch_y"] = paths, patches, xs, ys
    df["cell_class"] = df.cell_class.map(normalized_label)
    if "reference_source" not in df:
        df["reference_source"] = "unverified"
    df["reference_source"] = df.reference_source.fillna("unverified")
    unresolved = df.cell_class.eq("Tumor")
    if unresolved.any():
        raise ValueError("Supply reviewed three-class cell labels; unsplit Tumor labels cannot define PTA/NTA targets.")
    if not df.cell_class.isin(CLASSES).all():
        raise ValueError(f"Unrecognized cell classes: {df.loc[~df.cell_class.isin(CLASSES), 'cell_class'].unique().tolist()}")
    if set(df.cell_class) != set(CLASSES):
        raise ValueError("All three classes are required across the study.")
    for key in BBOX:
        df[key] = pd.to_numeric(df[key], errors="raise")
    if not np.isfinite(df[BBOX].to_numpy()).all() or ((df.bbox_xmax <= df.bbox_xmin) | (df.bbox_ymax <= df.bbox_ymin)).any():
        raise ValueError("Nonfinite or empty cell bounding boxes.")
    if "cell_id" not in df:
        df["cell_id"] = [fingerprint([r.patch_id, *[getattr(r, c) for c in BBOX]])[:24] for r in df.itertuples()]
    if df.cell_id.isna().any() or df.cell_id.astype(str).str.strip().eq("").any() or not df.cell_id.is_unique:
        raise ValueError("Missing/duplicate cell IDs. Deduplicate annotations explicitly before preparation.")
    if df.duplicated(["patch_id", *BBOX]).any():
        raise ValueError("Duplicate cell bounding boxes within a patch; audit duplicate exports.")
    # Never carry probabilities or previous model outputs into a new training table.
    keep = ["cell_id", "patient_id", "slide_id", "patch_id", "patch_path", "patch_x", "patch_y", *BBOX,
            "cell_class", "reference_source", "cell_index", "geojson_path", "patch_group", "reviewer_id",
            "image_width", "image_height", "patch_width", "patch_height", "coordinate_frame_width", "coordinate_frame_height"]
    return df[[c for c in keep if c in df]].copy(), patient_map


def patient_class_counts(df):
    return (df.groupby(["patient_id", "cell_class"]).size().unstack(fill_value=0)
            .reindex(columns=CLASSES, fill_value=0).sort_index())


def write_split_audit(df, out, stage):
    counts = patient_class_counts(df)
    counts.to_csv(out / f"patient_class_counts_{stage}.csv")
    return {"n_cells": len(df), "n_patients": df.patient_id.nunique(),
            "n_patches": df.patch_id.nunique(), "class_cell_counts": counts.sum().to_dict(),
            "class_patient_counts": counts.gt(0).sum().to_dict()}


def constrained_patient_parts(counts, n_folds, seed, val_fraction):
    """Exact search for small cohorts, using class counts only, never model scores."""
    if len(counts) > 12:
        raise ValueError("constrained-patient supports at most 12 patients (exact search). "
                         "Use stratified-group for larger cohorts or prespecify another patient-level design.")
    # Seeded order breaks objective ties without depending on input cell-row order.
    patients = np.random.default_rng(seed).permutation(counts.index.to_numpy()).tolist()
    values = counts.loc[patients].to_numpy(dtype=np.int64)
    total = values.sum(axis=0)
    all_indices = tuple(range(len(patients)))

    @lru_cache(maxsize=None)
    def inner_for_test(test):
        dev = tuple(i for i in all_indices if i not in test)
        if len(dev) < 2 or ((values[list(dev)] > 0).sum(axis=0) < 2).any():
            return None
        dev_total = values[list(dev)].sum(axis=0)
        target = int(np.ceil(len(dev) * val_fraction))
        sizes = sorted(range(1, len(dev)), key=lambda k: (abs(k - target), k))
        for size in sizes:
            best_val, best_score = None, float("inf")
            for val in combinations(dev, size):
                val_counts = values[list(val)].sum(axis=0)
                if (val_counts == 0).any() or (dev_total - val_counts == 0).any():
                    continue
                score = float(np.square(val_counts / dev_total - size / len(dev)).sum())
                if score < best_score:
                    best_val, best_score = val, score
            if best_val is not None:
                return tuple(i for i in dev if i not in best_val), best_val, test
        return None

    small, extra = divmod(len(patients), n_folds)
    sizes = tuple(sorted([small + 1] * extra + [small] * (n_folds - extra)))
    best_parts, best_score = None, float("inf")

    def search(remaining, remaining_sizes, parts, score):
        nonlocal best_parts, best_score
        if not remaining_sizes:
            if score < best_score:
                best_parts, best_score = parts, score
            return
        # Anchor each next group at the first unassigned patient: no fold permutations.
        for size in sorted(set(remaining_sizes)):
            next_sizes = list(remaining_sizes)
            next_sizes.remove(size)
            for rest in combinations(remaining[1:], size - 1):
                test = (remaining[0], *rest)
                next_score = score + float(np.square(values[list(test)].sum(axis=0) / total - size / len(patients)).sum())
                if next_score >= best_score:
                    continue
                partition = inner_for_test(test)
                if partition is not None:
                    search(tuple(i for i in remaining if i not in test), tuple(next_sizes),
                           [*parts, partition], next_score)

    search(all_indices, sizes, [], 0.0)
    if best_parts is None:
        raise ValueError("No feasible constrained-patient split with balanced outer test sizes "
                         "(differing by at most one patient) and all classes in inner train/validation. "
                         "Exact search exhausted these constraints; do not fall back to splitting cells. "
                         "Review patient_class_counts_after_thinning.csv and prespecify an alternative design.")
    return [{name: [patients[i] for i in indices] for name, indices in zip(["train", "val", "test"], part)}
            for part in best_parts]


def make_folds(df, n_folds=3, seed=2026, val_fraction=0.25, strategy="stratified-group"):
    if n_folds < 2 or df.patient_id.nunique() < max(4, n_folds):
        raise ValueError("Need at least four mapped patients and at least as many patients as folds.")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one.")
    counts = patient_class_counts(df)
    support = counts.gt(0).sum()
    # Every outer development set needs two distinct carriers per class. In two-fold
    # CV the development sets are complementary, so four carriers are necessary.
    minimum = 4 if n_folds == 2 else 3
    scarce = support[support < minimum]
    if len(scarce):
        detail = "; ".join(f"{c}: {int(n)} patient(s) {counts.index[counts[c] > 0].tolist()}" for c, n in scarce.items())
        raise ValueError(f"Insufficient patient-level class support after overlap policy: {detail}. "
                         f"{n_folds}-fold outer CV with all classes in every inner train/validation needs "
                         f"at least {minimum} distinct patients per class (necessary, not sufficient). "
                         "More cells from the same patients cannot fix this. Review the before/after thinning "
                         "audit; annotate missing classes on additional patients if present, or prespecify "
                         "a different study design. No random-cell fallback.")
    if strategy == "constrained-patient":
        partitions = constrained_patient_parts(counts, n_folds, seed, val_fraction)
    elif strategy == "stratified-group":
        outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        partitions = []
        for fold, (dev_idx, test_idx) in enumerate(outer.split(df, df.cell_class, df.patient_id)):
            dev, test = df.iloc[dev_idx], df.iloc[test_idx]
            inner = GroupShuffleSplit(n_splits=500, test_size=val_fraction, random_state=seed + fold)
            pair = None
            for ti, vi in inner.split(dev, groups=dev.patient_id):
                train, val = dev.iloc[ti], dev.iloc[vi]
                if set(train.cell_class) == set(CLASSES) and set(val.cell_class) == set(CLASSES):
                    pair = train, val
                    break
            if pair is None:
                support_dev = patient_class_counts(dev).gt(0).sum().to_dict()
                raise ValueError(f"Fold {fold}: no full-class inner split found at fixed validation size "
                                 f"{int(np.ceil(dev.patient_id.nunique() * val_fraction))} of "
                                 f"{dev.patient_id.nunique()} development patients. Class patient support: {support_dev}. "
                                 "This random search does not prove infeasibility. For <=12 patients, explicitly use "
                                 "--split-strategy constrained-patient before training to search balanced outer folds "
                                 "and permit the nearest feasible validation size. No cell-level fallback.")
            train, val = pair
            partitions.append({name: group.patient_id.unique().tolist()
                               for name, group in {"train": train, "val": val, "test": test}.items()})
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
    result = []
    for fold, partition in enumerate(partitions):
        parts = {name: df[df.patient_id.isin(ids)] for name, ids in partition.items()}
        assert_disjoint(parts)
        if sum(len(g) for g in parts.values()) != len(df) or any(g.empty for g in parts.values()):
            raise ValueError("Invalid patient partition: incomplete or empty split.")
        if any(set(parts[name].cell_class) != set(CLASSES) for name in ["train", "val"]):
            raise ValueError("Invalid patient partition: train/validation lacks a class.")
        n_dev = parts["train"].patient_id.nunique() + parts["val"].patient_id.nunique()
        n_val = parts["val"].patient_id.nunique()
        result.append({"fold": fold, "split_seed": seed, "split_strategy": strategy,
                       "inner_validation": {"requested_fraction": val_fraction,
                                            "requested_patients": int(np.ceil(n_dev * val_fraction)),
                                            "actual_patients": n_val, "actual_fraction": n_val / n_dev},
                       **{f"{name}_patients": sorted(g.patient_id.unique()) for name, g in parts.items()},
                       "class_counts": {name: g.cell_class.value_counts().to_dict() for name, g in parts.items()}})
    held_out = [p for fold in result for p in fold["test_patients"]]
    if sorted(held_out) != sorted(df.patient_id.unique()):
        raise ValueError("Each patient must appear in outer test exactly once.")
    return result


def audit_patch_images(df, expected_size, *, image_frame="original", native_image_sizes=None, slide_footprint_size=None):
    """Inspect every image/ROI frame before computing WSI footprints or allocating crops."""
    native = image_frame == "native"
    if image_frame not in {"original", "native"}:
        raise ValueError("Unknown image frame.")
    allowed_sizes = set(native_image_sizes or [518])
    if native and (not slide_footprint_size or slide_footprint_size < 1 or
                   any(s < 1 or s > slide_footprint_size for s in allowed_sizes)):
        raise ValueError("Native mode needs positive image sizes no larger than the confirmed slide footprint.")
    records = []
    for patch_id, group in df.groupby("patch_id", sort=True):
        path = group.patch_path.iloc[0]
        named_size = re.search(r"_w(\d+)_h(\d+)$", patch_id)
        named_width, named_height = map(int, named_size.groups()) if named_size else (None, None)
        record = {"patient_id": group.patient_id.iloc[0], "slide_id": group.slide_id.iloc[0],
                  "patch_id": patch_id, "patch_path": path, "n_cells": len(group),
                  "image_width": None, "image_height": None, "expected_patch_size": None if native else expected_size,
                  "patch_width": None, "patch_height": None, "image_to_slide_scale_x": None,
                  "image_to_slide_scale_y": None, "image_sha256": "", "image_frame_mode": image_frame,
                  "filename_width": named_width, "filename_height": named_height,
                  "bbox_min_x": group.bbox_xmin.min(), "bbox_min_y": group.bbox_ymin.min(),
                  "bbox_max_x": group.bbox_xmax.max(), "bbox_max_y": group.bbox_ymax.max(),
                  "n_cells_outside_image": None, "status": "ok", "error": ""}
        issues = []
        if group.patch_path.nunique() != 1:
            issues.append("multiple_paths_for_patch_id")
        try:
            with Image.open(path) as image:
                width, height = image.size
            footprint_width, footprint_height = (slide_footprint_size, slide_footprint_size) if native else (width, height)
            record.update(image_width=width, image_height=height, patch_width=footprint_width, patch_height=footprint_height,
                          image_to_slide_scale_x=footprint_width / width, image_to_slide_scale_y=footprint_height / height,
                          image_sha256=sha256(path))
            if native and (width != height or width not in allowed_sizes):
                issues.append("unexpected_native_image_size")
            elif not native and (width != expected_size or height != expected_size):
                issues.append("unexpected_image_size")
            if named_size and (footprint_width != named_width or footprint_height != named_height):
                issues.append("filename_footprint_conflict" if native else "filename_size_differs_from_image")
            for col, measured in [("image_width", width), ("image_height", height),
                                  ("coordinate_frame_width", width), ("coordinate_frame_height", height),
                                  ("patch_width", footprint_width), ("patch_height", footprint_height)]:
                if col in group and not pd.to_numeric(group[col], errors="raise").eq(measured).all():
                    issues.append(f"declared_{col}_conflict")
            outside = ((group.bbox_xmin < 0) | (group.bbox_ymin < 0) |
                       (group.bbox_xmax > width) | (group.bbox_ymax > height))
            record["n_cells_outside_image"] = int(outside.sum())
            if outside.any():
                issues.append("cell_bounds_outside_image")
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            issues.append("image_header_error")
            record["error"] = str(error)
        record["status"] = ";".join(issues) if issues else "ok"
        records.append(record)
    return pd.DataFrame(records)


def attach_image_frames(df, audit):
    if not audit.status.eq("ok").all() or not audit.patch_id.is_unique or set(audit.patch_id) != set(df.patch_id):
        raise ValueError("Every patch must pass the coordinate audit before attaching slide footprints.")
    result = df.copy()
    plan = audit.set_index("patch_id")
    for col in [*FRAME_COLUMNS[:4], "patch_width", "patch_height", "image_sha256"]:
        result[col] = result.patch_id.map(plan[col])
    result["annotation_frame"], result["spatial_frame"] = "image_pixels", "slide_pixels"
    validate_patch_frames(result, require_images=True)
    return result


def remove_overlapping_patches(df):
    """Deterministic spatial thinning; do not select patches by labels or model performance."""
    validate_patch_frames(df)
    keep, rejected = [], []
    for slide, group in df.groupby("slide_id", sort=True):
        rectangles = []
        patches = group.drop_duplicates("patch_id").sort_values(["patch_y", "patch_x", "patch_id"])
        for row in patches.itertuples():
            x0, y0, x1, y1 = row.patch_x, row.patch_y, row.patch_x + row.patch_width, row.patch_y + row.patch_height
            overlap = any(x0 < rx1 and x1 > rx0 and y0 < ry1 and y1 > ry0 for rx0, ry0, rx1, ry1 in rectangles)
            if overlap:
                rejected.append({"slide_id": slide, "patch_id": row.patch_id, "reason": "intersects_previously_retained_patch"})
            else:
                keep.append(row.patch_id)
                rectangles.append((x0, y0, x1, y1))
    return df[df.patch_id.isin(keep)].copy(), pd.DataFrame(rejected, columns=["slide_id", "patch_id", "reason"])


def cache_crops(df, out, image_size, padding, workers, expected_size):
    validate_patch_frames(df, require_images=True)
    required_bytes = len(df) * image_size * image_size * 3
    if shutil.disk_usage(out).free < required_bytes + 1024 ** 3:
        raise ValueError(f"Crop cache needs {required_bytes / 1024**3:.1f} GiB plus 1 GiB free. Use local NVMe.")
    target = out / "crops.uint8.npy"
    tmp = out / "crops.incomplete.npy"
    cache = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(df), image_size, image_size, 3))
    groups = list(df.groupby("patch_id", sort=True))

    def process(item):
        patch_id, group = item
        path = group.patch_path.iloc[0]
        frame = group.iloc[0]
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            if (width, height) != (frame.image_width, frame.image_height):
                raise ValueError(f"{path}: image dimensions changed after the coordinate audit.")
            if expected_size and (width != expected_size or height != expected_size):
                raise ValueError(f"{path}: got {width}x{height}, expected original {expected_size}x{expected_size} coordinate frame.")
            for row in group.itertuples():
                if row.bbox_xmin < 0 or row.bbox_ymin < 0 or row.bbox_xmax > width or row.bbox_ymax > height:
                    raise ValueError(f"Cell {row.cell_id}: ROI outside image. Repair GeoJSON coordinates before training.")
                box = cell_crop_box(row, padding)
                cache[row.cache_index] = np.asarray(image.crop(box).resize((image_size, image_size), Image.Resampling.BICUBIC))
        digest = sha256(path)
        if digest != frame.image_sha256:
            raise ValueError(f"{path}: image content changed after the coordinate audit.")
        return {"patch_id": patch_id, "image_sha256": digest,
                "crop_padding_image_x": padding / frame.image_to_slide_scale_x,
                "crop_padding_image_y": padding / frame.image_to_slide_scale_y}

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(tqdm(pool.map(process, groups), total=len(groups), desc="Decode each patch once"))
        cache.flush()
    finally:
        del cache
    tmp.replace(target)
    dimensions = pd.DataFrame(records).set_index("patch_id")
    for key in dimensions:
        df[key] = df.patch_id.map(dimensions[key])
    return df, required_bytes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell-labels-csv", required=True)
    ap.add_argument("--patient-map", help="CSV with slide_id,patient_id. Missing option writes a blank template and exits.")
    ap.add_argument("--patch-root", help="Unique image tree matching the annotations, searched recursively by exact filename.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--split-seed", type=int, default=2026)
    ap.add_argument("--val-fraction", type=float, default=0.25)
    ap.add_argument("--split-strategy", choices=["stratified-group", "constrained-patient"], default="stratified-group",
                    help="constrained-patient: exact class-aware splits for <=12 patients; allows nearest feasible validation size")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--crop-padding", type=float, default=40, help="Context on each side in SLIDE pixels; converted to native image pixels per patch")
    ap.add_argument("--image-frame", choices=["original", "native"], default="original")
    ap.add_argument("--native-image-sizes", type=int, nargs="+", default=[518], help="Allowed square image sizes in native mode; e.g. 518 1024 for mixed matching pairs")
    ap.add_argument("--slide-footprint-size", type=int, help="Original square patch footprint in the same slide-coordinate units as filename x/y")
    ap.add_argument("--confirm-native-annotations", action="store_true",
                    help="Confirm annotations match each actual image and images are full-field resizes of the declared footprint, without crop/shift/rotation")
    ap.add_argument("--expected-patch-size", type=int, default=1024,
                    help="Required source image/annotation size, not the resized model input size")
    ap.add_argument("--image-audit-only", action="store_true",
                    help="Write per-patch image/ROI dimension reports; no thinning, splits or crop cache. Exit 2 on mismatches.")
    ap.add_argument("--overlap-policy", choices=["drop", "allow"], default="drop", help="Default removes intersecting patches without using labels")
    ap.add_argument("--cache-workers", type=int, default=8)
    args = ap.parse_args()
    if args.image_frame == "native":
        if not args.confirm_native_annotations or not args.slide_footprint_size or args.slide_footprint_size < 1:
            ap.error("Native mode requires --slide-footprint-size and --confirm-native-annotations; bounds alone cannot establish registration.")
        if any(s < 1 or s > args.slide_footprint_size for s in args.native_image_sizes):
            ap.error("Native image sizes must be positive and no larger than the slide footprint.")
    elif args.slide_footprint_size is not None or args.confirm_native_annotations:
        ap.error("Use --image-frame native with native annotation/footprint options.")
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "study.json").exists() or (out / "cells.csv").exists():
        raise ValueError("Study exists. Use a new directory; do not alter locked folds or labels in place.")
    if args.image_size < 14 or not np.isfinite(args.crop_padding) or args.crop_padding < 0 or args.cache_workers < 1 or args.expected_patch_size < 1:
        raise ValueError("Invalid crop/cache settings.")
    raw = read_csv(args.cell_labels_csv)
    if not args.patient_map:
        require_columns(raw, ["slide_id"])
        template = out / "patient_map_TEMPLATE.csv"
        if template.exists():
            raise SystemExit(
                f"Patient map template already exists: {template}. It was not overwritten. "
                "Complete its patient IDs, then re-run with --patient-map pointing to that CSV."
            )
        pd.DataFrame({"slide_id": sorted(raw.slide_id.unique()), "patient_id": ""}).to_csv(template, index=False, mode="x")
        raise SystemExit("Fill patient_map_TEMPLATE.csv using actual case records, then re-run with --patient-map. Slide IDs are NOT assumed to be patient IDs.")
    print(f"Reading patient map: {Path(args.patient_map).expanduser().resolve()}", flush=True)
    df, mapping = canonicalize(raw, read_csv(args.patient_map), args.patch_root)
    image_audit = audit_patch_images(df, args.expected_patch_size, image_frame=args.image_frame,
                                     native_image_sizes=args.native_image_sizes, slide_footprint_size=args.slide_footprint_size)
    image_audit.to_csv(out / "patch_image_audit.csv", index=False)
    bad_images = image_audit[image_audit.status.ne("ok")]
    dimensions = [{"width": int(w), "height": int(h), "n_patches": int(n)}
                  for (w, h), n in image_audit.dropna(subset=["image_width", "image_height"])
                  .groupby(["image_width", "image_height"]).size().items()]
    write_json(out / "image_audit.json", {"status": "failed" if len(bad_images) else "passed_dimension_checks",
               "n_patches": len(image_audit), "n_problem_patches": len(bad_images), "image_sizes": dimensions,
               "expected_patch_size": args.expected_patch_size if args.image_frame == "original" else None,
               "image_frame_mode": args.image_frame, "native_image_sizes": args.native_image_sizes if args.image_frame == "native" else None,
               "slide_footprint_size": args.slide_footprint_size, "native_annotations_confirmed": args.confirm_native_annotations,
               "source_sha256": sha256(args.cell_labels_csv),
               "limitations": ["Image headers and cell bounds are checked, not registration or full image decoding.",
                               "Bounds fitting an image do not prove that annotations use its coordinate frame.",
                               "No image resizing, coordinate scaling or label changes are performed by this audit."]})
    print(f"Image audit: {len(image_audit)} patches, {len(bad_images)} with dimension/path issues. "
          f"Measured sizes: {dimensions}. Report: {out / 'patch_image_audit.csv'}", flush=True)
    if len(bad_images):
        examples = "\n".join(f"  {r.patch_id}: {r.image_width}x{r.image_height}; {r.status}" for r in bad_images.head(5).itertuples())
        frame_description = (f"allowed native image sizes {args.native_image_sizes}, slide footprint {args.slide_footprint_size}"
                             if args.image_frame == "native" else f"original {args.expected_patch_size}x{args.expected_patch_size}")
        message = (f"Image/annotation frame preflight failed before spatial thinning or crop allocation:\n{examples}\n"
                   f"Review {out / 'patch_image_audit.csv'}. Required: {frame_description}; --image-size only controls final cell crops. "
                   "Check the actual images used for QuPath review and their matching annotation frame. "
                   "Do not globally rescale mixed-frame annotations or merely rename/upsample resized patches. "
                   "Use --patch-root only with images AND annotations in that same frame. "
                   "Native mode requires confirmed full-field resizes and a separate slide footprint; no automatic coordinate scaling is performed.")
        if args.image_audit_only:
            print(message, flush=True)
            raise SystemExit(2)
        raise ValueError(message)
    if args.image_audit_only:
        print("Dimension audit complete. No overlap exclusions, folds, crop cache, or study.json were generated.")
        return
    df = attach_image_frames(df, image_audit)
    audit = {"split_strategy": args.split_strategy, "n_folds": args.folds, "split_seed": args.split_seed,
             "requested_val_fraction": args.val_fraction, "overlap_policy": args.overlap_policy,
             "before_thinning": write_split_audit(df, out, "before_thinning")}
    if args.overlap_policy == "drop":
        df, excluded = remove_overlapping_patches(df)
        excluded.to_csv(out / "overlap_exclusions.csv", index=False)
        print(f"Spatial thinning removed {len(excluded)} overlapping patches.")
    df = df.sort_values(["patient_id", "slide_id", "patch_id", "cell_id"]).reset_index(drop=True)
    df["cache_index"] = np.arange(len(df))
    audit["after_thinning"] = write_split_audit(df, out, "after_thinning")
    df.groupby(["patient_id", "slide_id", "cell_class", "reference_source"]).size().rename("n_cells").reset_index().to_csv(out / "cohort_audit.csv", index=False)
    print(f"Patients carrying each class after overlap policy: {audit['after_thinning']['class_patient_counts']}", flush=True)
    try:
        folds = make_folds(df, args.folds, args.split_seed, args.val_fraction, args.split_strategy)
    except ValueError as error:
        write_json(out / "split_diagnostics.json", {**audit, "status": "failed", "error": str(error)})
        raise ValueError(f"{error}\nDiagnostics saved in {out / 'split_diagnostics.json'}; "
                         "inspect patient_class_counts_before_thinning.csv and patient_class_counts_after_thinning.csv.") from error
    write_json(out / "split_diagnostics.json", {**audit, "status": "feasible", "folds": folds})
    for fold in folds:
        size = fold["inner_validation"]
        print(f"Fold {fold['fold']}: train={len(fold['train_patients'])}, val={len(fold['val_patients'])}, "
              f"test={len(fold['test_patients'])} patients; validation requested={size['requested_patients']}, "
              f"actual={size['actual_patients']} ({size['actual_fraction']:.1%} of development patients)", flush=True)
    df, size = cache_crops(df, out, args.image_size, args.crop_padding, args.cache_workers,
                           args.expected_patch_size if args.image_frame == "original" else None)
    if (df.groupby("image_sha256").patient_id.nunique() > 1).any():
        raise ValueError("Identical patch file content found under different patients. Audit duplicate images before training.")
    df.to_csv(out / "cells.csv", index=False)
    mapping.to_csv(out / "patient_map.csv", index=False)
    digest = sha256(out / "cells.csv")
    for fold in folds:
        write_json(out / f"fold_{fold['fold']:02d}.json", {**fold, "cells_sha256": digest})
    df.groupby(["image_width", "image_height", "patient_id", "cell_class"]).size().rename("n_cells").reset_index().to_csv(out / "resolution_class_counts.csv", index=False)
    write_json(out / "study.json", {"schema": 2, "cells_sha256": digest, "source_sha256": sha256(args.cell_labels_csv),
               "patient_map_sha256": sha256(out / "patient_map.csv"), "n_cells": len(df), "n_patients": df.patient_id.nunique(),
               "n_slides": df.slide_id.nunique(), "n_folds": args.folds, "image_size": args.image_size,
               "crop_padding": args.crop_padding, "crop_padding_units": "slide_pixels", "cache_bytes": size, "classes": CLASSES,
               "image_frame_mode": args.image_frame, "audited_image_sizes": dimensions,
               "coordinate_system": {"annotations": "image_pixels", "patch_positions_and_footprints": "slide_pixels",
                                     "image_to_slide": "patch_width/image_width and patch_height/image_height; full-field resize only"},
               "prepare_script_sha256": sha256(__file__), "contracts_sha256": sha256(Path(__file__).with_name("ptanta.py")),
               "overlap_policy": args.overlap_policy,
               "reference_sources": df.reference_source.value_counts().to_dict(), "preparation": vars(args),
               "limitations": ["Selected detections are not exhaustive whole-slide counts.",
                               "Overlapping patches can duplicate biological cells; no cross-patch instance deduplication is claimed.",
                               "Unverified/stain-heuristic labels are not independent membrane-staining references.",
                               "Native mode relies on user-confirmed annotation frames and full-field correspondence, not image registration.",
                               "Mixed source resolutions retain different image detail; see resolution_class_counts.csv. No lost detail is reconstructed."]})
    print(f"Prepared {len(df)} cells, {df.patient_id.nunique()} patients, {len(folds)} locked folds in {out}")


if __name__ == "__main__":
    main()
