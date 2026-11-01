"""Evaluate unguided and prototype-guided PTA-NTA on identical held-out cells."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ptanta import (PROBS, fingerprint, load_json, matched_labels, prediction_metrics,
                    prototype_scores, read_csv, sha256, validate_patch_frames, write_json)

VARIANTS = ["unguided", "prototype_guided"]


def read_predictions(directory, split, complete):
    report = complete["reports"][split]
    for suffix, key in [("predictions.csv.gz", "predictions_sha256"),
                        ("features.npy", "features_sha256")]:
        if sha256(directory / f"{split}_{suffix}") != report[key]:
            raise ValueError(f"Changed {split} export: {suffix}")
    df = read_csv(directory / f"{split}_predictions.csv.gz")
    features = np.load(directory / f"{split}_features.npy", mmap_mode="r")
    if fingerprint(df.cell_id.tolist()) != report["cell_ids_sha256"] or len(features) != len(df):
        raise ValueError("Features and cell records are not aligned.")
    validate_patch_frames(df, require_images=True)
    prediction_metrics(df)
    return df, features


def apply_policy(df, features, args, guided):
    scores, similarity, available, anchors = prototype_scores(
        df, features, tau=args.tau, beta=args.beta if guided else 0,
        mode=args.mode, min_anchors=args.min_anchors, gap=args.region_gap)
    result = df.copy()
    result["nta_score"] = scores
    result["pred_class"] = matched_labels(
        result[PROBS].to_numpy(), scores, args.positive_threshold, args.negative_threshold)
    result["prototype_available"] = available if guided else False
    result["prototype_similarity"] = np.where(available, similarity, np.nan) if guided else np.nan
    result["eligible_anchor_count"] = anchors if guided else 0
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", choices=["self", "leave_cell_out", "cross_region"], default="self")
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--min-anchors", type=int, default=10)
    ap.add_argument("--region-gap", type=float, default=0)
    ap.add_argument("--positive-threshold", type=float, default=0.5)
    ap.add_argument("--negative-threshold", type=float, default=0.5)
    args = ap.parse_args()
    numeric = [args.tau, args.beta, args.region_gap, args.positive_threshold, args.negative_threshold]
    if (not np.isfinite(numeric).all() or not 0 <= args.tau <= 1 or args.beta < 0
            or args.min_anchors < 1 or args.region_gap < 0
            or not 0 <= args.positive_threshold <= 1 or not 0 <= args.negative_threshold <= 1):
        ap.error("Invalid prototype or decision thresholds.")
    directory = Path(args.run_dir).resolve()
    run, complete = load_json(directory / "run.json"), load_json(directory / "complete.json")
    if run["identity"] != complete["identity"]:
        raise ValueError("Training identity mismatch.")
    out = directory / "evaluation"
    protocol = {"run_identity": run["identity"], "settings": vars(args),
                "code_sha256": sha256(__file__), "core_sha256": sha256(Path(__file__).with_name("ptanta.py")),
                "selection": "Fixed settings; no test-label tuning.",
                "anchors": "Predicted positive tumor from the same slide; reference labels are not used."}
    identity = fingerprint(protocol)
    if (out / "complete.json").exists():
        previous = load_json(out / "complete.json")
        if previous["identity"] != identity:
            raise ValueError("Evaluation settings changed. Use a new experiment directory.")
        for name, digest in previous["files_sha256"].items():
            if sha256(out / name) != digest:
                raise ValueError(f"Changed evaluation output: {name}")
        print(f"Already complete: {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "protocol.json", protocol)
    rows, patients, files, split_patients = [], [], {}, {}
    for split in ["val", "test"]:
        df, features = read_predictions(directory, split, complete)
        split_patients[split] = set(df.patient_id)
        if split == "test" and split_patients["val"] & split_patients["test"]:
            raise ValueError("Validation/test patient overlap.")
        for variant in VARIANTS:
            result = apply_policy(df, features, args, variant == "prototype_guided")
            rows.append({"split": split, "variant": variant, **prediction_metrics(result),
                         "prototype_coverage": float(result.prototype_available.mean())})
            if split == "test":
                filename = f"{variant}_test.csv.gz"
                result.to_csv(out / filename, index=False)
                files[filename] = sha256(out / filename)
                for patient, group in result.groupby("patient_id"):
                    patients.append({"variant": variant, "patient_id": patient, **prediction_metrics(group)})
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(patients).to_csv(out / "patient_metrics.csv", index=False)
    for name in ["metrics.csv", "patient_metrics.csv", "protocol.json"]:
        files[name] = sha256(out / name)
    write_json(out / "complete.json", {"identity": identity, "run_identity": run["identity"],
                                      "variants": VARIANTS, "files_sha256": files})
    print(f"Wrote cell-level evaluation: {out}")


if __name__ == "__main__":
    main()
