"""Summarize complete held-out predictions; use patients, not cells, for paired intervals."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import VARIANTS
from ptanta import (BBOX, FRAME_COLUMNS, load_json, load_split, load_study,
                    prediction_metrics, read_csv, sha256, validate_patch_frames, write_json)

TASKS = {"tumor": "Tumor AI", "pta": "PTA (reference tumor cells)", "nta": "NTA (all cells)"}
METRICS = {"accuracy": "Accuracy", "f1": "Binary F1", "auc": "AUROC", "ap": "Average precision"}
ENDPOINTS = [f"{task}_{metric}" for task in TASKS for metric in METRICS]


def paired_interval(differences, n_bootstrap, seed):
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    result = {"n_patients": len(values), "mean_delta": float(values.mean()) if len(values) else None,
              "ci_low": None, "ci_high": None}
    if len(values) >= 3:
        rng = np.random.default_rng(seed)
        means = values[rng.integers(0, len(values), (n_bootstrap, len(values)))].mean(axis=1)
        result.update(ci_low=float(np.quantile(means, 0.025)), ci_high=float(np.quantile(means, 0.975)))
    return result


def check_predictions(df, expected):
    if not df.cell_id.is_unique or set(df.cell_id) != set(expected.cell_id):
        raise ValueError("Incorrect or duplicate test-cell coverage.")
    ordered = expected.set_index("cell_id").loc[df.cell_id]
    for column in ["patient_id", "slide_id", "patch_id"]:
        if not np.array_equal(ordered[column].to_numpy(), df[column].to_numpy()):
            raise ValueError(f"Changed {column} in predictions.")
    if not np.array_equal(ordered.cell_class.to_numpy(), df.true_class.to_numpy()):
        raise ValueError("Prediction labels differ from the prepared reference.")
    validate_patch_frames(df, require_images=True)
    for column in ["patch_x", "patch_y", "patch_width", "patch_height", *FRAME_COLUMNS[:4], *BBOX]:
        if not np.allclose(ordered[column].to_numpy(float), df[column].to_numpy(float), rtol=0, atol=1e-10):
            raise ValueError(f"Changed annotation/slide coordinates: {column}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-root", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--include-smoke-tests", action="store_true")
    args = ap.parse_args()
    if args.n_bootstrap < 2:
        ap.error("At least two bootstrap samples are required.")
    root = Path(args.experiment_root).resolve()
    plan = load_json(root / "experiment_plan.json")
    cells, metadata = load_study(plan["study"])
    if metadata["cells_sha256"] != plan["cells_sha256"]:
        raise ValueError("Experiment plan and prepared cells differ.")
    jobs = plan["jobs"]
    models = {m["name"] for m in plan["protocol"]["models"]}
    wanted = {(m, f, s) for m in models for f in range(metadata["n_folds"]) for s in plan["seeds"]}
    actual = [(j["model"], j["fold"], j["seed"]) for j in jobs]
    if len(actual) != len(set(actual)) or set(actual) != wanted:
        raise ValueError("Missing or duplicate planned fold/seed jobs.")
    pieces, protocol_settings, input_hashes = {}, {}, {}
    for job in jobs:
        directory = Path(job["directory"])
        run = load_json(directory / "run.json")
        trained = load_json(directory / "complete.json")
        evaluation = load_json(directory / "evaluation/complete.json")
        if not run["paper_eligible"] and not args.include_smoke_tests:
            raise ValueError("Synthetic test results require --include-smoke-tests.")
        if trained["identity"] != run["identity"] or evaluation["run_identity"] != run["identity"]:
            raise ValueError("Mismatched run identities.")
        if evaluation["variants"] != VARIANTS:
            raise ValueError("Expected the same guided and unguided variants in every run.")
        parts, _, split = load_split(plan["study"], job["fold"])
        if run["split"] != split or run["settings"]["seed"] != job["seed"]:
            raise ValueError("Run does not match its assigned fold/seed.")
        for filename, digest in evaluation["files_sha256"].items():
            if sha256(directory / "evaluation" / filename) != digest:
                raise ValueError(f"Changed evaluation output: {directory / filename}")
        settings = load_json(directory / "evaluation/protocol.json")["settings"]
        settings = {k: v for k, v in settings.items() if k != "run_dir"}
        if job["model"] in protocol_settings and protocol_settings[job["model"]] != settings:
            raise ValueError("Do not mix different prototype settings.")
        protocol_settings[job["model"]] = settings
        for variant in VARIANTS:
            path = directory / "evaluation" / f"{variant}_test.csv.gz"
            df = read_csv(path)
            check_predictions(df, parts["test"])
            pieces.setdefault((job["model"], job["seed"], variant), []).append(df)
            input_hashes[path.relative_to(root).as_posix()] = sha256(path)
    pooled, patients = [], []
    for (model, seed, variant), frames in pieces.items():
        frame = pd.concat(frames, ignore_index=True)
        check_predictions(frame, cells)
        record = {"model": model, "seed": seed, "variant": variant}
        pooled.append({**record, **prediction_metrics(frame)})
        for patient, group in frame.groupby("patient_id"):
            patients.append({**record, "patient_id": patient, **prediction_metrics(group)})
    pooled, patients = pd.DataFrame(pooled), pd.DataFrame(patients)
    summary, contrasts = [], []
    for (model, variant), group in pooled.groupby(["model", "variant"]):
        if set(group.seed) != set(plan["seeds"]):
            raise ValueError("Incomplete seed coverage.")
        for endpoint in ENDPOINTS:
            values = group[endpoint].dropna()
            summary.append({"model": model, "variant": variant, "metric": endpoint,
                            "mean": values.mean(), "sd": values.std(ddof=1),
                            "n_defined_seeds": len(values), "n_planned_seeds": len(plan["seeds"])})
    units = patients.groupby(["model", "variant", "patient_id"])[ENDPOINTS].mean()
    for model in sorted(models):
        baseline = units.loc[(model, "unguided")]
        guided = units.loc[(model, "prototype_guided")]
        for metric in METRICS:
            endpoint = f"nta_{metric}"
            paired = pd.concat([baseline[endpoint], guided[endpoint]], axis=1, keys=["base", "guided"]).dropna()
            contrasts.append({"model": model, "metric": endpoint,
                              "n_patients_undefined": metadata["n_patients"] - len(paired),
                              **paired_interval(paired.guided - paired.base, args.n_bootstrap, args.seed)})
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(out / "metrics_by_seed.csv", index=False)
    patients.to_csv(out / "patient_metrics.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "metric_summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "guidance_difference.csv", index=False)
    write_json(out / "summary.json", {
        "n_cells": len(cells), "n_patients": metadata["n_patients"], "n_jobs": len(jobs),
        "seeds": plan["seeds"], "protocols": protocol_settings, "prediction_sha256": input_hashes,
        "synthetic": args.include_smoke_tests, "scope": "Internal patient-held-out cell classification",
        "interval": "Exploratory patient-paired percentile bootstrap after averaging seeds within patient.",
        "limitations": ["Seeds and cells are not independent patients.", "OOF models share training patients.",
                       "Missing-class AUROC/AP/F1 remain undefined.", "This is not whole-slide clinical validation."]})
    print(f"Wrote complete cell-level summaries: {out}")


if __name__ == "__main__":
    main()
