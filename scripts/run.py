"""Schedule independent folds/seeds, one process per GPU, without NCCL/DDP."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import yaml

from ptanta import fingerprint, load_json, load_split, load_study, write_json


def cli_args(values):
    result = []
    for key, value in values.items():
        key = "--" + key.replace("_", "-")
        if value is True:
            result.append(key)
        elif value is not False and value is not None:
            result.append(key)
            result.extend(map(str, value if isinstance(value, list) else [value]))
    return result


def validate_training_assets(settings, context):
    """Check local inputs without importing torch, downloading or loading large weights."""
    settings = {k.replace("_", "-"): v for k, v in settings.items()}
    encoder = settings.get("encoder", "dinov2")
    if encoder == "tiny":
        if not settings.get("smoke-test"):
            raise ValueError("Tiny encoder requires smoke-test: true; it is not a paper model.")
        return
    problems = []

    def check_file(value, key):
        if not value:
            problems.append(f"{key}: not configured")
            return
        path = Path(value)
        try:
            with path.open("rb") as stream:
                if not stream.read(1):
                    problems.append(f"{key}: empty file: {path.resolve()}")
        except OSError as error:
            problems.append(f"{key}: cannot read {path.resolve()}: {error}")

    check_file(settings.get("encoder-weights"), "encoder-weights")
    if encoder == "dinov2":
        repo = settings.get("dinov2-repo")
        check_file(Path(repo) / "hubconf.py" if repo else None, "dinov2-repo/hubconf.py")
    for key in ["ssl-provenance"]:
        if settings.get(key):
            check_file(settings[key], key)
    if problems:
        raise FileNotFoundError(
            f"Model asset preflight failed for {context}:\n  " + "\n  ".join(problems) +
            "\nNo GPU jobs were started and the experiment plan was not changed. "
            "Provide the local files named in the YAML (absolute paths recommended). "
            "Use a matching model-name and checkpoint; workers never download weights. "
            "If you change the YAML paths after a plan was locked, use a new output_root; "
            "the prepared study/cache can be reused."
        )


def make_plan(cfg, validate_assets=False):
    study = Path(cfg["study"]).expanduser().resolve()
    root = Path(cfg["output_root"]).expanduser().resolve()
    if not (study / "study.json").is_file():
        if (study / "cells.csv").is_file():
            action = (
                "cells.csv exists but study.json does not: preparation or transfer is incomplete. "
                "Check prepare_data.py's last error. Transfer the complete prepared directory, or fix "
                "the preparation error and run prepare_data.py with a NEW --out-dir."
            )
        elif (study / "patient_map_TEMPLATE.csv").is_file():
            action = (
                "Only the patient-map template stage may have finished. Fill patient IDs from "
                "actual case records, then run scripts/prepare_data.py with "
                "--patient-map pointing to the completed CSV and --out-dir pointing here."
            )
        else:
            action = (
                "Run scripts/prepare_data.py first with your cell-label CSV, "
                "patient map and matching image/annotation pairs, using --out-dir for this directory."
            )
        raise FileNotFoundError(
            f"Prepared study not found: {study / 'study.json'}\n{action}\n"
            "If prepare_data.py already succeeded elsewhere, set 'study:' in the experiment YAML "
            "to that prepared DIRECTORY (prefer an absolute path). Relative paths are resolved "
            "from the shell working directory. run.py does not create the study; do not "
            "create an empty study.json."
        )
    _, metadata = load_study(study)
    seeds = cfg.get("seeds", [17, 29, 43])
    models = cfg["models"]
    if not models or not seeds or len(set(seeds)) != len(seeds) or len({m["name"] for m in models}) != len(models):
        raise ValueError("Models/seeds must be nonempty and unique.")
    jobs = []
    scripts = Path(__file__).resolve().parent
    for model in models:
        if set(model) - {"name", "arguments"}:
            raise ValueError("Each model supports only name and arguments; check YAML keys.")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", model["name"]):
            raise ValueError("Model names may contain only letters, digits, underscores and hyphens.")
        for fold in range(metadata["n_folds"]):
            load_split(study, fold)
            for seed in seeds:
                settings = {**cfg.get("training", {}), **model.get("arguments", {})}
                if any(k.replace("_", "-") in {"study", "fold", "seed", "out-dir", "resume"} for k in settings):
                    raise ValueError("Model arguments cannot override the locked study, fold, seed or output directory.")
                settings = {k: v.format(fold=fold, seed=seed) if isinstance(v, str) else v for k, v in settings.items()}
                if validate_assets:
                    validate_training_assets(settings, f"{model['name']} fold={fold} seed={seed}")
                directory = root / model["name"] / f"fold_{fold:02d}" / f"seed_{seed}"
                train = [sys.executable, str(scripts / "train.py"), "--study", str(study),
                         "--fold", str(fold), "--seed", str(seed), "--out-dir", str(directory), "--resume", *cli_args(settings)]
                evaluate = [sys.executable, str(scripts / "evaluate.py"), "--run-dir", str(directory),
                            *cli_args(cfg.get("prototype", {}))]
                jobs.append({"model": model["name"], "fold": fold, "seed": seed, "directory": str(directory),
                             "commands": [train, evaluate]})
    return {"study": str(study), "output_root": str(root), "cells_sha256": metadata["cells_sha256"],
            "seeds": seeds, "n_folds": metadata["n_folds"], "jobs": jobs,
            "protocol": cfg, "identity": fingerprint({"config": cfg, "study": metadata})}


def stop_children(active):
    for item in active:
        p = item["process"]
        if p.poll() is None:
            if os.name == "posix":
                os.killpg(p.pid, signal.SIGTERM)
            else:
                p.terminate()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(x["process"].poll() is None for x in active):
        time.sleep(0.2)
    for item in active:
        p = item["process"]
        if p.poll() is None:
            if os.name == "posix":
                os.killpg(p.pid, signal.SIGKILL)
            else:
                p.kill()
        p.wait()
        item["log"].close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="One independent job per listed CUDA device")
    ap.add_argument("--dry-run", action="store_true", help="Validate folds and write the locked run plan without starting training")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    plan = make_plan(cfg, validate_assets=True)
    root = Path(plan["output_root"])
    if (root / "experiment_plan.json").exists() and load_json(root / "experiment_plan.json")["identity"] != plan["identity"]:
        raise ValueError("An incompatible experiment plan exists. Use a new output_root.")
    write_json(root / "experiment_plan.json", plan)
    print(f"Locked {len(plan['jobs'])} jobs ({plan['n_folds']} folds x {len(plan['seeds'])} seeds x {len(cfg['models'])} models).", flush=True)
    if args.dry_run:
        print(f"Plan: {root / 'experiment_plan.json'}")
        return
    free = [v.strip() for v in args.gpus.split(",") if v.strip()]
    if not free or len(set(free)) != len(free):
        raise ValueError("Provide distinct GPU IDs.")
    pending = list(plan["jobs"])
    active, failures = [], []

    def launch(job, gpu, phase):
        directory = Path(job["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        log = open(directory / ("train.log" if phase == 0 else "prototype.log"), "a", encoding="utf-8")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(job["commands"][phase], stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=os.name == "posix")
        print(f"GPU={gpu} PID={proc.pid} phase={phase} {job['model']} fold={job['fold']} seed={job['seed']}", flush=True)
        return {"job": job, "gpu": gpu, "phase": phase, "process": proc, "log": log}

    try:
        while pending or active:
            while pending and free:
                active.append(launch(pending.pop(0), free.pop(0), 0))
            for item in active[:]:
                code = item["process"].poll()
                if code is None:
                    continue
                item["log"].close()
                active.remove(item)
                if code:
                    failures.append({"directory": item["job"]["directory"], "phase": item["phase"], "exit_code": code})
                    free.append(item["gpu"])
                elif item["phase"] == 0:
                    active.append(launch(item["job"], item["gpu"], 1))
                else:
                    free.append(item["gpu"])
            time.sleep(1)
    except BaseException:
        stop_children(active)
        raise
    write_json(root / "launcher_status.json", {"failed_jobs": failures, "n_jobs": len(plan["jobs"]), "identity": plan["identity"]})
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed. Inspect launcher_status.json and job logs; rerun the same command to resume.")
    print("All jobs completed. Run scripts/summarize.py next.")


if __name__ == "__main__":
    main()
