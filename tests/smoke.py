"""One synthetic CPU end-to-end test. Outputs are not scientific results."""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ptanta import CLASSES, load_json, read_csv


def command(name, *arguments):
    subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, arguments)],
                   cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty directory for synthetic test data.")
    images = out / "images"
    images.mkdir(parents=True)
    rng = np.random.default_rng(17)
    rows = []
    for patient in range(6):
        for index, size in enumerate([518, 1024]):
            path = images / f"s{patient}_x{index * 1024}_y0_w1024_h1024.png"
            Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8)).save(path)
            for c, label in enumerate(CLASSES):
                rows.append({"slide_id": f"s{patient}", "patch_path": str(path),
                             "bbox_xmin": 10 + c * 20, "bbox_ymin": 10,
                             "bbox_xmax": 20 + c * 20, "bbox_ymax": 20,
                             "cell_class": label, "reference_source": "synthetic_test_only"})
    pd.DataFrame(rows).to_csv(out / "cells.csv", index=False)
    pd.DataFrame({"slide_id": [f"s{i}" for i in range(6)],
                  "patient_id": [f"p{i}" for i in range(6)]}).to_csv(out / "patients.csv", index=False)
    data, runs = out / "data", out / "runs"
    command("prepare_data.py", "--cell-labels-csv", out / "cells.csv", "--patient-map", out / "patients.csv",
            "--out-dir", data, "--image-frame", "native", "--native-image-sizes", 518, 1024,
            "--slide-footprint-size", 1024, "--confirm-native-annotations",
            "--folds", 3, "--split-strategy", "constrained-patient", "--image-size", 28, "--cache-workers", 2)
    cfg = {"study": str(data), "output_root": str(runs), "seeds": [17, 29],
           "training": {"encoder": "tiny", "smoke-test": True, "device": "cpu", "precision": "fp32",
                        "epochs": 1, "batch-size": 8, "num-workers": 0, "cpu-threads": 1},
           "prototype": {"tau": .2, "min-anchors": 1},
           "models": [{"name": "tiny", "arguments": {}}]}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    command("run.py", "--config", out / "config.yaml", "--gpus", "0,1")
    command("summarize.py", "--experiment-root", runs, "--n-bootstrap", 50, "--include-smoke-tests")
    command("plot.py", "--experiment-root", runs, "--model", "tiny")
    summary = load_json(runs / "summary/summary.json")
    assert summary["n_cells"] == 36 and summary["n_jobs"] == 6 and summary["synthetic"]
    cells = read_csv(data / "cells.csv")
    assert set(cells.image_width) == {518, 1024} and cells.patch_width.eq(1024).all()
    metrics = read_csv(runs / "summary/metric_summary.csv")
    assert set(metrics.variant) == {"unguided", "prototype_guided"}
    assert metrics.n_planned_seeds.eq(2).all()
    assert (runs / "figures/nta_curves.png").is_file()
    print("PASS: native-frame data, six CPU jobs, two PTA-NTA variants, patient summaries and charts.")


if __name__ == "__main__":
    main()
