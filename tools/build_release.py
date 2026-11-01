"""Build the compact source release from an explicit file allowlist."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "README.md", "LICENSE", "CITATION.cff", "THIRD_PARTY_NOTICES.md", ".gitignore", "requirements.txt",
    "configs/ptanta.yaml", "docs/DATA.md", "docs/METHODS.md",
    *[f"scripts/{name}.py" for name in ["ptanta", "prepare_data", "train", "evaluate", "run", "summarize", "plot"]],
    "tests/test_ptanta.py", "tests/smoke.py", "tools/plot_results.py", "tools/build_release.py",
    "results/README.md", "results/reported_metrics.csv", "results/guidance_ablation.csv",
    *[f"results/charts/{name}.{ext}" for name in ["framework", "reported_performance", "guidance_ablation"]
      for ext in ["png", "svg"]],
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT.parent / "PTA-NTA-final.zip")
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Archive already exists: {args.out}")
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing release files: {missing}")
    manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, "PTA-NTA/" + name)
        archive.writestr("PTA-NTA/RELEASE_MANIFEST.json",
                         json.dumps({"version": "1.0.0", "sha256": manifest}, indent=2) + "\n")
    with zipfile.ZipFile(args.out) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive integrity check failed.")
        for name, digest in manifest.items():
            if hashlib.sha256(archive.read("PTA-NTA/" + name)).hexdigest() != digest:
                raise ValueError(f"Archive hash mismatch: {name}")
    print(f"Wrote {args.out.resolve()}: {len(FILES)} files plus manifest.")


if __name__ == "__main__":
    main()
