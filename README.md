# PTA-NTA

**Cell-level tumor classification in gastric CLDN18.2 immunohistochemistry**

Research software accompanying *PTA-NTA: A Framework for Cell-Level Tumor
Classification in Gastric CLDN18.2 Immunohistochemistry*.

<img width="1710" height="593" alt="framework" src="https://github.com/user-attachments/assets/7bf9a818-eb15-4bc9-bf49-9df6f53d7fbf" />

## Why this problem matters

Brown staining alone does not identify tumor: normal epithelium may stain, and
tumor cells may remain unstained. Recognizing unstained tumor is essential to
the denominator of the CLDN18-positive tumor-cell fraction.

PTA-NTA uses a shared DINOv2 cell classifier with three classes:

| Class | Meaning |
| --- | --- |
| Non_tumor | Cells annotated as non-tumor |
| Tumor_CLDN18_positive | Tumor cells annotated as CLDN18-positive |
| Tumor_CLDN18_negative | Tumor cells annotated as CLDN18-negative |

**Tumor AI** distinguishes any tumor from non-tumor. **PTA** evaluates positive
versus negative staining among reference tumor cells. **NTA** identifies negative
tumor against all other cells. Prototype guidance uses predicted positive-tumor
anchors from the same slide to adjust NTA scores. The targets differ: NTA is
not expected to exceed conditional PTA merely because it uses PTA anchors.

This is cell classification on supplied objects, not validated detection,
whole-slide clinical scoring, or a treatment-selection device. Clinical
moderate/strong membrane scoring is a separate endpoint from these operational
annotation labels. See the [FDA assay guide](https://www.accessdata.fda.gov/cdrh_docs/pdf23/P230018C.pdf).

## What is included

- Native 518/1024 image-coordinate handling with a 1024 slide-pixel footprint.
- Patient-grouped training/validation/test splits, image audits, and crop caches.
- Frozen and fine-tuned DINOv2, a controlled multitask condition, and resume support.
- Matched unguided, same-slide, leave-one-cell-out, and cross-region prototypes.
- Calibrated tumor-minus-normal guidance and morphology-projection ablations.
- Tests, a fully synthetic CPU integration demo, aggregate results, and chart code.
- Optional QuPath/Cellpose preparation tools and explicitly marked legacy scripts.

**Not included:** patient records, real images/annotations, per-cell clinical
exports, trained study weights, or third-party model code.
CAD evaluation code, scores, and comparisons are excluded from this release.

## Installation

Use Python 3.11 or 3.12. On B200, keep the site's validated PyTorch/CUDA and
torchvision installation. The requirements file intentionally does not replace it.
For a new environment, select the appropriate installation from the
[official PyTorch instructions](https://pytorch.org/get-started/locally/).

~~~bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
~~~

DINOv2 repository and weights must be local before workers launch:

~~~bash
mkdir -p vendor models
git clone https://github.com/facebookresearch/dinov2.git vendor/dinov2
# Record and retain the exact DINOv2 commit used for each experiment.
git -C vendor/dinov2 rev-parse HEAD
~~~

Obtain the ViT-B/14 **with registers** checkpoint from the
[official DINOv2 model list](https://github.com/facebookresearch/dinov2#pretrained-models)
and place it at models/dinov2_vitb14_reg4_pretrain.pth. The configuration uses
dinov2_vitb14_reg. Workers do not download shared weights; loading is strict.
Third-party licenses apply independently.

## Quick test without clinical data

Install CPU PyTorch separately if no torch installation is present:

~~~bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -p "test_*.py" -v
python tests/smoke_reviewer_study.py --out-dir outputs/synthetic_demo --native-frames
~~~

The demo creates random images and synthetic labels, runs six tiny CPU training
jobs, evaluates prototypes, produces charts, and tests interrupted-run recovery.
It is a software test, **not evidence of scientific performance**. Use a new
empty demo directory for a fresh run.

## Run on eight B200 GPUs

Start from the repository root. Place private inputs under the ignored data/
directory. The CSV schema and annotation provenance are documented in
[Data and Annotations](docs/DATA.md).
Keep this release in a separate checkout from existing server experiments.
Do not overwrite scripts or reuse a locked output tree from a different code
version; point the new configuration at a new output directory.

~~~bash
python scripts/26_prepare_reviewer_study.py \
  --cell-labels-csv data/cell_labels.csv \
  --patient-map data/patient_map.csv \
  --patch-root data/reviewed_patches \
  --image-frame native --native-image-sizes 518 1024 \
  --slide-footprint-size 1024 --confirm-native-annotations \
  --folds 3 --split-strategy constrained-patient \
  --image-size 224 --crop-padding 40 --cache-workers 8 \
  --out-dir outputs/reviewer_study_native518

python scripts/31_run_reviewer_experiments.py \
  --config configs/reviewer_experiments_native518.yaml --dry-run

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/31_run_reviewer_experiments.py \
  --config configs/reviewer_experiments_native518.yaml \
  --gpus 0,1,2,3,4,5,6,7
~~~

The configuration schedules 27 independent jobs: three folds, three seeds, three
model conditions. **Do not use torchrun for this launcher.** Each job uses one GPU.
Defaults are BF16, batch 256, two data workers and one CPU compute thread per job.
Adjust only before creating the experiment plan. Runtime and memory must be
measured on your data; no B200 speed claim is made.

Ctrl+C stops the launcher's children. Rerun the identical command to resume.
Changing code, labels, weights, paths, or settings may change experiment identity;
use a new output directory instead of bypassing integrity checks.

~~~bash
python scripts/29_summarize_reviewer_study.py \
  --experiment-root outputs/reviewer_experiments_native518 --check-only
python scripts/29_summarize_reviewer_study.py \
  --experiment-root outputs/reviewer_experiments_native518 --n-bootstrap 2000
python scripts/33_plot_reviewer_study.py \
  --experiment-root outputs/reviewer_experiments_native518 --model dinov2_finetuned
~~~

Do not fabricate completion markers or omit failed folds. The detailed
[experiment guide](docs/EXPERIMENTS.md) includes calibration-only,
projection, and fold-restricted SSL commands.

## Results and charts

The bundled results describe **different experiments** and are never pooled.
The 100-slide research-program allocation is not evidence that every historical
result used that exact split. Available aggregate files retain their own scope.

| Result set | Status | Interpretation |
| --- | --- | --- |
| [Paper-reported aggregates](results/paper_reported/metrics.csv) | Author-reported; no matched exports | Descriptive only; unverified Tumor AI AUROC omitted |
| [Single fold / seed 17](results/single_fold/metrics.csv) | Imported recorded experiment | Fixed-decision-rule exploratory comparison |
| [Multi-fold NTA ablation](results/nta_ablation/nta_metrics_by_fold_seed.csv) | Imported aggregate exports | Validation-selected thresholds; nine fold-seed runs, not nine independent seeds |
| [Legacy held-out audit](results/legacy_heldout/model_comparison_metrics.csv) | Imported historical aggregates | Different protocol; do not merge with manuscript scores |

<img width="2307" height="999" alt="nta_ablation" src="https://github.com/user-attachments/assets/99b9696b-e73d-4d63-acf5-d790428af805" />

Only PTA-NTA variants are shown. Null or unfavorable guidance outcomes remain
included; removing an external comparator does not establish a guidance benefit.

Charts, source checksums, metric conventions, and limitations:
[Results catalogue](results/README.md).

~~~bash
python tools/plot_repository_results.py
~~~

This regenerates bundled charts from aggregate CSVs without images, model weights,
or GPUs. It does not recompute clinical metrics or ROC curves from aggregates.

## Repository map

| Directory | Contents |
| --- | --- |
| scripts/ | Existing numbered workflow and shared modules |
| configs/ | Portable example configurations |
| qupath/ | Project creation, pre-annotation bridge, reviewed export |
| tests/ | Unit/regression tests and synthetic integration demos |
| tools/ | Aggregate import, chart regeneration, release audit |
| examples/ | Synthetic schema examples only |
| results/ | Aggregate files and generated charts |
| docs/ | Methods, data contract, experiment guide, release checklist |

Recommended entry points are scripts 26-33. Scripts 15/16 and 20-25 are retained
for historical reproducibility; their estimates and decision rules are not
interchangeable with the matched workflow. The optional script 02 adapts a
DINOv2 backbone with a **SimSiam-style objective**, not the original full
DINOv2 pretraining recipe. See [Methods](docs/METHODS.md).

## Reproducibility and safe publication

Run the checks before publishing:

~~~bash
python tools/validate_release.py
~~~

[Local validation](docs/VALIDATION.md) records the passing tests, synthetic
integration workflows, tested CPU environment, and checks still requiring the
B200 server. Build a source-only archive with:

~~~bash
python tools/build_release.py --out ../PTA-NTA-source.zip
~~~

No remote repository is created by this command. Follow
[Publishing](docs/PUBLISHING.md) to create a repository under your own account.
Have the authors/institutions approve code and aggregate-data release first.
Never upload patient maps, image paths, raw predictions, or checkpoints without
separate authorization. Automatic checks are not a guarantee of anonymization.

## Citation and license

Use [CITATION.cff](CITATION.cff). Add the final paper DOI and repository URL when
available; no publication DOI has been invented.

Repository-authored software and documentation are MIT-licensed; see
[LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
The license does not grant access to clinical data or third-party model weights.
