# PTA-NTA

**Cell-level tumor classification in gastric CLDN18.2 immunohistochemistry**

PTA identifies CLDN18-positive tumor cells. Their DINOv2 morphology features
provide slide-specific prototypes for NTA, which identifies unstained tumor
cells. 

<img width="1710" height="593" alt="framework" src="https://github.com/user-attachments/assets/71d5c3b9-23da-4282-9b15-e485f3c57ba0" />

## Install

Use Python 3.11 or 3.12 with a matching PyTorch/torchvision installation.
On the B200 server, keep the site's validated CUDA-enabled PyTorch build.

~~~bash
python -m pip install -r requirements.txt
git clone https://github.com/facebookresearch/dinov2.git vendor/dinov2
git -C vendor/dinov2 rev-parse HEAD
~~~

Download the ViT-B/14 **with registers** weights from the
[official DINOv2 model list](https://github.com/facebookresearch/dinov2#pretrained-models)
to models/dinov2_vitb14_reg4_pretrain.pth.
The model name in configs/ptanta.yaml is dinov2_vitb14_reg.
Record the DINOv2 commit and checkpoint used. No weights download occurs inside GPU workers.

## Data

Use your QuPath-derived **cell-level** CSV and matching reviewed patch images.
The classes are Non_tumor, Tumor_CLDN18_positive, and Tumor_CLDN18_negative.
See [input format and coordinates](docs/DATA.md).
Clinical images, labels, patient records and trained weights are not distributed.

## Run on B200

Run from the repository root. Edit paths in configs/ptanta.yaml as needed.

~~~bash
python scripts/prepare_data.py \
  --cell-labels-csv data/cell_labels.csv \
  --patient-map data/patient_map.csv \
  --patch-root data/patches \
  --image-frame native --native-image-sizes 518 1024 \
  --slide-footprint-size 1024 --confirm-native-annotations \
  --folds 3 --split-strategy constrained-patient \
  --image-size 224 --crop-padding 40 --cache-workers 8 \
  --out-dir outputs/data

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/run.py --config configs/ptanta.yaml --gpus 0,1,2,3,4,5,6,7

python scripts/summarize.py --experiment-root outputs/ptanta
python scripts/plot.py --experiment-root outputs/ptanta
~~~

The configuration runs one DINOv2 model across three patient-held-out folds and
three seeds: **9 independent jobs**, one GPU per job. Do not use torchrun.
Use stratified-group instead of constrained-patient for more than 12 patients.
Image/annotation matching must be checked before confirming native coordinates.
Run directories are immutable experiment identities; use a new output_root
when changing data, code or settings. Interrupted training can resume by rerunning
the same launch command.

## Evaluation

The same cells and decision thresholds are used for unguided and guided NTA.
Prototype parameters are fixed in the configuration before test evaluation.
Outputs include accuracy, **positive-class binary F1**, AUROC and average
precision (AP), plus patient-paired guidance differences.

| Output under outputs/ptanta | Contents |
| --- | --- |
| summary/metrics_by_seed.csv | Complete out-of-fold cell metrics per seed |
| summary/metric_summary.csv | Mean and SD across seeds |
| summary/guidance_difference.csv | Guided-minus-unguided NTA; patient-paired intervals |
| figures/cell_performance.png | Tumor AI, conditional PTA and NTA metrics |
| figures/nta_curves.png | NTA ROC and precision-recall curves |
| figures/guidance_difference.png | NTA guidance effect |

## Files

- scripts/: prepare_data.py, train.py, evaluate.py, run.py, summarize.py, plot.py,
  and the shared ptanta.py module.
- configs/ptanta.yaml: the final training and prototype settings.
- tests/: focused unit tests and one synthetic end-to-end test.

~~~bash
python -m unittest discover -s tests -p "test_*.py"
python tests/smoke.py --out-dir outputs/synthetic
python tools/build_release.py --out ../PTA-NTA-final.zip
~~~

The synthetic test is a software check, not scientific evidence. The release
builder includes only source, documentation and aggregate charts; it excludes
runtime outputs, checkpoints, private data and development history.
Local verification: 16 unit tests and the six-job synthetic CPU workflow passed.
B200 training and clinical performance require execution on your server/data.

## Citation and License

Use [CITATION.cff](CITATION.cff). MIT license: [LICENSE](LICENSE).
Third-party models and libraries retain their own [licenses](THIRD_PARTY_NOTICES.md).
This is research software, not a validated diagnostic or prescribing device.

Sponsored by GensionBIO Inc.
