"""Train one locked fold/seed on ONE GPU. Launch independent jobs with run.py."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ptanta import CLASSES, PROBS, fingerprint, load_json, load_split, load_study, prediction_metrics, sha256, validate_patch_frames, write_json


class ArrayDataset(Dataset):
    def __init__(self, path, df):
        self.path = str(path)
        self.indices = df.cache_index.to_numpy(int)
        self.labels = pd.Categorical(df.cell_class, categories=CLASSES).codes.astype(np.int64)
        self.array = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r")
        return torch.from_numpy(np.array(self.array[self.indices[index]], copy=True)), int(self.labels[index])


def seed_worker(worker):
    np.random.seed(torch.initial_seed() % 2**32)
    random.seed(torch.initial_seed())


def loader(path, df, args, train=False, generator=None):
    workers = args.num_workers if train else 0
    kw = {"num_workers": workers, "pin_memory": args.device != "cpu"}
    if workers:
        kw.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(ArrayDataset(path, df), batch_size=args.batch_size, shuffle=train,
                      drop_last=False, generator=generator, worker_init_fn=seed_worker, **kw)


def input_tensor(x, device, train=False, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    x = x.to(device, non_blocking=True)
    if x.ndim == 2:
        return x.float()
    x = x.permute(0, 3, 1, 2).float().div_(255)
    if train:
        # Geometric-only augmentation preserves the color-dependent staining reference.
        horizontal = torch.rand(len(x), device=device) < 0.5
        vertical = torch.rand(len(x), device=device) < 0.5
        x[horizontal] = x[horizontal].flip(-1)
        x[vertical] = x[vertical].flip(-2)
    return (x - x.new_tensor(mean)[None, :, None, None]) / x.new_tensor(std)[None, :, None, None]


def make_encoder(args):
    if args.encoder == "tiny":
        if not args.smoke_test:
            raise ValueError("Tiny encoder is for --smoke-test only, never paper results.")
        return nn.Sequential(nn.Conv2d(3, 8, 3, stride=2), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    if not args.encoder_weights or not Path(args.encoder_weights).is_file():
        raise ValueError("Provide local --encoder-weights. This worker never downloads shared weights.")
    if args.encoder == "dinov2":
        if not args.dinov2_repo or not (Path(args.dinov2_repo) / "hubconf.py").is_file():
            raise ValueError("--dinov2-repo must contain the cloned hubconf.py.")
        if args.image_size % 14:
            raise ValueError("DINOv2 input size must be divisible by 14.")
        encoder = torch.hub.load(args.dinov2_repo, args.model_name, source="local", pretrained=False)
    state = torch.load(args.encoder_weights, map_location="cpu", weights_only=True)
    if args.weights_key:
        state = state[args.weights_key]
    elif "model" in state:
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]
    # Strict loading prevents quietly evaluating a random or only partly loaded encoder.
    encoder.load_state_dict(state, strict=True)
    return encoder


def encode(backbone, x, pooling):
    result = backbone(x)
    if isinstance(result, dict):
        if "x_norm_clstoken" not in result:
            raise ValueError("Unknown encoder output dictionary; provide an explicit adapter.")
        result = result["x_norm_clstoken"]
    if result.ndim == 3:
        if pooling == "cls":
            result = result[:, 0]
        else:
            raise ValueError("Token-valued encoder requires --pooling cls or a model-specific adapter. Do not guess register/patch token pooling.")
    if result.ndim != 2:
        raise ValueError("Encoder must output B x D features.")
    return result


class Classifier(nn.Module):
    def __init__(self, backbone, dim, pooling="vector"):
        super().__init__()
        self.backbone, self.pooling = backbone, pooling
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))

    def forward(self, x):
        features = x if x.ndim == 2 else encode(self.backbone, x, self.pooling)
        logits = self.head(features)
        return logits, torch.softmax(logits.float(), 1), features, None


def amp_context(device, precision):
    return torch.autocast(device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
                          enabled=device.type == "cuda" and precision != "fp32")


@torch.inference_mode()
def predict(model, batches, df, args, device, export_features=None):
    if "image_width" in df:
        validate_patch_frames(df, require_images=True)
    model.eval()
    probabilities, pos, feature_array = [], 0, None
    forward_seconds = 0.0
    wall_start = time.perf_counter()
    for x, _ in batches:
        x = input_tensor(x, device, mean=args.mean, std=args.std)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with amp_context(device, args.precision):
            _, p, z, _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_seconds += time.perf_counter() - start
        probabilities.append(p.cpu().numpy())
        if export_features:
            if feature_array is None:
                feature_array = np.lib.format.open_memmap(export_features, mode="w+", dtype=np.float32, shape=(len(df), z.shape[1]))
            feature_array[pos:pos + len(z)] = z.float().cpu().numpy()
            pos += len(z)
    if feature_array is not None:
        feature_array.flush()
        del feature_array
    result = df.copy().reset_index(drop=True)
    p = np.concatenate(probabilities)
    result[PROBS] = p
    result["true_class"] = result.cell_class
    result["pred_class"] = np.asarray(CLASSES)[p.argmax(1)]
    result["nta_score"] = p[:, 2]
    return result, {"n_cells": len(df), "model_seconds_including_first_batch": forward_seconds,
                    "model_cells_per_second": len(df) / max(forward_seconds, 1e-9),
                    "cached_crop_wall_seconds": time.perf_counter() - wall_start,
                    "scope": "cached_cell_crops_not_end_to_end_WSI"}


def validate_initialization(args, parts):
    if args.initialization == "ssl":
        if not args.ssl_provenance:
            raise ValueError("SSL initialization requires a provenance JSON (checkpoint_sha256, adaptation_patients); an all-slide checkpoint leaks outer test patients.")
        audit = load_json(args.ssl_provenance)
        if sha256(args.encoder_weights) != audit["checkpoint_sha256"]:
            raise ValueError("SSL checkpoint hash does not match provenance.")
        patients = set(map(str, audit["adaptation_patients"]))
        if not patients or not patients.issubset(set(parts["train"].patient_id)):
            raise ValueError("SSL adaptation patients must be a subset of THIS fold's inner training patients.")
    elif args.ssl_provenance:
        raise ValueError("Use --initialization ssl when supplying SSL provenance.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--encoder", choices=["dinov2", "tiny"], default="dinov2")
    ap.add_argument("--model-name", default="dinov2_vitb14_reg")
    ap.add_argument("--dinov2-repo")
    ap.add_argument("--encoder-weights")
    ap.add_argument("--weights-key")
    ap.add_argument("--pooling", choices=["vector", "cls"], default="vector")
    ap.add_argument("--mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    ap.add_argument("--std", type=float, nargs=3, default=[0.229, 0.224, 0.225])
    ap.add_argument("--initialization", choices=["pretrained", "ssl"], default="pretrained")
    ap.add_argument("--ssl-provenance")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--cpu-threads", type=int, default=2)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--class-weighted-loss", action="store_true")
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Do not use torchrun: run.py assigns independent fold/seed jobs to GPUs.")
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.cpu_threads < 1 or min(args.std) <= 0:
        raise ValueError("Invalid training arguments.")
    parts, study_meta, split = load_split(args.study, args.fold)
    all_cells, _ = load_study(args.study)
    args.image_size = study_meta["image_size"]
    print(f"Image frame={study_meta.get('image_frame_mode', 'original')}; "
          f"model crops={args.image_size}; source sizes={study_meta.get('audited_image_sizes', 'unspecified')}. "
          "Prediction exports retain native image coordinates and slide-pixel footprints.", flush=True)
    validate_initialization(args, parts)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings = {k: v for k, v in vars(args).items() if k != "resume"}
    identity = fingerprint({"settings": settings, "study": study_meta["cells_sha256"], "split": split,
                            "weights": sha256(args.encoder_weights) if args.encoder_weights else None,
                            "trainer_sha256": sha256(__file__), "contracts_sha256": sha256(Path(__file__).with_name("ptanta.py"))})
    if (out / "run.json").exists():
        if load_json(out / "run.json")["identity"] != identity:
            raise ValueError("Output contains a different experiment; use a new output directory.")
        if (out / "complete.json").exists():
            print(f"Already complete: {out}")
            return
        if not args.resume:
            raise ValueError("Incomplete run exists; use --resume or a new directory.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable. Install the B200-compatible PyTorch build or use --device cpu for smoke tests.")
    if device.type == "cuda" and args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is unavailable on this device.")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    generator = torch.Generator().manual_seed(args.seed)
    encoder = make_encoder(args).to(device).eval()
    with torch.no_grad(), amp_context(device, args.precision):
        dim = encode(encoder, torch.zeros(1, 3, args.image_size, args.image_size, device=device), args.pooling).shape[1]
    if args.freeze_encoder:
        encoder.requires_grad_(False)
    model = Classifier(encoder, dim, args.pooling).to(device)
    run = {"identity": identity, "settings": settings, "study": study_meta, "split": split,
           "feature_dim": dim, "selection": "inner_validation_three_class_macro_f1",
           "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                           "pandas": pd.__version__, "cuda": torch.version.cuda,
                           "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
                           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")},
           "paper_eligible": not args.smoke_test}
    write_json(out / "run.json", run)
    cache = Path(args.study) / "crops.uint8.npy"
    shape = np.load(cache, mmap_mode="r").shape
    if shape != (len(all_cells), args.image_size, args.image_size, 3):
        raise ValueError("Crop cache does not match study.")
    if args.freeze_encoder:
        feature_cache = out / "frozen_features.npy"
        if not feature_cache.exists():
            print("Extracting fixed encoder features once; subsequent epochs train only the head.", flush=True)
            temp = out / "frozen_features.incomplete.npy"
            predict(model, loader(cache, all_cells, args), all_cells, args, device, temp)
            temp.replace(feature_cache)
        cache = feature_cache
    train_loader = loader(cache, parts["train"], args, train=True, generator=generator)
    val_loader = loader(cache, parts["val"], args)
    params = [{"params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")], "lr": args.head_lr}]
    if not args.freeze_encoder:
        params.append({"params": model.backbone.parameters(), "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    weights = None
    if args.class_weighted_loss:
        counts = parts["train"].cell_class.value_counts()
        weights = torch.tensor([len(parts["train"]) / (3 * counts[c]) for c in CLASSES], device=device, dtype=torch.float32)
    start, best, stale, history = 0, -1.0, 0, []
    if args.resume and (out / "last.pt").exists():
        # Resume files are generated by this program; never load untrusted pickles.
        state = torch.load(out / "last.pt", map_location=device, weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Resume checkpoint belongs to another experiment.")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"].cpu())
        generator.set_state(state["loader_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        start, best, stale, history = state["epoch"], state["best"], state["stale"], state["history"]
    for epoch in range(start, args.epochs):
        if args.patience > 0 and stale >= args.patience:
            break
        model.train()
        if args.freeze_encoder:
            model.backbone.eval()
        loss_total = 0.0
        for x, y in tqdm(train_loader, desc=f"fold={args.fold} seed={args.seed} epoch={epoch + 1}"):
            x = input_tensor(x, device, train=True, mean=args.mean, std=args.std)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, args.precision):
                logits, _, _, extra = model(x)
                loss = nn.functional.cross_entropy(logits, y, weight=weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; run stopped without exporting test predictions.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_total += loss.item() * len(y)
        val, _ = predict(model, val_loader, parts["val"], args, device)
        metrics = prediction_metrics(val)
        score = metrics["macro_f1"]
        scheduler.step()
        record = {"epoch": epoch + 1, "train_loss": loss_total / len(parts["train"]), **metrics}
        history.append(record)
        print(record, flush=True)
        if score > best:
            best, stale = score, 0
            temp = out / "best.tmp.pt"
            torch.save({"model": model.state_dict(), "identity": identity, "epoch": epoch + 1}, temp)
            temp.replace(out / "best.pt")
        else:
            stale += 1
        state = {"identity": identity, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch + 1,
                 "best": best, "stale": stale, "history": history, "torch_rng": torch.get_rng_state(),
                 "loader_rng": generator.get_state(), "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None}
        temp = out / "last.tmp.pt"
        torch.save(state, temp)
        temp.replace(out / "last.pt")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    selected = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(selected["model"], strict=True)
    reports = {}
    # Test labels are evaluated only after checkpoint selection has finished.
    for name in ["val", "test"]:
        result, timing = predict(model, loader(cache, parts[name], args), parts[name], args, device, out / f"{name}_features.npy")
        result.to_csv(out / f"{name}_predictions.csv.gz", index=False)
        reports[name] = {"metrics": prediction_metrics(result), "timing": timing,
                         "cell_ids_sha256": fingerprint(result.cell_id.tolist()),
                         "features_sha256": sha256(out / f"{name}_features.npy"),
                         "predictions_sha256": sha256(out / f"{name}_predictions.csv.gz")}
    write_json(out / "complete.json", {"identity": identity, "best_epoch": selected["epoch"], "reports": reports,
                                      "note": "Outer-fold internal evaluation; not an external independent cohort."})
    print(f"Completed {out}")


if __name__ == "__main__":
    main()
