"""
CLDN18.2 GPU HoverNet + Cell Graph + Heatmap Pipeline
=====================================================

Modes
-----
hovernet     region Zarr -> nuclei CSV
patches      nuclei CSV + region Zarr -> cell-centered patch Zarr
embeddings   patch Zarr -> embeddings
graph        embeddings/nuclei -> per-slide graph NPZ
heatmap      nuclei/embeddings -> virtual biomarker heatmaps
evaluate     prediction scores + labels CSV -> F1, AUC, accuracy
all          run every stage in order

Recommended smoke-test run
--------------------------
python cldn18_gpu_hovernet_pipeline_revised.py \
  --mode all \
  --manifest region_zarr/all_regions_manifest.csv \
  --output_root cldn18_outputs \
  --num_gpus 8 \
  --batch_size 8 \
  --debug_detector \
  --embedding_backend basic

"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import zarr
from PIL import Image
from tqdm import tqdm

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - optional acceleration dependency
    cKDTree = None


@dataclass
class PipelineConfig:
    manifest: str
    output_root: str = "cldn18_outputs"
    mode: str = "all"

    num_gpus: int = 8
    batch_size: int = 8
    context_size: int = 1024
    patch_size: int = 224
    patch_batch_size: int = 512

    slide_id_col: str = "slide_id"
    zarr_path_col: str = "zarr_path"
    array_path_col: str = "array_path"
    index_col: str = "index"
    context_cols: Tuple[str, str, str, str] = ("context_x0", "context_y0", "context_x1", "context_y1")

    min_nucleus_area_px: int = 20
    max_nucleus_area_px: int = 5000
    debug_detector: bool = True
    hovernet_checkpoint: Optional[str] = None

    embedding_backend: str = "basic"  # basic or torchscript
    embedding_model_path: Optional[str] = None
    embedding_dim: int = 64

    graph_k: int = 8
    graph_radius_px: float = 160.0
    max_graph_cells_per_slide: Optional[int] = None

    heatmap_bin_px: int = 512
    heatmap_sigma_bins: float = 1.2
    positivity_score_col: str = "cldn18_morph_score"

    eval_labels: Optional[str] = None
    eval_label_col: str = "label"
    eval_score_col: str = "cldn18_morph_score"
    eval_join_cols: str = "slide_id"
    eval_level: str = "slide"  # slide or cell
    eval_agg: str = "mean"  # mean, max, median, top10_mean
    eval_threshold: Optional[float] = None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def output_dirs(cfg: PipelineConfig) -> Dict[str, Path]:
    root = Path(cfg.output_root)
    dirs = {
        "root": root,
        "nuclei": root / "nuclei_gpu",
        "patches": root / "cell_patches",
        "embeddings": root / "embeddings",
        "graphs": root / "graphs",
        "heatmaps": root / "heatmaps",
    }
    for d in dirs.values():
        ensure_dir(d)
    return dirs


def clean_manifest(cfg: PipelineConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.manifest)
    required = [cfg.slide_id_col, cfg.zarr_path_col, cfg.array_path_col, *cfg.context_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    for col in [cfg.slide_id_col, cfg.zarr_path_col, cfg.array_path_col]:
        df[col] = df[col].astype(str).str.strip()
    for c in cfg.context_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if cfg.index_col in df.columns:
        df[cfg.index_col] = pd.to_numeric(df[cfg.index_col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=list(cfg.context_cols)).copy()
    df = df[
        (df[cfg.zarr_path_col] != "")
        & (df[cfg.zarr_path_col].str.lower() != "nan")
        & (df[cfg.array_path_col] != "")
        & (df[cfg.array_path_col].str.lower() != "nan")
    ].copy()
    exists = df[cfg.zarr_path_col].apply(lambda p: Path(str(p)).exists())
    missing_paths = sorted(df.loc[~exists, cfg.zarr_path_col].drop_duplicates().tolist())
    df = df.loc[exists].copy()
    dedup_cols = [cfg.slide_id_col, cfg.zarr_path_col, cfg.array_path_col, *cfg.context_cols]
    if cfg.index_col in df.columns:
        dedup_cols.append(cfg.index_col)
    df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    dirs = output_dirs(cfg)
    if missing_paths:
        pd.DataFrame({"missing_zarr_path": missing_paths}).to_csv(dirs["root"] / "missing_zarr_paths.csv", index=False)
    df.to_csv(dirs["root"] / "cleaned_region_manifest.csv", index=False)
    print(f"Manifest rows: {before} -> {len(df)} valid contexts")
    if df.empty:
        raise ValueError("No valid manifest rows after cleaning.")
    return df


def open_zarr_array(zarr_path: str, array_path: str):
    root = zarr.open(str(zarr_path), mode="r")
    if array_path in root:
        return root[array_path]
    raise KeyError(f"{array_path} not found in {zarr_path}. Available: {list(root.keys())}")


def normalize_rgb(rgb: np.ndarray, size: Optional[int] = None) -> np.ndarray:
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[-1] not in (3, 4):
        rgb = np.moveaxis(rgb, 0, -1)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Invalid RGB shape: {rgb.shape}")
    rgb = rgb.astype(np.uint8)
    if size is not None and rgb.shape[:2] != (size, size):
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return rgb


def read_context(arr, row: pd.Series, cfg: PipelineConfig) -> np.ndarray:
    if cfg.index_col in row and not pd.isna(row[cfg.index_col]):
        rgb = np.asarray(arr[int(row[cfg.index_col])])
    else:
        x0, y0, x1, y1 = [int(row[c]) for c in cfg.context_cols]
        shape = arr.shape
        if len(shape) == 3 and shape[-1] in (3, 4):
            rgb = np.asarray(arr[y0:y1, x0:x1, :3])
        elif len(shape) == 3 and shape[0] in (3, 4):
            rgb = np.asarray(arr[:3, y0:y1, x0:x1])
            rgb = np.moveaxis(rgb, 0, -1)
        else:
            raise ValueError(f"Unsupported coordinate fallback for zarr shape {shape}; include index in manifest.")
    return normalize_rgb(rgb, cfg.context_size)


class HoverNetGPUAdapter:
    """Adapter boundary for real HoverNet.

    The debug detector is only for pipeline smoke tests. For production, load your
    HoverNet implementation/checkpoint in __init__ and return a list of DataFrames
    from predict_batch with local_x, local_y, area, type_id, confidence.
    """

    def __init__(self, device: torch.device, cfg: PipelineConfig):
        self.device = device
        self.cfg = cfg
        self.model = None
        if not cfg.debug_detector:
            if not cfg.hovernet_checkpoint:
                raise ValueError("--hovernet_checkpoint is required when --debug_detector is not set.")
            raise NotImplementedError(
                "Connect your production HoverNet model here. Keep predict_batch output schema unchanged."
            )

    @torch.no_grad()
    def predict_batch(self, images_rgb: List[np.ndarray]) -> List[pd.DataFrame]:
        if self.cfg.debug_detector:
            return [self._debug_detector(img) for img in images_rgb]
        raise NotImplementedError("Production HoverNet predict_batch is not wired yet.")

    def _debug_detector(self, rgb: np.ndarray) -> pd.DataFrame:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
        rows = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.cfg.min_nucleus_area_px or area > self.cfg.max_nucleus_area_px:
                continue
            cx, cy = cent[i]
            rows.append({"local_x": int(cx), "local_y": int(cy), "area": area, "type_id": -1, "confidence": 1.0})
        return pd.DataFrame(rows)


def append_prediction_records(cfg: PipelineConfig, row: pd.Series, pred: pd.DataFrame, records: List[Dict[str, Any]]) -> None:
    if pred is None or pred.empty:
        return
    slide_id = str(row[cfg.slide_id_col])
    x0, y0, x1, y1 = [int(row[c]) for c in cfg.context_cols]
    region_index = int(row[cfg.index_col]) if cfg.index_col in row and not pd.isna(row[cfg.index_col]) else -1
    for _, nrow in pred.iterrows():
        lx = int(nrow["local_x"])
        ly = int(nrow["local_y"])
        records.append({
            "slide_id": slide_id,
            "cell_id": -1,
            "x": int(x0 + lx),
            "y": int(y0 + ly),
            "local_x": lx,
            "local_y": ly,
            "area": int(nrow.get("area", -1)),
            "type_id": int(nrow.get("type_id", -1)),
            "confidence": float(nrow.get("confidence", 1.0)),
            "context_x0": x0,
            "context_y0": y0,
            "context_x1": x1,
            "context_y1": y1,
            "region_index": region_index,
            "zarr_path": str(row[cfg.zarr_path_col]),
            "array_path": str(row[cfg.array_path_col]),
            "error": "",
        })


def shard_dataframe(df: pd.DataFrame, rank: int, world_size: int) -> pd.DataFrame:
    return df.iloc[rank::world_size].reset_index(drop=True)


def hovernet_worker(rank: int, cfg_dict: dict, manifest_df: pd.DataFrame) -> None:
    cfg = PipelineConfig(**cfg_dict)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    dirs = output_dirs(cfg)
    rank_csv = dirs["nuclei"] / f"nuclei_rank{rank:02d}.csv"
    if rank_csv.exists():
        rank_csv.unlink()

    df = shard_dataframe(manifest_df, rank, cfg.num_gpus)
    print(f"[rank {rank}] device={device}; contexts={len(df)}", flush=True)
    detector = HoverNetGPUAdapter(device, cfg)
    zarr_cache: Dict[Tuple[str, str], object] = {}
    pending: List[Dict[str, Any]] = []
    header_written = False

    def flush() -> None:
        nonlocal pending, header_written
        if not pending:
            return
        pd.DataFrame(pending).to_csv(rank_csv, mode="a", index=False, header=not header_written)
        header_written = True
        pending.clear()

    batch_images: List[np.ndarray] = []
    batch_rows: List[pd.Series] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"GPU{rank} HoverNet"):
        try:
            key = (str(row[cfg.zarr_path_col]), str(row[cfg.array_path_col]))
            if key not in zarr_cache:
                zarr_cache[key] = open_zarr_array(*key)
            rgb = read_context(zarr_cache[key], row, cfg)
        except Exception as e:
            pending.append({
                "slide_id": str(row.get(cfg.slide_id_col, "")),
                "cell_id": -1,
                "x": -1,
                "y": -1,
                "error": f"read_context_failed: {str(e)[-300:]}",
            })
            continue

        batch_images.append(rgb)
        batch_rows.append(row)
        if len(batch_images) >= cfg.batch_size:
            preds = detector.predict_batch(batch_images)
            for brow, pred in zip(batch_rows, preds):
                append_prediction_records(cfg, brow, pred, pending)
            batch_images.clear()
            batch_rows.clear()
        if len(pending) >= max(1000, cfg.batch_size * 100):
            flush()

    if batch_images:
        preds = detector.predict_batch(batch_images)
        for brow, pred in zip(batch_rows, preds):
            append_prediction_records(cfg, brow, pred, pending)
    flush()
    print(f"[rank {rank}] saved {rank_csv}", flush=True)


def run_hovernet(cfg: PipelineConfig, manifest_df: pd.DataFrame) -> Path:
    dirs = output_dirs(cfg)
    with (dirs["nuclei"] / "gpu_hovernet_config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    available = torch.cuda.device_count()
    if available == 0:
        print("No CUDA detected; running one CPU worker.")
        cfg.num_gpus = 1
    elif cfg.num_gpus > available:
        print(f"Requested {cfg.num_gpus} GPUs, but only {available} visible; using {available}.")
        cfg.num_gpus = available

    mp.spawn(hovernet_worker, args=(asdict(cfg), manifest_df), nprocs=cfg.num_gpus, join=True)
    return merge_nuclei(cfg)


def merge_nuclei(cfg: PipelineConfig) -> Path:
    dirs = output_dirs(cfg)
    csvs = sorted(dirs["nuclei"].glob("nuclei_rank*.csv"))
    if not csvs:
        raise RuntimeError("No rank nuclei CSVs were produced.")
    dfs = [pd.read_csv(p) for p in csvs if p.stat().st_size > 0]
    if not dfs:
        raise RuntimeError("Rank nuclei CSVs are empty.")
    df = pd.concat(dfs, ignore_index=True)
    if "error" in df.columns:
        err = df["error"].fillna("").astype(str)
        errors = df[err != ""].copy()
        ok = df[err == ""].copy()
    else:
        errors = pd.DataFrame()
        ok = df.copy()
    if not errors.empty:
        errors.to_csv(dirs["nuclei"] / "nuclei_errors.csv", index=False)

    if not ok.empty:
        ok["x_round"] = (ok["x"] / 8).round().astype(int)
        ok["y_round"] = (ok["y"] / 8).round().astype(int)
        ok = ok.drop_duplicates(subset=["slide_id", "x_round", "y_round"]).drop(columns=["x_round", "y_round"])
        ok = ok.sort_values(["slide_id", "y", "x"]).reset_index(drop=True)
        ok["cell_id"] = ok.groupby("slide_id").cumcount()
    out = dirs["nuclei"] / "all_nuclei.csv"
    ok.to_csv(out, index=False)

    per_slide = dirs["nuclei"] / "per_slide"
    ensure_dir(per_slide)
    for slide_id, sdf in ok.groupby("slide_id"):
        sdf.to_csv(per_slide / f"{slide_id}_nuclei.csv", index=False)
    print(f"Nuclei merged: {out}; cells={len(ok)}")
    return out


def crop_patch_from_context(rgb: np.ndarray, cx: int, cy: int, patch_size: int) -> np.ndarray:
    half = patch_size // 2
    padded = np.pad(rgb, ((half, half), (half, half), (0, 0)), constant_values=255)
    px = int(cx) + half
    py = int(cy) + half
    patch = padded[py - half : py - half + patch_size, px - half : px - half + patch_size]
    return normalize_rgb(patch, patch_size)


def run_patches(cfg: PipelineConfig, nuclei_csv: Optional[Path] = None) -> Path:
    dirs = output_dirs(cfg)
    nuclei_csv = nuclei_csv or dirs["nuclei"] / "all_nuclei.csv"
    nuclei = pd.read_csv(nuclei_csv)
    if nuclei.empty:
        raise ValueError(f"No nuclei found in {nuclei_csv}")

    patch_store = dirs["patches"] / "cell_patches.zarr"
    if patch_store.exists():
        shutil_rmtree(patch_store)
    root = zarr.open(str(patch_store), mode="w")
    arr = root.create_dataset(
        "patches",
        shape=(0, cfg.patch_size, cfg.patch_size, 3),
        chunks=(1, cfg.patch_size, cfg.patch_size, 3),
        dtype="uint8",
        overwrite=True,
    )

    rows: List[Dict[str, Any]] = []
    zarr_cache: Dict[Tuple[str, str], object] = {}
    batch: List[np.ndarray] = []
    written = 0

    for _, row in tqdm(nuclei.iterrows(), total=len(nuclei), desc="cell patches"):
        try:
            key = (str(row["zarr_path"]), str(row["array_path"]))
            if key not in zarr_cache:
                zarr_cache[key] = open_zarr_array(*key)
            arr_context = zarr_cache[key]
            region_row = pd.Series({
                cfg.index_col: int(row["region_index"]),
                cfg.zarr_path_col: row["zarr_path"],
                cfg.array_path_col: row["array_path"],
                "context_x0": row["context_x0"],
                "context_y0": row["context_y0"],
                "context_x1": row["context_x1"],
                "context_y1": row["context_y1"],
            })
            rgb = read_context(arr_context, region_row, cfg)
            patch = crop_patch_from_context(rgb, int(row["local_x"]), int(row["local_y"]), cfg.patch_size)
        except Exception as e:
            rows.append({
                "slide_id": row.get("slide_id", ""),
                "cell_id": row.get("cell_id", -1),
                "patch_index": -1,
                "error": str(e)[-300:],
            })
            continue

        patch_index = written + len(batch)
        batch.append(patch)
        out_row = row.to_dict()
        out_row["patch_index"] = patch_index
        out_row["patch_zarr_path"] = str(patch_store)
        out_row["patch_array_path"] = "patches"
        out_row["error"] = ""
        rows.append(out_row)

        if len(batch) >= cfg.patch_batch_size:
            data = np.stack(batch, axis=0).astype(np.uint8)
            arr.resize((written + len(batch), cfg.patch_size, cfg.patch_size, 3))
            arr[written : written + len(batch)] = data
            written += len(batch)
            batch.clear()

    if batch:
        data = np.stack(batch, axis=0).astype(np.uint8)
        arr.resize((written + len(batch), cfg.patch_size, cfg.patch_size, 3))
        arr[written : written + len(batch)] = data
    meta = pd.DataFrame(rows)
    meta_path = dirs["patches"] / "cell_patches_manifest.csv"
    meta.to_csv(meta_path, index=False)
    print(f"Patches saved: {patch_store}; manifest={meta_path}")
    return meta_path


def shutil_rmtree(path: Path) -> None:
    import shutil
    shutil.rmtree(path)


class EmbeddingAdapter:
    def __init__(self, device: torch.device, cfg: PipelineConfig):
        self.device = device
        self.cfg = cfg
        self.model = None
        if cfg.embedding_backend == "torchscript":
            if not cfg.embedding_model_path:
                raise ValueError("--embedding_model_path is required for torchscript embedding backend.")
            self.model = torch.jit.load(cfg.embedding_model_path, map_location=device).eval()
        elif cfg.embedding_backend != "basic":
            raise ValueError(f"Unsupported embedding backend: {cfg.embedding_backend}")

    @torch.no_grad()
    def embed_batch(self, patches_rgb: np.ndarray) -> np.ndarray:
        if self.cfg.embedding_backend == "basic":
            return basic_patch_features(patches_rgb, self.cfg.embedding_dim)
        x = torch.from_numpy(patches_rgb).float().to(self.device) / 255.0
        x = x.permute(0, 3, 1, 2)
        x = (x - 0.5) / 0.5
        y = self.model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        return y.detach().cpu().float().numpy()


def basic_patch_features(patches: np.ndarray, out_dim: int) -> np.ndarray:
    feats = []
    for rgb in patches:
        rgb = normalize_rgb(rgb)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, s, v = cv2.split(hsv)
        brown = ((h >= 5) & (h <= 35) & (s > 35) & (v < 235)).astype(np.float32)
        purple = ((h >= 115) & (h <= 170) & (s > 25) & (v < 245)).astype(np.float32)
        tissue = (((gray < 235) & (s > 8)) | (s > 28)).astype(np.float32)
        color_mean = rgb.reshape(-1, 3).mean(axis=0) / 255.0
        color_std = rgb.reshape(-1, 3).std(axis=0) / 255.0
        hsv_mean = hsv.reshape(-1, 3).mean(axis=0) / np.array([179.0, 255.0, 255.0])
        moments = np.array([
            tissue.mean(), brown.mean(), purple.mean(),
            float(gray.mean() / 255.0), float(gray.std() / 255.0),
            float(cv2.Laplacian(gray, cv2.CV_32F).var() / 1000.0),
        ])
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).reshape(-1)
        hist = hist / max(hist.sum(), 1.0)
        f = np.concatenate([color_mean, color_std, hsv_mean, moments, hist]).astype(np.float32)
        if len(f) < out_dim:
            f = np.pad(f, (0, out_dim - len(f)))
        feats.append(f[:out_dim])
    return np.stack(feats, axis=0).astype(np.float32)


def run_embeddings(cfg: PipelineConfig, patch_manifest: Optional[Path] = None) -> Path:
    dirs = output_dirs(cfg)
    patch_manifest = patch_manifest or dirs["patches"] / "cell_patches_manifest.csv"
    meta = pd.read_csv(patch_manifest)
    err = meta["error"].fillna("").astype(str) if "error" in meta.columns else pd.Series([""] * len(meta))
    meta = meta[(meta["patch_index"] >= 0) & (err == "")].copy()
    if meta.empty:
        raise ValueError("No valid patches for embedding.")

    patch_zarr = str(meta.iloc[0]["patch_zarr_path"])
    patch_arr = zarr.open(patch_zarr, mode="r")["patches"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter = EmbeddingAdapter(device, cfg)

    all_feats = []
    rows = []
    for start in tqdm(range(0, len(meta), cfg.batch_size), desc="embeddings"):
        chunk = meta.iloc[start : start + cfg.batch_size]
        idx = chunk["patch_index"].astype(int).to_numpy()
        patches = np.asarray(patch_arr.get_orthogonal_selection((idx, slice(None), slice(None), slice(None))))
        feats = adapter.embed_batch(patches)
        all_feats.append(feats)
        rows.append(chunk.reset_index(drop=True))

    emb = np.concatenate(all_feats, axis=0).astype(np.float32)
    emb_path = dirs["embeddings"] / "cell_embeddings.npy"
    np.save(emb_path, emb)
    out_meta = pd.concat(rows, ignore_index=True)
    score = compute_cldn18_morph_score(out_meta, emb)
    out_meta[cfg.positivity_score_col] = score
    meta_path = dirs["embeddings"] / "cell_embeddings_manifest.csv"
    out_meta.to_csv(meta_path, index=False)
    print(f"Embeddings saved: {emb_path}; metadata={meta_path}; shape={emb.shape}")
    return meta_path


def compute_cldn18_morph_score(meta: pd.DataFrame, emb: np.ndarray) -> np.ndarray:
    # For real biomarker modeling, replace this with a trained classifier head.
    # The fallback score intentionally uses DAB-like color/texture features from
    # basic embeddings so heatmap generation can be tested end-to-end.
    if emb.shape[1] >= 12:
        raw = 2.2 * emb[:, 10] + 0.5 * emb[:, 9] - 0.4 * emb[:, 12]
    else:
        raw = emb.mean(axis=1)
    raw = raw.astype(np.float32)
    lo, hi = np.percentile(raw, [2, 98])
    if hi <= lo:
        return np.zeros_like(raw)
    return np.clip((raw - lo) / (hi - lo), 0, 1)


def run_graphs(cfg: PipelineConfig, embedding_manifest: Optional[Path] = None) -> Path:
    dirs = output_dirs(cfg)
    embedding_manifest = embedding_manifest or dirs["embeddings"] / "cell_embeddings_manifest.csv"
    meta = pd.read_csv(embedding_manifest)
    emb = np.load(dirs["embeddings"] / "cell_embeddings.npy")
    if len(meta) != len(emb):
        raise ValueError(f"Embedding metadata length {len(meta)} != embedding rows {len(emb)}")

    graph_index = []
    for slide_id, sdf in tqdm(meta.groupby("slide_id"), desc="graphs"):
        sdf = sdf.reset_index()
        if cfg.max_graph_cells_per_slide and len(sdf) > cfg.max_graph_cells_per_slide:
            sdf = sdf.sample(cfg.max_graph_cells_per_slide, random_state=13).sort_values(["y", "x"]).reset_index(drop=True)
        idx = sdf["index"].to_numpy()
        coords = sdf[["x", "y"]].to_numpy(dtype=np.float32)
        x_feat = emb[idx].astype(np.float32)
        scores = sdf[cfg.positivity_score_col].to_numpy(dtype=np.float32)
        edges = build_edges(coords, cfg.graph_k, cfg.graph_radius_px)
        out = dirs["graphs"] / f"{slide_id}_graph.npz"
        np.savez_compressed(
            out,
            x=x_feat,
            pos=coords,
            edge_index=edges,
            y_score=scores,
            cell_id=sdf["cell_id"].to_numpy(),
        )
        graph_index.append({"slide_id": slide_id, "graph_path": str(out), "num_cells": len(sdf), "num_edges": edges.shape[1]})
    index_path = dirs["graphs"] / "graph_manifest.csv"
    pd.DataFrame(graph_index).to_csv(index_path, index=False)
    print(f"Graphs saved: {index_path}")
    return index_path


def build_edges(coords: np.ndarray, k: int, radius: float) -> np.ndarray:
    n = len(coords)
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64)
    if cKDTree is not None:
        return build_edges_kdtree(coords, k, radius)

    src, dst = [], []
    for i in range(n):
        d = np.sqrt(((coords - coords[i]) ** 2).sum(axis=1))
        order = np.argsort(d)
        used = 0
        for j in order[1:]:
            if d[j] <= radius or used < k:
                src.append(i)
                dst.append(int(j))
                used += 1
            if used >= k and d[j] > radius:
                break
    return np.asarray([src, dst], dtype=np.int64)


def build_edges_kdtree(coords: np.ndarray, k: int, radius: float) -> np.ndarray:
    tree = cKDTree(coords)
    src: List[int] = []
    dst: List[int] = []
    query_k = min(max(k + 1, 2), len(coords))

    distances, indices = tree.query(coords, k=query_k, workers=-1)
    if query_k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    for i in range(len(coords)):
        used = set()
        for d, j in zip(distances[i], indices[i]):
            j = int(j)
            if j == i or j >= len(coords):
                continue
            src.append(i)
            dst.append(j)
            used.add(j)

        if radius > 0:
            for j in tree.query_ball_point(coords[i], r=radius):
                j = int(j)
                if j == i or j in used:
                    continue
                src.append(i)
                dst.append(j)

    return np.asarray([src, dst], dtype=np.int64)


def run_heatmaps(cfg: PipelineConfig, embedding_manifest: Optional[Path] = None) -> Path:
    dirs = output_dirs(cfg)
    embedding_manifest = embedding_manifest or dirs["embeddings"] / "cell_embeddings_manifest.csv"
    meta = pd.read_csv(embedding_manifest)
    out_rows = []
    for slide_id, sdf in tqdm(meta.groupby("slide_id"), desc="heatmaps"):
        x = sdf["x"].to_numpy(dtype=np.float32)
        y = sdf["y"].to_numpy(dtype=np.float32)
        score = sdf[cfg.positivity_score_col].to_numpy(dtype=np.float32)
        max_x = max(float(x.max()), 1.0)
        max_y = max(float(y.max()), 1.0)
        w = int(math.ceil(max_x / cfg.heatmap_bin_px)) + 1
        h = int(math.ceil(max_y / cfg.heatmap_bin_px)) + 1
        value = np.zeros((h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)
        bx = np.clip((x / cfg.heatmap_bin_px).astype(int), 0, w - 1)
        by = np.clip((y / cfg.heatmap_bin_px).astype(int), 0, h - 1)
        for xx, yy, ss in zip(bx, by, score):
            value[yy, xx] += ss
            count[yy, xx] += 1
        heat = value / np.maximum(count, 1.0)
        heat[count == 0] = 0.0
        if cfg.heatmap_sigma_bins > 0:
            heat = cv2.GaussianBlur(heat, (0, 0), cfg.heatmap_sigma_bins)
        npy_path = dirs["heatmaps"] / f"{slide_id}_cldn18_heatmap.npy"
        png_path = dirs["heatmaps"] / f"{slide_id}_cldn18_heatmap.png"
        np.save(npy_path, heat.astype(np.float32))
        save_heatmap_png(heat, png_path)
        out_rows.append({
            "slide_id": slide_id,
            "heatmap_npy": str(npy_path),
            "heatmap_png": str(png_path),
            "num_cells": len(sdf),
            "mean_score": float(score.mean()) if len(score) else 0.0,
        })
    manifest = dirs["heatmaps"] / "heatmap_manifest.csv"
    pd.DataFrame(out_rows).to_csv(manifest, index=False)
    print(f"Heatmaps saved: {manifest}")
    return manifest


def save_heatmap_png(heat: np.ndarray, path: Path) -> None:
    if heat.size == 0:
        Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)).save(path)
        return
    h = heat.astype(np.float32)
    if h.max() > h.min():
        h = (h - h.min()) / (h.max() - h.min())
    img = (h * 255).astype(np.uint8)
    color = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    scale = max(1, int(1200 / max(color.shape[:2])))
    color = cv2.resize(color, (color.shape[1] * scale, color.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    Image.fromarray(color).save(path)


def run_evaluation(cfg: PipelineConfig, embedding_manifest: Optional[Path] = None) -> Path:
    if not cfg.eval_labels:
        raise ValueError("--eval_labels is required for --mode evaluate or evaluation after --mode all.")
    dirs = output_dirs(cfg)
    embedding_manifest = embedding_manifest or dirs["embeddings"] / "cell_embeddings_manifest.csv"
    pred = pd.read_csv(embedding_manifest)
    labels = pd.read_csv(cfg.eval_labels)

    if cfg.eval_score_col not in pred.columns:
        raise ValueError(f"Prediction manifest missing score column: {cfg.eval_score_col}")
    if cfg.eval_label_col not in labels.columns:
        raise ValueError(f"Labels CSV missing label column: {cfg.eval_label_col}")

    join_cols = [c.strip() for c in cfg.eval_join_cols.split(",") if c.strip()]
    missing_pred = [c for c in join_cols if c not in pred.columns]
    missing_lab = [c for c in join_cols if c not in labels.columns]
    if missing_pred or missing_lab:
        raise ValueError(f"Join column mismatch. Missing in predictions={missing_pred}; missing in labels={missing_lab}")

    if cfg.eval_level == "slide":
        pred_eval = aggregate_predictions(pred, join_cols, cfg.eval_score_col, cfg.eval_agg)
    elif cfg.eval_level == "cell":
        pred_eval = pred[join_cols + [cfg.eval_score_col]].copy()
    else:
        raise ValueError("--eval_level must be slide or cell")

    label_eval = labels[join_cols + [cfg.eval_label_col]].drop_duplicates(subset=join_cols).copy()
    merged = pred_eval.merge(label_eval, on=join_cols, how="inner")
    if merged.empty:
        raise ValueError("No matching rows between predictions and labels. Check --eval_join_cols.")

    y_true = merged[cfg.eval_label_col].astype(int).to_numpy()
    y_score = merged[cfg.eval_score_col].astype(float).to_numpy()
    metrics, curve = binary_classification_metrics(y_true, y_score, cfg.eval_threshold)

    eval_dir = dirs["root"] / "evaluation"
    ensure_dir(eval_dir)
    merged_path = eval_dir / f"{cfg.eval_level}_evaluation_rows.csv"
    metrics_path = eval_dir / f"{cfg.eval_level}_metrics.csv"
    curve_path = eval_dir / f"{cfg.eval_level}_threshold_curve.csv"
    merged.to_csv(merged_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    curve.to_csv(curve_path, index=False)

    print(f"Evaluation rows: {merged_path}")
    print(f"Evaluation metrics: {metrics_path}")
    print(
        "Metrics: "
        f"AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, "
        f"accuracy={metrics['accuracy']:.4f}, threshold={metrics['threshold']:.4f}, n={metrics['n']}"
    )
    return metrics_path


def aggregate_predictions(pred: pd.DataFrame, join_cols: List[str], score_col: str, agg: str) -> pd.DataFrame:
    if agg == "mean":
        return pred.groupby(join_cols, as_index=False)[score_col].mean()
    if agg == "max":
        return pred.groupby(join_cols, as_index=False)[score_col].max()
    if agg == "median":
        return pred.groupby(join_cols, as_index=False)[score_col].median()
    if agg == "top10_mean":
        rows = []
        for key, sdf in pred.groupby(join_cols):
            scores = sdf[score_col].astype(float).sort_values(ascending=False)
            n_top = max(1, int(math.ceil(len(scores) * 0.10)))
            if not isinstance(key, tuple):
                key = (key,)
            rows.append({**dict(zip(join_cols, key)), score_col: float(scores.iloc[:n_top].mean())})
        return pd.DataFrame(rows)
    raise ValueError("--eval_agg must be mean, max, median, or top10_mean")


def binary_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: Optional[float],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    y_true = y_true.astype(int)
    if set(np.unique(y_true)) - {0, 1}:
        raise ValueError("Labels must be binary 0/1 for F1, AUC, and accuracy.")
    auc = roc_auc_score_manual(y_true, y_score)

    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], y_score)))
    rows = []
    for th in thresholds:
        row = metrics_at_threshold(y_true, y_score, float(th))
        rows.append(row)
    curve = pd.DataFrame(rows).sort_values(["f1", "accuracy"], ascending=False).reset_index(drop=True)
    chosen = metrics_at_threshold(y_true, y_score, threshold) if threshold is not None else curve.iloc[0].to_dict()
    chosen["auc"] = auc
    chosen["n"] = int(len(y_true))
    chosen["positives"] = int(y_true.sum())
    chosen["negatives"] = int((1 - y_true).sum())
    return chosen, curve


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(y_true), 1)
    sensitivity = recall
    specificity = tn / max(tn + fp, 1)
    return {
        "threshold": float(threshold),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = ranks[y_true == 1].sum()
    n_pos = len(pos)
    n_neg = len(neg)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLDN18.2 GPU HoverNet + embedding + graph + heatmap pipeline")
    p.add_argument("--mode", choices=["hovernet", "patches", "embeddings", "graph", "heatmap", "evaluate", "all"], default="all")
    p.add_argument("--manifest", default=None)
    p.add_argument("--output_root", default="cldn18_outputs")
    p.add_argument("--num_gpus", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--context_size", type=int, default=1024)
    p.add_argument("--patch_size", type=int, default=224)
    p.add_argument("--debug_detector", action="store_true", help="Use classical detector for pipeline smoke test.")
    p.add_argument("--real_hovernet", action="store_true", help="Use production HoverNet adapter implementation.")
    p.add_argument("--hovernet_checkpoint", default=None)
    p.add_argument("--embedding_backend", choices=["basic", "torchscript"], default="basic")
    p.add_argument("--embedding_model_path", default=None)
    p.add_argument("--embedding_dim", type=int, default=64)
    p.add_argument("--graph_k", type=int, default=8)
    p.add_argument("--graph_radius_px", type=float, default=160.0)
    p.add_argument("--max_graph_cells_per_slide", type=int, default=None)
    p.add_argument("--heatmap_bin_px", type=int, default=512)
    p.add_argument("--eval_labels", default=None, help="CSV with ground-truth labels.")
    p.add_argument("--eval_label_col", default="label")
    p.add_argument("--eval_score_col", default="cldn18_morph_score")
    p.add_argument("--eval_join_cols", default="slide_id", help="Comma-separated keys, e.g. slide_id or slide_id,cell_id.")
    p.add_argument("--eval_level", choices=["slide", "cell"], default="slide")
    p.add_argument("--eval_agg", choices=["mean", "max", "median", "top10_mean"], default="mean")
    p.add_argument("--eval_threshold", type=float, default=None, help="If omitted, threshold maximizing F1 is used.")
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        manifest=args.manifest,
        output_root=args.output_root,
        mode=args.mode,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        context_size=args.context_size,
        patch_size=args.patch_size,
        debug_detector=(args.debug_detector or not args.real_hovernet),
        hovernet_checkpoint=args.hovernet_checkpoint,
        embedding_backend=args.embedding_backend,
        embedding_model_path=args.embedding_model_path,
        embedding_dim=args.embedding_dim,
        graph_k=args.graph_k,
        graph_radius_px=args.graph_radius_px,
        max_graph_cells_per_slide=args.max_graph_cells_per_slide,
        heatmap_bin_px=args.heatmap_bin_px,
        eval_labels=args.eval_labels,
        eval_label_col=args.eval_label_col,
        eval_score_col=args.eval_score_col,
        eval_join_cols=args.eval_join_cols,
        eval_level=args.eval_level,
        eval_agg=args.eval_agg,
        eval_threshold=args.eval_threshold,
    )


def main() -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    cfg = config_from_args(parse_args())
    dirs = output_dirs(cfg)
    with (dirs["root"] / "cldn18_pipeline_config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    manifest_df = None
    if cfg.mode not in ("evaluate",):
        if not cfg.manifest:
            raise ValueError("--manifest is required unless --mode evaluate")
        manifest_df = clean_manifest(cfg)

    nuclei_csv: Optional[Path] = None
    patch_manifest: Optional[Path] = None
    embedding_manifest: Optional[Path] = None

    if cfg.mode in ("hovernet", "all"):
        assert manifest_df is not None
        nuclei_csv = run_hovernet(cfg, manifest_df)
    if cfg.mode in ("patches", "all"):
        patch_manifest = run_patches(cfg, nuclei_csv)
    if cfg.mode in ("embeddings", "all"):
        embedding_manifest = run_embeddings(cfg, patch_manifest)
    if cfg.mode in ("graph", "all"):
        run_graphs(cfg, embedding_manifest)
    if cfg.mode in ("heatmap", "all"):
        run_heatmaps(cfg, embedding_manifest)
    if cfg.mode in ("evaluate", "all") and cfg.eval_labels:
        run_evaluation(cfg, embedding_manifest)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
