"""
CLDN18.2 iSyntax -> Context Zarr Converter
==========================================

Recommended run
---------------
python3.10 isyntax_direct_context_zarr_converter_revised.py \
  --slides_dir /data/isyntax \
  --out_dir /data/cldn18_region_zarr \
  --read_levels 0,1 \
  --region_size 1024 \
  --stride 1024 \
  --use_process_timeout \
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import tempfile
import time
import uuid
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import zarr
from PIL import Image
from tqdm import tqdm

try:
    import isyntax
except Exception as e:  # pragma: no cover - server dependency
    raise SystemExit(
        "Could not import pyisyntax module `isyntax`.\n"
        "Use Python >= 3.10 and install pyisyntax, pillow, numpy, pandas, zarr, opencv-python, tqdm.\n"
        f"Original error: {e}"
    )


@dataclass
class Config:
    slides_dir: Path = Path("./slides")
    out_dir: Path = Path("./region_zarr")
    qc_dir: Path = Path("./region_zarr_qc")
    log_file: Path = Path("./context_zarr_conversion_log.csv")

    region_size: int = 1024
    stride: int = 1024
    read_levels: Tuple[int, ...] = (0, 1)

    use_process_timeout: bool = False
    mp_start_method: str = "auto"
    region_read_timeout_sec: int = 45
    smoke_test_timeout_sec: int = 45
    timeout_chunk_size: int = 8
    chunk_read_timeout_sec: int = 30
    bisect_timeout_chunks: bool = False
    max_consecutive_region_failures: int = 4
    max_total_region_failures_per_slide: int = 500
    abort_on_failure_limit: bool = False
    skip_rest_of_row_after_consecutive_failures: bool = True

    white_threshold: int = 245
    max_white_fraction: float = 0.82
    min_tissue_fraction: float = 0.04
    min_stain_fraction: float = 0.002
    max_pen_fraction: float = 0.12
    min_region_component_size: int = 4
    keep_only_main_component: bool = True

    append_batch_size: int = 16
    zarr_region_chunk_n: int = 1
    overwrite: bool = False
    save_qc: bool = True
    max_qc_regions_to_draw: int = 5000

    max_slides: Optional[int] = None
    max_regions_per_slide: Optional[int] = None
    progress_every_regions: int = 25


REGION_COLUMNS = [
    "index",
    "slide_id",
    "x0",
    "y0",
    "x1",
    "y1",
    "context_x0",
    "context_y0",
    "context_x1",
    "context_y1",
    "region_size",
    "read_level",
    "white_fraction",
    "tissue_fraction",
    "stain_fraction",
    "brown_fraction",
    "purple_fraction",
    "pen_fraction",
    "edge_fraction",
    "object_component_id",
    "object_component_regions",
    "zarr_path",
    "array_path",
]

FAILED_REGION_COLUMNS = [
    "slide_id",
    "x0",
    "y0",
    "x1",
    "y1",
    "read_level",
    "error",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    return obj


def open_slide(slide_path: Path):
    return isyntax.ISyntax.open(str(slide_path))


def get_slide_dimensions(slide) -> Tuple[int, int]:
    for attr in ("dimensions",):
        if hasattr(slide, attr):
            dims = getattr(slide, attr)
            return int(dims[0]), int(dims[1])
    if hasattr(slide, "level_dimensions"):
        dims = slide.level_dimensions[0]
        return int(dims[0]), int(dims[1])
    for w_attr, h_attr in (("width", "height"), ("Width", "Height"), ("w", "h")):
        if hasattr(slide, w_attr) and hasattr(slide, h_attr):
            return int(getattr(slide, w_attr)), int(getattr(slide, h_attr))
    raise RuntimeError("Could not determine slide dimensions.")


def normalize_rgb(obj: Any, w: int, h: int) -> np.ndarray:
    arr = np.asarray(obj)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Invalid RGB shape: {arr.shape}")
    arr = arr.astype(np.uint8)
    if arr.shape[:2] == (h, w):
        return arr
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    hh = min(h, arr.shape[0])
    ww = min(w, arr.shape[1])
    out[:hh, :ww] = arr[:hh, :ww]
    return out


def read_region_from_open_slide(slide, x: int, y: int, w: int, h: int, level: int) -> np.ndarray:
    pixels = slide.read_region(int(x), int(y), int(w), int(h), level=int(level))
    return normalize_rgb(pixels, w, h)


def choose_mp_context(method: str):
    if method == "auto":
        method = "fork" if os.name == "posix" else "spawn"
    return mp.get_context(method)


def _reader_worker(slide_path: str, x: int, y: int, w: int, h: int, level: int, temp_path: str, conn) -> None:
    try:
        with isyntax.ISyntax.open(slide_path) as slide:
            rgb = read_region_from_open_slide(slide, x, y, w, h, level)
        np.save(temp_path, rgb, allow_pickle=False)
        conn.send(("ok", temp_path, ""))
    except Exception as e:
        conn.send(("error", "", str(e)[-1000:]))
    finally:
        conn.close()


def read_region_timeout(
    slide_path: Path,
    x: int,
    y: int,
    w: int,
    h: int,
    level: int,
    timeout_sec: int,
    mp_start_method: str,
) -> np.ndarray:
    ctx = choose_mp_context(mp_start_method)
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    temp_dir = Path(tempfile.gettempdir()) / "cldn18_isyntax_region_reads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{slide_path.stem}_L{level}_x{x}_y{y}_{uuid.uuid4().hex}.npy"
    proc = ctx.Process(target=_reader_worker, args=(str(slide_path), x, y, w, h, level, str(temp_path), child_conn))
    proc.start()
    child_conn.close()
    deadline = time.time() + timeout_sec
    status = None
    payload = ""
    msg = ""

    # Do not send 1024 RGB arrays through multiprocessing IPC. Some servers hang
    # in Queue/Pipe feeder cleanup with multi-MB payloads. The child writes a .npy
    # file and sends only the path through the pipe.
    while time.time() < deadline:
        if parent_conn.poll(0.25):
            status, payload, msg = parent_conn.recv()
            break
        if not proc.is_alive():
            proc.join(1)
            parent_conn.close()
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"reader exited with code {proc.exitcode} and no result")

    if status is None:
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join(5)
        parent_conn.close()
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise TimeoutError(f"read timeout level={level}, x={x}, y={y}")

    proc.join(5)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
    parent_conn.close()

    if status != "ok":
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise RuntimeError(msg)
    try:
        rgb = np.load(payload, allow_pickle=False)
        return normalize_rgb(rgb, w, h)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def read_region_safe(slide_path: Path, slide, x: int, y: int, w: int, h: int, level: int, cfg: Config) -> np.ndarray:
    if cfg.use_process_timeout:
        return read_region_timeout(slide_path, x, y, w, h, level, cfg.region_read_timeout_sec, cfg.mp_start_method)
    return read_region_from_open_slide(slide, x, y, w, h, level)


def _chunk_reader_worker(
    slide_path: str,
    slide_id: str,
    coords: List[Tuple[int, int]],
    read_level: int,
    cfg_dict: Dict[str, Any],
    temp_dir: str,
    conn,
) -> None:
    """Read/filter a chunk in one process.

    This avoids launching one process per tile. Large RGB arrays are saved as
    temporary .npy files; only metadata and paths cross IPC.
    """
    cfg = Config(**cfg_dict)
    accepted: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    try:
        with isyntax.ISyntax.open(slide_path) as slide:
            for x, y in coords:
                try:
                    rgb = read_region_from_open_slide(slide, x, y, cfg.region_size, cfg.region_size, read_level)
                    metrics = region_metrics(rgb, cfg)
                    if not accept_region(metrics, cfg):
                        continue
                    temp_path = Path(temp_dir) / f"{slide_id}_L{read_level}_x{x}_y{y}_{uuid.uuid4().hex}.npy"
                    np.save(temp_path, rgb, allow_pickle=False)
                    accepted.append({
                        "index": -1,
                        "slide_id": slide_id,
                        "x0": x,
                        "y0": y,
                        "x1": x + cfg.region_size,
                        "y1": y + cfg.region_size,
                        "context_x0": x,
                        "context_y0": y,
                        "context_x1": x + cfg.region_size,
                        "context_y1": y + cfg.region_size,
                        "region_size": cfg.region_size,
                        "read_level": read_level,
                        "temp_path": str(temp_path),
                        **{k: round(v, 6) for k, v in metrics.items()},
                    })
                except Exception as e:
                    failed.append({
                        "slide_id": slide_id,
                        "x0": x,
                        "y0": y,
                        "x1": x + cfg.region_size,
                        "y1": y + cfg.region_size,
                        "read_level": read_level,
                        "error": str(e)[-600:],
                    })
        conn.send(("ok", accepted, failed, ""))
    except Exception as e:
        conn.send(("error", accepted, failed, str(e)[-1000:]))
    finally:
        conn.close()


def read_filter_chunk_timeout(
    slide_path: Path,
    coords: List[Tuple[int, int]],
    read_level: int,
    cfg: Config,
    timeout_sec: int,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, int], np.ndarray], List[Dict[str, Any]]]:
    ctx = choose_mp_context(cfg.mp_start_method)
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    temp_dir = Path(tempfile.gettempdir()) / "cldn18_isyntax_chunk_reads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = asdict(cfg)
    proc = ctx.Process(
        target=_chunk_reader_worker,
        args=(str(slide_path), slide_path.stem, coords, read_level, cfg_dict, str(temp_dir), child_conn),
    )
    proc.start()
    child_conn.close()

    deadline = time.time() + timeout_sec
    status = None
    accepted: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    msg = ""
    while time.time() < deadline:
        if parent_conn.poll(0.25):
            status, accepted, failed, msg = parent_conn.recv()
            break
        if not proc.is_alive():
            proc.join(1)
            parent_conn.close()
            raise RuntimeError(f"chunk reader exited with code {proc.exitcode} and no result")

    if status is None:
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join(5)
        parent_conn.close()
        failed = []
        for x, y in coords:
            failed.append({
                "slide_id": slide_path.stem,
                "x0": x,
                "y0": y,
                "x1": x + cfg.region_size,
                "y1": y + cfg.region_size,
                "read_level": read_level,
                "error": f"chunk read timeout after {timeout_sec}s",
            })
        return [], {}, failed

    proc.join(5)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
    parent_conn.close()

    cache: Dict[Tuple[int, int], np.ndarray] = {}
    cleaned: List[Dict[str, Any]] = []
    for rec in accepted:
        temp_path = Path(str(rec.pop("temp_path")))
        try:
            rgb = np.load(temp_path, allow_pickle=False)
            cache[(int(rec["x0"]), int(rec["y0"]))] = normalize_rgb(rgb, cfg.region_size, cfg.region_size)
            rec["index"] = len(cleaned)
            cleaned.append(rec)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    if status != "ok":
        failed.append({
            "slide_id": slide_path.stem,
            "x0": -1,
            "y0": -1,
            "x1": -1,
            "y1": -1,
            "read_level": read_level,
            "error": msg,
        })
    return cleaned, cache, failed


def read_filter_chunk_with_bisect(
    slide_path: Path,
    coords: List[Tuple[int, int]],
    read_level: int,
    cfg: Config,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, int], np.ndarray], List[Dict[str, Any]]]:
    timeout = cfg.chunk_read_timeout_sec
    records, cache, failed = read_filter_chunk_timeout(slide_path, coords, read_level, cfg, timeout)
    timed_out = bool(failed) and all("chunk read timeout" in str(f.get("error", "")) for f in failed)
    if timed_out and cfg.bisect_timeout_chunks and len(coords) > 1:
        mid = len(coords) // 2
        left_r, left_c, left_f = read_filter_chunk_with_bisect(slide_path, coords[:mid], read_level, cfg)
        right_r, right_c, right_f = read_filter_chunk_with_bisect(slide_path, coords[mid:], read_level, cfg)
        left_r.extend(right_r)
        left_c.update(right_c)
        left_f.extend(right_f)
        return left_r, left_c, left_f
    return records, cache, failed


def smoke_test_level(slide_path: Path, slide, width: int, height: int, level: int, cfg: Config) -> Tuple[bool, str]:
    size = min(cfg.region_size, width, height)
    if size <= 0:
        return False, "invalid_slide_dimensions"
    raw_points = [
        (width // 2 - size // 2, height // 2 - size // 2),
        (width // 4, height // 4),
        (width // 2, height // 4),
        (width // 4, height // 2),
        (1024, 1024),
    ]
    errors = []
    for x, y in raw_points:
        x = int(max(0, min(width - size, x)))
        y = int(max(0, min(height - size, y)))
        try:
            if cfg.use_process_timeout:
                rgb = read_region_timeout(slide_path, x, y, size, size, level, cfg.smoke_test_timeout_sec, cfg.mp_start_method)
            else:
                rgb = read_region_from_open_slide(slide, x, y, size, size, level)
            if rgb.shape[:2] == (size, size):
                return True, f"level_{level}_ok_at_{x}_{y}"
        except Exception as e:
            errors.append(f"{x},{y}:{str(e)[-180:]}")
    return False, " | ".join(errors[:5])


def choose_read_level(slide_path: Path, slide, width: int, height: int, cfg: Config) -> Tuple[Optional[int], str]:
    for level in cfg.read_levels:
        ok, reason = smoke_test_level(slide_path, slide, width, height, int(level), cfg)
        if ok:
            return int(level), reason
        print(f"[{slide_path.stem}] level {level} smoke test failed: {reason}")
    return None, "all_requested_levels_failed"


def region_metrics(rgb: np.ndarray, cfg: Config) -> Dict[str, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    white = np.all(rgb >= cfg.white_threshold, axis=2)
    tissue = ((gray < 235) & (s > 8)) | (s > 28)

    # DAB/brown and hematoxylin/purple-blue are intentionally broad. This keeps
    # Ventana-positive tissue while rejecting pure colored scanner/background noise.
    brown = ((h >= 5) & (h <= 35) & (s > 35) & (v < 235))
    purple = ((h >= 115) & (h <= 170) & (s > 25) & (v < 245))
    stain = (brown | purple) & tissue

    # Common pen/edge artifacts: very saturated blue/green/red thin marks.
    pen_blue = ((h >= 85) & (h <= 130) & (s > 90) & (v > 60))
    pen_green = ((h >= 40) & (h <= 85) & (s > 90) & (v > 60))
    pen_red = (((h <= 5) | (h >= 175)) & (s > 100) & (v > 60))
    pen = pen_blue | pen_green | pen_red

    edges = cv2.Canny(gray, 80, 160)
    line_fraction = float((edges > 0).mean())

    return {
        "white_fraction": float(white.mean()),
        "tissue_fraction": float(tissue.mean()),
        "stain_fraction": float(stain.mean()),
        "brown_fraction": float((brown & tissue).mean()),
        "purple_fraction": float((purple & tissue).mean()),
        "pen_fraction": float(pen.mean()),
        "edge_fraction": line_fraction,
    }


def accept_region(metrics: Dict[str, float], cfg: Config) -> bool:
    if metrics["white_fraction"] > cfg.max_white_fraction:
        return False
    if metrics["tissue_fraction"] < cfg.min_tissue_fraction:
        return False
    if metrics["stain_fraction"] < cfg.min_stain_fraction:
        return False
    if metrics["pen_fraction"] > cfg.max_pen_fraction and metrics["tissue_fraction"] < 0.20:
        return False
    return True


def build_candidate_grid(width: int, height: int, cfg: Config) -> Iterable[Tuple[int, int]]:
    for y in range(0, height - cfg.region_size + 1, cfg.stride):
        for x in range(0, width - cfg.region_size + 1, cfg.stride):
            yield x, y


def component_filter(records: List[Dict[str, Any]], cfg: Config) -> List[Dict[str, Any]]:
    if not records:
        return []
    xs = sorted({int(r["x0"]) for r in records})
    ys = sorted({int(r["y0"]) for r in records})
    x_to_i = {x: i for i, x in enumerate(xs)}
    y_to_i = {y: i for i, y in enumerate(ys)}
    mask = np.zeros((len(ys), len(xs)), dtype=np.uint8)
    for r in records:
        mask[y_to_i[int(r["y0"])], x_to_i[int(r["x0"])]] = 1

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return records

    valid_labels = [
        i for i in range(1, n_labels)
        if int(stats[i, cv2.CC_STAT_AREA]) >= cfg.min_region_component_size
    ]
    if not valid_labels:
        return []
    if cfg.keep_only_main_component:
        valid_labels = [max(valid_labels, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))]

    kept = []
    valid = set(valid_labels)
    for r in records:
        grid_y = y_to_i[int(r["y0"])]
        grid_x = x_to_i[int(r["x0"])]
        label = int(labels[grid_y, grid_x])
        if label in valid:
            r = dict(r)
            r["object_component_id"] = label
            r["object_component_regions"] = int(stats[label, cv2.CC_STAT_AREA])
            kept.append(r)
    for i, r in enumerate(kept):
        r["index"] = i
    print(f"Component filter kept {len(kept)} / {len(records)} candidate regions")
    return kept


def create_zarr_store(out_path: Path, cfg: Config):
    root = zarr.open(str(out_path), mode="w")
    arr = root.create_dataset(
        "regions",
        shape=(0, cfg.region_size, cfg.region_size, 3),
        chunks=(cfg.zarr_region_chunk_n, cfg.region_size, cfg.region_size, 3),
        dtype="uint8",
        overwrite=True,
    )
    return root, arr


def append_batch(arr, start_idx: int, batch: List[np.ndarray]) -> int:
    if not batch:
        return start_idx
    data = np.stack(batch, axis=0).astype(np.uint8)
    new_n = start_idx + data.shape[0]
    arr.resize((new_n,) + arr.shape[1:])
    arr[start_idx:new_n] = data
    return new_n


def save_qc_overlay(cfg: Config, slide_id: str, width: int, height: int, records: List[Dict[str, Any]]) -> None:
    if not cfg.save_qc:
        return
    ensure_dir(cfg.qc_dir)
    scale = max(width / 2400, height / 2400, 1)
    canvas = np.full((int(height / scale) + 1, int(width / scale) + 1, 3), 255, dtype=np.uint8)
    draw_records = records
    if len(draw_records) > cfg.max_qc_regions_to_draw:
        idx = np.linspace(0, len(draw_records) - 1, cfg.max_qc_regions_to_draw).astype(int)
        draw_records = [draw_records[i] for i in idx]
    for rec in draw_records:
        x0 = int(rec["x0"] / scale)
        y0 = int(rec["y0"] / scale)
        x1 = int(rec["x1"] / scale)
        y1 = int(rec["y1"] / scale)
        color = (0, 150, 0) if rec.get("object_component_id", 1) == 1 else (0, 110, 220)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 1)
    Image.fromarray(canvas).save(cfg.qc_dir / f"{slide_id}_accepted_regions.png")


def collect_regions(
    slide_path: Path,
    slide,
    width: int,
    height: int,
    read_level: int,
    cfg: Config,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, int], np.ndarray], List[Dict[str, Any]]]:
    if cfg.use_process_timeout and cfg.timeout_chunk_size > 1:
        return collect_regions_chunked(slide_path, width, height, read_level, cfg)

    records: List[Dict[str, Any]] = []
    cache: Dict[Tuple[int, int], np.ndarray] = {}
    failed: List[Dict[str, Any]] = []
    consecutive_failures = 0

    coords = list(build_candidate_grid(width, height, cfg))
    for region_i, (x, y) in enumerate(tqdm(coords, desc=f"{slide_path.stem} read/filter"), start=1):
        if cfg.progress_every_regions > 0 and region_i % cfg.progress_every_regions == 0:
            print(
                f"[{slide_path.stem}] region {region_i}/{len(coords)} "
                f"x={x}, y={y}, accepted={len(records)}, failed={len(failed)}",
                flush=True,
            )
        try:
            rgb = read_region_safe(slide_path, slide, x, y, cfg.region_size, cfg.region_size, read_level, cfg)
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            failed.append({
                "slide_id": slide_path.stem,
                "x0": x,
                "y0": y,
                "x1": x + cfg.region_size,
                "y1": y + cfg.region_size,
                "read_level": read_level,
                "error": str(e)[-600:],
            })
            if consecutive_failures >= cfg.max_consecutive_region_failures and cfg.abort_on_failure_limit:
                raise RuntimeError(f"too many consecutive unreadable regions ({consecutive_failures})")
            if len(failed) >= cfg.max_total_region_failures_per_slide and cfg.abort_on_failure_limit:
                raise RuntimeError(f"too many unreadable regions ({len(failed)})")
            continue

        metrics = region_metrics(rgb, cfg)
        if not accept_region(metrics, cfg):
            continue

        rec = {
            "index": len(records),
            "slide_id": slide_path.stem,
            "x0": x,
            "y0": y,
            "x1": x + cfg.region_size,
            "y1": y + cfg.region_size,
            "context_x0": x,
            "context_y0": y,
            "context_x1": x + cfg.region_size,
            "context_y1": y + cfg.region_size,
            "region_size": cfg.region_size,
            "read_level": read_level,
            **{k: round(v, 6) for k, v in metrics.items()},
        }
        records.append(rec)
        cache[(x, y)] = rgb
        if cfg.max_regions_per_slide is not None and len(records) >= cfg.max_regions_per_slide:
            break
    return records, cache, failed


def collect_regions_chunked(
    slide_path: Path,
    width: int,
    height: int,
    read_level: int,
    cfg: Config,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, int], np.ndarray], List[Dict[str, Any]]]:
    coords = list(build_candidate_grid(width, height, cfg))
    records: List[Dict[str, Any]] = []
    cache: Dict[Tuple[int, int], np.ndarray] = {}
    failed_regions: List[Dict[str, Any]] = []
    consecutive_failures = 0
    skip_y: Optional[int] = None

    pbar = tqdm(total=len(coords), desc=f"{slide_path.stem} chunk read/filter")
    i = 0
    while i < len(coords):
        x, y = coords[i]
        if skip_y is not None and y == skip_y:
            failed_regions.append({
                "slide_id": slide_path.stem,
                "x0": x,
                "y0": y,
                "x1": x + cfg.region_size,
                "y1": y + cfg.region_size,
                "read_level": read_level,
                "error": "skipped rest of row after consecutive read timeouts",
            })
            i += 1
            pbar.update(1)
            continue
        if skip_y is not None and y != skip_y:
            skip_y = None
            consecutive_failures = 0

        chunk = coords[i : i + cfg.timeout_chunk_size]
        chunk_records, chunk_cache, chunk_failed = read_filter_chunk_with_bisect(slide_path, chunk, read_level, cfg)

        for rec in chunk_records:
            rec["index"] = len(records)
            records.append(rec)
        cache.update(chunk_cache)
        failed_regions.extend(chunk_failed)

        if chunk_failed and not chunk_records:
            consecutive_failures += len(chunk_failed)
        elif chunk_records:
            consecutive_failures = 0

        if (
            cfg.skip_rest_of_row_after_consecutive_failures
            and consecutive_failures >= cfg.max_consecutive_region_failures
        ):
            skip_y = y
            print(
                f"[{slide_path.stem}] {consecutive_failures} consecutive unreadable regions at y={y}; "
                "skipping rest of this row.",
                flush=True,
            )
            consecutive_failures = 0

        if cfg.abort_on_failure_limit and len(failed_regions) >= cfg.max_total_region_failures_per_slide:
            raise RuntimeError(f"too many unreadable regions ({len(failed_regions)})")

        i += len(chunk)
        pbar.update(len(chunk))
        if cfg.progress_every_regions > 0 and i % cfg.progress_every_regions == 0:
            print(
                f"[{slide_path.stem}] region {min(i, len(coords))}/{len(coords)} "
                f"accepted={len(records)}, failed={len(failed_regions)}",
                flush=True,
            )
        if cfg.max_regions_per_slide is not None and len(records) >= cfg.max_regions_per_slide:
            break

    pbar.close()
    return records, cache, failed_regions


def write_regions(slide_path: Path, out_path: Path, records: List[Dict[str, Any]], cache: Dict[Tuple[int, int], np.ndarray], width: int, height: int, read_level: int, cfg: Config) -> None:
    root, arr = create_zarr_store(out_path, cfg)
    root.attrs.update({
        "slide_id": slide_path.stem,
        "source_isyntax": str(slide_path),
        "width": int(width),
        "height": int(height),
        "region_size": int(cfg.region_size),
        "stride": int(cfg.stride),
        "read_level": int(read_level),
        "filter": "tissue+stain+largest_component",
    })
    written = 0
    batch: List[np.ndarray] = []
    for rec in tqdm(records, desc=f"{slide_path.stem} zarr write"):
        rgb = cache.get((int(rec["x0"]), int(rec["y0"])))
        if rgb is None:
            continue
        batch.append(rgb)
        if len(batch) >= cfg.append_batch_size:
            written = append_batch(arr, written, batch)
            batch.clear()
    if batch:
        append_batch(arr, written, batch)


def convert_one_slide(slide_path: Path, cfg: Config) -> Tuple[str, str, str, str, int]:
    slide_id = slide_path.stem
    out_path = cfg.out_dir / f"{slide_id}.zarr"
    region_csv = cfg.out_dir / f"{slide_id}_regions.csv"
    failed_csv = cfg.out_dir / f"{slide_id}_failed_regions.csv"

    if out_path.exists() and region_csv.exists() and not cfg.overwrite:
        try:
            n_existing = len(pd.read_csv(region_csv))
        except pd.errors.EmptyDataError:
            n_existing = 0
        status = "skipped_exists" if n_existing > 0 else "skipped_exists_no_regions"
        return slide_id, status, str(out_path), "existing output reused", n_existing

    if cfg.overwrite:
        if out_path.exists():
            shutil.rmtree(out_path)
        for p in (region_csv, failed_csv):
            if p.exists():
                p.unlink()

    started = time.time()
    try:
        slide_context = open_slide(slide_path)
    except Exception as e:
        return slide_id, "skipped_open_failed", str(out_path), str(e)[-800:], 0

    with slide_context as slide:
        width, height = get_slide_dimensions(slide)
        read_level, level_reason = choose_read_level(slide_path, slide, width, height, cfg)
        if read_level is None:
            return slide_id, "skipped_no_readable_level", str(out_path), level_reason, 0
        records, cache, failed = collect_regions(slide_path, slide, width, height, read_level, cfg)

    records = component_filter(records, cfg)
    keep_keys = {(int(r["x0"]), int(r["y0"])) for r in records}
    cache = {k: v for k, v in cache.items() if k in keep_keys}
    for i, rec in enumerate(records):
        rec["index"] = i
        rec["zarr_path"] = str(out_path)
        rec["array_path"] = "regions"

    write_regions(slide_path, out_path, records, cache, width, height, int(records[0]["read_level"]) if records else -1, cfg)
    pd.DataFrame(records, columns=REGION_COLUMNS).to_csv(region_csv, index=False)
    if failed:
        pd.DataFrame(failed, columns=FAILED_REGION_COLUMNS).to_csv(failed_csv, index=False)
    save_qc_overlay(cfg, slide_id, width, height, records)

    msg = f"elapsed={time.time() - started:.1f}s; regions={len(records)}; failed_regions={len(failed)}; {level_reason}"
    status = "success" if records else "success_no_regions"
    return slide_id, status, str(out_path), msg, len(records)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert CLDN18.2 iSyntax slides to filtered context-region Zarr.")
    p.add_argument("--slides_dir", default="./slides")
    p.add_argument("--out_dir", default="./region_zarr")
    p.add_argument("--qc_dir", default="./region_zarr_qc")
    p.add_argument("--log_file", default="./context_zarr_conversion_log.csv")
    p.add_argument("--region_size", type=int, default=1024)
    p.add_argument("--stride", type=int, default=1024)
    p.add_argument("--read_levels", default="0,1", help="Comma-separated levels to try, e.g. 0,1")
    p.add_argument("--use_process_timeout", action="store_true")
    p.add_argument(
        "--mp_start_method",
        choices=["auto", "fork", "spawn", "forkserver"],
        default="auto",
        help="Multiprocessing start method for timed iSyntax reads. auto uses fork on Linux and spawn on Windows.",
    )
    p.add_argument("--region_read_timeout_sec", type=int, default=45)
    p.add_argument("--smoke_test_timeout_sec", type=int, default=45)
    p.add_argument("--timeout_chunk_size", type=int, default=8, help="Number of regions read per timeout worker. Use 1 for old per-region timeout behavior.")
    p.add_argument("--chunk_read_timeout_sec", type=int, default=30, help="Timeout for one chunk worker.")
    p.add_argument("--bisect_timeout_chunks", action="store_true", help="Split timed-out chunks to salvage readable regions. Slower on unstable slides.")
    p.add_argument("--max_consecutive_region_failures", type=int, default=4)
    p.add_argument("--abort_on_failure_limit", action="store_true", help="Abort slide when consecutive/total failure limits are reached.")
    p.add_argument("--no_skip_rest_of_row", action="store_true", help="Do not skip remaining regions in a row after repeated read timeouts.")
    p.add_argument("--max_white_fraction", type=float, default=0.82)
    p.add_argument("--min_tissue_fraction", type=float, default=0.04)
    p.add_argument("--min_stain_fraction", type=float, default=0.002)
    p.add_argument("--keep_all_components", action="store_true")
    p.add_argument("--min_region_component_size", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_qc", action="store_true")
    p.add_argument("--max_slides", type=int, default=None)
    p.add_argument("--max_regions_per_slide", type=int, default=None)
    p.add_argument("--progress_every_regions", type=int, default=25)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        slides_dir=Path(args.slides_dir),
        out_dir=Path(args.out_dir),
        qc_dir=Path(args.qc_dir),
        log_file=Path(args.log_file),
        region_size=args.region_size,
        stride=args.stride,
        read_levels=tuple(int(x.strip()) for x in args.read_levels.split(",") if x.strip() != ""),
        use_process_timeout=args.use_process_timeout,
        mp_start_method=args.mp_start_method,
        region_read_timeout_sec=args.region_read_timeout_sec,
        smoke_test_timeout_sec=args.smoke_test_timeout_sec,
        timeout_chunk_size=args.timeout_chunk_size,
        chunk_read_timeout_sec=args.chunk_read_timeout_sec,
        bisect_timeout_chunks=args.bisect_timeout_chunks,
        max_consecutive_region_failures=args.max_consecutive_region_failures,
        abort_on_failure_limit=args.abort_on_failure_limit,
        skip_rest_of_row_after_consecutive_failures=not args.no_skip_rest_of_row,
        max_white_fraction=args.max_white_fraction,
        min_tissue_fraction=args.min_tissue_fraction,
        min_stain_fraction=args.min_stain_fraction,
        keep_only_main_component=not args.keep_all_components,
        min_region_component_size=args.min_region_component_size,
        overwrite=args.overwrite,
        save_qc=not args.no_qc,
        max_slides=args.max_slides,
        max_regions_per_slide=args.max_regions_per_slide,
        progress_every_regions=args.progress_every_regions,
    )


def main() -> None:
    cfg = build_config(parse_args())
    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.qc_dir)
    with (cfg.out_dir / "direct_context_zarr_config.json").open("w") as f:
        json.dump(json_safe(asdict(cfg)), f, indent=2)

    slides = sorted(cfg.slides_dir.glob("*.isyntax"))
    if cfg.max_slides is not None:
        slides = slides[: cfg.max_slides]
    if not slides:
        raise SystemExit(f"No .isyntax files found in {cfg.slides_dir}")

    manifest_path = cfg.out_dir / "all_regions_manifest.csv"
    if cfg.overwrite and manifest_path.exists():
        manifest_path.unlink()
    manifest_has_header = manifest_path.exists() and manifest_path.stat().st_size > 0
    log_has_header = cfg.log_file.exists() and cfg.log_file.stat().st_size > 0

    with cfg.log_file.open("a", newline="") as log_f:
        log_writer = csv.writer(log_f)
        if not log_has_header:
            log_writer.writerow(["slide_id", "status", "zarr_path", "message", "num_regions"])

        for i, slide_path in enumerate(slides, start=1):
            print(f"\n[{i}/{len(slides)}] {slide_path.name}")
            try:
                slide_id, status, zarr_path, message, n_regions = convert_one_slide(slide_path, cfg)
            except Exception as e:
                slide_id = slide_path.stem
                status = "error"
                zarr_path = str(cfg.out_dir / f"{slide_id}.zarr")
                message = f"{str(e)[-1000:]}\n{traceback.format_exc()[-2000:]}"
                n_regions = 0
            print(f"[{slide_id}] {status}: {message}")
            log_writer.writerow([slide_id, status, zarr_path, message, n_regions])
            log_f.flush()

            region_csv = cfg.out_dir / f"{slide_id}_regions.csv"
            if status in ("success", "skipped_exists") and region_csv.exists() and region_csv.stat().st_size > 0:
                try:
                    rdf = pd.read_csv(region_csv)
                except pd.errors.EmptyDataError:
                    rdf = pd.DataFrame(columns=REGION_COLUMNS)
                if not rdf.empty:
                    rdf.to_csv(manifest_path, mode="a", index=False, header=not manifest_has_header)
                    manifest_has_header = True

    print(f"\nDone. Manifest: {manifest_path}")
    print("Next:")
    print("python3.10 cldn18_gpu_hovernet_pipeline_revised.py --mode all --manifest region_zarr/all_regions_manifest.csv --output_root cldn18_outputs --num_gpus 8")


if __name__ == "__main__":
    main()
