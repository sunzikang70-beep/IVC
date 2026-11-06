"""Evaluate controlled BOP-dataset refinement trials with BOP pose errors.

This script follows the BOP19 VSD/MSSD/MSPD definitions and thresholds, but
scores the project's controlled corruption trials rather than the official
T-LESS target list. The resulting values are therefore BOP-style metrics for
the controlled protocol, not a BOP leaderboard submission score.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import torch

from bop_thresholds import attach_image_widths, normalized_mspd


# Dataset and toolkit roots come from the environment; the original study
# machine's paths are deliberately not hard-coded.
#   BOP_DATASETS=<root containing tless/>
#   BOP_TOOLKIT=<checkout containing bop_toolkit_lib/>
DEFAULT_DATASET = (Path(os.environ["BOP_DATASETS"]) / "tless") if os.environ.get("BOP_DATASETS") else None
DEFAULT_TOOLKIT = Path(os.environ["BOP_TOOLKIT"]) if os.environ.get("BOP_TOOLKIT") else None
DEFAULT_METHODS = (
    "protocolB_e2e_initial_pose,"
    "protocolB_e2e_open3d_point_to_point_icp,"
    "protocolB_e2e_open3d_point_to_plane_icp,"
    "protocolB_e2e_open3d_tukey_icp,"
    "protocolB_e2e_open3d_generalized_icp,"
    "protocolB_e2e_geometry_icp,"
    "protocolB_e2e_direct_local_icp,"
    "protocolB_e2e_geometry_then_local_icp,"
    "protocolB_e2e_reliability_gated_icp"
)

VSD_TAUS = np.arange(0.05, 0.51, 0.05)
VSD_CORRECTNESS = np.arange(0.05, 0.51, 0.05)
MSSD_CORRECTNESS = np.arange(0.05, 0.51, 0.05)
MSPD_CORRECTNESS = np.arange(5.0, 51.0, 5.0)
FLIP_Y = np.diag([1.0, -1.0, 1.0])
BASE_KEY_COLUMNS = [
    "object_id",
    "scene_id",
    "img_id",
    "inst_id",
    "corruption_type",
    "requested_corruption_fraction",
    "trial",
]
KEY_COLUMNS = BASE_KEY_COLUMNS.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate BOP19-style AR on controlled BOP-dataset trials."
    )
    parser.add_argument("--trials-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dataset-name", default="T-LESS")
    parser.add_argument("--dataset-split", default="test_primesense")
    parser.add_argument("--depth-extension", default=".png")
    parser.add_argument("--toolkit-root", type=Path, default=DEFAULT_TOOLKIT)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument(
        "--metrics",
        default="vsd,mssd,mspd",
        help="Comma-separated subset of vsd,mssd,mspd.",
    )
    parser.add_argument(
        "--frames-per-object",
        type=int,
        default=10,
        help="Planned frame count used by the source experiment.",
    )
    parser.add_argument("--gpu-batch-size", type=int, default=16)
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto"
    )
    parser.add_argument(
        "--max-conditions",
        type=int,
        default=None,
        help="Smoke-test limit applied before expanding across methods.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--extra-key-columns",
        default="",
        help="Additional comma-separated condition columns, e.g. mask_variant.",
    )
    return parser.parse_args()


def activate_toolkit(toolkit_root: Path) -> None:
    if not (toolkit_root / "bop_toolkit_lib").is_dir():
        raise FileNotFoundError(f"BOP Toolkit not found: {toolkit_root}")
    sys.path.insert(0, str(toolkit_root))


def parse_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def read_trials(args: argparse.Namespace) -> pd.DataFrame:
    data = pd.read_csv(args.trials_csv)
    required = set(KEY_COLUMNS + ["method", "R_00", "R_22", "t_x_mm", "t_z_mm"])
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Trials CSV is missing pose columns: {missing}")

    requested = parse_list(args.methods)
    available = set(data["method"].unique())
    selected = [method for method in requested if method in available]
    unavailable = [method for method in requested if method not in available]
    if unavailable:
        print("Skipping methods absent from CSV:", ", ".join(unavailable))
    if not selected:
        raise ValueError("None of the requested methods is present in the trials CSV.")

    data = data[data["method"].isin(selected)].copy()
    if args.max_conditions is not None:
        conditions = data[KEY_COLUMNS].drop_duplicates().head(args.max_conditions)
        data = data.merge(conditions, on=KEY_COLUMNS, how="inner")

    duplicate = data.duplicated(KEY_COLUMNS + ["method"])
    if duplicate.any():
        raise ValueError(f"Found {int(duplicate.sum())} duplicate condition/method rows.")

    expected = len(selected)
    counts = data.groupby(KEY_COLUMNS, dropna=False)["method"].nunique()
    if not counts.eq(expected).all():
        raise ValueError("At least one condition does not contain every selected method.")
    return data.reset_index(drop=True)


def load_frame_metadata(
    dataset_root: Path, row: pd.Series, dataset_split: str = "test_primesense"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    scene_path = dataset_root / dataset_split / f"{int(row.scene_id):06d}"
    key = str(int(row.img_id))
    camera = json.loads((scene_path / "scene_camera.json").read_text())[key]
    poses = json.loads((scene_path / "scene_gt.json").read_text())[key]
    pose = poses[int(row.inst_id)]
    if int(pose["obj_id"]) != int(row.object_id):
        raise ValueError(f"GT object mismatch for {tuple(row[c] for c in KEY_COLUMNS[:4])}")
    K = np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3)
    R_gt = np.asarray(pose["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
    t_gt = np.asarray(pose["cam_t_m2c"], dtype=np.float64).reshape(3, 1)
    return K, R_gt, t_gt, float(camera["depth_scale"])


def internal_pose_to_bop(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    R_internal = np.asarray(
        [
            [row.R_00, row.R_01, row.R_02],
            [row.R_10, row.R_11, row.R_12],
            [row.R_20, row.R_21, row.R_22],
        ],
        dtype=np.float64,
    )
    t_internal = np.asarray(
        [row.t_x_mm, row.t_y_mm, row.t_z_mm], dtype=np.float64
    ).reshape(3, 1)
    R_bop = FLIP_Y @ R_internal
    t_bop = FLIP_Y @ t_internal
    if not np.isclose(np.linalg.det(R_bop), 1.0, atol=2e-4):
        raise ValueError(f"Converted rotation is not proper for row {row.name}.")
    return R_bop, t_bop


def select_evenly(candidates: list[tuple[int, int, int]], count: int) -> list[tuple[int, int, int]]:
    if len(candidates) <= count:
        return candidates.copy()
    indices = np.round(np.linspace(0, len(candidates) - 1, count)).astype(int)
    selected = []
    seen = set()
    for index in indices:
        index = int(index)
        if index not in seen:
            selected.append(candidates[index])
            seen.add(index)
    return selected


def selected_frame_indices(
    dataset_root: Path,
    object_ids: list[int],
    frames_per_object: int,
    dataset_split: str = "test_primesense",
) -> dict[tuple[int, int, int, int], int]:
    mapping: dict[tuple[int, int, int, int], int] = {}
    test_root = dataset_root / dataset_split
    for object_id in object_ids:
        candidates = []
        for scene_path in sorted(test_root.iterdir(), key=lambda p: int(p.name)):
            if not scene_path.is_dir() or not scene_path.name.isdigit():
                continue
            gt = json.loads((scene_path / "scene_gt.json").read_text())
            info_path = scene_path / "scene_gt_info.json"
            info = json.loads(info_path.read_text()) if info_path.exists() else None
            for image_key in sorted(gt, key=int):
                matches = []
                for instance_id, pose in enumerate(gt[image_key]):
                    if int(pose["obj_id"]) != object_id:
                        continue
                    detail = info[image_key][instance_id] if info is not None else {}
                    visib = float(detail.get("visib_fract", 1.0))
                    pixels = int(detail.get("px_count_visib", -1))
                    if visib >= 0.30 and (pixels < 0 or pixels >= 500):
                        matches.append((visib, pixels, int(scene_path.name), int(image_key), instance_id))
                if matches:
                    best = max(matches, key=lambda item: (item[0], item[1]))
                    candidates.append((best[2], best[3], best[4]))
        candidates.sort()
        for frame_index, (scene_id, image_id, instance_id) in enumerate(
            select_evenly(candidates, frames_per_object)
        ):
            mapping[(object_id, scene_id, image_id, instance_id)] = frame_index
    return mapping


def reconstruct_corrupted_depth(
    dataset_root: Path,
    row: pd.Series,
    frame_index: int,
    dataset_split: str = "test_primesense",
    depth_extension: str = ".png",
) -> tuple[np.ndarray, np.ndarray]:
    scene_path = dataset_root / dataset_split / f"{int(row.scene_id):06d}"
    image_id = int(row.img_id)
    instance_id = int(row.inst_id)
    image_name = f"{image_id:06d}.{depth_extension.lstrip('.')}"
    camera_key = str(image_id)
    camera = json.loads((scene_path / "scene_camera.json").read_text())[camera_key]
    depth_raw = cv2.imread(str(scene_path / "depth" / image_name), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(
        str(scene_path / "mask_visib" / f"{image_id:06d}_{instance_id:06d}.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    if depth_raw is None or mask is None:
        raise FileNotFoundError(f"Missing depth or mask for scene {row.scene_id}, image {image_id}")

    depth_mm = depth_raw.astype(np.float64) * float(camera["depth_scale"])
    mask_bool = mask > 0
    if row.corruption_type == "clean_depth":
        return depth_mm, np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3)

    valid = mask_bool & (depth_mm > 0)
    v_valid, u_valid = np.nonzero(valid)
    fraction = float(row.requested_corruption_fraction)
    if {"patch_center_u", "patch_center_v"}.issubset(row.index) and int(row.patch_center_u) >= 0:
        center_u = int(row.patch_center_u)
        center_v = int(row.patch_center_v)
    else:
        patch_seed = 200000 + int(row.object_id) * 1000000 + frame_index * 10000 + int(row.trial)
        rng = np.random.default_rng(patch_seed)
        center_index = int(rng.integers(0, len(v_valid)))
        center_v, center_u = int(v_valid[center_index]), int(u_valid[center_index])
    k = max(1, min(int(np.round(fraction * len(v_valid))), len(v_valid)))
    dist2 = (v_valid - center_v) ** 2 + (u_valid - center_u) ** 2
    selected = np.arange(len(v_valid)) if k == len(v_valid) else np.argpartition(dist2, k - 1)[:k]
    patch = np.zeros_like(mask_bool)
    patch[v_valid[selected], u_valid[selected]] = True

    corrupted = depth_mm.copy()
    # Match run_frozen_real_inputs_validation.py for the external-input
    # sensitivity runs. Legacy rows without a recorded condition seed retain
    # the historical deterministic formula.
    if row.corruption_type == "local_outlier" and "patch_center_u" in row.index:
        value_seed = 910000 + int(row.object_id) * 1000000 + frame_index * 10000
    else:
        value_seed = (
            500000
            + int(row.object_id) * 1000000
            + frame_index * 10000
            + int(row.trial) * 10
            + ["local_bias", "local_outlier", "local_hole"].index(row.corruption_type)
        )
    value_rng = np.random.default_rng(value_seed)
    if row.corruption_type == "local_bias":
        sign = 1.0 if value_rng.random() >= 0.5 else -1.0
        corrupted[patch] += sign * 10.0
    elif row.corruption_type == "local_outlier":
        magnitudes = value_rng.uniform(10.0, 50.0, size=int(patch.sum()))
        signs = value_rng.choice(np.asarray([-1.0, 1.0]), size=int(patch.sum()))
        corrupted[patch] += signs * magnitudes
    elif row.corruption_type == "local_hole":
        corrupted[patch] = 0.0
    else:
        raise ValueError(f"Unsupported corruption type: {row.corruption_type}")
    corrupted[corrupted <= 1.0] = 0.0
    K = np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3)
    return corrupted, K


class Open3DRaycastRenderer:
    """BOP renderer interface backed by CPU ray casting."""

    def __init__(self, width: int, height: int, cache_size: int = 16):
        self.width = width
        self.height = height
        self.cache_size = cache_size
        self.scenes: dict[int, o3d.t.geometry.RaycastingScene] = {}
        self.cache: OrderedDict[tuple, np.ndarray] = OrderedDict()

    def add_object(self, obj_id: int, model_path: str) -> None:
        legacy = o3d.io.read_triangle_mesh(model_path)
        vertices = np.asarray(legacy.vertices, dtype=np.float32)
        triangles = np.asarray(legacy.triangles, dtype=np.uint32)
        if len(vertices) == 0 or len(triangles) == 0:
            raise ValueError(f"Model has no mesh geometry: {model_path}")
        mesh = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(vertices), o3d.core.Tensor(triangles)
        )
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)
        self.scenes[int(obj_id)] = scene

    def render_object(self, obj_id, R, t, fx, fy, cx, cy):
        pose_key = np.asarray(np.hstack([R.reshape(-1), np.asarray(t).reshape(-1)]), dtype=np.float64).tobytes()
        key = (int(obj_id), float(fx), float(fy), float(cx), float(cy), pose_key)
        if key in self.cache:
            depth = self.cache.pop(key)
            self.cache[key] = depth
            return {"depth": depth}

        K = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(K),
            o3d.core.Tensor(np.eye(4, dtype=np.float32)),
            self.width,
            self.height,
        ).numpy()
        R = np.asarray(R, dtype=np.float32).reshape(3, 3)
        t = np.asarray(t, dtype=np.float32).reshape(3)
        rays[..., :3] = (-t @ R).reshape(1, 1, 3)
        rays[..., 3:] = rays[..., 3:] @ R
        hits = self.scenes[int(obj_id)].cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        hits[~np.isfinite(hits)] = 0.0
        depth = hits.astype(np.float32, copy=False)
        self.cache[key] = depth
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return {"depth": depth}


def prepare_pose_arrays(
    data: pd.DataFrame,
    dataset_root: Path,
    dataset_split: str = "test_primesense",
) -> dict[str, np.ndarray]:
    arrays = {
        "R_est": [],
        "t_est": [],
        "R_gt": [],
        "t_gt": [],
        "K": [],
    }
    metadata_cache = {}
    for _, row in data.iterrows():
        frame_key = (int(row.object_id), int(row.scene_id), int(row.img_id), int(row.inst_id))
        if frame_key not in metadata_cache:
            metadata_cache[frame_key] = load_frame_metadata(
                dataset_root, row, dataset_split
            )
        K, R_gt, t_gt, _ = metadata_cache[frame_key]
        R_est, t_est = internal_pose_to_bop(row)
        arrays["R_est"].append(R_est)
        arrays["t_est"].append(t_est)
        arrays["R_gt"].append(R_gt)
        arrays["t_gt"].append(t_gt)
        arrays["K"].append(K)
    return {key: np.asarray(value) for key, value in arrays.items()}


def compute_surface_projection_errors(
    data: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    dataset_root: Path,
    metrics: set[str],
    device_name: str,
    batch_size: int,
):
    from bop_toolkit_lib import inout, misc, pose_error, pose_error_gpu

    model_info = inout.load_json(
        str(dataset_root / "models_eval" / "models_info.json"), keys_to_int=True
    )
    mssd_values = np.full(len(data), np.nan)
    mspd_values = np.full(len(data), np.nan)
    device = torch.device(device_name)

    for object_id in sorted(data["object_id"].unique()):
        indices = np.flatnonzero(data["object_id"].to_numpy() == object_id)
        model = inout.load_ply(str(dataset_root / "models_eval" / f"obj_{int(object_id):06d}.ply"))
        points = np.asarray(model["pts"], dtype=np.float32)
        symmetries = misc.get_symmetry_transformations(model_info[int(object_id)], 0.01)
        print(f"Object {object_id}: {len(indices)} estimates, {len(points)} points, {len(symmetries)} symmetries")

        if device.type == "cuda":
            pts_t = torch.as_tensor(points, device=device)
            syms_t = [
                {
                    "R": torch.as_tensor(sym["R"], dtype=torch.float32, device=device),
                    "t": torch.as_tensor(sym["t"], dtype=torch.float32, device=device),
                }
                for sym in symmetries
            ]
            R_est = torch.as_tensor(arrays["R_est"][indices], dtype=torch.float32, device=device)
            t_est = torch.as_tensor(arrays["t_est"][indices], dtype=torch.float32, device=device)
            R_gt = torch.as_tensor(arrays["R_gt"][indices], dtype=torch.float32, device=device)
            t_gt = torch.as_tensor(arrays["t_gt"][indices], dtype=torch.float32, device=device)
            K = torch.as_tensor(arrays["K"][indices], dtype=torch.float32, device=device)
            if "mssd" in metrics:
                mssd_values[indices] = pose_error_gpu.mssd_by_batch(
                    R_est, t_est, R_gt, t_gt, pts_t, syms_t, max_batch_size=batch_size
                ).cpu().numpy()
            if "mspd" in metrics:
                mspd_values[indices] = pose_error_gpu.mspd_by_batch(
                    R_est, t_est, R_gt, t_gt, K, pts_t, syms_t, max_batch_size=batch_size
                ).cpu().numpy()
        else:
            for local_index, row_index in enumerate(indices):
                if "mssd" in metrics:
                    mssd_values[row_index] = pose_error.mssd(
                        arrays["R_est"][row_index], arrays["t_est"][row_index],
                        arrays["R_gt"][row_index], arrays["t_gt"][row_index], points, symmetries,
                    )
                if "mspd" in metrics:
                    mspd_values[row_index] = pose_error.mspd(
                        arrays["R_est"][row_index], arrays["t_est"][row_index],
                        arrays["R_gt"][row_index], arrays["t_gt"][row_index],
                        arrays["K"][row_index], points, symmetries,
                    )
                if (local_index + 1) % 20 == 0:
                    print(f"  CPU progress: {local_index + 1}/{len(indices)}")
    return mssd_values, mspd_values


def compute_vsd_errors(
    data: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    dataset_root: Path,
    output_dir: Path,
    frames_per_object: int,
    checkpoint_every: int,
    resume: bool,
    dataset_split: str = "test_primesense",
    depth_extension: str = ".png",
) -> np.ndarray:
    from bop_toolkit_lib import inout, pose_error

    model_info = inout.load_json(
        str(dataset_root / "models_eval" / "models_info.json"), keys_to_int=True
    )
    first = data.iloc[0]
    first_depth_path = (
        dataset_root
        / dataset_split
        / f"{int(first.scene_id):06d}"
        / "depth"
        / f"{int(first.img_id):06d}.{depth_extension.lstrip('.')}"
    )
    first_depth = cv2.imread(str(first_depth_path), cv2.IMREAD_UNCHANGED)
    if first_depth is None:
        raise FileNotFoundError(f"Missing depth image: {first_depth_path}")
    height, width = first_depth.shape[:2]
    renderer = Open3DRaycastRenderer(width=width, height=height)
    for object_id in sorted(data["object_id"].unique()):
        renderer.add_object(
            int(object_id),
            str(dataset_root / "models_eval" / f"obj_{int(object_id):06d}.ply"),
        )

    if "selected_frame_index" in data.columns:
        frame_rows = data[
            ["object_id", "scene_id", "img_id", "inst_id", "selected_frame_index"]
        ].drop_duplicates()
        if frame_rows.duplicated(["object_id", "scene_id", "img_id", "inst_id"]).any():
            raise ValueError("A frame maps to multiple selected_frame_index values.")
        frame_indices = {
            (int(row.object_id), int(row.scene_id), int(row.img_id), int(row.inst_id)):
            int(row.selected_frame_index)
            for _, row in frame_rows.iterrows()
        }
    else:
        frame_indices = selected_frame_indices(
            dataset_root,
            [int(value) for value in sorted(data["object_id"].unique())],
            frames_per_object,
            dataset_split,
        )
    checkpoint_path = output_dir / "vsd_checkpoint.csv"
    done = {}
    if resume and checkpoint_path.exists():
        previous = pd.read_csv(checkpoint_path)
        for _, item in previous.iterrows():
            done[int(item.row_index)] = [item[f"VSD_tau_{tau:.2f}"] for tau in VSD_TAUS]
        print(f"Resuming {len(done)} VSD rows from {checkpoint_path}")

    values = np.full((len(data), len(VSD_TAUS)), np.nan)
    pose_cache = {}
    rendered_count = 0
    reused_count = 0
    for row_index, row in data.iterrows():
        pose_key = (
            *(row[column] for column in BASE_KEY_COLUMNS),
            arrays["R_est"][row_index].tobytes(),
            arrays["t_est"][row_index].tobytes(),
        )
        if row_index in done:
            values[row_index] = done[row_index]
            pose_cache[pose_key] = values[row_index].copy()
            continue
        if pose_key in pose_cache:
            values[row_index] = pose_cache[pose_key]
            reused_count += 1
            continue
        frame_key = (int(row.object_id), int(row.scene_id), int(row.img_id), int(row.inst_id))
        if frame_key not in frame_indices:
            raise ValueError(f"Frame is not in reconstructed selection: {frame_key}")
        depth_test, K = reconstruct_corrupted_depth(
            dataset_root,
            row,
            frame_indices[frame_key],
            dataset_split,
            depth_extension,
        )
        diameter = float(model_info[int(row.object_id)]["diameter"])
        values[row_index] = pose_error.vsd(
            arrays["R_est"][row_index],
            arrays["t_est"][row_index],
            arrays["R_gt"][row_index],
            arrays["t_gt"][row_index],
            depth_test,
            K,
            15.0,
            list(VSD_TAUS),
            True,
            diameter,
            renderer,
            int(row.object_id),
            cost_type="step",
        )
        pose_cache[pose_key] = values[row_index].copy()
        rendered_count += 1
        if (row_index + 1) % checkpoint_every == 0 or row_index + 1 == len(data):
            checkpoint = pd.DataFrame({"row_index": np.arange(len(data))})
            for tau_index, tau in enumerate(VSD_TAUS):
                checkpoint[f"VSD_tau_{tau:.2f}"] = values[:, tau_index]
            checkpoint = checkpoint[checkpoint[f"VSD_tau_{VSD_TAUS[0]:.2f}"].notna()]
            checkpoint.to_csv(checkpoint_path, index=False)
            print(f"VSD progress: {row_index + 1}/{len(data)}")
    checkpoint = pd.DataFrame({"row_index": np.arange(len(data))})
    for tau_index, tau in enumerate(VSD_TAUS):
        checkpoint[f"VSD_tau_{tau:.2f}"] = values[:, tau_index]
    checkpoint.to_csv(checkpoint_path, index=False)
    print(f"VSD poses rendered: {rendered_count}; exact pose reuses: {reused_count}")
    return values


def average_recall(group: pd.DataFrame, metric: str) -> float:
    if metric == "mssd":
        errors = group["MSSD_normalized"].to_numpy()
        return float(np.mean([np.mean(errors < threshold) for threshold in MSSD_CORRECTNESS]))
    if metric == "mspd":
        errors = normalized_mspd(group)
        return float(np.mean([np.mean(errors < threshold) for threshold in MSPD_CORRECTNESS]))
    recalls = []
    for tau in VSD_TAUS:
        errors = group[f"VSD_tau_{tau:.2f}"].to_numpy()
        recalls.extend(np.mean(errors < threshold) for threshold in VSD_CORRECTNESS)
    return float(np.mean(recalls))


def aggregate_scores(data: pd.DataFrame, metrics: set[str]) -> pd.DataFrame:
    rows = []
    scopes = [("all", data)] + [
        (name, group) for name, group in data.groupby("corruption_type", sort=True)
    ]
    for scope, scoped in scopes:
        for method, group in scoped.groupby("method", sort=False):
            row = {
                "scope": scope,
                "method": method,
                "conditions": len(group),
                "independent_frames": group["frame_id"].nunique(),
            }
            available = []
            for metric in ["vsd", "mssd", "mspd"]:
                if metric in metrics:
                    score = average_recall(group, metric)
                    row[f"AR_{metric.upper()}"] = score
                    available.append(score)
            row["AR_mean_available"] = float(np.mean(available))
            rows.append(row)
    return pd.DataFrame(rows)


def per_object_scores(data: pd.DataFrame, metrics: set[str]) -> pd.DataFrame:
    rows = []
    for (object_id, method), group in data.groupby(["object_id", "method"], sort=True):
        row = {
            "object_id": int(object_id),
            "method": method,
            "conditions": len(group),
        }
        available = []
        for metric in ["vsd", "mssd", "mspd"]:
            if metric in metrics:
                score = average_recall(group, metric)
                row[f"AR_{metric.upper()}"] = score
                available.append(score)
        row["AR_mean_available"] = float(np.mean(available))
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_recalls(data: pd.DataFrame, metrics: set[str]) -> pd.DataFrame:
    rows = []
    for method, group in data.groupby("method", sort=False):
        if "mssd" in metrics:
            for threshold in MSSD_CORRECTNESS:
                rows.append({"method": method, "metric": "MSSD", "tau": np.nan, "correctness_threshold": threshold, "recall": np.mean(group["MSSD_normalized"] < threshold)})
        if "mspd" in metrics:
            for threshold in MSPD_CORRECTNESS:
                rows.append({"method": method, "metric": "MSPD", "tau": np.nan, "correctness_threshold": threshold, "recall": np.mean(normalized_mspd(group) < threshold)})
        if "vsd" in metrics:
            for tau in VSD_TAUS:
                for threshold in VSD_CORRECTNESS:
                    rows.append({"method": method, "metric": "VSD", "tau": tau, "correctness_threshold": threshold, "recall": np.mean(group[f"VSD_tau_{tau:.2f}"] < threshold)})
    return pd.DataFrame(rows)


def dataframe_to_markdown(data: pd.DataFrame, decimals: int = 6) -> str:
    columns = list(data.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in data.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("" if not np.isfinite(value) else f"{value:.{decimals}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(args, data, scores, per_object, metrics, device_name):
    all_scores = scores[scores["scope"].eq("all")].sort_values(
        "AR_mean_available", ascending=False
    )
    metric_text = ", ".join(metric.upper() for metric in sorted(metrics))
    integrity_lines = [
        f"- Rows evaluated: {len(data)}",
        f"- Independent RGB-D frames: {data['frame_id'].nunique()}",
        f"- Objects: {data['object_id'].nunique()}",
        "- Internal Y-up estimates were converted back to OpenCV/BOP Y-down coordinates before evaluation.",
    ]
    if {"mssd", "mspd"}.intersection(metrics):
        integrity_lines.append(
            "- MSSD/MSPD use `models_eval` and symmetries discretized with `max_sym_disc_step=0.01`."
        )
    if "vsd" in metrics:
        integrity_lines.append(
            "- VSD uses delta=15 mm and the synthetically corrupted full test depth reconstructed from frozen seeds."
        )
    if "mspd" in metrics:
        integrity_lines.append(
            "- MSPD is normalized by 640/image_width before applying 5:5:50 thresholds; image widths are measured from dataset depth images."
        )

    lines = [
        "# Controlled-Protocol BOP Metric Report",
        "",
        "## Material Passport",
        "",
        "- Origin: experiment-agent / BOP metric extension",
        "- Verification Status: ANALYZED",
        f"- Protocol: controlled {args.dataset_name} local-depth corruption evaluation",
        f"- Source: `{args.trials_csv}`",
        f"- Metrics: {metric_text}",
        f"- Compute device: {device_name}",
        "",
        "## Scope Warning",
        "",
        f"These scores use the BOP19 pose-error definitions, official thresholds, {args.dataset_name} eval models, and official symmetry discretization. They are calculated on custom repeated corruption conditions rather than an official BOP target list; they must be reported as controlled-protocol BOP-style AR, not as an official BOP leaderboard score.",
        "",
        "## Main Scores",
        "",
        dataframe_to_markdown(all_scores),
        "",
        "## Integrity Checks",
        "",
        *integrity_lines,
        "",
        "## Per-Object Scores",
        "",
        dataframe_to_markdown(per_object),
        "",
    ]
    (args.output_dir / "bop_metric_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global KEY_COLUMNS
    args = parse_args()
    extra_key_columns = parse_list(args.extra_key_columns)
    duplicate_keys = sorted(set(extra_key_columns).intersection(BASE_KEY_COLUMNS))
    if duplicate_keys:
        raise ValueError(f"Extra key columns duplicate base keys: {duplicate_keys}")
    KEY_COLUMNS = BASE_KEY_COLUMNS + extra_key_columns
    if args.dataset_root is None:
        raise SystemExit("--dataset-root is required (or set BOP_DATASETS).")
    if args.toolkit_root is None:
        raise SystemExit("--toolkit-root is required (or set BOP_TOOLKIT).")
    activate_toolkit(args.toolkit_root)
    metrics = set(parse_list(args.metrics))
    unsupported = metrics.difference({"vsd", "mssd", "mspd"})
    if unsupported:
        raise ValueError(f"Unsupported metrics: {sorted(unsupported)}")
    if not (args.dataset_root / "models_eval" / "models_info.json").exists():
        raise FileNotFoundError("models_eval is required for BOP metrics.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = read_trials(args)
    arrays = prepare_pose_arrays(data, args.dataset_root, args.dataset_split)
    determinants = np.linalg.det(arrays["R_est"])
    if not np.allclose(determinants, 1.0, atol=2e-4):
        raise ValueError("Pose-coordinate conversion determinant check failed.")

    mssd, mspd = compute_surface_projection_errors(
        data, arrays, args.dataset_root, metrics, device_name, args.gpu_batch_size
    )
    if "mssd" in metrics:
        data["MSSD_mm"] = mssd
        data["MSSD_normalized"] = mssd / data["diameter_mm"].to_numpy()
    if "mspd" in metrics:
        data["MSPD_px"] = mspd
        data = attach_image_widths(data, args.dataset_root, args.dataset_split, args.depth_extension)
    if "vsd" in metrics:
        vsd = compute_vsd_errors(
            data,
            arrays,
            args.dataset_root,
            args.output_dir,
            args.frames_per_object,
            args.checkpoint_every,
            args.resume,
            args.dataset_split,
            args.depth_extension,
        )
        for index, tau in enumerate(VSD_TAUS):
            data[f"VSD_tau_{tau:.2f}"] = vsd[:, index]

    scores = aggregate_scores(data, metrics)
    objects = per_object_scores(data, metrics)
    recalls = threshold_recalls(data, metrics)
    data.to_csv(args.output_dir / "bop_pose_errors.csv", index=False)
    scores.to_csv(args.output_dir / "table_bop_ar_summary.csv", index=False)
    objects.to_csv(args.output_dir / "table_bop_ar_per_object.csv", index=False)
    recalls.to_csv(args.output_dir / "table_bop_threshold_recalls.csv", index=False)
    write_report(args, data, scores, objects, metrics, device_name)
    print("BOP-style evaluation complete:", args.output_dir)


if __name__ == "__main__":
    main()
