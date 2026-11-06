"""Extract GT-free render-and-compare features for Protocol-G candidates."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from evaluate_bop_metrics_controlled import (
    Open3DRaycastRenderer,
    internal_pose_to_bop,
)
from extract_real_gtfree_acceptance_features import masks_by_source_index
import run_p2p_reliability_filter_development as support
from protocolG_observation import (
    OBSERVATION_FIELDS, OBSERVATION_VERSION, replay_method_depth, validate_condition_rows,
)
from run_protocolG_multisolver_smoke import SOLVER_VERSION
from run_frozen_real_inputs_validation import (
    MASK_JSON,
    candidate_from_manifest,
    decode_uncompressed_rle,
)


KEYS = ["object_id", "scene_id", "img_id", "inst_id", "frame_id", "input_depth_condition", "method"]


def load_sensor_frame(dataset_root, row, dataset_split, depth_extension):
    """Read only the measured depth and intrinsics, never a GT mask or pose."""
    scene = dataset_root / dataset_split / f"{int(row.scene_id):06d}"
    camera = json.loads((scene / "scene_camera.json").read_text(encoding="utf-8"))[str(int(row.img_id))]
    depth = cv2.imread(
        str(scene / "depth" / f"{int(row.img_id):06d}.{depth_extension.lstrip('.')}"),
        cv2.IMREAD_UNCHANGED,
    )
    if depth is None:
        raise FileNotFoundError(f"Missing sensor depth for {row.frame_id}")
    return {
        "depth": depth.astype(np.float64) * float(camera["depth_scale"]) / 1000.0,
        "K": np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3),
    }


def boundary(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return mask.astype(bool) & ~eroded.astype(bool)


def symmetric_boundary_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    edge_a = boundary(a)
    edge_b = boundary(b)
    if not edge_a.any() or not edge_b.any():
        return float("inf"), float("inf"), float("inf")
    distance_to_a = cv2.distanceTransform((~edge_a).astype(np.uint8), cv2.DIST_L2, 3)
    distance_to_b = cv2.distanceTransform((~edge_b).astype(np.uint8), cv2.DIST_L2, 3)
    a_to_b = float(np.mean(distance_to_b[edge_a]))
    b_to_a = float(np.mean(distance_to_a[edge_b]))
    return 0.5 * (a_to_b + b_to_a), a_to_b, b_to_a


def compare(rendered_depth_mm: np.ndarray, observed_depth_mm: np.ndarray, predicted_mask: np.ndarray) -> dict:
    rendered_mask = rendered_depth_mm > 0
    predicted_mask = predicted_mask.astype(bool)
    intersection = rendered_mask & predicted_mask
    union = rendered_mask | predicted_mask
    rendered_pixels = int(rendered_mask.sum())
    predicted_pixels = int(predicted_mask.sum())
    intersection_pixels = int(intersection.sum())
    silhouette_iou = intersection_pixels / max(int(union.sum()), 1)
    silhouette_precision = intersection_pixels / max(rendered_pixels, 1)
    silhouette_recall = intersection_pixels / max(predicted_pixels, 1)
    silhouette_f1 = 2.0 * silhouette_precision * silhouette_recall / max(
        silhouette_precision + silhouette_recall, 1e-12
    )
    boundary_symmetric_px, boundary_render_to_mask_px, boundary_mask_to_render_px = (
        symmetric_boundary_distance(rendered_mask, predicted_mask)
    )
    valid = intersection & (observed_depth_mm > 0)
    residual = np.abs(rendered_depth_mm[valid] - observed_depth_mm[valid])
    signed = observed_depth_mm[valid] - rendered_depth_mm[valid]
    if len(residual):
        depth_median_mm = float(np.median(residual))
        depth_q75_mm = float(np.quantile(residual, 0.75))
        depth_q90_mm = float(np.quantile(residual, 0.90))
        depth_inlier_5mm = float(np.mean(residual <= 5.0))
        depth_inlier_10mm = float(np.mean(residual <= 10.0))
        depth_inlier_20mm = float(np.mean(residual <= 20.0))
        rendered_front_fraction = float(np.mean(signed > 10.0))
        rendered_behind_fraction = float(np.mean(signed < -15.0))
    else:
        depth_median_mm = depth_q75_mm = depth_q90_mm = float("inf")
        depth_inlier_5mm = depth_inlier_10mm = depth_inlier_20mm = 0.0
        rendered_front_fraction = rendered_behind_fraction = 1.0
    return {
        "rendered_pixels": rendered_pixels,
        "predicted_mask_pixels": predicted_pixels,
        "render_mask_area_ratio": rendered_pixels / max(predicted_pixels, 1),
        "silhouette_iou": silhouette_iou,
        "silhouette_precision": silhouette_precision,
        "silhouette_recall": silhouette_recall,
        "silhouette_f1": silhouette_f1,
        "boundary_symmetric_px": boundary_symmetric_px,
        "boundary_render_to_mask_px": boundary_render_to_mask_px,
        "boundary_mask_to_render_px": boundary_mask_to_render_px,
        "render_depth_valid_pixels": int(valid.sum()),
        "render_depth_valid_fraction": float(valid.sum() / max(intersection_pixels, 1)),
        "render_depth_median_mm": depth_median_mm,
        "render_depth_q75_mm": depth_q75_mm,
        "render_depth_q90_mm": depth_q90_mm,
        "render_depth_inlier_5mm": depth_inlier_5mm,
        "render_depth_inlier_10mm": depth_inlier_10mm,
        "render_depth_inlier_20mm": depth_inlier_20mm,
        "rendered_front_fraction": rendered_front_fraction,
        "rendered_behind_fraction": rendered_behind_fraction,
    }


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    trials = pd.read_csv(args.root / "se3_trust_region_trials.csv", low_memory=False)
    if "solver_version" not in trials or not trials.solver_version.eq(SOLVER_VERSION).all():
        raise ValueError("Render input solver version differs; regenerate the corrected bank")
    if "solver_config_sha256" not in trials or trials.solver_config_sha256.nunique() != 1:
        raise ValueError("Missing or mixed solver configuration hashes")
    frames = pd.read_csv(args.root / "frame_manifest.csv")
    frame_lookup = frames.set_index("frame_id")
    masks = None
    namespace = support.load_experiment_namespace(
        sorted(trials.object_id.astype(int).unique().tolist()),
        max(1, len(frames)), 1, [0.20, 0.30, 0.40],
        dataset_root=args.dataset_root, dataset_split=args.dataset_split,
        depth_extension=args.depth_extension,
    )
    if not args.controlled_visible_mask:
        masks = masks_by_source_index(args.mask_json, set(frames.mask_source_index.astype(int)))

    first_frame = frames.iloc[0]
    first_depth = cv2.imread(
        str(
            args.dataset_root
            / args.dataset_split
            / f"{int(first_frame.scene_id):06d}"
            / "depth"
            / f"{int(first_frame.img_id):06d}.{args.depth_extension.lstrip('.')}"
        ),
        cv2.IMREAD_UNCHANGED,
    )
    if first_depth is None:
        raise FileNotFoundError("Cannot determine render dimensions from the first depth frame")
    height, width = first_depth.shape[:2]
    renderer = Open3DRaycastRenderer(width=width, height=height, cache_size=64)
    for object_id in sorted(trials.object_id.unique()):
        renderer.add_object(
            int(object_id),
            str(args.dataset_root / "models_eval" / f"obj_{int(object_id):06d}.ply"),
        )

    rows = []
    grouped = trials.groupby(["frame_id", "input_depth_condition"], sort=False)
    for group_index, ((frame_id, condition), group) in enumerate(grouped, start=1):
        frame_row = frame_lookup.loc[str(frame_id)]
        validate_condition_rows(group)
        if args.controlled_visible_mask:
            assert namespace is not None
            namespace["configure_object"](int(frame_row.object_id))
            frame = namespace["load_frame_data"](
                candidate_from_manifest(frame_row, args.dataset_root, args.dataset_split)
            )
            predicted_mask = frame["mask_bool"]
        else:
            assert masks is not None
            predicted_mask = decode_uncompressed_rle(masks[int(frame_row.mask_source_index)])
            frame = load_sensor_frame(args.dataset_root, frame_row, args.dataset_split, args.depth_extension)
        reference = group.iloc[0]
        prepared = replay_method_depth(namespace, frame, predicted_mask, reference, int(frame_row.selected_index))
        observed_depth_mm = prepared["depth"] * 1000.0
        K = frame["K"]
        condition_start = time.perf_counter_ns()
        condition_row_indices = []
        for _, candidate in group.iterrows():
            R, t = internal_pose_to_bop(candidate)
            render_start = time.perf_counter_ns()
            rendered = renderer.render_object(
                int(candidate.object_id),
                R,
                t,
                float(K[0, 0]),
                float(K[1, 1]),
                float(K[0, 2]),
                float(K[1, 2]),
            )["depth"]
            render_ms = (time.perf_counter_ns() - render_start) / 1_000_000.0
            compare_start = time.perf_counter_ns()
            comparison = compare(rendered, observed_depth_mm, predicted_mask)
            comparison_ms = (time.perf_counter_ns() - compare_start) / 1_000_000.0
            rows.append({
                **{key: candidate[key] for key in KEYS},
                **prepared["metadata"],
                "solver_version": candidate.solver_version,
                "solver_config_sha256": candidate.solver_config_sha256,
                **comparison,
                "render_ms": render_ms,
                "render_comparison_ms": comparison_ms,
                "render_compare_ms": render_ms + comparison_ms,
            })
            condition_row_indices.append(len(rows) - 1)
        condition_ms = (time.perf_counter_ns() - condition_start) / 1_000_000.0
        for row_index in condition_row_indices:
            rows[row_index]["condition_render_compare_ms"] = condition_ms
        if group_index % 20 == 0 or group_index == len(grouped):
            print(f"Render-compare conditions: {group_index}/{len(grouped)}", flush=True)

    features = pd.DataFrame(rows)
    if len(features) != len(trials):
        raise RuntimeError(f"Expected {len(trials)} feature rows, got {len(features)}")
    if features.duplicated(KEYS).any():
        raise RuntimeError("Duplicate render-compare feature rows")
    features.to_csv(args.output / "render_compare_features.csv", index=False)
    summary = features.groupby(["input_depth_condition", "method"], sort=False).agg(
        conditions=("silhouette_iou", "size"),
        silhouette_iou=("silhouette_iou", "mean"),
        boundary_symmetric_px=("boundary_symmetric_px", "mean"),
        render_depth_median_mm=("render_depth_median_mm", "mean"),
        render_depth_inlier_10mm=("render_depth_inlier_10mm", "mean"),
    ).reset_index()
    summary.to_csv(args.output / "table_render_compare_summary.csv", index=False)
    manifest = {
        "status": "render_compare_features_complete",
        "observation_version": OBSERVATION_VERSION,
        "solver_version": SOLVER_VERSION,
        "solver_config_sha256": str(trials.solver_config_sha256.iloc[0]),
        "observed_depth_policy": "Replay recorded method input seeds and verify input hashes; never use evaluation-depth policy.",
        "gt_free": True,
        "candidate_rows": len(features),
        "conditions": int(features[["frame_id", "input_depth_condition"]].drop_duplicates().shape[0]),
        "feature_columns": [column for column in features.columns if column not in (*KEYS, *OBSERVATION_FIELDS, "solver_version", "solver_config_sha256")],
        "provenance_only_columns": [*OBSERVATION_FIELDS, "solver_version", "solver_config_sha256"],
        "renderer": "Open3D RaycastingScene using T-LESS models_eval meshes",
        "mask_source": "BOP visible mask" if args.controlled_visible_mask else "external predicted mask",
        "timing": "perf_counter_ns; excludes dataset loading and renderer mesh initialization",
    }
    (args.output / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("PROTOCOL-G RENDER-COMPARE FEATURES COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-json", type=Path, default=MASK_JSON)
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\6D_pose\bop_datasets\tless"))
    parser.add_argument("--dataset-split", default="test_primesense")
    parser.add_argument("--depth-extension", default=".png")
    parser.add_argument("--controlled-visible-mask", action="store_true")
    run(parser.parse_args())
