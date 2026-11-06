"""Generate SE(3) trust-region P2P candidates and GT-free geometry scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import run_p2p_reliability_filter_development as support
from protocolG_observation import replay_method_depth, validate_condition_rows
from extract_real_gtfree_acceptance_features import masks_by_source_index
from run_frozen_real_inputs_validation import (
    INITIAL,
    MASK_JSON,
    PROPOSED,
    TRUNK,
    candidate_from_manifest,
    decode_uncompressed_rle,
)


SCALES = (0.25, 0.50, 1.00)
MAX_ROTATION_DEG = 1.0
MAX_TRANSLATION_MM = 5.0
SCORE_MARGIN = 0.02
SCORE_COLUMNS = ["score_bidir", "score_bidir_visibility", "score_bidir_visibility_freespace"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restricted_pose(initial: np.ndarray, full: np.ndarray, scale: float) -> tuple[np.ndarray, dict]:
    relative_rotation = full[:3, :3] @ initial[:3, :3].T
    rotvec = Rotation.from_matrix(relative_rotation).as_rotvec()
    full_rotation_deg = float(np.linalg.norm(rotvec) * 180.0 / np.pi)
    rotation_cap = MAX_ROTATION_DEG * scale
    rotation_scale = min(1.0, rotation_cap / max(full_rotation_deg, 1e-12))

    translation_delta = full[:3, 3] - initial[:3, 3]
    full_translation_mm = float(np.linalg.norm(translation_delta) * 1000.0)
    translation_cap = MAX_TRANSLATION_MM * scale
    translation_scale = min(1.0, translation_cap / max(full_translation_mm, 1e-12))

    candidate = np.eye(4, dtype=np.float64)
    candidate[:3, :3] = Rotation.from_rotvec(rotvec * rotation_scale).as_matrix() @ initial[:3, :3]
    candidate[:3, 3] = initial[:3, 3] + translation_scale * translation_delta
    return candidate, {
        "trust_scale": scale,
        "full_update_rotation_deg": full_rotation_deg,
        "full_update_translation_mm": full_translation_mm,
        "applied_rotation_deg": full_rotation_deg * rotation_scale,
        "applied_translation_mm": full_translation_mm * translation_scale,
        "rotation_scale": rotation_scale,
        "translation_scale": translation_scale,
    }


def robust_cost(distances: np.ndarray, weights: np.ndarray | None = None) -> float:
    if len(distances) < 20:
        return np.inf
    distances = np.minimum(np.asarray(distances, dtype=np.float64), 0.030)
    costs = np.log1p((distances / 0.005) ** 2)
    if weights is None:
        return float(np.mean(costs))
    weights = np.clip(np.asarray(weights, dtype=np.float64), 0.05, 1.0)
    return float(np.sum(weights * costs) / max(np.sum(weights), 1e-12))


def free_space_diagnostics(ns: dict, transform: np.ndarray, depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> dict:
    visible_model, _, _ = ns["generate_estimated_visible_model"](
        ns["full_model_points"], transform, mask, K
    )
    if len(visible_model) < 20:
        return {
            "front_free_space_fraction": 1.0,
            "behind_surface_fraction": 1.0,
            "signed_depth_valid_fraction": 0.0,
        }
    points = ns["transform_points"](visible_model, transform)
    z = points[:, 2]
    u = np.rint(K[0, 0] * points[:, 0] / np.maximum(z, 1e-9) + K[0, 2]).astype(int)
    v = np.rint(K[1, 2] - K[1, 1] * points[:, 1] / np.maximum(z, 1e-9)).astype(int)
    h, w = depth.shape
    inside = (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx = np.flatnonzero(inside)
    if len(idx) < 20:
        return {
            "front_free_space_fraction": 1.0,
            "behind_surface_fraction": 1.0,
            "signed_depth_valid_fraction": 0.0,
        }
    observed = depth[v[idx], u[idx]]
    valid = mask[v[idx], u[idx]] & (observed > 0)
    idx = idx[valid]
    observed = observed[valid]
    if len(idx) < 20:
        return {
            "front_free_space_fraction": 1.0,
            "behind_surface_fraction": 1.0,
            "signed_depth_valid_fraction": float(len(idx) / max(len(visible_model), 1)),
        }
    signed = observed - z[idx]
    return {
        "front_free_space_fraction": float(np.mean(signed > 0.010)),
        "behind_surface_fraction": float(np.mean(signed < -0.015)),
        "signed_depth_valid_fraction": float(len(idx) / max(len(visible_model), 1)),
    }


def geometry_scores(
    ns: dict,
    source_model: np.ndarray,
    target_points: np.ndarray,
    target_reliability: np.ndarray,
    target_tree: cKDTree,
    transform: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    target_cloud,
) -> dict:
    if len(source_model) < 20 or len(target_points) < 20:
        return {column: np.inf for column in SCORE_COLUMNS}
    source_camera = ns["transform_points"](source_model, transform)
    forward_distance, forward_index = target_tree.query(source_camera, k=1)
    source_tree = cKDTree(source_camera)
    reverse_distance, _ = source_tree.query(target_points, k=1)
    forward = robust_cost(forward_distance, target_reliability[forward_index])
    reverse = robust_cost(reverse_distance, target_reliability)
    bidir = 0.5 * (forward + reverse)
    _, _, visibility = ns["generate_observation_supported_visible_model"](
        ns["full_model_points"], transform, depth, mask, K, target_cloud
    )
    soft_support = float(np.nan_to_num(visibility["mean_soft_support"], nan=0.0, posinf=0.0))
    joint_fraction = float(
        visibility["joint_supported_points"] / max(visibility["zbuffer_candidates"], 1)
    )
    visibility_score = bidir + 0.75 * (1.0 - np.clip(soft_support, 0.0, 1.0)) + 0.25 * (1.0 - np.clip(joint_fraction, 0.0, 1.0))
    free = free_space_diagnostics(ns, transform, depth, mask, K)
    full_score = visibility_score + 2.0 * free["front_free_space_fraction"] + 0.5 * free["behind_surface_fraction"]
    return {
        "score_bidir": bidir,
        "score_bidir_visibility": visibility_score,
        "score_bidir_visibility_freespace": full_score,
        "forward_robust_cost": forward,
        "reverse_robust_cost": reverse,
        "candidate_mean_soft_support": soft_support,
        "candidate_joint_support_fraction": joint_fraction,
        **free,
    }


def method_name(direction: str, scale: float) -> str:
    return f"protocolF_se3_trust_{direction}_s{int(round(scale * 100)):03d}"


def unrestricted_method_name(direction: str) -> str:
    return f"protocolF_ablation_full_{direction}"


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    frames = pd.read_csv(args.input_root / "frame_manifest.csv")
    matched = frames[frames.matched_prediction.eq(1)].copy()
    if args.max_targets is not None:
        matched = matched.head(args.max_targets).copy()
    trials = pd.read_csv(args.input_root / "real_inputs_validation_trials.csv")
    wanted_frames = set(matched.frame_id.astype(str))
    trials = trials[trials.frame_id.astype(str).isin(wanted_frames)].copy()
    masks = masks_by_source_index(args.mask_json, set(matched.mask_source_index.astype(int)))
    objects = sorted(matched.object_id.astype(int).unique().tolist())
    ns = support.load_experiment_namespace(
        objects, max(1, len(matched)), 1, [0.30],
        dataset_root=args.dataset_root,
        dataset_split=args.dataset_split,
        depth_extension=".png",
    )
    output_trials = []
    score_rows = []
    conditions = [("native_depth", 0.0), ("local_outlier_30pct", 0.30)]
    directions = [("p2p", TRUNK), ("filtered", PROPOSED)]
    for object_id in objects:
        ns["configure_object"](object_id)
        for _, frame_row in matched[matched.object_id.eq(object_id)].iterrows():
            frame = ns["load_frame_data"](
                candidate_from_manifest(frame_row, args.dataset_root, args.dataset_split)
            )
            predicted_mask = decode_uncompressed_rle(masks[int(frame_row.mask_source_index)])
            for condition_name, fraction in conditions:
                condition = trials[
                    trials.frame_id.astype(str).eq(str(frame_row.frame_id))
                    & trials.input_depth_condition.eq(condition_name)
                ]
                if set(condition.method) != {INITIAL, TRUNK, PROPOSED}:
                    raise RuntimeError(f"Incomplete input candidates for {frame_row.frame_id}/{condition_name}")
                validate_condition_rows(condition)
                initial_row = condition[condition.method.eq(INITIAL)].iloc[0]
                initial = support.transform_from_row(initial_row)
                prepared = replay_method_depth(ns, frame, predicted_mask, initial_row, int(frame_row.selected_index))
                target, uv = ns["depth_to_cloud"](prepared["depth"], predicted_mask, frame["K"])
                reliability = ns["compute_depth_reliability"](prepared["depth"], predicted_mask, uv)
                gate = ns["compute_reliability_gate_features"](prepared["depth"], predicted_mask)
                source, _, _ = ns["generate_observation_supported_visible_model"](
                    ns["full_model_points"], initial, prepared["depth"], predicted_mask, frame["K"], target
                )
                target_points = np.asarray(target.points, dtype=np.float64)
                target_tree = cKDTree(target_points)
                generation_start = time.perf_counter()
                candidates = [(INITIAL, initial, {"direction": "initial", "trust_scale": 0.0,
                                                   "full_update_rotation_deg": 0.0, "full_update_translation_mm": 0.0,
                                                   "applied_rotation_deg": 0.0, "applied_translation_mm": 0.0,
                                                   "rotation_scale": 0.0, "translation_scale": 0.0})]
                for direction_name, input_method in directions:
                    full = support.transform_from_row(condition[condition.method.eq(input_method)].iloc[0])
                    if args.include_unrestricted:
                        candidates.append((unrestricted_method_name(direction_name), full, {
                            "direction": direction_name,
                            "trust_scale": 1.0,
                            "full_update_rotation_deg": 0.0,
                            "full_update_translation_mm": 0.0,
                            "applied_rotation_deg": 0.0,
                            "applied_translation_mm": 0.0,
                            "rotation_scale": 1.0,
                            "translation_scale": 1.0,
                        }))
                    for scale in SCALES:
                        pose, diagnostics = restricted_pose(initial, full, scale)
                        diagnostics["direction"] = direction_name
                        candidates.append((method_name(direction_name, scale), pose, diagnostics))
                candidate_generation_ms = (time.perf_counter() - generation_start) * 1000.0
                geometry_scoring_ms = 0.0
                for candidate_method, pose, diagnostics in candidates:
                    record = initial_row.to_dict()
                    record["method"] = candidate_method
                    record["termination_reason"] = "se3_trust_region_candidate"
                    record["runtime_ms"] = 0.0
                    record.update(prepared["metadata"])
                    support.update_pose_metrics(record, ns, pose, frame)
                    output_trials.append(record)
                    score_start = time.perf_counter()
                    scores = geometry_scores(
                        ns, source, target_points, reliability, target_tree, pose,
                        prepared["depth"], predicted_mask, frame["K"], target,
                    )
                    candidate_score_ms = (time.perf_counter() - score_start) * 1000.0
                    geometry_scoring_ms += candidate_score_ms
                    score_rows.append({
                        "object_id": object_id,
                        "scene_id": int(frame_row.scene_id),
                        "img_id": int(frame_row.img_id),
                        "inst_id": int(frame_row.inst_id),
                        "frame_id": str(frame_row.frame_id),
                        "input_depth_condition": condition_name,
                        "method": candidate_method,
                        **prepared["metadata"],
                        "gate_triggered": int(gate["trigger"]),
                        "candidate_generation_ms": candidate_generation_ms,
                        "candidate_score_ms": candidate_score_ms,
                        **diagnostics,
                        **scores,
                    })
                for score_row in score_rows[-len(candidates):]:
                    score_row["geometry_scoring_ms"] = geometry_scoring_ms
                    score_row["protocolF_extra_ms"] = candidate_generation_ms + geometry_scoring_ms
        print(f"Object {object_id}: trust-region candidates complete", flush=True)
        pd.DataFrame(output_trials).to_csv(args.output / "candidate_trials_checkpoint.csv", index=False)
        pd.DataFrame(score_rows).to_csv(args.output / "candidate_scores_checkpoint.csv", index=False)

    trial_table = pd.DataFrame(output_trials)
    score_table = pd.DataFrame(score_rows)
    methods = [INITIAL]
    if args.include_unrestricted:
        methods += [unrestricted_method_name(direction) for direction, _ in directions]
    methods += [method_name(direction, scale) for direction, _ in directions for scale in SCALES]
    expected = len(matched) * len(conditions) * len(methods)
    if len(trial_table) != expected or len(score_table) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(trial_table)} trials/{len(score_table)} scores")
    if score_table.duplicated(["frame_id", "input_depth_condition", "method"]).any():
        raise RuntimeError("Duplicate candidate score rows")
    trial_table.to_csv(args.output / "se3_trust_region_trials.csv", index=False)
    score_table.to_csv(args.output / "se3_trust_region_scores.csv", index=False)
    (args.output / "methods.txt").write_text(",".join(methods), encoding="ascii")
    source_paths = [Path(__file__), args.input_root / "frame_manifest.csv", args.input_root / "real_inputs_validation_trials.csv", args.mask_json]
    manifest = {
        "status": "development_candidates_complete",
        "matched_targets": len(matched),
        "source_total_targets": len(frames),
        "evaluation_target_denominator": len(matched) if args.max_targets is not None else len(frames),
        "objects": objects,
        "conditions": [name for name, _ in conditions],
        "methods": methods,
        "scales": SCALES,
        "max_rotation_deg": MAX_ROTATION_DEG,
        "max_translation_mm": MAX_TRANSLATION_MM,
        "score_margin": SCORE_MARGIN,
        "include_unrestricted_ablation_candidates": bool(args.include_unrestricted),
        "score_columns": SCORE_COLUMNS,
        "hashes": {str(path): digest(path) for path in source_paths},
    }
    (args.output / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    snapshots = args.output / "source_snapshot"
    snapshots.mkdir()
    shutil.copy2(Path(__file__), snapshots / Path(__file__).name)
    print("SE3 TRUST-REGION CANDIDATES COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-json", type=Path, default=MASK_JSON)
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\6D_pose\bop_datasets\tless"))
    parser.add_argument("--dataset-split", default="test_primesense")
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--include-unrestricted", action="store_true")
    run(parser.parse_args())
