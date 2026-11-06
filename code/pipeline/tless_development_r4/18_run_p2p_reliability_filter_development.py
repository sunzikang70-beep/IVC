"""Develop a reliability-filtered point-to-point ICP method on frozen trials.

The existing Open3D point-to-point result is the safety trunk. The reliability
gate is computed without ground truth. When the gate is closed, or filtering is
not viable, the trunk pose is returned unchanged. Gate-open trials compare a
direct filtered registration with a filtered refinement initialized by the
trunk pose.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd


_DEFAULT_SOURCE = Path(
    r"D:\6D_pose\reliability_weighted_icp_hard_validation"
    r"\reliability_weighted_icp_hard_validation.py"
)
_BUNDLED_SOURCE = Path(__file__).resolve().with_name(
    "reliability_weighted_icp_hard_validation.py"
)
# Explicit overrides win; existing local installs retain the original source.
SOURCE = Path(
    os.environ.get("P2P_LEGACY_SOURCE")
    or (_DEFAULT_SOURCE if _DEFAULT_SOURCE.is_file() else _BUNDLED_SOURCE)
).expanduser()
OUTPUT_ROOT = Path(
    os.environ.get("P2P_OUTPUT_ROOT")
    or Path(__file__).resolve().parent.parent / "outputs"
).expanduser()
DEFAULT_EXISTING = (
    OUTPUT_ROOT
    / "strong_baselines_external_10obj_10f_raw"
    / "reliability_weighted_icp_hard_validation_trials.csv"
)
TRUNK_METHOD = "protocolB_e2e_open3d_point_to_point_icp"
REFERENCE_METHODS = [
    "protocolB_e2e_initial_pose",
    TRUNK_METHOD,
    "protocolB_e2e_geometry_icp",
    "protocolB_e2e_reliability_gated_icp",
]
KEY_COLUMNS = [
    "object_id",
    "scene_id",
    "img_id",
    "inst_id",
    "corruption_type",
    "requested_corruption_fraction",
    "trial",
]
THRESHOLDS = (0.15, 0.25, 0.35)
ROUTES = ("direct", "staged")
MIN_FILTERED_POINTS = 300
MIN_RETENTION_RATIO = 0.25


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", default="2,4,6,7,9,11,14,17,20,30")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--fractions", default="0.20,0.30,0.40")
    parser.add_argument("--thresholds", default="0.15,0.25,0.35")
    parser.add_argument("--routes", default="direct,staged")
    parser.add_argument("--existing-trials-csv", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument(
        "--label", default="p2p_reliability_filter_development_10obj_10f_raw"
    )
    parser.add_argument(
        "--filter-source-support",
        action="store_true",
        help="Apply the reliability mask to source visibility support as well.",
    )
    parser.add_argument("--checkpoint", type=int, default=5)
    return parser.parse_args()


def parse_values(value, cast):
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def method_name(route, threshold, filter_source_support=False):
    strategy = "dual_filter" if filter_source_support else "filter"
    return (
        f"protocolB_e2e_gate_p2p_{route}_{strategy}_"
        f"r{int(threshold * 100):03d}"
    )


def _is_legacy_output_mkdir(statement):
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    call = statement.value
    if not (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "os"
        and call.func.attr == "makedirs"
    ):
        return False
    directory = call.args[0] if call.args else next(
        (kw.value for kw in call.keywords if kw.arg == "name"), None
    )
    return isinstance(directory, ast.Name) and directory.id == "output_dir"


def load_experiment_namespace(
    objects,
    frames,
    repeats,
    fractions,
    dataset_root=None,
    dataset_split="test_primesense",
    depth_extension=".png",
):
    source = SOURCE.read_text(encoding="utf-8")
    if dataset_root is not None:
        replacement = f'root = r"{Path(dataset_root)}"'
        source, count = re.subn(
            r'^root = r"[^"]+"$',
            lambda _: replacement,
            source,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise RuntimeError("Could not replace the source dataset root.")
        if not dataset_split or any(char in dataset_split for char in "/\\"):
            raise ValueError("dataset_split must be a single directory name.")
        source, count = re.subn(
            r'(?m)^(\s*)"test_primesense"$',
            rf'\1"{dataset_split}"',
            source,
            count=1,
        )
        if count != 1:
            raise RuntimeError("Could not replace the source dataset split.")
        extension = str(depth_extension).lower().lstrip(".")
        if extension not in {"png", "tif", "tiff"}:
            raise ValueError("depth_extension must be png, tif, or tiff.")
        source, count = re.subn(
            r'f"\{img_id:06d\}\.png"',
            f'f"{{img_id:06d}}.{extension}"',
            source,
            count=1,
        )
        if count != 1:
            raise RuntimeError("Could not replace the source depth extension.")
    marker = "# ==================================================\n# Methods / protocols\n"
    if marker not in source:
        raise RuntimeError("Could not find the source preamble boundary.")
    filename = str(SOURCE) + "::support"
    preamble = ast.parse(source.split(marker, 1)[0], filename=filename)
    # Filter only module-level legacy output creation, never function bodies.
    # The source file stays untouched so provenance hashes still identify it.
    preamble.body = [
        statement for statement in preamble.body
        if not _is_legacy_output_mkdir(statement)
    ]
    namespace = {"__name__": "p2p_filter_support"}
    exec(
        compile(preamble, filename, "exec"),
        namespace,
    )
    namespace["OBJECT_IDS"] = objects
    namespace["FRAMES_PER_OBJECT"] = frames
    namespace["REPEATS_PER_LEVEL"] = repeats
    namespace["corruption_fractions"] = fractions
    return namespace


def transform_from_row(row):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [
            [row.R_00, row.R_01, row.R_02],
            [row.R_10, row.R_11, row.R_12],
            [row.R_20, row.R_21, row.R_22],
        ],
        dtype=np.float64,
    )
    transform[:3, 3] = np.array(
        [row.t_x_mm, row.t_y_mm, row.t_z_mm], dtype=np.float64
    ) / 1000.0
    return transform


def pose_fields(transform):
    return {
        "R_00": transform[0, 0],
        "R_01": transform[0, 1],
        "R_02": transform[0, 2],
        "R_10": transform[1, 0],
        "R_11": transform[1, 1],
        "R_12": transform[1, 2],
        "R_20": transform[2, 0],
        "R_21": transform[2, 1],
        "R_22": transform[2, 2],
        "t_x_mm": transform[0, 3] * 1000.0,
        "t_y_mm": transform[1, 3] * 1000.0,
        "t_z_mm": transform[2, 3] * 1000.0,
    }


def condition_key(object_id, frame, corruption, fraction, trial):
    return (
        int(object_id),
        int(frame["scene_id"]),
        int(frame["img_id"]),
        int(frame["inst_id"]),
        corruption,
        float(fraction),
        int(trial),
    )


def filtered_cloud(target_cloud, reliability, threshold):
    points = np.asarray(target_cloud.points, dtype=np.float64)
    keep = np.asarray(reliability, dtype=np.float64) > threshold
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[keep])
    return cloud, int(np.count_nonzero(keep)), float(np.mean(keep))


def filtered_depth_support(corrupted, threshold):
    depth = corrupted["corrupted_depth"].copy()
    uv = np.asarray(corrupted["target_uv"], dtype=np.int64)
    keep = np.asarray(corrupted["depth_reliability"], dtype=np.float64) > threshold
    rejected_uv = uv[~keep]
    depth[rejected_uv[:, 1], rejected_uv[:, 0]] = 0.0
    return depth


def run_point_to_point(ns, source_points, target_cloud, initial_transform):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target_cloud,
        ns["max_correspondence_distance"],
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=ns["max_iterations"]
        ),
    )
    return result


def update_pose_metrics(record, ns, transform, frame, result=None):
    add_s = ns["compute_add_s"](ns["evaluation_model_points"], transform, frame["T_gt"])
    record.update(pose_fields(transform))
    record.update(
        {
            "ADD_S_mm": float(add_s * 1000.0),
            "rotation_error_deg": ns["rotation_error_deg"](transform, frame["T_gt"]),
            "translation_error_mm": ns["translation_error_mm"](
                transform, frame["T_gt"]
            ),
            "success": float(add_s < ns["success_threshold_m"]),
        }
    )
    if result is not None:
        record["RMSE_mm"] = float(result.inlier_rmse * 1000.0)
        record["fitness"] = float(result.fitness)


def annotate_gate(record, gate):
    record.update(
        {
            "gate_triggered": int(gate["trigger"]),
            "gate_reason": gate["reason"],
            "gate_mean_reliability": gate["mean_reliability"],
            "gate_median_reliability": gate["median_reliability"],
            "gate_q10_reliability": gate["q10_reliability"],
            "gate_q25_reliability": gate["q25_reliability"],
            "gate_strong_low_fraction": gate["strong_low_fraction"],
            "gate_opened_low_fraction": gate["opened_low_fraction"],
            "gate_largest_component_fraction": gate[
                "largest_component_fraction"
            ],
            "gate_largest_component_share_of_low": gate[
                "largest_component_share_of_low"
            ],
            "gate_mean_local_diff_mm": gate["mean_local_diff_mm"],
            "gate_q90_local_diff_mm": gate["q90_local_diff_mm"],
        }
    )


def make_candidate_record(
    trunk_row,
    ns,
    frame,
    gate,
    method,
    route,
    threshold,
    filtered_count,
    retention,
    gate_runtime_ms,
    candidate_runtime_ms=0.0,
    transform=None,
    result=None,
    reason="gate_closed:p2p_trunk",
):
    record = trunk_row.to_dict()
    record["method"] = method
    record["filter_route"] = route
    record["filter_threshold"] = float(threshold)
    record["filtered_target_points"] = int(filtered_count)
    record["filtered_retention_ratio"] = float(retention)
    record["candidate_used"] = int(transform is not None)
    record["source_support_filtered"] = int("dual_filter" in method)
    record["termination_reason"] = reason
    record["trunk_runtime_ms"] = float(trunk_row.runtime_ms)
    record["gate_filter_runtime_ms"] = float(gate_runtime_ms + candidate_runtime_ms)
    record["runtime_ms"] = float(
        trunk_row.runtime_ms + gate_runtime_ms + candidate_runtime_ms
    )
    annotate_gate(record, gate)
    if transform is not None:
        update_pose_metrics(record, ns, transform, frame, result=result)
    return record


def main():
    args = parse_args()
    objects = parse_values(args.objects, int)
    fractions = parse_values(args.fractions, float)
    thresholds = parse_values(args.thresholds, float)
    routes = parse_values(args.routes, str)
    if not thresholds or any(value <= 0.0 or value >= 1.0 for value in thresholds):
        raise ValueError("Thresholds must be values strictly between zero and one.")
    if not routes or any(value not in ROUTES for value in routes):
        raise ValueError(f"Routes must be selected from {ROUTES}.")
    existing = pd.read_csv(args.existing_trials_csv)
    references = existing[existing["method"].isin(REFERENCE_METHODS)].copy()
    trunk = existing[existing["method"] == TRUNK_METHOD].copy()
    if trunk.duplicated(KEY_COLUMNS).any():
        raise ValueError("Point-to-point trunk contains duplicate condition keys.")
    trunk_lookup = {tuple(row[column] for column in KEY_COLUMNS): row for _, row in trunk.iterrows()}

    output_dir = OUTPUT_ROOT / args.label
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "p2p_reliability_filter_candidates.csv"
    combined_path = output_dir / "reliability_weighted_icp_hard_validation_trials.csv"
    if combined_path.exists():
        raise FileExistsError(f"Completed output already exists: {combined_path}")

    ns = load_experiment_namespace(objects, args.frames, args.repeats, fractions)
    records = []
    processed_frames = 0
    skipped_frames = 0
    for object_index, object_id in enumerate(objects):
        ns["configure_object"](object_id)
        selected = ns["select_evenly"](ns["discover_candidates"](), args.frames)
        print(f"Object {object_id} ({object_index + 1}/{len(objects)}): {len(selected)} selected")
        for frame_index, candidate in enumerate(selected):
            try:
                frame = ns["load_frame_data"](candidate)
                T_initial = ns["generate_initial_pose"](frame["T_gt"])
            except Exception as error:
                skipped_frames += 1
                print("  SKIP:", repr(error))
                continue
            expected_frame_keys = [
                condition_key(object_id, frame, corruption, fraction, trial)
                for fraction in fractions
                for trial in range(args.repeats)
                for corruption in ns["corruption_types"]
            ]
            present_count = sum(key in trunk_lookup for key in expected_frame_keys)
            if present_count == 0:
                skipped_frames += 1
                print("  SKIP: frame absent from the frozen P2P trunk")
                continue
            if present_count != len(expected_frame_keys):
                raise KeyError(
                    f"Frozen P2P trunk is incomplete for object {object_id}, "
                    f"scene {frame['scene_id']}, image {frame['img_id']}: "
                    f"{present_count}/{len(expected_frame_keys)} conditions present."
                )
            processed_frames += 1
            for fraction_index, fraction in enumerate(fractions):
                for trial in range(args.repeats):
                    patch_seed = (
                        200000
                        + object_id * 1000000
                        + frame_index * 10000
                        + fraction_index * 100
                        + trial
                    )
                    patch, _, _, _ = ns["select_local_patch"](
                        frame["depth"], frame["mask_bool"], fraction, patch_seed
                    )
                    for corruption_index, corruption in enumerate(ns["corruption_types"]):
                        key = condition_key(object_id, frame, corruption, fraction, trial)
                        if key not in trunk_lookup:
                            raise KeyError(f"Missing point-to-point trunk condition: {key}")
                        trunk_row = trunk_lookup[key]
                        value_seed = (
                            500000
                            + object_id * 1000000
                            + frame_index * 10000
                            + fraction_index * 1000
                            + trial * 10
                            + corruption_index
                        )
                        start_gate = time.perf_counter()
                        corrupted = ns["make_local_corrupted_target"](
                            frame["depth"],
                            frame["mask_bool"],
                            frame["K"],
                            patch,
                            corruption,
                            value_seed,
                        )
                        gate = ns["compute_reliability_gate_features"](
                            corrupted["corrupted_depth"], frame["mask_bool"]
                        )
                        gate_runtime_ms = 1000.0 * (time.perf_counter() - start_gate)

                        prepared = {}
                        for threshold in thresholds:
                            cloud, count, retention = filtered_cloud(
                                corrupted["target_cloud"],
                                corrupted["depth_reliability"],
                                threshold,
                            )
                            support_depth = (
                                filtered_depth_support(corrupted, threshold)
                                if args.filter_source_support
                                else corrupted["corrupted_depth"]
                            )
                            prepared[threshold] = (
                                cloud,
                                count,
                                retention,
                                support_depth,
                            )

                        source_points_by_threshold = {}
                        if gate["trigger"]:
                            for threshold in thresholds:
                                cloud, _, _, support_depth = prepared[threshold]
                                source_points, _, _ = ns[
                                    "generate_observation_supported_visible_model"
                                ](
                                    ns["full_model_points"],
                                    T_initial,
                                    support_depth,
                                    frame["mask_bool"],
                                    frame["K"],
                                    cloud if args.filter_source_support else corrupted["target_cloud"],
                                )
                                source_points_by_threshold[threshold] = source_points
                        T_trunk = transform_from_row(trunk_row)
                        for threshold in thresholds:
                            cloud, count, retention, _ = prepared[threshold]
                            source_points = source_points_by_threshold.get(threshold)
                            viable = (
                                count >= MIN_FILTERED_POINTS
                                and retention >= MIN_RETENTION_RATIO
                            )
                            for route in routes:
                                name = method_name(
                                    route, threshold, args.filter_source_support
                                )
                                if not gate["trigger"]:
                                    record = make_candidate_record(
                                        trunk_row,
                                        ns,
                                        frame,
                                        gate,
                                        name,
                                        route,
                                        threshold,
                                        count,
                                        retention,
                                        gate_runtime_ms,
                                    )
                                elif not viable or len(source_points) < MIN_FILTERED_POINTS:
                                    record = make_candidate_record(
                                        trunk_row,
                                        ns,
                                        frame,
                                        gate,
                                        name,
                                        route,
                                        threshold,
                                        count,
                                        retention,
                                        gate_runtime_ms,
                                        reason="filter_not_viable:p2p_trunk",
                                    )
                                else:
                                    initial = T_initial if route == "direct" else T_trunk
                                    start_candidate = time.perf_counter()
                                    try:
                                        result = run_point_to_point(
                                            ns, source_points, cloud, initial
                                        )
                                        candidate_runtime_ms = 1000.0 * (
                                            time.perf_counter() - start_candidate
                                        )
                                        record = make_candidate_record(
                                            trunk_row,
                                            ns,
                                            frame,
                                            gate,
                                            name,
                                            route,
                                            threshold,
                                            count,
                                            retention,
                                            gate_runtime_ms,
                                            candidate_runtime_ms,
                                            result.transformation,
                                            result,
                                            "gate_open:filtered_candidate",
                                        )
                                    except Exception as error:
                                        candidate_runtime_ms = 1000.0 * (
                                            time.perf_counter() - start_candidate
                                        )
                                        record = make_candidate_record(
                                            trunk_row,
                                            ns,
                                            frame,
                                            gate,
                                            name,
                                            route,
                                            threshold,
                                            count,
                                            retention,
                                            gate_runtime_ms,
                                            candidate_runtime_ms,
                                            reason=(
                                                "candidate_failed:"
                                                + type(error).__name__
                                                + ":p2p_trunk"
                                            ),
                                        )
                                record["filtered_source_points"] = (
                                    np.nan
                                    if source_points is None
                                    else int(len(source_points))
                                )
                                if source_points is not None:
                                    record["source_points"] = int(len(source_points))
                                records.append(record)
            print(
                f"  frame {processed_frames}: complete "
                f"({skipped_frames} skipped, {len(records)} candidate rows)"
            )
            if processed_frames % args.checkpoint == 0:
                pd.DataFrame(records).to_csv(candidate_path, index=False)
                print("  Checkpoint:", candidate_path)

    candidates = pd.DataFrame(records)
    candidates.to_csv(candidate_path, index=False)
    candidate_keys = candidates[KEY_COLUMNS].drop_duplicates()
    references = references.merge(candidate_keys, on=KEY_COLUMNS, how="inner")
    combined = pd.concat([references, candidates], ignore_index=True, sort=False)
    if combined.duplicated(KEY_COLUMNS + ["method"]).any():
        raise ValueError("Combined output contains duplicate method-condition rows.")
    expected_methods = len(REFERENCE_METHODS) + len(thresholds) * len(routes)
    expected_rows = len(candidate_keys) * expected_methods
    if len(combined) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(combined)}.")
    combined.to_csv(combined_path, index=False)

    runtime = candidates.groupby("method", as_index=False).agg(
        N=("runtime_ms", "size"),
        candidate_used=("candidate_used", "sum"),
        fallback_count=("candidate_used", lambda value: int((value == 0).sum())),
        mean_total_runtime_ms=("runtime_ms", "mean"),
        median_total_runtime_ms=("runtime_ms", "median"),
        mean_incremental_runtime_ms=("gate_filter_runtime_ms", "mean"),
    )
    runtime.to_csv(output_dir / "table_p2p_filter_runtime.csv", index=False)
    print(f"Completed: {len(candidates)} candidate rows, {len(combined)} combined rows")
    print("Combined trials:", combined_path)


if __name__ == "__main__":
    main()
