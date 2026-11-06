"""Run the frozen refinement with public initializer poses and predicted masks.

The prediction selector is deliberately GT-free. Ground-truth pose and visible
mask are loaded only after a prediction has been selected, for evaluation and
post-hoc diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import run_frozen_unseen_validation as frozen
import run_p2p_reliability_filter_development as support
from protocolG_observation import OBSERVATION_VERSION, prepare_depth
from run_frozen_mask_robustness import filtered_depth, mask_quality
from run_strong_baselines_only import make_output, make_record


DATASET_ROOT = Path(r"D:\6D_pose\bop_datasets\tless")
POSE_CSV = Path(
    r"D:\6D_pose\external_predictions\megapose_cnos_tless_2023"
    r"\cnos-fastsammegapose_tless-test_94e046a0-42af-495f-8a35-11ce8ee6f217.csv"
)
MASK_JSON = Path(
    r"D:\6D_pose\external_predictions\megapose_cnos_tless_2023\fastSAM_pbr_tless.json"
)
CONFIG = frozen.CONFIG
INITIAL = frozen.INITIAL
PROPOSED = frozen.PROPOSED
TRUNK = support.TRUNK_METHOD
METHODS = [INITIAL, TRUNK, PROPOSED]
INPUT_CONDITIONS = (("native_depth", 0.0), ("local_outlier_30pct", 0.30))
POSE_COLUMNS = list(support.pose_fields(np.eye(4)))
FLIP4 = np.diag([1.0, -1.0, 1.0, 1.0])
OFFICIAL_POSE_URL = (
    "https://bop.felk.cvut.cz/media/subs/"
    "cnos-fastsammegapose_tless-test_94e046a0-42af-495f-8a35-11ce8ee6f217.csv"
)
OFFICIAL_MASK_SOURCE = (
    "https://drive.google.com/drive/folders/1yGRKpz1RI4h5-u0drusVeXPuAsg_GIO5"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iter_json_array(path: Path, chunk_size: int = 1 << 20):
    """Yield objects from a JSON array without materializing the 75 MB file."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as stream:
        buffer = ""
        position = 0
        started = False
        finished = False
        while not finished:
            if position >= len(buffer) - 1:
                buffer = buffer[position:] + stream.read(chunk_size)
                position = 0
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"Expected a JSON array in {path}")
                position += 1
                started = True
                continue
            while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                finished = True
                continue
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                more = stream.read(chunk_size)
                if not more:
                    raise
                buffer = buffer[position:] + more
                position = 0
                continue
            yield value
            position = end


def decode_uncompressed_rle(segmentation: dict) -> np.ndarray:
    counts = segmentation["counts"]
    height, width = map(int, segmentation["size"])
    if not isinstance(counts, list):
        raise TypeError("Only uncompressed COCO RLE counts are supported.")
    flat = np.zeros(height * width, dtype=bool)
    cursor = 0
    foreground = False
    for raw_count in counts:
        count = int(raw_count)
        if count < 0 or cursor + count > len(flat):
            raise ValueError("Invalid COCO RLE run length.")
        if foreground:
            flat[cursor : cursor + count] = True
        cursor += count
        foreground = not foreground
    if cursor != len(flat):
        raise ValueError(f"COCO RLE covers {cursor} pixels, expected {len(flat)}.")
    return flat.reshape((height, width), order="F")


def bbox_iou_xywh(first, second) -> float:
    ax, ay, aw, ah = map(float, first)
    bx, by, bw, bh = map(float, second)
    if not np.isfinite([ax, ay, aw, ah, bx, by, bw, bh]).all():
        return 0.0
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - intersection
    return intersection / union if union > 0 else 0.0


def parse_vector(value, count: int) -> np.ndarray:
    values = np.fromstring(str(value), sep=" ", dtype=np.float64)
    if len(values) != count:
        raise ValueError(f"Expected {count} values, found {len(values)}: {value}")
    return values


def bop_pose_from_row(row) -> tuple[np.ndarray, np.ndarray]:
    return parse_vector(row.R, 9).reshape(3, 3), parse_vector(row.t, 3)


def internal_pose_from_bop(rotation: np.ndarray, translation_mm: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float64)
    pose[:3, 3] = np.asarray(translation_mm, dtype=np.float64) / 1000.0
    return FLIP4 @ pose


def projected_bbox_xywh(
    model_points_m: np.ndarray,
    rotation_bop: np.ndarray,
    translation_mm: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[float, float, float, float]:
    camera = (rotation_bop @ (np.asarray(model_points_m) * 1000.0).T).T
    camera += np.asarray(translation_mm, dtype=np.float64)
    camera = camera[camera[:, 2] > 1e-6]
    if len(camera) == 0:
        return (np.nan, np.nan, 0.0, 0.0)
    u = camera[:, 0] * camera_matrix[0, 0] / camera[:, 2] + camera_matrix[0, 2]
    v = camera[:, 1] * camera_matrix[1, 1] / camera[:, 2] + camera_matrix[1, 2]
    left, right = float(np.min(u)), float(np.max(u))
    top, bottom = float(np.min(v)), float(np.max(v))
    return (left, top, right - left, bottom - top)


def select_external_candidates(
    pose_rows: list[dict],
    mask_rows: list[dict],
    model_points_m: np.ndarray,
    camera_matrix: np.ndarray,
    n_top: int,
) -> list[dict]:
    """Pair top-N poses and masks one-to-one without consulting ground truth."""
    if not pose_rows or not mask_rows or n_top <= 0:
        return []
    poses = sorted(
        pose_rows,
        key=lambda item: (-float(item["score"]), int(item["source_index"])),
    )[:n_top]
    remaining_masks = list(mask_rows)
    selections = []
    for pose_rank, pose in enumerate(poses, start=1):
        if not remaining_masks:
            break
        rotation, translation = bop_pose_from_row(type("PoseRow", (), pose))
        projected = projected_bbox_xywh(model_points_m, rotation, translation, camera_matrix)
        ranked_masks = [
            (
                bbox_iou_xywh(projected, item["bbox"]),
                float(item["score"]),
                -int(item["source_index"]),
                item,
            )
            for item in remaining_masks
        ]
        overlap, _, _, mask = max(ranked_masks, key=lambda item: item[:3])
        remaining_masks.remove(mask)
        selections.append(
            {
                "pose": pose,
                "mask": mask,
                "pose_rank": pose_rank,
                "rotation_bop": rotation,
                "translation_bop_mm": translation,
                "projected_bbox": projected,
                "bbox_iou": float(overlap),
            }
        )
    return selections


def select_external_prediction(
    pose_rows: list[dict],
    mask_rows: list[dict],
    model_points_m: np.ndarray,
    camera_matrix: np.ndarray,
) -> dict | None:
    """Backward-compatible single-instance selector used by unit tests."""
    selections = select_external_candidates(
        pose_rows, mask_rows, model_points_m, camera_matrix, n_top=1
    )
    return selections[0] if selections else None


def associate_evaluation_instance(
    selections: list[dict], gt_visible_mask: np.ndarray, expected_shape: tuple[int, int]
) -> dict | None:
    """Associate independent detections to a GT target only for evaluation.

    The method would produce the same pose for each detection if all detections
    were refined first. This association therefore cannot change a refinement
    result; it only identifies which independently processed detection is scored.
    """
    associated = []
    for selection in selections:
        predicted_mask = decode_uncompressed_rle(selection["mask"]["segmentation"])
        if predicted_mask.shape != expected_shape:
            raise ValueError(
                f"Mask shape {predicted_mask.shape} != expected shape {expected_shape}"
            )
        quality = mask_quality(predicted_mask, gt_visible_mask)
        associated.append(
            (
                float(quality["mask_iou"]),
                float(selection["pose"]["score"]),
                -int(selection["pose"]["source_index"]),
                selection,
                predicted_mask,
                quality,
            )
        )
    if not associated:
        return None
    target_iou, _, _, selection, predicted_mask, quality = max(
        associated, key=lambda item: item[:3]
    )
    result = dict(selection)
    result["predicted_mask"] = predicted_mask
    result["mask_quality"] = quality
    result["evaluation_association_iou"] = target_iou
    return result


def load_bop_target_counts(dataset_root: Path) -> dict[tuple[int, int, int], int]:
    path = dataset_root / "test_targets_bop19.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        (int(row["scene_id"]), int(row["im_id"]), int(row["obj_id"])): int(row["inst_count"])
        for row in rows
    }


def load_pose_index(path: Path, target_keys: set[tuple[int, int, int]]):
    table = pd.read_csv(path)
    index = defaultdict(list)
    for source_index, row in table.iterrows():
        key = (int(row.scene_id), int(row.im_id), int(row.obj_id))
        if key in target_keys:
            item = row.to_dict()
            item["source_index"] = int(source_index)
            index[key].append(item)
    return index, len(table)


def load_mask_index(path: Path, target_keys: set[tuple[int, int, int]]):
    index = defaultdict(list)
    total = 0
    for source_index, item in enumerate(iter_json_array(path)):
        total += 1
        key = (int(item["scene_id"]), int(item["image_id"]), int(item["category_id"]))
        if key in target_keys:
            retained = dict(item)
            retained["source_index"] = int(source_index)
            index[key].append(retained)
    return index, total


def candidate_from_manifest(row, dataset_root: Path, dataset_split: str) -> dict:
    scene_id, image_id, instance_id = int(row.scene_id), int(row.img_id), int(row.inst_id)
    info_path = dataset_root / dataset_split / f"{scene_id:06d}" / "scene_gt_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    visib = float(info[str(image_id)][instance_id].get("visib_fract", 1.0))
    return {
        "scene_name": f"{scene_id:06d}",
        "img_id": image_id,
        "img_key": str(image_id),
        "inst_id": instance_id,
        "visib_fract": visib,
    }


SEED_SALT_UNIT = 10 ** 9  # pre-registered multi-realization step for the corruption seeds


def realization_salt(realization: int) -> int:
    """Seed salt for corruption realization ``r`` (pre-registered, see prereg_P1_20260924.json).

    The salt is added to both arithmetic corruption seeds, so realization r draws an
    independent patch and an independent displacement. Realization 0 returns 0 and is
    therefore bit-identical to the frozen single-realization protocol.
    """
    if realization < 0:
        raise ValueError("realization must be >= 0")
    return int(realization) * SEED_SALT_UNIT


def condition_specs(outlier_fractions: tuple[float, ...]) -> tuple[tuple[str, float], ...]:
    """Build deterministic input-condition labels while preserving the default protocol."""
    specs = [("native_depth", 0.0)]
    for fraction in sorted(set(float(value) for value in outlier_fractions)):
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"Outlier fractions must lie in (0, 1), got {fraction}")
        label = f"local_outlier_{int(round(100.0 * fraction)):02d}pct"
        specs.append((label, fraction))
    return tuple(specs)


def perturb_initial_pose(initial: np.ndarray, rotation_deg: float, translation_mm: float,
                         object_id: int, selected_index: int, seed: int) -> tuple[np.ndarray, int]:
    """Apply a deterministic, GT-free camera-frame perturbation to an external pose."""
    if rotation_deg == 0.0 and translation_mm == 0.0:
        return initial, int(seed)
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(int(seed) + int(object_id) * 1000003 + int(selected_index) * 9176)
    axis = rng.normal(size=3)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    direction = rng.normal(size=3)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = Rotation.from_rotvec(axis * np.deg2rad(rotation_deg)).as_matrix()
    delta[:3, 3] = direction * (float(translation_mm) / 1000.0)
    return delta @ initial, int(seed)


def run_methods(ns, frame, meta, initial, predicted_mask, prepared, frame_id, selection, selected_index):
    depth = prepared["depth"]
    start = time.perf_counter()
    target, uv = ns["depth_to_cloud"](depth, predicted_mask, frame["K"])
    target_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    reliability = ns["compute_depth_reliability"](depth, predicted_mask, uv)
    gate = ns["compute_reliability_gate_features"](depth, predicted_mask)
    gate_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    source, _, _ = ns["generate_observation_supported_visible_model"](
        ns["full_model_points"], initial, depth, predicted_mask, frame["K"], target
    )
    source_ms = (time.perf_counter() - start) * 1000.0
    trunk_t, trunk_result, trunk_reason, trunk_icp_ms = frozen.fit(ns, source, target, initial)
    trunk_ms = target_ms + source_ms + trunk_icp_ms

    chosen, chosen_result = trunk_t, trunk_result
    reason = "gate_closed:p2p_trunk"
    filtered_source_count = len(source)
    filtered_target_count = len(target.points)
    retention = 1.0
    used = 0
    start = time.perf_counter()
    if gate["trigger"]:
        filtered_target, filtered_target_count, retention = support.filtered_cloud(
            target, reliability, 0.35
        )
        support_depth = filtered_depth(depth, uv, reliability)
        filtered_source, _, _ = ns["generate_observation_supported_visible_model"](
            ns["full_model_points"], initial, support_depth, predicted_mask, frame["K"], filtered_target
        )
        filtered_source_count = len(filtered_source)
        reason = "filter_not_viable:p2p_trunk"
        if min(filtered_source_count, filtered_target_count) >= 300 and retention >= 0.25:
            candidate_t, result, status, _ = frozen.fit(ns, filtered_source, filtered_target, initial)
            if status == "completed":
                chosen, chosen_result = candidate_t, result
                used, reason = 1, "gate_open:filtered_candidate"
            else:
                reason = status + ":p2p_trunk"
    filter_ms = (time.perf_counter() - start) * 1000.0
    outputs = [
        (INITIAL, initial, None, "external_initializer", 0.0, len(source)),
        (TRUNK, trunk_t, trunk_result, trunk_reason, trunk_ms, len(source)),
        (PROPOSED, chosen, chosen_result, reason, trunk_ms + gate_ms + filter_ms,
         filtered_source_count if used else len(source)),
    ]
    records = []
    condition_name = prepared["condition_name"]
    for method, transform, result, status, runtime, source_count in outputs:
        evaluated = make_output(
            ns,
            transform,
            frame["T_gt"],
            result.fitness if result is not None else np.nan,
            result.inlier_rmse if result is not None else np.nan,
            status,
        )
        # BOP VSD must use the native scene depth. The actual refinement input is
        # recorded separately in input_depth_condition.
        record = make_record(
            ns, meta, frame, frame_id, "clean_depth", 0.0,
            prepared["actual_fraction"], prepared["patch"], prepared["center_u"],
            prepared["center_v"], 0, method, source_count, evaluated, runtime,
        )
        record.update(
            {
                "input_depth_condition": condition_name,
                "input_outlier_fraction": float(prepared["requested_fraction"]),
                "selected_frame_index": int(selected_index),
                "patch_seed": int(prepared["patch_seed"]),
                "value_seed": int(prepared["value_seed"]),
                "initializer_score": float(selection["pose"]["score"]),
                "initializer_source_index": int(selection["pose"]["source_index"]),
                "mask_score": float(selection["mask"]["score"]),
                "mask_source_index": int(selection["mask"]["source_index"]),
                "pose_mask_bbox_iou": float(selection["bbox_iou"]),
                "target_points": int(len(target.points)),
                "target_preparation_ms": target_ms,
                "trunk_source_ms": source_ms,
                "trunk_icp_ms": trunk_icp_ms,
                **prepared["metadata"],
            }
        )
        if method == PROPOSED:
            support.annotate_gate(record, gate)
            record.update(
                {
                    "candidate_used": used,
                    "filtered_source_points": filtered_source_count,
                    "filtered_target_points": filtered_target_count,
                    "filtered_retention_ratio": retention,
                    "gate_ms": gate_ms,
                    "filter_ms": filter_ms,
                    "trunk_runtime_ms": trunk_ms,
                }
            )
        records.append(record)
    return records


def run(args: argparse.Namespace) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["method"] != PROPOSED or config["reliability_keep_rule"] != "reliability > 0.35":
        raise ValueError("Frozen method configuration does not match the implemented route.")
    source_manifest = pd.read_csv(args.frame_manifest)
    source_manifest = source_manifest[source_manifest["included"].astype(bool)].copy()
    if args.max_targets is not None:
        source_manifest = source_manifest.head(args.max_targets).copy()
    if source_manifest.empty:
        raise ValueError("No included targets were found in the source frame manifest.")
    if source_manifest.duplicated(["scene_id", "img_id", "object_id"]).any():
        raise ValueError("Target keys must be unique at scene/image/object level.")
    objects = sorted(source_manifest.object_id.astype(int).unique().tolist())
    target_keys = set(
        zip(
            source_manifest.scene_id.astype(int),
            source_manifest.img_id.astype(int),
            source_manifest.object_id.astype(int),
        )
    )
    source_paths = [
        Path(__file__), Path(frozen.__file__), Path(support.__file__),
        Path(__file__).with_name("run_frozen_mask_robustness.py"),
        Path(__file__).with_name("run_strong_baselines_only.py"), CONFIG,
        Path(__file__).with_name("evaluate_bop_metrics_controlled.py"),
        Path(__file__).with_name("analyze_frozen_real_inputs_validation.py"),
        Path(__file__).with_name("protocolG_observation.py"), support.SOURCE,
        args.dataset_root / "test_targets_bop19.json",
        args.frame_manifest, args.pose_csv, args.mask_json,
    ]
    conditions = condition_specs(tuple(args.outlier_fractions))
    expected_manifest = {
        "status": "protocol_locked_real_initializer_predicted_mask_v3_before_bop_results",
        "observation_version": OBSERVATION_VERSION,
        "cohort_label": args.cohort_label,
        "dataset_name": "T-LESS",
        "dataset_root": str(args.dataset_root),
        "dataset_split": args.dataset_split,
        "source_frame_manifest": str(args.frame_manifest),
        "corruption_realization": int(args.realization),
        "corruption_seed_salt": int(realization_salt(int(args.realization))),
        "corruption_seed_rule": (
            "patch_seed = 810000 + r*10^9 + 10^6*object_id + 10^4*selected_index; "
            "value_seed = 910000 + r*10^9 + 10^6*object_id + 10^4*selected_index; "
            "r=0 reproduces the frozen single-realization seeds exactly"
        ),
        "target_count": len(source_manifest),
        "objects": objects,
        "methods": METHODS,
        "input_conditions": [list(item) for item in conditions],
        "initialization_perturbation": {
            "rotation_deg": float(args.init_rotation_deg),
            "translation_mm": float(args.init_translation_mm),
            "seed": int(args.init_seed),
            "direction_rule": "per-target isotropic directions from a deterministic NumPy generator",
        },
        "prediction_selection": (
            "Use BOP19 inst_count as N; retain top-N pose scores per scene/image/object; "
            "greedily pair each pose to one unused mask by projected-CAD-bbox IoU, then mask score "
            "and source index. Pose-mask pairing uses predictions, CAD and K only. The evaluation "
            "harness associates these independent detections to the selected GT instance by visible-mask "
            "IoU; this association is never used by the refinement method."
        ),
        "missing_policy": "Missing pose or mask remains in the all-target denominator with zero AR.",
        "invalid_mask_policy": "Return initializer/P2P fallback; never remove the target after selection.",
        "primary_endpoint": (
            "Unseen12 local_outlier_30pct all-target delta AR, frozen dual filter minus P2P, "
            "with RGB-D-image clustered 95% CI and sign-flip test."
        ),
        "secondary_endpoints": (
            "Native-depth safety, matched-only refinement, AR-VSD/MSSD/MSPD, per-object effects, "
            "coverage, mask IoU, gate behavior and initializer-quality strata."
        ),
        "vsd_depth_policy": (
            "VSD visibility uses native test depth for both conditions; input_depth_condition "
            "records the depth supplied to refinement."
        ),
        "frozen_method_config": config,
        "pose_source_url": OFFICIAL_POSE_URL,
        "mask_source_url": OFFICIAL_MASK_SOURCE,
        "hashes": {str(path): digest(path) for path in source_paths},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol_manifest.json"
    if protocol_path.exists():
        if not args.resume:
            raise FileExistsError(f"Use --resume or a new output directory: {args.output}")
        actual = json.loads(protocol_path.read_text(encoding="utf-8"))
        for field in [
            "status", "cohort_label", "dataset_root", "dataset_split", "target_count",
            "objects", "methods", "input_conditions", "prediction_selection", "missing_policy",
            "observation_version", "corruption_realization", "corruption_seed_salt",
            "initialization_perturbation",
        ]:
            if actual.get(field) != expected_manifest[field]:
                raise ValueError(f"Resume protocol mismatch for {field}")
        for path in source_paths:
            if actual["hashes"].get(str(path)) != digest(path):
                raise ValueError(f"Source changed since protocol lock: {path}")
    else:
        protocol_path.write_text(json.dumps(expected_manifest, indent=2), encoding="utf-8")
        snapshots = args.output / "source_snapshot"
        snapshots.mkdir()
        for path in source_paths:
            shutil.copy2(path, snapshots / path.name)

    target_counts = load_bop_target_counts(args.dataset_root)
    missing_target_counts = sorted(target_keys.difference(target_counts))
    if missing_target_counts:
        raise ValueError(f"BOP19 target counts missing for {len(missing_target_counts)} keys")
    pose_index, total_pose_rows = load_pose_index(args.pose_csv, target_keys)
    mask_index, total_mask_rows = load_mask_index(args.mask_json, target_keys)
    print(
        f"Indexed official predictions: {total_pose_rows} poses, {total_mask_rows} masks; "
        f"targets={len(target_keys)}",
        flush=True,
    )
    ns = support.load_experiment_namespace(
        objects, max(1, len(source_manifest)), 1,
        [fraction for _, fraction in conditions if fraction > 0.0] or [0.30],
        dataset_root=args.dataset_root,
        dataset_split=args.dataset_split,
        depth_extension=".png",
    )

    trials_path = args.output / "trials_checkpoint.csv"
    frame_path = args.output / "frame_manifest.csv"
    trials = pd.read_csv(trials_path).to_dict("records") if args.resume and trials_path.exists() else []
    frame_rows = pd.read_csv(frame_path).to_dict("records") if args.resume and frame_path.exists() else []
    completed = {str(row["frame_id"]) for row in frame_rows if str(row.get("status", "")) != "pending"}
    for _, source_row in source_manifest.iterrows():
        frame_id = str(source_row.frame_id)
        if frame_id in completed:
            print(f"RESUME-SKIP {frame_id}", flush=True)
            continue
        object_id = int(source_row.object_id)
        key = (int(source_row.scene_id), int(source_row.img_id), object_id)
        meta = ns["configure_object"](object_id)
        candidate = candidate_from_manifest(source_row, args.dataset_root, args.dataset_split)
        frame = ns["load_frame_data"](candidate)
        n_top = target_counts[key]
        external_candidates = select_external_candidates(
            pose_index.get(key, []), mask_index.get(key, []), ns["full_model_points"], frame["K"], n_top
        )
        selection = associate_evaluation_instance(
            external_candidates, frame["mask_bool"], frame["depth"].shape
        )
        row = {
            "object_id": object_id,
            "frame_id": frame_id,
            "scene_id": key[0],
            "img_id": key[1],
            "inst_id": int(source_row.inst_id),
            "selected_index": int(source_row.selected_index),
            "pose_candidates": len(pose_index.get(key, [])),
            "mask_candidates": len(mask_index.get(key, [])),
            "bop_inst_count_n_top": n_top,
            "paired_external_candidates": len(external_candidates),
            "pose_prediction_found": int(bool(pose_index.get(key))),
            "mask_prediction_found": int(bool(mask_index.get(key))),
            "matched_prediction": int(selection is not None),
            "status": "pending",
        }
        if selection is None:
            row["status"] = "missing_pose" if not pose_index.get(key) else "missing_mask"
            frame_rows.append(row)
            pd.DataFrame(frame_rows).to_csv(frame_path, index=False)
            print(f"{frame_id}: {row['status']}", flush=True)
            continue
        predicted_mask = selection["predicted_mask"]
        quality = selection["mask_quality"]
        initial = internal_pose_from_bop(
            selection["rotation_bop"], selection["translation_bop_mm"]
        )
        initial, init_seed = perturb_initial_pose(
            initial,
            float(args.init_rotation_deg),
            float(args.init_translation_mm),
            object_id,
            int(source_row.selected_index),
            int(args.init_seed),
        )
        valid_predicted_depth = int(np.count_nonzero(predicted_mask & (frame["depth"] > 0)))
        row.update(
            {
                "status": "matched",
                "initializer_score": float(selection["pose"]["score"]),
                "initializer_source_index": int(selection["pose"]["source_index"]),
                "mask_score": float(selection["mask"]["score"]),
                "mask_source_index": int(selection["mask"]["source_index"]),
                "selected_pose_rank_within_n_top": int(selection["pose_rank"]),
                "pose_mask_bbox_iou": float(selection["bbox_iou"]),
                "evaluation_instance_mask_iou": float(selection["evaluation_association_iou"]),
                "init_perturb_rotation_deg": float(args.init_rotation_deg),
                "init_perturb_translation_mm": float(args.init_translation_mm),
                "init_perturb_seed": int(init_seed),
                "predicted_valid_depth_pixels": valid_predicted_depth,
                **quality,
            }
        )
        projected = selection["projected_bbox"]
        mask_bbox = selection["mask"]["bbox"]
        for prefix, values in [("projected_bbox", projected), ("mask_bbox", mask_bbox)]:
            for suffix, value in zip(("x", "y", "w", "h"), values):
                row[f"{prefix}_{suffix}"] = float(value)
        selected_index = int(source_row.selected_index)
        frame_trials = []
        try:
            for condition_name, fraction in conditions:
                prepared = prepare_depth(
                    ns, frame, predicted_mask, object_id, selected_index, condition_name, fraction,
                    realization_salt(int(args.realization)),
                )
                prepared["condition_name"] = condition_name
                prepared["requested_fraction"] = fraction
                condition_records = run_methods(
                    ns, frame, meta, initial, predicted_mask, prepared, frame_id, selection, selected_index
                )
                for record in condition_records:
                    record.update(quality)
                    record.update(
                        {
                            "init_perturb_rotation_deg": float(args.init_rotation_deg),
                            "init_perturb_translation_mm": float(args.init_translation_mm),
                            "init_perturb_seed": int(init_seed),
                            "corruption_realization": int(args.realization),
                        }
                    )
                frame_trials.extend(condition_records)
        except RuntimeError as error:
            if "No valid object depth pixels" not in str(error):
                raise
            row["matched_prediction"] = 0
            row["status"] = "invalid_object_depth"
            row["failure_reason"] = str(error)
            frame_rows.append(row)
            pd.DataFrame(frame_rows).to_csv(frame_path, index=False)
            print(f"{frame_id}: invalid_object_depth", flush=True)
            continue
        trials.extend(frame_trials)
        frame_rows.append(row)
        pd.DataFrame(trials).to_csv(trials_path, index=False)
        pd.DataFrame(frame_rows).to_csv(frame_path, index=False)
        print(
            f"{frame_id}: matched; mask IoU={quality['mask_iou']:.3f}; "
            f"valid predicted depth={valid_predicted_depth}",
            flush=True,
        )

    frame_table = pd.DataFrame(frame_rows)
    trial_table = pd.DataFrame(trials)
    matched = int(frame_table.matched_prediction.sum())
    expected_rows = matched * len(conditions) * len(METHODS)
    if len(trial_table) != expected_rows:
        raise ValueError(f"Expected {expected_rows} trial rows, found {len(trial_table)}")
    if not trial_table.empty:
        keys = support.KEY_COLUMNS + ["input_depth_condition", "method"]
        if trial_table.duplicated(keys).any():
            raise ValueError("Duplicate trial condition/method rows detected.")
        if not np.isfinite(trial_table[POSE_COLUMNS + ["ADD_S_mm"]]).all().all():
            raise ValueError("Non-finite pose or ADD-S value detected.")
    final_trials = args.output / "real_inputs_validation_trials.csv"
    trial_table.to_csv(final_trials, index=False)
    frame_table.to_csv(frame_path, index=False)
    completion = {
        "targets": len(frame_table),
        "matched_targets": matched,
        "coverage": matched / len(frame_table),
        "missing_pose": int((frame_table.status == "missing_pose").sum()),
        "missing_mask": int((frame_table.status == "missing_mask").sum()),
        "trial_rows": len(trial_table),
    }
    (args.output / "run_complete.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    print("COMPLETE", args.output, json.dumps(completion), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--pose-csv", type=Path, default=POSE_CSV)
    parser.add_argument("--mask-json", type=Path, default=MASK_JSON)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--dataset-split", default="test_primesense")
    parser.add_argument("--cohort-label", required=True)
    parser.add_argument(
        "--outlier-fractions", default="0.30",
        help="Comma-separated local-outlier fractions; native_depth is always included.",
    )
    parser.add_argument("--init-rotation-deg", type=float, default=0.0)
    parser.add_argument("--init-translation-mm", type=float, default=0.0)
    parser.add_argument("--init-seed", type=int, default=20260920)
    parser.add_argument(
        "--realization",
        type=int,
        default=0,
        help=(
            "pre-registered corruption realization index r; the corruption seeds gain "
            "r*10^9, so r=0 reproduces the frozen single-realization seeds exactly"
        ),
    )
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.outlier_fractions = tuple(
        float(value.strip()) for value in str(parsed.outlier_fractions).split(",") if value.strip()
    )
    run(parsed)
