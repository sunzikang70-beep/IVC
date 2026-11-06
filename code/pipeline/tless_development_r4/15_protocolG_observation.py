"""One recorded observation contract for refinement, geometry and rendering.

Depth is in meters here. Evaluation-only corruption fields are deliberately
ignored: VSD can use native depth without granting it to the method.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


OBSERVATION_VERSION = "protocolG-observation-v3"
HASH_FIELDS = ("method_depth_sha256", "method_patch_sha256", "method_mask_sha256", "method_K_sha256")
OBSERVATION_FIELDS = (
    "observation_version", "method_depth_unit", "patch_generator",
    "patch_seed", "value_seed", *HASH_FIELDS,
)


def array_digest(array, dtype) -> str:
    canonical = np.ascontiguousarray(array, dtype=dtype)
    header = json.dumps([canonical.shape, canonical.dtype.str], separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + b"\n" + canonical.tobytes()).hexdigest()


def observation_metadata(prepared, mask, K) -> dict:
    return {
        "observation_version": OBSERVATION_VERSION,
        "method_depth_unit": "m",
        "patch_generator": "legacy_nearest_valid_pixels",
        "patch_seed": int(prepared["patch_seed"]),
        "value_seed": int(prepared["value_seed"]),
        "method_depth_sha256": array_digest(prepared["depth"], "<f8"),
        "method_patch_sha256": array_digest(prepared["patch"], "u1"),
        "method_mask_sha256": array_digest(mask, "u1"),
        "method_K_sha256": array_digest(K, "<f8"),
    }


def prepare_depth(ns, frame, predicted_mask, object_id, selected_index, name, fraction,
                  seed_salt=0, *, patch_seed=None, value_seed=None):
    depth = np.asarray(frame["depth"], dtype=np.float64)
    mask = np.asarray(predicted_mask, dtype=bool)
    if depth.shape != mask.shape:
        raise ValueError("Method depth and mask must have the same shape")
    if not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("Method depth must be finite, nonnegative meters")
    if name == "native_depth":
        if float(fraction) != 0:
            raise ValueError("native_depth must have zero outlier fraction")
        prepared = {
            "depth": depth.copy(), "patch": np.zeros_like(mask),
            "center_u": -1, "center_v": -1, "actual_fraction": 0.0,
            "patch_seed": -1, "value_seed": -1,
        }
    else:
        expected = f"local_outlier_{int(round(100 * float(fraction))):02d}pct"
        if not 0 < float(fraction) < 1 or name != expected:
            raise ValueError(f"Unsupported method observation: {name}/{fraction}")
        if (patch_seed is None) != (value_seed is None):
            raise ValueError("Both recorded corruption seeds are required")
        if patch_seed is None:
            patch_seed = 810000 + int(seed_salt) + int(object_id) * 1000000 + int(selected_index) * 10000
            value_seed = 910000 + int(seed_salt) + int(object_id) * 1000000 + int(selected_index) * 10000
        if min(int(patch_seed), int(value_seed)) < 0:
            raise ValueError("Corrupted observations require nonnegative seeds")
        patch, center_u, center_v, actual = ns["select_local_patch"](
            depth, mask, float(fraction), int(patch_seed)
        )
        corrupted = ns["make_local_corrupted_target"](
            depth, mask, frame["K"], patch, "local_outlier", int(value_seed)
        )
        prepared = {
            "depth": corrupted["corrupted_depth"], "patch": patch,
            "center_u": center_u, "center_v": center_v, "actual_fraction": actual,
            "patch_seed": int(patch_seed), "value_seed": int(value_seed),
        }
    prepared["condition_name"] = str(name)
    prepared["requested_fraction"] = float(fraction)
    prepared["metadata"] = observation_metadata(prepared, mask, frame["K"])
    return prepared


def validate_condition_rows(rows) -> None:
    """Candidates in one condition cannot silently use different observations."""
    for field in (*OBSERVATION_FIELDS, "input_depth_condition", "input_outlier_fraction",
                  "selected_frame_index", "corruption_realization"):
        if field in rows and rows[field].nunique(dropna=False) != 1:
            raise ValueError(f"Inconsistent condition observation field: {field}")
    for field in ("solver_version", "solver_config_sha256"):
        if field in rows and rows[field].nunique(dropna=False) != 1:
            raise ValueError(f"Mixed solver provenance: {field}")


def replay_method_depth(ns, frame, predicted_mask, row, selected_index):
    name = str(row["input_depth_condition"])
    fraction = float(row["input_outlier_fraction"])
    if "selected_frame_index" in row and int(row["selected_frame_index"]) != int(selected_index):
        raise ValueError("Selected frame index differs from recorded observation")
    seeds = {}
    if name != "native_depth":
        for field in ("patch_seed", "value_seed"):
            if field not in row or not np.isfinite(float(row[field])):
                raise ValueError(f"Missing recorded {field}; refusing to guess realization 0")
            seeds[field] = int(row[field])
    for field in OBSERVATION_FIELDS:
        if field not in row:
            raise ValueError(f"Missing observation contract field: {field}; regenerate upstream inputs")
    if row["observation_version"] != OBSERVATION_VERSION:
        raise ValueError("Unsupported observation contract version")
    prepared = prepare_depth(
        ns, frame, predicted_mask, int(row["object_id"]), int(selected_index), name, fraction, **seeds
    )
    for field in OBSERVATION_FIELDS:
        if row[field] != prepared["metadata"][field]:
            raise ValueError(f"Observation replay mismatch: {field}")
    return prepared
