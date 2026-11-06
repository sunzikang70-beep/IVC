"""Observation-only local-observability state for the decoupled development bank.

For every development condition (case x input-depth condition), this script replays the
recorded refinement observation and computes, without ground truth:

* point-to-plane information-matrix (6x6) observability of the observed visible model at
  T0 and at each of the six bounded candidates (eigen-spectrum, weak directions,
  linearised pose standard deviation);
* how well the observation constrains the direction of each candidate update;
* multi-start point-to-point ICP convergence spread from K fixed perturbations of T0, using
  the unchanged trunk ICP settings, and each candidate's motion towards the multi-start
  consensus.

Nothing is trained here and the existing bank (inputs, trials, labels) is read-only.
Run order: observability_residual_gate.py --stage lock  ->  this script  ->  gate --stage evaluate.
"""
from __future__ import annotations

import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import zlib

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(r"D:\6D_pose\codex_paper_workspace")
BANK = ROOT / "outputs" / "decoupled_development_20261020_r4"
OUTPUT = ROOT / "outputs" / "observability_state_dev_20261105"

INITIAL = "protocolB_e2e_initial_pose"
TRUNK = "protocolB_e2e_open3d_point_to_point_icp"
PROPOSED = "protocolB_e2e_gate_p2p_direct_dual_filter_r035"
CANDIDATES = [f"protocolF_se3_trust_{d}_s{s}" for d in ("p2p", "filtered") for s in ("025", "050", "100")]
TRUNK_OF = {"p2p": TRUNK, "filtered": PROPOSED}
SCALE_OF = {"025": 0.25, "050": 0.50, "100": 1.00}
KEYS = ["cohort", "object_id", "scene_id", "img_id", "inst_id", "frame_id", "input_depth_condition"]

CONFIG = dict(
    version="observability-state-v1",
    inlier_threshold_m=0.010,        # point-to-plane correspondences used for the information matrix
    min_inliers=30,
    normal_knn=30,                   # PCA normals on the observed cloud (Open3D default neighbourhood size)
    weak_ratio=0.05,                 # eigen-direction is weak if lambda < weak_ratio * lambda_max
    multistart_k=8,
    multistart_rotation_deg=3.0,     # fixed-magnitude perturbations of T0, R' = Q R, t' = t + u
    multistart_translation_mm=3.0,
    multistart_seed=20261105,
    agree_fraction_of_diameter=0.02,
    model_subsample=500,             # model points used for ADD-S distances between poses
)

POSE_FEATURES = ["inliers", "inlier_fraction", "p2plane_rms_mm", "inv_cond", "log10_lmin", "n_weak",
                 "rot_block_min", "trans_block_min", "std_weak_mm", "std_trace_mm"]
UPDATE_FEATURES = ["upd_rayleigh", "upd_weak_fraction", "upd_std_along_mm", "upd_snr"]
MULTISTART_FEATURES = ["ms_valid_runs", "ms_spread_norm", "ms_to_trunk_norm", "ms_frac_agree", "ms_rmse_std_mm"]
CANDIDATE_MS_FEATURES = ["ms_cand_to_medoid_norm", "ms_cand_toward_medoid_norm"]


def config_digest(config=CONFIG) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- pure numerics

def estimate_normals(points: np.ndarray, knn: int) -> np.ndarray:
    """PCA normals (smallest-eigenvalue direction of the k-NN covariance); sign is irrelevant here."""
    points = np.asarray(points, dtype=np.float64)
    k = int(min(knn, len(points)))
    _, index = cKDTree(points).query(points, k=k)
    neighbours = points[index]
    centred = neighbours - neighbours.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centred, centred) / max(k - 1, 1)
    _, vectors = np.linalg.eigh(covariance)
    return vectors[:, :, 0]


def transform(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def safe_inverse(values: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return (vectors / np.maximum(values, 1e-12)) @ vectors.T


def pose_observability(source_model, pose, target_points, target_normals, tree, centre, length, config):
    """Point-to-plane information matrix H = mean(J^T J), J = [((x-c) x n)/L, n] at a pose."""
    empty = {name: np.nan for name in POSE_FEATURES}
    if len(source_model) < config["min_inliers"] or len(target_points) < config["min_inliers"]:
        return empty, None
    camera = transform(source_model, pose)
    distance, index = tree.query(camera, k=1)
    inlier = distance < config["inlier_threshold_m"]
    count = int(inlier.sum())
    if count < config["min_inliers"]:
        return dict(empty, inliers=count, inlier_fraction=count / len(source_model)), None
    x = camera[inlier]
    y = target_points[index[inlier]]
    n = target_normals[index[inlier]]
    J = np.hstack([np.cross(x - centre, n) / length, n])
    H = J.T @ J / count
    residual = np.einsum("ij,ij->i", x - y, n)
    sigma = float(np.sqrt(np.mean(residual ** 2)))
    values, vectors = np.linalg.eigh(H)
    top = max(float(values[-1]), 1e-18)
    low = max(float(values[0]), 0.0)
    rot = np.linalg.eigvalsh(H[:3, :3])[0]
    trans = np.linalg.eigvalsh(H[3:, 3:])[0]
    inverse = safe_inverse(values, vectors)
    features = dict(
        inliers=count, inlier_fraction=count / len(source_model), p2plane_rms_mm=sigma * 1000.,
        inv_cond=low / top, log10_lmin=float(np.log10(max(low, 1e-12))),
        n_weak=int(np.sum(values < config["weak_ratio"] * top)),
        rot_block_min=float(rot) / top, trans_block_min=float(trans) / top,
        std_weak_mm=float(sigma / np.sqrt(count * max(low, 1e-12)) * 1000.),
        std_trace_mm=float(sigma * np.sqrt(np.trace(inverse) / count) * 1000.),
    )
    return features, dict(H=H, values=values, vectors=vectors, inverse=inverse, sigma=sigma, count=count)


def update_twist(initial, candidate, centre, length):
    """Twist of T0 -> candidate written as a rotation about the visible-model centre c (metres)."""
    delta = candidate[:3, :3] @ initial[:3, :3].T
    omega = Rotation.from_matrix(delta).as_rotvec()
    shift = candidate[:3, 3] - delta @ initial[:3, 3] - centre + delta @ centre
    return np.concatenate([omega * length, shift])


def update_observability(xi, info, config):
    empty = {name: np.nan for name in UPDATE_FEATURES}
    norm = float(np.linalg.norm(xi))
    if info is None or norm < 1e-12:
        return empty
    unit = xi / norm
    top = max(float(info["values"][-1]), 1e-18)
    weak = info["values"] < config["weak_ratio"] * top
    projection = info["vectors"].T @ unit
    along = float(info["sigma"] * np.sqrt(max(unit @ info["inverse"] @ unit, 0.) / info["count"]) * 1000.)
    return dict(
        upd_rayleigh=float(unit @ info["H"] @ unit) / top,
        upd_weak_fraction=float(np.sum(projection[weak] ** 2)),
        upd_std_along_mm=along,
        upd_snr=norm * 1000. / max(along, 1e-9),
    )


def adds_distance(points, first, second):
    """Mean closest-point distance between the model under two poses (symmetry-tolerant)."""
    a = transform(points, first)
    b = transform(points, second)
    return 0.5 * (cKDTree(b).query(a, k=1)[0].mean() + cKDTree(a).query(b, k=1)[0].mean())


def perturbed_starts(initial, config, key):
    seed = [config["multistart_seed"], zlib.crc32(key.encode("utf-8"))]
    rng = np.random.default_rng(seed)
    starts = []
    for _ in range(config["multistart_k"]):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        start = np.eye(4)
        start[:3, :3] = Rotation.from_rotvec(axis * np.radians(config["multistart_rotation_deg"])).as_matrix() @ initial[:3, :3]
        start[:3, 3] = initial[:3, 3] + direction * config["multistart_translation_mm"] / 1000.
        starts.append(start)
    return starts


def multistart_features(points, converged, rmse, trunk, initial, candidates, diameter_m, config):
    target = {name: np.nan for name in MULTISTART_FEATURES}
    per_candidate = {name: {f: np.nan for f in CANDIDATE_MS_FEATURES} for name in candidates}
    target["ms_valid_runs"] = len(converged)
    if len(converged) < 2:
        return target, per_candidate
    k = len(converged)
    pair = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            pair[i, j] = pair[j, i] = adds_distance(points, converged[i], converged[j])
    to_trunk = np.array([adds_distance(points, pose, trunk) for pose in converged])
    medoid = converged[int(np.argmin(pair.sum(axis=1)))]
    initial_to_medoid = adds_distance(points, initial, medoid)
    target.update(
        ms_spread_norm=float(pair[np.triu_indices(k, 1)].mean() / diameter_m),
        ms_to_trunk_norm=float(to_trunk.mean() / diameter_m),
        ms_frac_agree=float(np.mean(to_trunk < config["agree_fraction_of_diameter"] * diameter_m)),
        ms_rmse_std_mm=float(np.std(rmse) * 1000.),
    )
    for name, pose in candidates.items():
        distance = adds_distance(points, pose, medoid)
        per_candidate[name] = dict(ms_cand_to_medoid_norm=float(distance / diameter_m),
                                   ms_cand_toward_medoid_norm=float((initial_to_medoid - distance) / diameter_m))
    return target, per_candidate


def condition_features(source_model, target_points, poses, trunk_pose, model_points, diameter_m,
                       config, key, converged=None, rmse=None):
    """All features of one condition. `converged`/`rmse` come from the multi-start ICP runs."""
    initial = poses[INITIAL]
    length = diameter_m / 2.
    usable = len(source_model) >= config["min_inliers"] and len(target_points) >= config["min_inliers"]
    if usable:
        centre = transform(source_model, initial).mean(axis=0)
        normals = estimate_normals(target_points, config["normal_knn"])
        tree = cKDTree(target_points)
    rows, info0 = {}, None
    for name, pose in poses.items():
        if usable:
            values, info = pose_observability(source_model, pose, target_points, normals, tree, centre, length, config)
        else:
            values, info = {k: np.nan for k in POSE_FEATURES}, None
            centre = np.zeros(3)
        if name == INITIAL:
            info0 = info
        rows[name] = {"obs_" + k: v for k, v in values.items()}
    for name in CANDIDATES:
        rows[name].update(update_observability(update_twist(initial, poses[name], centre, length), info0, config))
    target, per_candidate = multistart_features(model_points, converged or [], rmse or [], trunk_pose, initial,
                                                {n: poses[n] for n in CANDIDATES}, diameter_m, config)
    for name in poses:
        rows[name].update(target)
        rows[name].update(per_candidate.get(name, {f: np.nan for f in CANDIDATE_MS_FEATURES}))
        if name == INITIAL:
            rows[name].update({f: np.nan for f in UPDATE_FEATURES})
    return rows


# ----------------------------------------------------------------------------- bank replay

def process_object(task):
    """Worker: replay every condition of one object and return feature rows (no labels read)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_frozen_real_inputs_validation as real
    import run_se3_trust_region_geometric_development as trust
    import prepare_winner_instance_expansion as p
    from protocolG_observation import replay_method_depth

    oid, cases, inputs, trials, segmentations, dataset, n_manifest, config = task
    ns = real.support.load_experiment_namespace(p.OBJECTS, n_manifest, 1, [.3], dataset_root=Path(dataset))
    ns["configure_object"](int(oid))
    model = np.asarray(ns["full_model_points"], dtype=np.float64)
    model_points = model[::max(1, int(np.ceil(len(model) / config["model_subsample"])))]
    out, timing = [], []
    for case in cases:
        frame = ns["load_frame_data"](real.candidate_from_manifest(pd.Series(case), Path(dataset), "test_primesense"))
        frame["T_gt"] = np.full((4, 4), np.nan)          # label poison: any use of ground truth fails loudly
        mask = real.decode_uncompressed_rle(segmentations[int(case["mask_source_index"])])
        for condition in ("native_depth", "local_outlier_30pct"):
            start_time = time.perf_counter()
            select = (inputs.cohort.eq(case["cohort"]) & inputs.frame_id.eq(case["frame_id"])
                      & inputs.input_depth_condition.eq(condition))
            records = {row.method: row for _, row in inputs.loc[select].iterrows()}
            tselect = (trials.cohort.eq(case["cohort"]) & trials.frame_id.eq(case["frame_id"])
                       & trials.input_depth_condition.eq(condition))
            bank = {row.method: row for _, row in trials.loc[tselect].iterrows()}
            if set(records) != {INITIAL, TRUNK, PROPOSED} or set(bank) != {INITIAL, *CANDIDATES}:
                raise ValueError(f"Incomplete bank rows for {case['cohort']} {case['frame_id']} {condition}")
            prepared = replay_method_depth(ns, frame, mask, records[INITIAL], int(case["selected_index"]))
            depth, K = prepared["depth"], frame["K"]
            target, _ = ns["depth_to_cloud"](depth, mask, K)
            target_points = np.asarray(target.points, dtype=np.float64)
            poses = {name: real.support.transform_from_row(bank[name]) for name in [INITIAL, *CANDIDATES]}
            initial = poses[INITIAL]
            if not np.allclose(initial, real.support.transform_from_row(records[INITIAL]), atol=1e-9):
                raise ValueError("Initial pose differs between inputs and trials")
            for direction, trunk_name in TRUNK_OF.items():
                full = real.support.transform_from_row(records[trunk_name])
                for suffix, scale in SCALE_OF.items():
                    expected, _ = trust.restricted_pose(initial, full, scale)
                    if not np.allclose(expected, poses[f"protocolF_se3_trust_{direction}_s{suffix}"], atol=1e-9):
                        raise ValueError("Candidate pose does not reproduce from its trunk")
            source, _, _ = ns["generate_observation_supported_visible_model"](ns["full_model_points"], initial, depth, mask, K, target)
            source = np.asarray(source, dtype=np.float64)
            if len(source) != int(records[TRUNK].source_points):
                raise ValueError("Visible-model source differs from the recorded trunk input")
            key = f"{case['cohort']}|{case['frame_id']}|{condition}"
            converged, rmse = [], []
            ms_start = time.perf_counter()
            if len(source) >= 3 and len(target_points) >= 3:
                for start in perturbed_starts(initial, config, key):
                    result = real.support.run_point_to_point(ns, source, target, start)
                    if np.isfinite(result.transformation).all() and len(result.correspondence_set) >= 3:
                        converged.append(np.asarray(result.transformation))
                        rmse.append(float(result.inlier_rmse))
            ms_ms = (time.perf_counter() - ms_start) * 1000.
            diameter_m = float(records[INITIAL].diameter_mm) / 1000.
            rows = condition_features(source, target_points, poses, real.support.transform_from_row(records[TRUNK]),
                                      model_points, diameter_m, config, key, converged, rmse)
            total_ms = (time.perf_counter() - start_time) * 1000.
            for name, values in rows.items():
                out.append(dict(**{k: case[k] for k in KEYS if k != "input_depth_condition"},
                                input_depth_condition=condition, method=name, **values))
            timing.append(dict(cohort=case["cohort"], frame_id=case["frame_id"], input_depth_condition=condition,
                               multistart_ms=ms_ms, total_ms=total_ms, target_points=len(target_points), source_points=len(source)))
    return int(oid), out, timing


def run(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_frozen_real_inputs_validation as real
    from extract_real_gtfree_acceptance_features import masks_by_source_index
    from winner_training_data_io import read_csv

    output = args.output
    if args.smoke and output == OUTPUT:
        output = OUTPUT.with_name(OUTPUT.name + "_smoke")
    objects_dir = output / "objects"
    state_path = output / "state.json"
    if output.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_sha256") != config_digest() or state.get("script_sha256") != file_digest(Path(__file__)):
            raise FileExistsError("Existing output was produced by a different script/config; use a new --output")
        if state.get("status") == "COMPLETE":
            raise FileExistsError("Already complete; nothing to do")
    else:
        objects_dir.mkdir(parents=True)
        state = dict(status="RUNNING", created_at_utc=datetime.now(timezone.utc).isoformat(), config=CONFIG,
                     config_sha256=config_digest(), script_sha256=file_digest(Path(__file__)),
                     bank=str(args.bank), smoke=bool(args.smoke),
                     input_sha256={name: file_digest(args.bank / name) for name in
                                   ["locked_development_instances.csv", "inputs.csv", "trials.csv"]})
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    manifest = read_csv(args.bank / "locked_development_instances.csv")
    matched = manifest.loc[manifest.matched_prediction.eq(1)].copy()
    inputs = read_csv(args.bank / "inputs.csv")
    trials = read_csv(args.bank / "trials.csv")
    if len(matched) * 2 * 3 != len(inputs) or len(matched) * 2 * 7 != len(trials):
        raise ValueError("Bank size differs from the locked development bank")
    if args.smoke:
        first = sorted(matched.object_id.unique())[0]
        matched = matched.loc[matched.object_id.eq(first)].head(2)
    segmentations = masks_by_source_index(real.MASK_JSON, set(matched.mask_source_index.astype(int)))
    tasks = []
    for oid, group in matched.groupby("object_id", sort=True):
        if (objects_dir / f"object_{int(oid):02d}.csv").exists():
            print("skip complete object", int(oid), flush=True)
            continue
        keys = set(group.frame_id)
        tasks.append((int(oid), group.to_dict("records"),
                      inputs.loc[inputs.frame_id.isin(keys)].copy(), trials.loc[trials.frame_id.isin(keys)].copy(),
                      {int(i): segmentations[int(i)] for i in group.mask_source_index},
                      str(args.dataset), len(manifest), CONFIG))
    started = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_object, task) for task in tasks]
            for future in as_completed(futures):
                oid, rows, timing = future.result()
                pd.DataFrame(timing).to_csv(objects_dir / f"timing_{oid:02d}.csv", index=False)
                pd.DataFrame(rows).to_csv(objects_dir / f"object_{oid:02d}.csv", index=False)
                print(f"OBJECT {oid:02d} done: {len(rows)} rows; elapsed {time.monotonic()-started:.0f}s", flush=True)
        parts = sorted(objects_dir.glob("object_*.csv"))
        features = pd.concat([pd.read_csv(path) for path in parts], ignore_index=True)
        timing = pd.concat([pd.read_csv(path) for path in sorted(objects_dir.glob("timing_*.csv"))], ignore_index=True)
        expected = len(matched) * 2 * 7
        if len(features) != expected or features.duplicated(["cohort", "frame_id", "input_depth_condition", "method"]).any():
            raise ValueError(f"Feature rows {len(features)} != expected {expected} or duplicated")
        features.to_csv(output / "observability_features.csv", index=False)
        timing.to_csv(output / "timing.csv", index=False)
        state.update(status="COMPLETE", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                     rows=len(features), conditions=len(timing),
                     features_sha256=file_digest(output / "observability_features.csv"),
                     median_condition_ms=float(timing.total_ms.median()),
                     median_multistart_ms=float(timing.multistart_ms.median()),
                     missing_fraction={c: float(features[c].isna().mean()) for c in features.columns if c.startswith(("obs_", "upd_", "ms_"))})
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print("COMPLETE", len(features), "rows;", output / "observability_features.csv", flush=True)
    except Exception:
        state.update(status="FAILED", error=traceback.format_exc())
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, default=BANK)
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\6D_pose\bop_datasets\tless"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="two cases of the first object, for a check run")
    run(parser.parse_args())
