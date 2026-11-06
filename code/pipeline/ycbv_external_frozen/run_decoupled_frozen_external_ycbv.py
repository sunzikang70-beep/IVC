"""Frozen r4 external validation on YCB-V (BOP19 test split, public CosyPose inputs).

Nothing is fitted, tuned or selected on YCB-V. The deployment artifact is the
22-object T-LESS development model that was applied unchanged to the 8 unseen
T-LESS objects (decoupled_development_20261020_r4/group_oof). YCB-V adds a new
dataset domain: 21 textured household objects, a different RGB-D sensor and a
different public initializer (CosyPose ECCV'20, synt+real, single view, with its
own instance masks).

Cohort (locked before any r4 output exists on YCB-V)
  Every visibility-qualified BOP19 YCB-V test target already enumerated for the
  September full-pool run (visib_fract >= 0.30, px_count_visib >= 500, highest-
  visibility instance per image and object): 4,102 targets in 900 images, of
  which 4,024 have a matched public CosyPose pose and mask. As on T-LESS, the
  matched targets are evaluated; coverage is reported separately.
  Each matched target is run under the same three decoupled initialisations as
  T-LESS (native, mild 5 deg / 5 mm, hard 15 deg / 15 mm; identical seeds) and the
  same two depth conditions (native, 30 % local outlier patch). The outlier
  realisation is the one recorded by the September run, replayed and verified.

Stages (each resumable; --stage all runs them in order and skips finished work)
  prepare   lock cohort, shard plan, protocol and primary endpoint
  generate  per shard: decoupled initial pose -> P2P / filtered P2P -> 6 SE(3)
            trust-region candidates -> geometry scores -> render-and-compare tails
  merge     concatenate shards; assemble the frozen 191-feature schema (GT-blind check)
  bop       BOP19 VSD/MSSD/MSPD for initial + 6 candidates (labels only now)
  apply     frozen policies (D = conditional_residual_joint) and same-bank arms A-D
  analyze   object-equal summaries, 21-object cluster bootstrap, gate diagnostics

Usage (Windows, pose environment; run from the work folder)
  python run_decoupled_frozen_external_ycbv.py --stage all --smoke
  python run_decoupled_frozen_external_ycbv.py --stage all --shards 8 --workers 8 --bop-parts 4 --bop-workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

for _name, _value in {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}.items():
    os.environ.setdefault(_name, _value)

WORK = Path(__file__).resolve().parent
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import extract_bop_aware_tail_features as tail
import extract_real_gtfree_acceptance_features as feature_support
import run_decoupled_development as development
import run_decoupled_frozen_unseen5 as u5
import run_frozen_real_inputs_validation as real
import run_se3_trust_region_geometric_development as trust
import train_bop_aware_group_oof as base
import train_bop_aware_selected_risk as selected_risk
import train_bop_aware_state_residual as residual
import train_decoupled_group_oof as decoupled_oof
from decoupled_checkpoint_recovery import INITIAL_DIAGNOSTICS
from pose_perturbation_protocols import CAMERA, DECOUPLED, noise, paired_poses
from protocolG_observation import HASH_FIELDS, observation_metadata
from winner_training_data_io import read_csv

ROOT = WORK.parent
YCBV = Path(os.environ.get("YCBV_ROOT", r"D:\6D_pose\bop_datasets\YCB-V"))
DEFAULT_DATASET = YCBV / "protocolG_runtime"
DEFAULT_POSE_CSV = YCBV / "cosypose-eccv20-syntreal-1view_ycbv-test_e0fe67ff-0117-401d-9830-791ee9a353e6.csv"
DEFAULT_MASK_JSON = YCBV / "cosypose-eccv20-syntreal-1view_ycbv-test_8dde7029-85bc-49a0-bae1-15cab9b505dd.json"
DEFAULT_TOOLKIT = Path(os.environ.get("BOP_TOOLKIT", r"D:\6D_pose\tools\bop_toolkit"))
SOURCE_RUN = ROOT / "outputs" / "ycbv_full_pool_external_20260924" / "merged"
SAME_BANK = ROOT / "outputs" / "same_bank_residual_ablation_20261026_r1"
DEFAULT_OUTPUT = ROOT / "outputs" / "decoupled_frozen_ycbv_external_20261104"
SMOKE_OUTPUT = ROOT / "outputs" / "decoupled_frozen_ycbv_external_20261104_smoke"

DATASET_NAME = "YCB-V"
SPLIT = "test"
COHORT_PREFIX = "ycbv_external_decoupled_v1"
EXPECTED_POOL, EXPECTED_MATCHED, EXPECTED_OBJECTS, EXPECTED_IMAGES = 4102, 4024, 21, 900
SMOKE_OBJECTS, SMOKE_FRAMES = (1, 14), 2
REGIMES = dict(u5.REGIMES)
if REGIMES != {"native": (0.0, 0.0, 20261016), "mild": (5.0, 5.0, 20261016), "hard": (15.0, 15.0, 20261017)}:
    raise RuntimeError("T-LESS decoupled regimes changed; refusing to run")
REGIME_ORDER = {name: i for i, name in enumerate(REGIMES)}
METHODS = [base.INITIAL, *base.ACTIONS]
CONDITIONS = [name for name, _ in real.INPUT_CONDITIONS]
TABLES = ["inputs", "trials", "scores", "tails", "observations", "geometry"]
SHARED_POLICY = dict(risk_limit=.3, component_loss=.01, continuous_weight=.005, disabled=False)

digest, save, now = u5.digest, u5.save, u5.now


def load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def update_state(out: Path, **values) -> dict:
    path = out / "state.json"
    state = load(path) if path.exists() else {}
    state.update(values)
    save(path, state)
    return state


def shard_dir(out: Path, shard: int) -> Path:
    return out / "shards" / f"shard_{int(shard):02d}"


# --------------------------------------------------------------------- prepare
def stage_prepare(args, out: Path) -> None:
    if (out / "locked_protocol.json").exists():
        cfg = load(out / "run_config.json")
        if bool(cfg["smoke"]) != bool(args.smoke):
            raise ValueError(f"{out} was prepared with smoke={cfg['smoke']}; use the matching --output")
        log(f"PREPARE already locked: {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    for path in [args.dataset / SPLIT, args.dataset / "models_eval" / "models_info.json", args.pose_csv,
                 args.mask_json, args.toolkit / "bop_toolkit_lib", SOURCE_RUN / "frame_manifest.csv",
                 SOURCE_RUN / "real_inputs_validation_trials.csv", SAME_BANK / "frozen_development_policies.json"]:
        if not path.exists():
            raise FileNotFoundError(path)
    dev_state, model, model_meta = u5.verify_frozen_development()
    for variant in ["conditional_residual_joint", "conditional_residual_legacy"]:
        if model["policies"][variant] != SHARED_POLICY:
            raise ValueError(f"Frozen {variant} policy changed")

    pool = read_csv(SOURCE_RUN / "frame_manifest.csv")
    if pool.duplicated("frame_id").any() or pool.duplicated(["scene_id", "img_id", "object_id"]).any():
        raise ValueError("Full-pool manifest has duplicate targets")
    images_all = pool.groupby(["scene_id", "img_id"]).ngroups
    if (len(pool), pool.object_id.nunique(), images_all) != (EXPECTED_POOL, EXPECTED_OBJECTS, EXPECTED_IMAGES):
        raise ValueError("Full-pool manifest differs from the September cohort")
    matched = pool.loc[pool.matched_prediction.astype(int).eq(1)].copy()
    if len(matched) != EXPECTED_MATCHED:
        raise ValueError("Matched-target count differs from the September cohort")
    if args.smoke:
        matched = (matched.loc[matched.object_id.isin(SMOKE_OBJECTS)].sort_values(["object_id", "frame_id"])
                   .groupby("object_id").head(SMOKE_FRAMES).copy())

    source = pd.read_csv(SOURCE_RUN / "real_inputs_validation_trials.csv",
                         usecols=["frame_id", "method", "input_depth_condition"])
    source = source.loc[source.method.eq(real.INITIAL)]
    if source.duplicated(["frame_id", "input_depth_condition"]).any():
        raise ValueError("Duplicate recorded observation realisations")
    have = set(zip(source.frame_id, source.input_depth_condition))
    missing = [(f, c) for f in matched.frame_id for c in CONDITIONS if (f, c) not in have]
    if missing:
        raise ValueError(f"{len(missing)} matched target/depth conditions lack a recorded observation")

    # One shard owns every target (and every regime) of an image; longest images first.
    shards = 1 if args.smoke else int(args.shards)
    sizes = matched.groupby(["scene_id", "img_id"]).size()
    loads, owner = [0] * shards, {}
    for key in sorted(sizes.index, key=lambda k: (-int(sizes[k]), k)):
        j = loads.index(min(loads))
        owner[key] = j
        loads[j] += int(sizes[key])
    matched["shard"] = [owner[(s, i)] for s, i in zip(matched.scene_id, matched.img_id)]
    pieces = []
    for regime in REGIMES:
        piece = matched.copy()
        piece["init_regime"] = regime
        piece["cohort"] = f"{COHORT_PREFIX}_{regime}"
        piece["original_input_directory"] = str(SOURCE_RUN)
        pieces.append(piece)
    manifest = pd.concat(pieces, ignore_index=True)
    if manifest.duplicated(["cohort", "frame_id"]).any():
        raise ValueError("Duplicate cohort/frame identity")
    manifest.to_csv(out / "locked_external_instances.csv", index=False)

    cfg = dict(dataset=str(args.dataset), split=SPLIT, dataset_name=DATASET_NAME, pose_csv=str(args.pose_csv),
               mask_json=str(args.mask_json), toolkit=str(args.toolkit), source_run=str(SOURCE_RUN),
               same_bank=str(SAME_BANK), shards=shards, smoke=bool(args.smoke),
               model_sha256=model_meta["model_sha256"], created_at_utc=now())
    save(out / "run_config.json", cfg)

    snapshot = out / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    tracked = [Path(__file__).resolve(), WORK / "analyze_decoupled_frozen_external_ycbv.py",
               WORK / "run_decoupled_frozen_unseen5.py", WORK / "run_same_bank_residual_ablation.py",
               WORK / "evaluate_bop_metrics_controlled.py", WORK / "bop_aware_state_residual_plan_20261019.json",
               WORK / "bop_aware_selected_risk_plan_20261019.json", SAME_BANK / "frozen_development_policies.json"]
    for path in tracked:
        shutil.copy2(path, snapshot / path.name)
    hashed = tracked + [u5.DEV_OOF / "development_candidate.joblib", u5.DEV_OOF / "state.json",
                        args.pose_csv, args.mask_json, SOURCE_RUN / "frame_manifest.csv",
                        SOURCE_RUN / "real_inputs_validation_trials.csv"]
    hashes = {str(path): digest(path) for path in hashed}
    save(out / "input_hashes.json", hashes)

    protocol = dict(
        status="LOCKED_BEFORE_ANY_R4_OUTPUT_ON_YCBV",
        locked_at_utc=now(), smoke=bool(args.smoke),
        purpose="Independent external validation of the frozen r4 deployment artifact on a new dataset/sensor/initializer.",
        dataset=dict(name=DATASET_NAME, split=SPLIT, objects=sorted(int(o) for o in matched.object_id.unique()),
                     pool_targets=int(len(pool)), matched_targets=int(len(matched)),
                     unmatched_targets=int(len(pool) - EXPECTED_MATCHED), images=int(matched.groupby(["scene_id", "img_id"]).ngroups)),
        initializer="CosyPose ECCV'20 synt+real single-view public BOP submission, with its instance masks",
        regimes={k: dict(rotation_deg=v[0], translation_mm=v[1], seed=v[2]) for k, v in REGIMES.items()},
        depth_conditions=CONDITIONS, pose_protocol=DECOUPLED, candidate_actions=list(base.ACTIONS),
        deployment_model=dict(path=str(u5.DEV_OOF / "development_candidate.joblib"), sha256=model_meta["model_sha256"],
                              training_objects=model["bundle"]["training_objects"], policies=model["policies"]),
        fitting_on_ycbv="none", tuning_on_ycbv="none",
        primary_endpoint=("Object-equal BOP-style AR (mean of VSD/MSSD/MSPD recalls), D = conditional_residual_joint "
                          "minus the initializer, pooled over the 3 regimes x 2 depth conditions; 95% CI from a "
                          "21-object cluster bootstrap (20000 draws, seed 20261104)."),
        primary_decision_rule=dict(gain="CI lower bound > 0", harm="CI upper bound < 0",
                                   non_inferior=f"CI includes 0 and lower bound >= {-0.5} pp",
                                   inconclusive="CI includes 0 and lower bound < -0.5 pp"),
        secondary=["D minus each fixed trust-region candidate", "D minus the best fixed candidate (chosen post hoc on YCB-V, favouring the baseline)",
                   "D minus A (ordinary ranking, same bank and policy)", "update rate and accepted ADD-S regression rate",
                   "regime and depth strata", "per-gate pass rates (domain gate expected to matter)"],
        evaluation="BOP19 VSD/MSSD/MSPD with official thresholds; VSD visibility from native sensor depth for both depth conditions.",
        prior_use_of_ycbv="YCB-V was the external set of the earlier Protocol G method; r4 features, candidates and policies were fixed on T-LESS and no r4 result on YCB-V has been inspected.",
    )
    save(out / "locked_protocol.json", protocol)
    update_state(out, status="PREPARED", prepared_at_utc=now(), model_sha256=model_meta["model_sha256"],
                 development_status=dev_state["status"], targets=int(len(matched)), cases=int(len(manifest)),
                 shard_cases={int(k): int(v) for k, v in manifest.groupby("shard").size().items()})
    log(f"PREPARE locked: {len(matched)} matched targets x {len(REGIMES)} regimes = {len(manifest)} cases "
        f"({len(manifest) * len(CONDITIONS)} target conditions) in {shards} shard(s)")


# -------------------------------------------------------------------- generate
def generate_shard(out: Path, shard: int) -> None:
    cfg = load(out / "run_config.json")
    sdir = shard_dir(out, shard)
    if (sdir / "shard_complete.json").exists():
        return
    sdir.mkdir(parents=True, exist_ok=True)
    dataset = Path(cfg["dataset"])
    manifest = read_csv(out / "locked_external_instances.csv")
    part = manifest.loc[manifest.shard.astype(int).eq(int(shard))].copy()
    part["_regime_order"] = part.init_regime.map(REGIME_ORDER)
    part = part.sort_values(["object_id", "frame_id", "_regime_order"]).drop(columns="_regime_order")
    poses = read_csv(Path(cfg["pose_csv"]))
    masks = feature_support.masks_by_source_index(Path(cfg["mask_json"]), set(part.mask_source_index.astype(int)))
    objects = sorted(int(o) for o in part.object_id.unique())
    ns = real.support.load_experiment_namespace(objects, len(part), 1, [0.3], dataset_root=dataset,
                                                dataset_split=SPLIT, depth_extension=".png")
    recorded = read_csv(Path(cfg["source_run"]) / "real_inputs_validation_trials.csv")
    recorded = recorded.loc[recorded.method.eq(real.INITIAL) & recorded.frame_id.isin(set(part.frame_id))]
    old_rows = {(str(r.frame_id), str(r.input_depth_condition)): r for _, r in recorded.iterrows()}

    renderer = None
    points_cache: dict[int, np.ndarray] = {}
    model_hashes: dict[str, str] = {}
    total, done, started = len(part), 0, time.perf_counter()

    def progress() -> None:
        elapsed = time.perf_counter() - started
        save(sdir / "progress.json", dict(shard=int(shard), done=done, total=total, elapsed_s=elapsed,
                                          updated_at_utc=now()))

    for oid, group in part.groupby("object_id", sort=True):
        oid = int(oid)
        checkpoint = sdir / "objects" / f"object_{oid:02d}"
        if (checkpoint / "complete.json").exists():
            done += len(group)
            continue
        rows = {name: [] for name in TABLES}
        meta = ns["configure_object"](oid)
        for row in group.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            frame = ns["load_frame_data"](real.candidate_from_manifest(row_series, dataset, SPLIT))
            mask = real.decode_uncompressed_rle(masks[int(row.mask_source_index)])
            public = poses.iloc[int(row.initializer_source_index)]
            if tuple(map(int, [public.scene_id, public.im_id, public.obj_id])) != (int(row.scene_id), int(row.img_id), oid):
                raise ValueError("Public pose identity changed")
            rotation, translation = real.bop_pose_from_row(public)
            reference = real.internal_pose_from_bop(rotation, translation)
            angle, increment, seed = REGIMES[str(row.init_regime)]
            q, u, metadata = noise(angle, increment, oid, int(row.selected_index), seed)
            paired = paired_poses(reference, q, u)
            initial = paired[DECOUPLED]
            if not np.isclose(1000.0 * np.linalg.norm(initial[:3, 3] - reference[:3, 3]), increment, atol=1e-9):
                raise ValueError("Decoupled translation magnitude changed")
            rows["geometry"].append(dict(
                cohort=row.cohort, frame_id=row.frame_id, object_id=oid, init_regime=row.init_regime,
                selected_index=int(row.selected_index),
                actual_origin_shift_mm=float(1000.0 * np.linalg.norm(initial[:3, 3] - reference[:3, 3])),
                nominal_increment_mm=increment,
                original_camera_origin_shift_mm=float(1000.0 * np.linalg.norm(paired[CAMERA][:3, 3] - reference[:3, 3])),
                noise_rotation_json=json.dumps(q.tolist()), noise_translation_m_json=json.dumps(u.tolist()), **metadata,
            ))
            for condition in CONDITIONS:
                old_initial = old_rows[(str(row.frame_id), condition)]
                # Reuse only the recorded observation realisation; the pose is the new decoupled one.
                prepared, replay_status = development.replay(ns, frame, mask, old_initial, int(row.selected_index))
                hashes = {key: prepared["metadata"][key] for key in HASH_FIELDS}
                selection = dict(
                    pose=dict(source_index=int(row.initializer_source_index), score=float(public.score)),
                    mask=dict(source_index=int(row.mask_source_index), score=float(row.mask_score)),
                    bbox_iou=float(row.pose_mask_bbox_iou),
                )
                generated = real.run_methods(ns, frame, meta, initial, mask, prepared, row.frame_id, selection,
                                             int(row.selected_index))
                for record in generated:
                    record.update(
                        cohort=row.cohort, init_regime=row.init_regime, pose_protocol=DECOUPLED,
                        init_perturb_rotation_deg=angle, init_perturb_translation_mm=increment,
                        init_perturb_seed=seed, corruption_realization=0,
                    )
                rows["inputs"].extend(generated)
                records = {record["method"]: record for record in generated}
                initial_record = records[real.INITIAL]
                target, uv = ns["depth_to_cloud"](prepared["depth"], mask, frame["K"])
                reliability = ns["compute_depth_reliability"](prepared["depth"], mask, uv)
                source, _, _ = ns["generate_observation_supported_visible_model"](
                    ns["full_model_points"], initial, prepared["depth"], mask, frame["K"], target,
                )
                target_points = np.asarray(target.points, dtype=float)
                tree = cKDTree(target_points)
                gate = ns["compute_reliability_gate_features"](prepared["depth"], mask)
                candidates = [(base.INITIAL, initial, dict(INITIAL_DIAGNOSTICS))]
                for direction, trunk in [("p2p", real.TRUNK), ("filtered", real.PROPOSED)]:
                    full = development.pose_from_record(records[trunk])
                    for scale in trust.SCALES:
                        pose, diagnostics = trust.restricted_pose(initial, full, scale)
                        candidates.append((trust.method_name(direction, scale), pose, dict(direction=direction, **diagnostics)))
                observed = prepared["depth"] * 1000.0
                K = frame["K"]
                if renderer is None:
                    h, w = observed.shape
                    renderer = tail.Open3DRaycastRenderer(w, h, cache_size=24)
                if oid not in points_cache:
                    model_path = dataset / "models_eval" / f"obj_{oid:06d}.ply"
                    renderer.add_object(oid, str(model_path))
                    vertices = np.asarray(__import__("open3d").io.read_triangle_mesh(str(model_path)).vertices, dtype=float)
                    points_cache[oid] = vertices[::max(1, int(np.ceil(len(vertices) / 1000)))]
                    model_hashes[str(model_path)] = digest(model_path)
                rendered, absolute, current = {}, {}, []
                for method, pose, diagnostics in candidates:
                    record = dict(initial_record, method=method, termination_reason="se3_trust_region_candidate", runtime_ms=0.0)
                    real.support.update_pose_metrics(record, ns, pose, frame)
                    rows["trials"].append(record)
                    current.append(record)
                    start = time.perf_counter()
                    values = trust.geometry_scores(
                        ns, source, target_points, reliability, tree, pose,
                        prepared["depth"], mask, K, target,
                    )
                    rows["scores"].append(dict(
                        **{key: record[key] for key in base.KEYS}, cohort=row.cohort, method=method,
                        gate_triggered=int(gate["trigger"]), **prepared["metadata"], **diagnostics, **values,
                        candidate_score_ms=(time.perf_counter() - start) * 1000.0,
                    ))
                    rb, tb = tail.internal_pose_to_bop(pd.Series(record))
                    rendered[method] = renderer.render_object(
                        oid, rb, tb, K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                    )["depth"]
                    absolute[method] = tail.geometry_features(rendered[method], observed, mask, K)
                for record in current:
                    method = record["method"]
                    values = absolute[method]
                    entry = dict(
                        **{key: record[key] for key in base.KEYS}, cohort=row.cohort,
                        method=method, replay_status=replay_status, **prepared["metadata"],
                    )
                    for key, value in values.items():
                        entry["tail_" + key] = value
                        entry["tail_delta_" + key] = value - absolute[base.INITIAL][key]
                    entry.update(tail.common_support(rendered[base.INITIAL], rendered[method], observed, mask))
                    entry.update(tail.model_motion(
                        points_cache[oid], pd.Series(record), pd.Series(initial_record), K,
                        float(initial_record["diameter_mm"]), renderer.width,
                    ))
                    rows["tails"].append(entry)
                actual = observation_metadata(prepared, mask, K)
                if any(actual[key] != expected for key, expected in hashes.items()):
                    raise ValueError("Shared observation mutated")
                rows["observations"].append(dict(
                    cohort=row.cohort, frame_id=row.frame_id, object_id=oid,
                    input_depth_condition=condition, replay_status=replay_status,
                    **prepared["metadata"],
                ))
            done += 1
            if done % 5 == 0:
                progress()
        checkpoint.mkdir(parents=True, exist_ok=True)
        for name in TABLES:
            pd.DataFrame(rows[name]).to_csv(checkpoint / f"{name}.csv", index=False)
        save(checkpoint / "complete.json", dict(cases=len(group), rows={k: len(v) for k, v in rows.items()},
                                                completed_at_utc=now()))
        progress()
        log(f"shard {shard:02d}: object {oid:02d} complete ({done}/{total} cases)")

    tables = {}
    for name in TABLES:
        parts = [read_csv(d / f"{name}.csv") for d in sorted((sdir / "objects").glob("object_*"))]
        tables[name] = pd.concat(parts, ignore_index=True)
        tables[name].to_csv(sdir / f"{name}.csv", index=False)
    expected = total * len(CONDITIONS) * len(METHODS)
    for name in ["trials", "scores", "tails"]:
        if len(tables[name]) != expected:
            raise ValueError(f"shard {shard}: {name} has {len(tables[name])} rows, expected {expected}")
    save(sdir / "shard_complete.json", dict(shard=int(shard), cases=total, rows={k: len(v) for k, v in tables.items()},
                                           model_hashes=model_hashes, completed_at_utc=now()))
    log(f"shard {shard:02d}: COMPLETE {total} cases")


def stage_generate(args, out: Path) -> None:
    cfg = load(out / "run_config.json")
    todo = [i for i in range(int(cfg["shards"])) if not (shard_dir(out, i) / "shard_complete.json").exists()]
    if not todo:
        log("GENERATE already complete")
        return
    update_state(out, stage="generate", generate_started_at_utc=now())
    if int(cfg["shards"]) == 1:
        generate_shard(out, 0)
        return
    (out / "logs").mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def launch(i: int) -> int:
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--stage", "generate-shard",
               "--shard", str(i), "--output", str(out)]
        with (out / "logs" / f"generate_shard_{i:02d}.log").open("a", encoding="utf-8") as fh:
            return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(WORK)).returncode

    manifest = pd.read_csv(out / "locked_external_instances.csv", usecols=["shard"])
    total = len(manifest)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, min(int(args.workers), len(todo)))) as pool:
        futures = {pool.submit(launch, i): i for i in todo}
        pending = set(futures)
        while pending:
            finished, pending = wait(pending, timeout=60)
            done = 0
            for i in range(int(cfg["shards"])):
                d = shard_dir(out, i)
                if (d / "shard_complete.json").exists():
                    done += int(load(d / "shard_complete.json")["cases"])
                elif (d / "progress.json").exists():
                    done += int(load(d / "progress.json")["done"])
            elapsed = time.perf_counter() - started
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate / 60 if rate > 0 else float("nan")
            log(f"GENERATE {done}/{total} cases ({100 * done / total:.1f}%), elapsed {elapsed / 60:.1f} min, ETA {eta:.0f} min")
            for future in finished:
                if future.result() != 0:
                    log(f"shard {futures[future]:02d} FAILED; see logs/generate_shard_{futures[future]:02d}.log")
    failed = [i for i in range(int(cfg["shards"])) if not (shard_dir(out, i) / "shard_complete.json").exists()]
    if failed:
        raise RuntimeError(f"Generation incomplete for shards {failed}; rerun the same command to resume")
    update_state(out, generate_completed_at_utc=now(), generate_minutes=(time.perf_counter() - started) / 60)


# ----------------------------------------------------------------------- merge
def stage_merge(args, out: Path) -> None:
    if (out / "observable_candidate_features.csv").exists():
        log("MERGE already complete")
        return
    cfg = load(out / "run_config.json")
    manifest = read_csv(out / "locked_external_instances.csv")
    tables = {}
    for name in TABLES:
        tables[name] = pd.concat([read_csv(shard_dir(out, i) / f"{name}.csv") for i in range(int(cfg["shards"]))],
                                 ignore_index=True)
    expected = len(manifest) * len(CONDITIONS) * len(METHODS)
    for name in ["trials", "scores", "tails"]:
        if len(tables[name]) != expected:
            raise ValueError(f"{name}: {len(tables[name])} rows, expected {expected}")
        if tables[name].duplicated(["cohort", *base.KEYS, "method"]).any():
            raise ValueError(f"Duplicate {name} rows")
    if set(tables["trials"].method) != set(METHODS):
        raise ValueError("Candidate method set changed")
    features, names = u5.append_candidate_features({k: tables[k] for k in ["trials", "scores", "tails"]})
    if (len(features) != len(manifest) * len(CONDITIONS) * len(base.ACTIONS)
            or features.frame_id.nunique() != manifest.frame_id.nunique()):
        raise ValueError("Feature coverage mismatch")
    for name in ["inputs", "trials", "scores", "tails"]:
        tables[name].to_csv(out / f"{name}.csv", index=False)
    tables["observations"].to_csv(out / "observation_replay_checks.csv", index=False)
    tables["geometry"].to_csv(out / "initialization_geometry.csv", index=False)
    save(out / "feature_schema.json", {"feature_names": names, "observation_features_only": True,
                                       "csv_parser": "float_precision=round_trip"})
    features.to_csv(out / "observable_candidate_features.csv", index=False)
    replay = tables["observations"].replay_status.value_counts().to_dict()
    update_state(out, stage="merged", candidate_pose_rows=len(tables["trials"]), candidate_feature_rows=len(features),
                 replay_status=replay, merged_at_utc=now())
    log(f"MERGE: {len(tables['trials'])} candidate poses, {len(features)} feature rows, replay {replay}")


# ------------------------------------------------------------------------- bop
def stage_bop(args, out: Path) -> None:
    if (out / "external_bop_labels.csv").exists():
        log("BOP already complete")
        return
    from bop_thresholds import attach_image_widths, row_recalls

    cfg = load(out / "run_config.json")
    trials = read_csv(out / "trials.csv")
    root = out / "bop_parts"
    root.mkdir(exist_ok=True)
    plan_path = root / "part_plan.json"
    if not plan_path.exists():
        parts = 1 if cfg["smoke"] else int(args.bop_parts)
        sizes = trials.groupby(["scene_id", "img_id"]).size()
        loads, owner = [0] * parts, {}
        for key in sorted(sizes.index, key=lambda k: (-int(sizes[k]), k)):
            j = loads.index(min(loads))
            owner[key] = j
            loads[j] += int(sizes[key])
        part_of = np.array([owner[(s, i)] for s, i in zip(trials.scene_id, trials.img_id)])
        for j in range(parts):
            d = root / f"part_{j:02d}"
            d.mkdir(exist_ok=True)
            trials.loc[part_of == j].to_csv(d / "trials_part.csv", index=False)
        save(plan_path, dict(parts=parts, rows=[int(x) for x in loads], created_at_utc=now()))
    plan = load(plan_path)
    todo = [j for j in range(int(plan["parts"])) if not (root / f"part_{j:02d}" / "bop_pose_errors.csv").exists()]
    (out / "logs").mkdir(exist_ok=True)
    update_state(out, stage="bop", bop_started_at_utc=now())
    started = time.perf_counter()

    def launch(j: int) -> int:
        d = root / f"part_{j:02d}"
        cmd = [sys.executable, "-u", str(WORK / "evaluate_bop_metrics_controlled.py"),
               "--trials-csv", str(d / "trials_part.csv"), "--output-dir", str(d),
               "--dataset-root", cfg["dataset"], "--dataset-name", DATASET_NAME, "--dataset-split", SPLIT,
               "--toolkit-root", cfg["toolkit"], "--methods", ",".join(METHODS),
               "--extra-key-columns", "cohort,input_depth_condition", "--device", args.device,
               "--gpu-batch-size", str(args.gpu_batch_size), "--checkpoint-every", str(args.checkpoint_every),
               "--resume"]
        with (out / "logs" / f"bop_part_{j:02d}.log").open("a", encoding="utf-8") as fh:
            return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(WORK)).returncode

    if todo:
        log(f"BOP: evaluating {len(todo)} part(s) of {plan['parts']} ({sum(plan['rows'])} pose rows)")
        with ThreadPoolExecutor(max_workers=max(1, min(int(args.bop_workers), len(todo)))) as pool:
            futures = {pool.submit(launch, j): j for j in todo}
            pending = set(futures)
            while pending:
                finished, pending = wait(pending, timeout=120)
                done = []
                for j in range(int(plan["parts"])):
                    ck = root / f"part_{j:02d}" / "vsd_checkpoint.csv"
                    if (root / f"part_{j:02d}" / "bop_pose_errors.csv").exists():
                        done.append("done")
                    elif ck.exists():
                        done.append(f"vsd~{sum(1 for _ in ck.open(encoding='utf-8')) - 1}")
                    else:
                        done.append("mssd/mspd")
                log(f"BOP parts: {', '.join(done)}; elapsed {(time.perf_counter() - started) / 60:.1f} min")
                for future in finished:
                    if future.result() != 0:
                        log(f"BOP part {futures[future]:02d} FAILED; see logs/bop_part_{futures[future]:02d}.log")
    missing = [j for j in range(int(plan["parts"])) if not (root / f"part_{j:02d}" / "bop_pose_errors.csv").exists()]
    if missing:
        raise RuntimeError(f"BOP incomplete for parts {missing}; rerun the same command to resume")

    errors = pd.concat([read_csv(root / f"part_{j:02d}" / "bop_pose_errors.csv") for j in range(int(plan["parts"]))],
                       ignore_index=True)
    key = ["cohort", *base.KEYS, "method"]
    if len(errors) != len(trials) or errors.duplicated(key).any():
        raise ValueError("BOP rows do not match the candidate trials one-to-one")
    if set(map(tuple, errors[key].astype(str).values)) != set(map(tuple, trials[key].astype(str).values)):
        raise ValueError("BOP row identities differ from the candidate trials")
    labels = row_recalls(attach_image_widths(errors, Path(cfg["dataset"]), SPLIT))
    labels["evaluation_depth_policy"] = "native_sensor_depth"
    labels["pose_protocol"] = DECOUPLED
    (out / "bop").mkdir(exist_ok=True)
    errors.to_csv(out / "bop" / "bop_pose_errors.csv", index=False)
    features = read_csv(out / "observable_candidate_features.csv")
    attached = base.attach_labels(features, labels)
    if len(attached) != len(features):
        raise ValueError("Labels failed to attach")
    labels.to_csv(out / "external_bop_labels.csv", index=False)
    update_state(out, stage="labelled", labels_rows=len(labels), bop_minutes=(time.perf_counter() - started) / 60,
                 bop_completed_at_utc=now())
    log(f"BOP: {len(labels)} labelled pose rows")


# ----------------------------------------------------------------------- apply
def stage_apply(args, out: Path) -> None:
    targets = [out / "frozen_outcomes.csv", out / "same_bank_outcomes.csv", out / "candidate_gates.csv.gz"]
    if all(p.exists() for p in targets):
        log("APPLY already complete")
        return
    import run_same_bank_residual_ablation as sba  # sets BLAS threads to 1; harmless at this stage

    cfg = load(out / "run_config.json")
    started = time.perf_counter()
    dev_state, model, model_meta = u5.verify_frozen_development()
    if model_meta["model_sha256"] != cfg["model_sha256"]:
        raise ValueError("Deployment model changed after the protocol was locked")
    residual_plan = load(WORK / "bop_aware_state_residual_plan_20261019.json")
    winner_plan = load(WORK / "bop_aware_selected_risk_plan_20261019.json")
    manifest = read_csv(out / "locked_external_instances.csv")
    features = read_csv(out / "observable_candidate_features.csv")
    regimes = manifest[["cohort", "frame_id", "init_regime"]].drop_duplicates()
    features = features.merge(regimes, on=["cohort", "frame_id"], validate="many_to_one")
    data = base.attach_labels(features, read_csv(out / "external_bop_labels.csv")).reset_index(drop=True)
    if not data.groupby("target_id").action.agg(list).map(lambda x: sorted(x) == sorted(base.ACTIONS)).all():
        raise ValueError("Incomplete or duplicate action bank")
    predictions = residual.predict_bundle(model["bundle"], data)

    outcomes = []
    for variant in ["conditional_residual_legacy", "conditional_residual_joint"]:
        choices = residual.decisions(data, predictions, model["policies"][variant], variant, residual_plan)
        selected = residual.v2.evaluate_choices(data, choices)
        selected["method"] = variant
        outcomes.append(selected)
    for variant in ["stratified_only", "selected_risk_only", "selected_risk_stratified"]:
        pool = selected_risk.predict_selected_risk(
            model["selected_risk"], selected_risk.winner_pool(data, predictions, winner_plan),
        )
        selected = selected_risk.selected_frame(data, pool, model["policies"][variant]["policy_id"],
                                                model["policies"][variant], variant)
        selected["method"] = variant
        outcomes.append(selected)
    for method in METHODS:
        baseline = base.baseline_rows(data, method)
        baseline["method"] = method
        outcomes.append(baseline)
    outcome_table = pd.concat(outcomes, ignore_index=True)
    ids = data[["target_id", "init_regime"]].drop_duplicates("target_id")
    initial_columns = data.drop_duplicates("target_id")[["target_id", *["initial_" + c for c in base.AR]]]
    outcome_table = outcome_table.merge(ids, on="target_id", validate="many_to_one")
    outcome_table = outcome_table.merge(initial_columns, on="target_id", validate="many_to_one")
    outcome_table.to_csv(out / "frozen_outcomes.csv", index=False)
    decoupled_oof.planned_summary(outcome_table, manifest).to_csv(out / "frozen_summary.csv", index=False)

    common = model["policies"]["conditional_residual_joint"]
    checks = sba.verify_replay(data, predictions, model["bundle"], residual_plan, outcome_table, common)
    nested = load(SAME_BANK / "frozen_development_policies.json")
    arms = sba.produce(data, predictions, model["bundle"], {a: common for a in sba.ARMS}, residual_plan, "ycbv21", "shared_r4")
    arms += sba.produce(data, predictions, model["bundle"], nested, residual_plan, "ycbv21", "nested_equal_budget")
    pd.concat(arms, ignore_index=True).to_csv(out / "same_bank_outcomes.csv", index=False)

    gates, utility, branch = residual.gate_arrays(data, predictions, common, "conditional_residual_joint", residual_plan)
    gate_table = data[["target_id", "cohort", "init_regime", "object_id", "frame_id", "input_depth_condition", "action"]].copy()
    for name, values in gates.items():
        gate_table["gate_" + name] = np.asarray(values, dtype=bool)
    gate_table["joint_utility"] = utility
    gate_table["branch"] = branch
    numeric = predictions.select_dtypes("number")
    for column in numeric.columns:
        gate_table["p_" + column] = numeric[column].to_numpy()
    gate_table["true_delta_AR_mean"] = data[list(base.TARGETS[:3])].mean(axis=1).to_numpy()
    gate_table["true_delta_ADD_S_mm"] = data.delta_ADD_S_mm.to_numpy()
    gate_table.to_csv(out / "candidate_gates.csv.gz", index=False, compression="gzip")

    save(out / "frozen_model_receipt.json", dict(
        model_path=model_meta["model_path"], model_sha256=model_meta["model_sha256"],
        training_objects=model["bundle"]["training_objects"], external_dataset=DATASET_NAME,
        policies=model["policies"], same_bank_nested_policies=nested, policy_tuning="none_on_ycbv",
        residual_plan_sha256=digest(WORK / "bop_aware_state_residual_plan_20261019.json"),
        winner_plan_sha256=digest(WORK / "bop_aware_selected_risk_plan_20261019.json"),
        replay_checks=checks))
    update_state(out, stage="applied", apply_minutes=(time.perf_counter() - started) / 60, applied_at_utc=now(),
                 method_promoted=False)
    log(f"APPLY: frozen policies and same-bank arms evaluated on {data.target_id.nunique()} target conditions")


# --------------------------------------------------------------------- analyze
def stage_analyze(args, out: Path) -> None:
    import analyze_decoupled_frozen_external_ycbv as analysis

    checks = analysis.analyze(out)
    state = update_state(out, status="COMPLETE_FROZEN_EXTERNAL_YCBV_VALIDATION", completed_at_utc=now(),
                         primary_endpoint=checks["primary_endpoint"],
                         primary_classification=checks["primary_classification"])
    state["artifact_hashes"] = {str(p.relative_to(out)): digest(p) for p in (out / "analysis").rglob("*") if p.is_file()}
    save(out / "state.json", state)
    print((out / "analysis" / "report_zh.md").read_text(encoding="utf-8"), flush=True)


STAGES = {"prepare": stage_prepare, "generate": stage_generate, "merge": stage_merge, "bop": stage_bop,
          "apply": stage_apply, "analyze": stage_analyze}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=[*STAGES, "all", "generate-shard"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="2 objects x 2 targets x 3 regimes, end to end")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pose-csv", type=Path, default=DEFAULT_POSE_CSV)
    parser.add_argument("--mask-json", type=Path, default=DEFAULT_MASK_JSON)
    parser.add_argument("--toolkit", type=Path, default=DEFAULT_TOOLKIT)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8, help="concurrent generation processes")
    parser.add_argument("--threads", type=int, default=2, help="OMP threads per generation process")
    parser.add_argument("--shard", type=int)
    parser.add_argument("--bop-parts", type=int, default=4)
    parser.add_argument("--bop-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--gpu-batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    args = parser.parse_args()
    out = args.output or (SMOKE_OUTPUT if args.smoke else DEFAULT_OUTPUT)

    if args.stage == "generate-shard":
        generate_shard(out, int(args.shard))
        return
    order = list(STAGES) if args.stage == "all" else [args.stage]
    if order[0] != "prepare" and not (out / "locked_protocol.json").exists():
        raise SystemExit(f"Run --stage prepare first (no locked protocol in {out})")
    t0 = time.perf_counter()
    for name in order:
        t = time.perf_counter()
        log(f"=== {name} ===")
        try:
            STAGES[name](args, out)
        except Exception:
            if out.exists():
                update_state(out, status=f"FAILED_AT_{name.upper()}_RESUMABLE", error=traceback.format_exc(),
                             failed_at_utc=now())
            raise
        log(f"=== {name} done in {(time.perf_counter() - t) / 60:.1f} min ===")
    log(f"FROZEN EXTERNAL YCB-V {'SMOKE ' if args.smoke else ''}RUN COMPLETE in {(time.perf_counter() - t0) / 60:.1f} min: {out}")


if __name__ == "__main__":
    main()
