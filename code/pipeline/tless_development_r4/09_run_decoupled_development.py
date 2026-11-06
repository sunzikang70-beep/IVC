"""Freeze and regenerate the development bank under the explicit model-origin pivot."""
from __future__ import annotations

import os
for _key, _value in {"OMP_NUM_THREADS":"4", "OPENBLAS_NUM_THREADS":"1", "MKL_NUM_THREADS":"1"}.items():
    os.environ[_key] = _value

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import prepare_winner_instance_expansion as p
import run_frozen_real_inputs_validation as real
import run_se3_trust_region_geometric_development as trust
import extract_bop_aware_tail_features as tail
import train_bop_aware_group_oof as base
from pose_perturbation_protocols import CAMERA, DECOUPLED, noise, paired_poses
from protocolG_observation import HASH_FIELDS, replay_method_depth, prepare_depth, observation_metadata
from extract_real_gtfree_acceptance_features import masks_by_source_index
from winner_training_data_io import read_csv
from audit_initialization_perturbation_protocol import verify_prior_delivery
from decoupled_checkpoint_recovery import INITIAL_DIAGNOSTICS, recover

ROOT = Path(r"D:\6D_pose\codex_paper_workspace")
SPECS = [
    ("legacy_native", "native", "gtfree_real_acceptance_development_inputs_20260915", None),
    ("legacy_mild", "mild", "gtfree_unseen_full_inputs_20261016_mild", None),
    ("legacy_hard", "hard", "gtfree_unseen_full_inputs_20261016_hard", None),
    *[("expanded_"+regime, regime, "winner_instance_expansion_20261020_r1/inputs", regime) for regime in p.REGIMES],
]
METHODS = [base.INITIAL, *base.ACTIONS]


def pose_from_record(record):
    return real.support.transform_from_row(pd.Series(record))


def now():
    return datetime.now(timezone.utc).isoformat()


def check_boundaries(manifest, reserved):
    if not set(manifest.object_id).issubset(p.OBJECTS):
        raise ValueError("Nondevelopment object in locked data")
    if not set(manifest.init_regime).issubset(p.REGIMES):
        raise ValueError("Unknown initialization regime")
    active = manifest.loc[manifest.matched_prediction.eq(1)] if "matched_prediction" in manifest else manifest
    images = set(map(tuple, active[["scene_id", "img_id"]].astype(int).to_numpy()))
    embargo = set(map(tuple, reserved[["scene_id", "img_id"]].astype(int).to_numpy()))
    if images & embargo:
        raise ValueError("Reserved RGB-D images entered development")
    if manifest.duplicated(["cohort", "frame_id"]).any():
        raise ValueError("Repeated case inside cohort")


def load_sources(root):
    frames, sources = [], []
    for cohort, regime, directory, subset in SPECS:
        folder = root/"outputs"/directory
        path = folder/"frame_manifest.csv"
        frame = read_csv(path)
        if subset is not None:
            frame = frame.loc[frame.init_regime.eq(subset)].copy()
        frame["cohort"] = "decoupled_v1_"+cohort
        frame["init_regime"] = regime
        frame["original_input_directory"] = str(folder)
        frames.append(frame)
        sources.extend([path, folder/"real_inputs_validation_trials.csv"])
    manifest = pd.concat(frames, ignore_index=True)
    reserve = root/"outputs/winner_instance_plan_20261020_r1/reserved_instances_do_not_evaluate.csv"
    check_boundaries(manifest, read_csv(reserve))
    if len(manifest) != 867 or manifest.loc[manifest.matched_prediction.eq(1), "frame_id"].nunique() != 629:
        raise ValueError("Frozen full-development identity coverage changed")
    return manifest, list(dict.fromkeys([*sources, reserve]))


def run_command(output, name, command, state, timeout=10800):
    print("STAGE", name, flush=True)
    state.update(stage=name, stage_started_at_utc=now())
    state.setdefault("commands", []).append(command)
    p.save(output/"state.json", state)
    path = output/(name+".log")
    with path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, PYTHONUNBUFFERED="1"))
        state["child_pid"] = process.pid
        p.save(output/"state.json", state)
        start = time.monotonic()
        offset = 0
        while process.poll() is None:
            time.sleep(5)
            with path.open(encoding="utf-8", errors="replace") as source:
                source.seek(offset)
                content = source.read()
                offset = source.tell()
            if content:
                print(content, end="", flush=True)
            state["last_monitor_at_utc"] = now()
            state["stage_elapsed_seconds"] = time.monotonic()-start
            p.save(output/"state.json", state)
            if time.monotonic()-start > timeout:
                print("HARD TIMEOUT: stopping only this experiment child", flush=True)
                process.kill()
                process.wait()
                raise TimeoutError(name+" exceeded locked hard timeout")
        with path.open(encoding="utf-8", errors="replace") as source:
            source.seek(offset)
            print(source.read(), end="", flush=True)
        if process.returncode:
            raise RuntimeError(f"{name} failed ({process.returncode}); no automatic retry")


def freeze(args):
    if args.output.exists():
        raise FileExistsError("Never overwrite a locked experiment")
    if verify_prior_delivery(args.root/"outputs"):
        raise ValueError("Prior delivery hashes changed")
    smoke = args.root/"outputs/paired_pivot_smoke_20261020_r1"
    receipt = json.loads((smoke/"state.json").read_text())
    if receipt["status"] != "SMOKE_COMPLETE_NOT_PROMOTED":
        raise ValueError("Paired pivot smoke did not complete")
    for relative, expected in receipt["artifact_hashes"].items():
        if p.digest(smoke/relative) != expected:
            raise ValueError("Paired smoke artifact changed: "+relative)
    manifest, sources = load_sources(args.root)
    work = Path(__file__).parent
    sources += [work/name for name in [
        "run_decoupled_development.py", "train_decoupled_group_oof.py", "test_decoupled_development.py", "decoupled_checkpoint_recovery.py",
        "pose_perturbation_protocols.py", "winner_training_data_io.py", "protocolG_observation.py",
        "run_frozen_real_inputs_validation.py", "run_frozen_unseen_validation.py",
        "run_p2p_reliability_filter_development.py", "run_se3_trust_region_geometric_development.py",
        "extract_bop_aware_tail_features.py", "extract_protocolG_render_compare_features.py",
        "train_bop_aware_group_oof.py", "train_bop_aware_tail_risk_v2.py",
        "train_bop_aware_selective_calibration.py", "train_bop_aware_state_residual.py",
        "train_bop_aware_selected_risk.py", "evaluate_bop_metrics_controlled.py", "bop_thresholds.py",
        "bop_aware_tail_risk_v2_plan_20261019.json", "bop_aware_state_residual_plan_20261019.json",
        "bop_aware_selected_risk_plan_20261019.json",
    ]]
    sources += [real.POSE_CSV, real.MASK_JSON, real.CONFIG, real.support.SOURCE, smoke/"state.json"]
    if args.reuse_checkpoints is not None:
        sources += [args.reuse_checkpoints/"state.json", args.reuse_checkpoints/"locked_protocol.json",
                    args.reuse_checkpoints/"evaluation_sources_pre_BOP.json"]
    args.output.mkdir(parents=True)
    snapshot = args.output/"source_snapshot"
    snapshot.mkdir()
    for i, source in enumerate(sources):
        if source.suffix in {".py", ".json"} and source not in [real.MASK_JSON]:
            shutil.copy2(source, snapshot/f"{i:02d}_{source.name}")
    manifest.to_csv(args.output/"locked_development_instances.csv", index=False)
    split_data = manifest.loc[manifest.matched_prediction.eq(1)].reset_index(drop=True)
    splits = []
    for fold, (a, b) in enumerate(base.folds(split_data, 5)):
        splits.append(dict(fold=fold, training_objects=sorted(map(int, split_data.iloc[a].object_id.unique())),
                           validation_objects=sorted(map(int, split_data.iloc[b].object_id.unique()))))
    plan = dict(status="LOCKED_BEFORE_NEW_CANDIDATES_AND_LABELS", protocol=DECOUPLED,
        equation="R_prime=Q@R; t_prime=t+u; Q/u in existing internal camera basis; translations meters internally",
        rotation_pivot="model-coordinate origin, not scene camera origin or newly estimated centroid",
        regimes=p.REGIMES, coupling="original identity/regime/selected_index/namespace_seed and Q/u unchanged",
        observation="reuse original public pose and predicted mask indexes and replay original method depth; no reassociation",
        candidates=dict(methods=METHODS, scales=trust.SCALES, max_rotation_deg=trust.MAX_ROTATION_DEG,
                        max_translation_mm=trust.MAX_TRANSLATION_MM, unchanged_trunks=True),
        training_objects=p.OBJECTS, excluded_objects=p.EXCLUDED, planned_cases=len(manifest),
        matched_cases=int(manifest.matched_prediction.sum()), matched_physical_instances=629,
        conditions=real.INPUT_CONDITIONS, expected_bop_labels=int(manifest.matched_prediction.sum())*2*7,
        label_budget="all seven poses in every matched condition; no old/new model winner filtering",
        evaluation_depth="native_sensor_depth; original BOP19 120 thresholds and image-width-normalized MSPD",
        feature_schema="existing 191 observable geometry/state/tail features; round-trip CSV; inf normalized to missing",
        outer_folds=splits, inner_folds=3, winner_calibration_deep_folds=3,
        models="retrain existing raw heads/selective calibration/state residual guards and selected-risk models from scratch",
        primary_method="selected_risk_stratified", controls=["conditional_residual_legacy", "conditional_residual_joint", "stratified_only", "selected_risk_only"],
        tuning="original policy grids and constraints; training folds only; disabled if none feasible; no outer retuning",
        scope="inspected development only; not independent confirmation; not official benchmark leaderboard",
        missing="all planned cases retained without replacement; matched and planned-denominator AR reported",
        preserve="old outputs and reserved88 unchanged; no cloud/real-platform/excluded-object inference",
        ledger_overlap="three original missing-pose ledger rows share later-reserved images; denominator only, no observation loading/inference",
        timeout_seconds_per_child_stage=10800, created_at_utc=now())
    state = dict(status="RUNNING", stage="locked", started_at_utc=now(), pid=os.getpid(),
                 method_promoted=False, new_final_holdout=False,
                 source_hashes={str(path):p.digest(path) for path in dict.fromkeys(sources)},
                 environment={"python":sys.version, **{name:version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "torch", "open3d"]}})
    plan["manifest_sha256"] = p.digest(args.output/"locked_development_instances.csv")
    plan["source_hashes"] = state["source_hashes"]
    p.save(args.output/"locked_protocol.json", plan)
    p.save(args.output/"state.json", state)
    return manifest, plan, state


def replay(ns, frame, mask, row, selected_index):
    if "observation_version" in row.index and pd.notna(row.observation_version):
        return replay_method_depth(ns, frame, mask, row, selected_index), "RECORDED_V3_HASHES_VERIFIED"
    condition = row.input_depth_condition
    seeds = {} if condition == "native_depth" else dict(patch_seed=int(row.patch_seed), value_seed=int(row.value_seed))
    prepared = prepare_depth(ns, frame, mask, int(row.object_id), selected_index,
                             condition, float(row.input_outlier_fraction), **seeds)
    if (prepared["center_u"] != int(row.patch_center_u) or prepared["center_v"] != int(row.patch_center_v)
            or int(prepared["patch"].sum()) != int(row.patch_pixels)):
        raise ValueError("Legacy patch geometry changed")
    return prepared, "LEGACY_SEEDS_PATCH_VERIFIED_NO_RETROSPECTIVE_HASH_CLAIM"


def generate(args, manifest, state):
    matched = manifest.loc[manifest.matched_prediction.eq(1)].copy()
    poses = read_csv(real.POSE_CSV)
    masks = masks_by_source_index(real.MASK_JSON, set(matched.mask_source_index.astype(int)))
    ns = real.support.load_experiment_namespace(p.OBJECTS, len(manifest), 1, [.3], dataset_root=args.dataset)
    inputs, trials, scores, tails, observations, geometry = [], [], [], [], [], []
    reused = set()
    if args.reuse_checkpoints is not None:
        tables, geometry, observations, reused = recover(args.reuse_checkpoints, args.output, manifest, poses)
        inputs, trials, scores, tails = (tables[name] for name in ["inputs", "trials", "scores", "tails"])
        state["approved_checkpoint_reuse"] = json.loads((args.output/"approved_checkpoint_reuse.json").read_text())
        state["completed_cases"] = len(reused)
        print("REUSING", len(reused), "cases; generating", len(matched)-len(reused), "remaining", flush=True)
    raw_tables = {name:read_csv(Path(name)/"real_inputs_validation_trials.csv") for name in matched.original_input_directory.unique()}
    renderer = None
    points_cache = {}
    import open3d as o3d
    state["stage"] = "regenerating_inputs_candidates_and_tails"
    p.save(args.output/"state.json", state)
    for oid, group in matched.groupby("object_id", sort=True):
        meta = ns["configure_object"](int(oid))
        for row in group.itertuples(index=False):
            if (row.cohort, row.frame_id) in reused:
                continue
            cohort = row.cohort
            frame = ns["load_frame_data"](real.candidate_from_manifest(row, args.dataset, "test_primesense"))
            mask = real.decode_uncompressed_rle(masks[int(row.mask_source_index)])
            public = poses.iloc[int(row.initializer_source_index)]
            if tuple(map(int, [public.scene_id, public.im_id, public.obj_id])) != (int(row.scene_id), int(row.img_id), int(oid)):
                raise ValueError("Cached public pose identity changed")
            R, t = real.bop_pose_from_row(public)
            reference = real.internal_pose_from_bop(R, t)
            angle, increment, seed = p.REGIMES[row.init_regime]
            q, u, metadata = noise(angle, increment, int(oid), int(row.selected_index), seed)
            paired = paired_poses(reference, q, u)
            initial = paired[DECOUPLED]
            shift = float(1000*np.linalg.norm(initial[:3, 3]-reference[:3, 3]))
            if not np.isclose(shift, increment, atol=1e-9):
                raise ValueError("Decoupled translation magnitude changed")
            geometry.append(dict(cohort=cohort, frame_id=row.frame_id, object_id=int(oid), init_regime=row.init_regime,
                selected_index=int(row.selected_index), actual_origin_shift_mm=shift, nominal_increment_mm=increment,
                original_camera_origin_shift_mm=float(1000*np.linalg.norm(paired[CAMERA][:3,3]-reference[:3,3])),
                noise_rotation_json=json.dumps(q.tolist()), noise_translation_m_json=json.dumps(u.tolist()), **metadata))
            selection = dict(pose=dict(source_index=int(row.initializer_source_index), score=float(public.score)),
                mask=dict(source_index=int(row.mask_source_index), score=float(row.mask_score)), bbox_iou=float(row.pose_mask_bbox_iou))
            old = raw_tables[row.original_input_directory]
            for condition, _ in real.INPUT_CONDITIONS:
                old_group = old.loc[old.frame_id.eq(row.frame_id)&old.input_depth_condition.eq(condition)]
                old_initial = old_group.loc[old_group.method.eq(real.INITIAL)].iloc[0]
                if not np.allclose(real.support.transform_from_row(old_initial), paired[CAMERA], rtol=1e-10, atol=1e-10):
                    raise ValueError("Q/u does not reproduce original input")
                prepared, replay_status = replay(ns, frame, mask, old_initial, int(row.selected_index))
                hashes = {key:prepared["metadata"][key] for key in HASH_FIELDS}
                generated = real.run_methods(ns, frame, meta, initial, mask, prepared, row.frame_id, selection, int(row.selected_index))
                for record in generated:
                    record.update({key:old_initial[key] for key in ["mask_pixels", "gt_mask_pixels", "mask_iou", "mask_precision", "mask_recall", "mask_area_ratio"] if key in old_initial})
                    record.update(cohort=cohort, init_regime=row.init_regime, pose_protocol=DECOUPLED,
                        init_perturb_rotation_deg=angle, init_perturb_translation_mm=increment, init_perturb_seed=seed, corruption_realization=0)
                inputs.extend(generated)
                records = {record["method"]:record for record in generated}
                initial_record = records[real.INITIAL]
                target, uv = ns["depth_to_cloud"](prepared["depth"], mask, frame["K"])
                reliability = ns["compute_depth_reliability"](prepared["depth"], mask, uv)
                source, _, _ = ns["generate_observation_supported_visible_model"](ns["full_model_points"], initial, prepared["depth"], mask, frame["K"], target)
                target_points = np.asarray(target.points, dtype=float)
                tree = cKDTree(target_points)
                gate = ns["compute_reliability_gate_features"](prepared["depth"], mask)
                candidates = [(base.INITIAL, initial, dict(INITIAL_DIAGNOSTICS))]
                for direction, trunk in [("p2p", real.TRUNK), ("filtered", real.PROPOSED)]:
                    full = pose_from_record(records[trunk])
                    for scale in trust.SCALES:
                        pose, diagnostics = trust.restricted_pose(initial, full, scale)
                        candidates.append((trust.method_name(direction, scale), pose, dict(direction=direction, **diagnostics)))
                observed = prepared["depth"]*1000.
                K = frame["K"]
                if renderer is None:
                    h, w = observed.shape
                    renderer = tail.Open3DRaycastRenderer(w, h, cache_size=16)
                if int(oid) not in points_cache:
                    path = args.dataset/"models_eval"/f"obj_{int(oid):06d}.ply"
                    renderer.add_object(int(oid), str(path))
                    vertices = np.asarray(o3d.io.read_triangle_mesh(str(path)).vertices, dtype=float)
                    points_cache[int(oid)] = vertices[::max(1, int(np.ceil(len(vertices)/1000)))]
                    state.setdefault("dataset_hashes", {})[str(path)] = p.digest(path)
                rendered, absolute, current = {}, {}, []
                for method, pose, diagnostics in candidates:
                    record = dict(initial_record, method=method, termination_reason="se3_trust_region_candidate", runtime_ms=0.)
                    real.support.update_pose_metrics(record, ns, pose, frame)
                    trials.append(record)
                    current.append(record)
                    start = time.perf_counter()
                    values = trust.geometry_scores(ns, source, target_points, reliability, tree, pose,
                                                   prepared["depth"], mask, K, target)
                    scores.append(dict(**{key:record[key] for key in base.KEYS}, cohort=cohort, method=method,
                        gate_triggered=int(gate["trigger"]), **prepared["metadata"], **diagnostics, **values,
                        candidate_score_ms=(time.perf_counter()-start)*1000.))
                    rb, tb = tail.internal_pose_to_bop(pd.Series(record))
                    rendered[method] = renderer.render_object(int(oid), rb, tb, K[0,0], K[1,1], K[0,2], K[1,2])["depth"]
                    absolute[method] = tail.geometry_features(rendered[method], observed, mask, K)
                for record in current:
                    method = record["method"]
                    values = absolute[method]
                    entry = dict(**{key:record[key] for key in base.KEYS}, cohort=cohort, method=method,
                                 replay_status=replay_status, **prepared["metadata"])
                    for key, value in values.items():
                        entry["tail_"+key] = value
                        entry["tail_delta_"+key] = value-absolute[base.INITIAL][key]
                    entry.update(tail.common_support(rendered[base.INITIAL], rendered[method], observed, mask))
                    entry.update(tail.model_motion(points_cache[int(oid)], pd.Series(record), pd.Series(initial_record), K,
                                                  float(initial_record["diameter_mm"]), renderer.width))
                    tails.append(entry)
                actual = observation_metadata(prepared, mask, K)
                if any(actual[key] != expected for key, expected in hashes.items()):
                    raise ValueError("Shared observation mutated")
                observations.append(dict(cohort=cohort, frame_id=row.frame_id, object_id=int(oid),
                    input_depth_condition=condition, replay_status=replay_status, **prepared["metadata"]))
            state["completed_cases"] = len(geometry)
            if len(geometry)%10 == 0:
                p.save(args.output/"state.json", state)
                print(f"DECOUPLED BANK {len(geometry)}/{len(matched)} cases", flush=True)
        checkpoint = args.output/"generation_checkpoints"/f"object_{int(oid):02d}"
        checkpoint.mkdir(parents=True)
        for name, rows in [("inputs", inputs), ("trials", trials), ("scores", scores), ("tails", tails)]:
            pd.DataFrame([record for record in rows if int(record["object_id"]) == int(oid)]).to_csv(checkpoint/(name+".csv"), index=False)
        p.save(args.output/"state.json", state)
        print("OBJECT COMPLETE", int(oid), len(geometry), "cases", flush=True)
    tables = {name:pd.DataFrame(rows) for name, rows in [("inputs", inputs), ("trials", trials), ("scores", scores), ("tails", tails)]}
    expected = len(matched)*2*7
    for name in ["trials", "scores", "tails"]:
        if len(tables[name]) != expected or tables[name].duplicated(["cohort", *base.KEYS, "method"]).any():
            raise ValueError("Incomplete/duplicate regenerated bank: "+name)
    all_features, names = [], None
    metadata_names = {"cohort", *base.KEYS, "method", "replay_status", *prepared["metadata"]}
    tail_names = [key for key in tables["tails"] if key not in metadata_names]
    for cohort, group in tables["trials"].groupby("cohort", sort=True):
        features, raw_names = base.build_features(tables["scores"].loc[tables["scores"].cohort.eq(cohort)], group, cohort)
        shadow = group.copy()
        for key in ["ADD_S_mm", "mask_iou", "rotation_error_deg", "translation_error_mm"]:
            shadow[key] = 99999.
        replayed, _ = base.build_features(tables["scores"].loc[tables["scores"].cohort.eq(cohort)], shadow, cohort)
        np.testing.assert_allclose(features[raw_names], replayed[raw_names], equal_nan=True)
        keys = ["cohort", *base.KEYS, "action"]
        features = features.merge(tables["tails"].rename(columns={"method":"action"})[keys+tail_names], on=keys, validate="one_to_one")
        feature_names = raw_names+tail_names
        if names is not None and names != feature_names:
            raise ValueError("Feature schema differs between cohorts")
        names = feature_names
        features[names] = features[names].replace([np.inf, -np.inf], np.nan)
        all_features.append(features)
    features = pd.concat(all_features, ignore_index=True)
    features["target_id"] = features.groupby(["cohort", *base.KEYS], sort=False, dropna=False).ngroup()
    features["candidate_index"] = np.arange(len(features))
    if len(names) != 191 or features.frame_id.nunique() != 629:
        raise ValueError("Frozen feature/physical-instance schema changed")
    for name, table in tables.items():
        table.to_csv(args.output/(name+".csv"), index=False)
    features.to_csv(args.output/"observable_candidate_features.csv", index=False)
    pd.DataFrame(observations).to_csv(args.output/"observation_replay_checks.csv", index=False)
    pd.DataFrame(geometry).to_csv(args.output/"initialization_geometry.csv", index=False)
    p.save(args.output/"feature_schema.json", dict(feature_names=names, tail_names=tail_names,
        observation_features_only=True, label_blind_base_replay=True, csv_parser="float_precision=round_trip"))
    state.update(candidate_pose_rows=expected, candidate_feature_rows=len(features), matched_cases=len(matched),
                 matched_physical_instances=features.frame_id.nunique())
    p.save(args.output/"state.json", state)


def finalize_bop(args):
    from bop_thresholds import row_recalls, attach_image_widths
    labels = row_recalls(attach_image_widths(read_csv(args.output/"bop/bop_pose_errors.csv"), args.dataset, "test_primesense"))
    features = read_csv(args.output/"observable_candidate_features.csv")
    trials = read_csv(args.output/"trials.csv")
    keys = ["cohort", *base.KEYS, "method"]
    if len(labels) != len(trials) or labels.duplicated(keys).any():
        raise ValueError("BOP coverage differs from locked bank")
    merged = labels.merge(trials[keys+["ADD_S_mm"]], on=keys, suffixes=("", "_trial"), validate="one_to_one")
    np.testing.assert_allclose(merged.ADD_S_mm, merged.ADD_S_mm_trial, rtol=1e-7, atol=1e-7)
    labels["evaluation_depth_policy"] = "native_sensor_depth"
    labels["pose_protocol"] = DECOUPLED
    labels.to_csv(args.output/"development_bop_labels.csv", index=False)
    data = base.attach_labels(features, labels)
    if len(data) != len(features) or data.object_id.nunique() != 22:
        raise ValueError("Development labels failed to attach")
    data.to_csv(args.output/"labeled_candidate_data.csv", index=False)
    print("ALL CANDIDATE LABELS COMPLETE", len(labels), "poses", len(data), "candidate rows", flush=True)


def run(args):
    manifest, plan, state = freeze(args)
    try:
        run_command(args.output, "tests", [sys.executable, "-m", "unittest", "-v", "test_decoupled_development",
            "test_paired_pivot_smoke", "test_initialization_perturbation_protocol", "test_winner_instance_expansion"], state)
        generate(args, manifest, state)
        work = Path(__file__).parent
        run_command(args.output, "bop", [sys.executable, "-u", str(work/"evaluate_bop_metrics_controlled.py"),
            "--trials-csv", str(args.output/"trials.csv"), "--output-dir", str(args.output/"bop"),
            "--dataset-root", str(args.dataset), "--toolkit-root", str(args.toolkit), "--methods", ",".join(METHODS),
            "--extra-key-columns", "cohort,input_depth_condition", "--device", "cuda", "--gpu-batch-size", "1", "--checkpoint-every", "50"], state)
        finalize_bop(args)
        state["stage"] = "labels_complete"
        state["labels_sha256"] = p.digest(args.output/"development_bop_labels.csv")
        p.save(args.output/"state.json", state)
        run_command(args.output, "group_oof", [sys.executable, "-u", str(work/"train_decoupled_group_oof.py"),
                    "--data", str(args.output), "--output", str(args.output/"group_oof")], state)
        if verify_prior_delivery(args.root/"outputs"):
            raise ValueError("Prior delivery changed")
        for path, expected in {**state["source_hashes"], **state.get("dataset_hashes", {})}.items():
            if p.digest(Path(path)) != expected:
                raise ValueError("Frozen source changed: "+path)
        state.update(status="COMPLETE_DECOUPLED_DEVELOPMENT_NOT_PROMOTED", completed_at_utc=now())
        state["artifact_hashes"] = {str(path.relative_to(args.output)):p.digest(path) for path in args.output.rglob("*")
            if path.is_file() and path != args.output/"state.json"}
        p.save(args.output/"state.json", state)
        print("DECOUPLED DEVELOPMENT COMPLETE; not promoted", flush=True)
    except Exception:
        state.update(status="FAILED_PRESERVED_NO_AUTOMATIC_RETRY", error=traceback.format_exc())
        p.save(args.output/"state.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/decoupled_development_20261020_r1")
    parser.add_argument("--dataset", type=Path, default=real.DATASET_ROOT)
    parser.add_argument("--toolkit", type=Path, default=Path(r"D:\6D_pose\tools\bop_toolkit"))
    parser.add_argument("--reuse-checkpoints", type=Path, help="Explicitly author-approved stopped pre-BOP object checkpoints only")
    run(parser.parse_args())
