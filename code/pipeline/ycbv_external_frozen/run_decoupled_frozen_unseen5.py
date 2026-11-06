"""Run the frozen decoupled candidate bank on five untouched T-LESS objects.

This script deliberately uses the deployment artifact produced from the 22-object
development bank. It never fits, tunes, or selects a policy with the five held-out
objects. Labels are requested only after the candidate bank is complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
import traceback

for _name, _value in {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}.items():
    os.environ.setdefault(_name, _value)

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import extract_real_gtfree_acceptance_features as feature_support
import extract_bop_aware_tail_features as tail
import prepare_winner_instance_expansion as expansion
import run_decoupled_development as development
import run_frozen_real_inputs_validation as real
import run_p2p_reliability_filter_development as support
import run_se3_trust_region_geometric_development as trust
import train_bop_aware_group_oof as base
import train_bop_aware_selected_risk as selected_risk
import train_bop_aware_state_residual as residual
import train_decoupled_group_oof as decoupled_oof
from decoupled_checkpoint_recovery import INITIAL_DIAGNOSTICS
from pose_perturbation_protocols import DECOUPLED, CAMERA, noise, paired_poses
from protocolG_observation import HASH_FIELDS, OBSERVATION_FIELDS, observation_metadata
from winner_training_data_io import read_csv


ROOT = Path(r"D:\6D_pose\codex_paper_workspace")
DEV = ROOT / "outputs" / "decoupled_development_20261020_r4"
DEV_OOF = DEV / "group_oof"
MILD_INPUTS = ROOT / "outputs" / "gtfree_unseen5_full_inputs_20261016_mild"
HARD_INPUTS = ROOT / "outputs" / "gtfree_unseen5_full_inputs_20261016_hard"
OUTPUT = ROOT / "outputs" / "decoupled_frozen_unseen5_20261021_r1"
HOLDOUT_OBJECTS = [5, 13, 18, 21, 27]
REGIMES = {"native": (0.0, 0.0, 20261016), "mild": (5.0, 5.0, 20261016), "hard": (15.0, 15.0, 20261017)}
METHODS = [base.INITIAL, *base.ACTIONS]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_frozen_development() -> tuple[dict, dict, dict]:
    state = json.loads((DEV_OOF / "state.json").read_text(encoding="utf-8"))
    if state.get("status") != "COMPLETE_NESTED_OBJECT_OOF_NOT_PROMOTED":
        raise ValueError("Frozen development OOF is not complete")
    for relative, expected in state.get("artifact_hashes", {}).items():
        path = DEV_OOF / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Frozen development artifact hash mismatch: {path}")
    for raw_path, expected in state.get("source_hashes", {}).items():
        path = Path(raw_path)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Frozen development source hash mismatch: {path}")
    model_path = DEV_OOF / "development_candidate.joblib"
    model = joblib.load(model_path)
    if model.get("status") != "DEVELOPMENT_ONLY_NOT_PROMOTED":
        raise ValueError("Deployment model status changed")
    if model.get("protocol") != DECOUPLED:
        raise ValueError("Deployment model protocol changed")
    bundle = model["bundle"]
    if bundle.get("training_objects") != expansion.OBJECTS:
        raise ValueError("Deployment model is not trained on exactly the 22 development objects")
    if len(bundle.get("feature_names", [])) != 191:
        raise ValueError("Deployment feature schema changed")
    if set(model["policies"]) != {
        "conditional_residual_legacy", "conditional_residual_joint", "stratified_only",
        "selected_risk_only", "selected_risk_stratified",
    }:
        raise ValueError("Deployment policy set changed")
    return state, model, {"model_path": str(model_path), "model_sha256": digest(model_path)}


def build_holdout_manifest(output: Path) -> pd.DataFrame:
    sources = {"mild": MILD_INPUTS, "hard": HARD_INPUTS}
    frames = {}
    pieces = []
    for regime, folder in sources.items():
        table = pd.read_csv(folder / "frame_manifest.csv")
        if set(table.object_id.astype(int)) != set(HOLDOUT_OBJECTS):
            raise ValueError(f"Unexpected objects in {folder}")
        table = table.loc[table.matched_prediction.astype(int).eq(1)].copy()
        if table.groupby("object_id").size().min() < 5:
            raise ValueError(f"A held-out object has fewer than five matched {regime} frames")
        table["init_regime"] = regime
        table["cohort"] = f"holdout_decoupled_v1_{regime}"
        table["original_input_directory"] = str(folder)
        frames[regime] = table[["frame_id", "object_id", "scene_id", "img_id", "inst_id", "selected_index"]].copy()
        pieces.append(table)
    left = frames["mild"].sort_values(["object_id", "frame_id"]).reset_index(drop=True)
    right = frames["hard"].sort_values(["object_id", "frame_id"]).reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Mild and hard held-out frame identities differ")
    native = pieces[0].copy()
    native["init_regime"] = "native"
    native["cohort"] = "holdout_decoupled_v1_native"
    native["original_input_directory"] = str(MILD_INPUTS)
    manifest = pd.concat([native, *pieces], ignore_index=True)
    if manifest.duplicated(["cohort", "frame_id"]).any():
        raise ValueError("Duplicate holdout cohort/frame identity")
    if set(manifest.object_id.astype(int)) & set(expansion.OBJECTS):
        raise ValueError("Holdout objects overlap development objects")
    manifest.to_csv(output / "locked_holdout_instances.csv", index=False)
    return manifest


def append_candidate_features(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    all_features, names = [], None
    # Keep the frozen 191-feature contract: observation hashes/seeds are provenance,
    # not model inputs. The development runner excludes this same OBSERVATION_FIELDS set.
    metadata_names = {"cohort", *base.KEYS, "method", "replay_status", *OBSERVATION_FIELDS}
    tail_names = [name for name in tables["tails"] if name not in metadata_names]
    for cohort, group in tables["trials"].groupby("cohort", sort=True):
        scores = tables["scores"].loc[tables["scores"].cohort.eq(cohort)]
        features, raw_names = base.build_features(scores, group, cohort)
        shadow = group.copy()
        for key in ["ADD_S_mm", "mask_iou", "rotation_error_deg", "translation_error_mm"]:
            shadow[key] = 99999.0
        replayed, _ = base.build_features(scores, shadow, cohort)
        np.testing.assert_allclose(features[raw_names], replayed[raw_names], equal_nan=True)
        keys = ["cohort", *base.KEYS, "action"]
        features = features.merge(
            tables["tails"].rename(columns={"method": "action"})[keys + tail_names],
            on=keys, validate="one_to_one",
        )
        current_names = raw_names + tail_names
        if names is not None and names != current_names:
            raise ValueError("Holdout feature schema differs between cohorts")
        names = current_names
        features[names] = features[names].replace([np.inf, -np.inf], np.nan)
        all_features.append(features)
    features = pd.concat(all_features, ignore_index=True)
    features["target_id"] = features.groupby(["cohort", *base.KEYS], sort=False, dropna=False).ngroup()
    features["candidate_index"] = np.arange(len(features))
    if len(names or []) != 191:
        raise ValueError(f"Expected 191 observable features, found {len(names or [])}")
    return features, names


def load_candidate_checkpoints(checkpoint_root: Path) -> dict[str, pd.DataFrame]:
    """Load only the completed candidate tables from a preserved generation run."""
    directories = sorted(checkpoint_root.glob("object_*"))
    if len(directories) != len(HOLDOUT_OBJECTS):
        raise ValueError("Candidate checkpoint object coverage is incomplete")
    tables = {}
    for name in ["trials", "scores", "tails"]:
        parts = [read_csv(directory / f"{name}.csv") for directory in directories]
        if any(part.empty for part in parts):
            raise ValueError(f"Empty candidate checkpoint: {name}")
        tables[name] = pd.concat(parts, ignore_index=True)
    if tables["trials"].duplicated(["cohort", *base.KEYS, "method"]).any():
        raise ValueError("Duplicate trial rows in reused checkpoints")
    if tables["scores"].duplicated(["cohort", *base.KEYS, "method"]).any():
        raise ValueError("Duplicate score rows in reused checkpoints")
    if tables["tails"].duplicated(["cohort", *base.KEYS, "method"]).any():
        raise ValueError("Duplicate tail rows in reused checkpoints")
    return tables


def generate_candidates(args: argparse.Namespace, output: Path, manifest: pd.DataFrame, state: dict) -> None:
    if (output / "observable_candidate_features.csv").exists():
        return
    if args.reuse_checkpoints is not None:
        tables = load_candidate_checkpoints(args.reuse_checkpoints / "generation_checkpoints")
        expected_trials = len(manifest) * 2 * len(METHODS)
        if any(len(tables[name]) != expected_trials for name in ["trials", "scores", "tails"]):
            raise ValueError("Reused candidate checkpoint coverage differs from locked holdout")
        features, names = append_candidate_features(tables)
        if len(features) != len(manifest) * 2 * len(base.ACTIONS) or features.frame_id.nunique() != manifest.frame_id.nunique():
            raise ValueError("Reused holdout feature coverage mismatch")
        for name, table in tables.items():
            table.to_csv(output / f"{name}.csv", index=False)
        features.to_csv(output / "observable_candidate_features.csv", index=False)
        save(output / "feature_schema.json", {"feature_names": names, "observation_features_only": True,
                                               "csv_parser": "float_precision=round_trip"})
        save(output / "checkpoint_reuse_receipt.json", {
            "source": str(args.reuse_checkpoints),
            "source_state_sha256": digest(args.reuse_checkpoints / "state.json"),
            "objects": HOLDOUT_OBJECTS,
            "candidate_pose_rows": len(tables["trials"]),
            "feature_rows": len(features),
            "feature_names": len(names),
            "purpose": "reuse completed GT-free candidate generation after metadata-contract repair",
        })
        state.update(stage="reused_frozen_holdout_candidates", completed_cases=660,
                     candidate_pose_rows=len(tables["trials"]), candidate_feature_rows=len(features),
                     matched_physical_instances=features.frame_id.nunique())
        save(output / "state.json", state)
        return
    poses = read_csv(real.POSE_CSV)
    matched = manifest.copy()
    masks = feature_support.masks_by_source_index(real.MASK_JSON, set(matched.mask_source_index.astype(int)))
    ns = real.support.load_experiment_namespace(
        HOLDOUT_OBJECTS, len(manifest), 1, [0.3], dataset_root=args.dataset,
    )
    raw_tables = {
        str(folder): read_csv(folder / "real_inputs_validation_trials.csv")
        for folder in [MILD_INPUTS, HARD_INPUTS]
    }
    rows = {name: [] for name in ["inputs", "trials", "scores", "tails", "observations", "geometry"]}
    renderer = None
    points_cache: dict[int, np.ndarray] = {}
    state.update(stage="generating_frozen_holdout_candidates", stage_started_at_utc=now())
    save(output / "state.json", state)
    for oid, group in matched.groupby("object_id", sort=True):
        meta = ns["configure_object"](int(oid))
        for row in group.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            frame = ns["load_frame_data"](real.candidate_from_manifest(row_series, args.dataset, "test_primesense"))
            mask = real.decode_uncompressed_rle(masks[int(row.mask_source_index)])
            public = poses.iloc[int(row.initializer_source_index)]
            if tuple(map(int, [public.scene_id, public.im_id, public.obj_id])) != (int(row.scene_id), int(row.img_id), int(oid)):
                raise ValueError("Public pose identity changed")
            rotation, translation = real.bop_pose_from_row(public)
            reference = real.internal_pose_from_bop(rotation, translation)
            angle, increment, seed = REGIMES[str(row.init_regime)]
            q, u, metadata = noise(angle, increment, int(oid), int(row.selected_index), seed)
            paired = paired_poses(reference, q, u)
            initial = paired[DECOUPLED]
            if not np.isclose(1000.0 * np.linalg.norm(initial[:3, 3] - reference[:3, 3]), increment, atol=1e-9):
                raise ValueError("Decoupled translation magnitude changed")
            rows["geometry"].append(dict(
                cohort=row.cohort, frame_id=row.frame_id, object_id=int(oid), init_regime=row.init_regime,
                selected_index=int(row.selected_index), actual_origin_shift_mm=float(1000.0 * np.linalg.norm(initial[:3, 3] - reference[:3, 3])),
                nominal_increment_mm=increment, original_camera_origin_shift_mm=float(1000.0 * np.linalg.norm(paired[CAMERA][:3, 3] - reference[:3, 3])),
                noise_rotation_json=json.dumps(q.tolist()), noise_translation_m_json=json.dumps(u.tolist()), **metadata,
            ))
            old_table = raw_tables[str(row.original_input_directory)]
            for condition, _ in real.INPUT_CONDITIONS:
                old_initial = old_table.loc[
                    old_table.frame_id.eq(row.frame_id) & old_table.input_depth_condition.eq(condition)
                    & old_table.method.eq(real.INITIAL)
                ].iloc[0]
                # Reuse only the frozen observation realization. The previous
                # initializer pose is not used; the new decoupled pose is used.
                prepared, replay_status = development.replay(ns, frame, mask, old_initial, int(row.selected_index))
                hashes = {key: prepared["metadata"][key] for key in HASH_FIELDS}
                selection = dict(
                    pose=dict(source_index=int(row.initializer_source_index), score=float(public.score)),
                    mask=dict(source_index=int(row.mask_source_index), score=float(row.mask_score)),
                    bbox_iou=float(row.pose_mask_bbox_iou),
                )
                generated = real.run_methods(ns, frame, meta, initial, mask, prepared, row.frame_id, selection, int(row.selected_index))
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
                    renderer = tail.Open3DRaycastRenderer(w, h, cache_size=16)
                if int(oid) not in points_cache:
                    model_path = args.dataset / "models_eval" / f"obj_{int(oid):06d}.ply"
                    renderer.add_object(int(oid), str(model_path))
                    vertices = np.asarray(__import__("open3d").io.read_triangle_mesh(str(model_path)).vertices, dtype=float)
                    points_cache[int(oid)] = vertices[::max(1, int(np.ceil(len(vertices) / 1000)))]
                    state.setdefault("dataset_hashes", {})[str(model_path)] = digest(model_path)
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
                        int(oid), rb, tb, K[0, 0], K[1, 1], K[0, 2], K[1, 2],
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
                        points_cache[int(oid)], pd.Series(record), pd.Series(initial_record), K,
                        float(initial_record["diameter_mm"]), renderer.width,
                    ))
                    rows["tails"].append(entry)
                actual = observation_metadata(prepared, mask, K)
                if any(actual[key] != expected for key, expected in hashes.items()):
                    raise ValueError("Shared observation mutated")
                rows["observations"].append(dict(
                    cohort=row.cohort, frame_id=row.frame_id, object_id=int(oid),
                    input_depth_condition=condition, replay_status=replay_status,
                    **prepared["metadata"],
                ))
            state["completed_cases"] = int(state.get("completed_cases", 0)) + 1
            if state["completed_cases"] % 10 == 0:
                save(output / "state.json", state)
                print(f"HOLDOUT CANDIDATES {state['completed_cases']}/{len(matched)} cases", flush=True)
        checkpoint = output / "generation_checkpoints" / f"object_{int(oid):02d}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        for name in ["inputs", "trials", "scores", "tails"]:
            pd.DataFrame([r for r in rows[name] if int(r["object_id"]) == int(oid)]).to_csv(checkpoint / f"{name}.csv", index=False)
        save(output / "state.json", state)
        print(f"OBJECT COMPLETE {int(oid)}; {state['completed_cases']} cases", flush=True)
    tables = {name: pd.DataFrame(rows[name]) for name in rows}
    expected_trials = len(matched) * 2 * len(METHODS)
    if len(tables["trials"]) != expected_trials or len(tables["scores"]) != expected_trials or len(tables["tails"]) != expected_trials:
        raise ValueError("Incomplete frozen holdout candidate bank")
    features, names = append_candidate_features(tables)
    if len(features) != len(matched) * 2 * len(base.ACTIONS) or features.frame_id.nunique() != manifest.frame_id.nunique():
        raise ValueError("Frozen holdout feature coverage mismatch")
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    features.to_csv(output / "observable_candidate_features.csv", index=False)
    tables["observations"].to_csv(output / "observation_replay_checks.csv", index=False)
    tables["geometry"].to_csv(output / "initialization_geometry.csv", index=False)
    save(output / "feature_schema.json", {"feature_names": names, "observation_features_only": True, "csv_parser": "float_precision=round_trip"})
    state.update(candidate_pose_rows=len(tables["trials"]), candidate_feature_rows=len(features), matched_physical_instances=features.frame_id.nunique())
    save(output / "state.json", state)


def evaluate_bop(args: argparse.Namespace, output: Path, state: dict) -> None:
    if (output / "holdout_bop_labels.csv").exists():
        return
    if args.reuse_bop is not None:
        source = args.reuse_bop
        required = [source / "holdout_bop_labels.csv", source / "labeled_candidate_data.csv",
                    source / "bop" / "bop_pose_errors.csv"]
        if any(not path.is_file() for path in required):
            raise ValueError("Reused BOP output is incomplete")
        shutil.copy2(source / "holdout_bop_labels.csv", output / "holdout_bop_labels.csv")
        shutil.copy2(source / "labeled_candidate_data.csv", output / "labeled_candidate_data.csv")
        shutil.copytree(source / "bop", output / "bop")
        if (source / "bop.log").is_file():
            shutil.copy2(source / "bop.log", output / "bop.log")
        labels = read_csv(output / "holdout_bop_labels.csv")
        data = read_csv(output / "labeled_candidate_data.csv")
        state.update(stage="reused_completed_frozen_holdout_bop", labels_rows=len(labels),
                     labeled_feature_rows=len(data), reused_bop_from=str(source))
        save(output / "state.json", state)
        return
    work = Path(__file__).parent
    state.update(stage="evaluating_frozen_holdout_bop", stage_started_at_utc=now())
    save(output / "state.json", state)
    command = [sys.executable, "-u", str(work / "evaluate_bop_metrics_controlled.py"),
               "--trials-csv", str(output / "trials.csv"), "--output-dir", str(output / "bop"),
               "--dataset-root", str(args.dataset), "--toolkit-root", str(args.toolkit),
               "--methods", ",".join(METHODS), "--extra-key-columns", "cohort,input_depth_condition",
               "--device", args.device, "--gpu-batch-size", "1", "--checkpoint-every", "50"]
    with (output / "bop.log").open("w", encoding="utf-8") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    from bop_thresholds import attach_image_widths, row_recalls
    errors = read_csv(output / "bop" / "bop_pose_errors.csv")
    labels = row_recalls(attach_image_widths(errors, args.dataset, "test_primesense"))
    labels["evaluation_depth_policy"] = "native_sensor_depth"
    labels["pose_protocol"] = DECOUPLED
    labels.to_csv(output / "holdout_bop_labels.csv", index=False)
    features = read_csv(output / "observable_candidate_features.csv")
    data = base.attach_labels(features, labels)
    if len(data) != len(features) or set(data.object_id.astype(int)) != set(HOLDOUT_OBJECTS):
        raise ValueError("Frozen holdout labels failed to attach")
    data.to_csv(output / "labeled_candidate_data.csv", index=False)
    state.update(labels_rows=len(labels), labeled_feature_rows=len(data))
    save(output / "state.json", state)


def apply_frozen_model(output: Path, manifest: pd.DataFrame, model: dict, model_meta: dict, state: dict) -> None:
    if (output / "frozen_summary.csv").exists():
        return
    state.update(stage="applying_frozen_deployment_policy", stage_started_at_utc=now())
    save(output / "state.json", state)
    work = Path(__file__).parent
    raw_plan = json.loads((work / "bop_aware_tail_risk_v2_plan_20261019.json").read_text(encoding="utf-8"))
    residual_plan = json.loads((work / "bop_aware_state_residual_plan_20261019.json").read_text(encoding="utf-8"))
    winner_plan = json.loads((work / "bop_aware_selected_risk_plan_20261019.json").read_text(encoding="utf-8"))
    features = read_csv(output / "observable_candidate_features.csv")
    regimes = manifest[["cohort", "frame_id", "init_regime"]].drop_duplicates()
    features = features.merge(regimes, on=["cohort", "frame_id"], validate="many_to_one")
    data = base.attach_labels(features, read_csv(output / "holdout_bop_labels.csv")).reset_index(drop=True)
    names = model["bundle"]["feature_names"]
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
        selected = selected_risk.selected_frame(data, pool, model["policies"][variant]["policy_id"], model["policies"][variant], variant)
        selected["method"] = variant
        outcomes.append(selected)
    for method in [base.INITIAL, *base.ACTIONS]:
        baseline = base.baseline_rows(data, method)
        baseline["method"] = method
        outcomes.append(baseline)
    outcome_table = pd.concat(outcomes, ignore_index=True)
    targets = data[["target_id", "cohort", "frame_id", "object_id", "init_regime"]].drop_duplicates("target_id")
    initial_columns = data.drop_duplicates("target_id")[["target_id", *["initial_" + c for c in base.AR]]]
    outcome_table = outcome_table.merge(targets[["target_id", "init_regime"]], on="target_id", validate="many_to_one")
    outcome_table = outcome_table.merge(initial_columns, on="target_id", validate="many_to_one")
    outcome_table.to_csv(output / "frozen_outcomes.csv", index=False)
    summary = decoupled_oof.planned_summary(outcome_table, manifest)
    summary.to_csv(output / "frozen_summary.csv", index=False)
    paired = decoupled_oof.paired_bootstrap(outcome_table)
    paired.to_csv(output / "paired_object_bootstrap.csv", index=False)
    save(output / "frozen_model_receipt.json", {
        "model_path": model_meta["model_path"], "model_sha256": model_meta["model_sha256"],
        "training_objects": model["bundle"]["training_objects"], "holdout_objects": HOLDOUT_OBJECTS,
        "policies": model["policies"], "policy_tuning": "none_on_holdout",
        "raw_plan_sha256": digest(work / "bop_aware_tail_risk_v2_plan_20261019.json"),
        "residual_plan_sha256": digest(work / "bop_aware_state_residual_plan_20261019.json"),
        "winner_plan_sha256": digest(work / "bop_aware_selected_risk_plan_20261019.json"),
    })
    validation_status = "COMPLETE_FROZEN_HOLDOUT_VALIDATION"
    object_label = f"{len(HOLDOUT_OBJECTS)}-object"
    lines = [
        f"# {object_label} frozen decoupled validation", "", "## Material Passport",
        f"- Status: {validation_status}; Verification Status: ANALYZED",
        f"- {len(HOLDOUT_OBJECTS)} held-out T-LESS objects: {', '.join(map(str, HOLDOUT_OBJECTS))}.",
        "- The deployment artifact was fit only on the 22 development objects; no holdout label was used for training, threshold selection, or policy tuning.",
        "- The same selected frames, public pose/mask sources, depth realizations, candidate scales, and native-sensor BOP evaluation policy were used across methods.",
        "- Native is evaluated on the fixed held-out frame list; mild and hard use the pre-locked held-out frame lists. The repeated difficulty rows are paired conditions, not independent objects.",
        "", "## Frozen deployment policy", "```text", json.dumps(model["policies"], indent=2), "```",
        "", "## Summary", "```text", summary.loc[summary.axis.isin(["all", "regime", "depth"])].to_string(index=False), "```",
        "", "## Interpretation boundary",
        "- This is an independent object-level validation of the frozen deployment artifact, not a leaderboard submission.",
        f"- The {len(HOLDOUT_OBJECTS)} objects were not used by the development OOF or the final development refit.",
        "- The selected_risk_stratified policy remains disabled if that is what the frozen development receipt specifies; this is reported rather than re-enabled on the holdout.",
        "- Confidence intervals and any claim of generalization must use object-cluster resampling; rows within an object and difficulty are paired.",
    ]
    (output / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    state.update(status=validation_status, completed_at_utc=now(), model_sha256=model_meta["model_sha256"], method_promoted=False)
    state["artifact_hashes"] = {str(path.relative_to(output)): digest(path) for path in output.rglob("*") if path.is_file() and path != output / "state.json"}
    save(output / "state.json", state)


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Preserve prior output: {args.output}")
    args.output.mkdir(parents=True)
    state = {"status": "RUNNING", "started_at_utc": now(), "pid": os.getpid(), "holdout_objects": HOLDOUT_OBJECTS}
    save(args.output / "state.json", state)
    try:
        dev_state, model, model_meta = verify_frozen_development()
        manifest = build_holdout_manifest(args.output)
        state.update(development_status=dev_state["status"], model_receipt=model_meta)
        save(args.output / "state.json", state)
        snapshots = args.output / "source_snapshot"
        snapshots.mkdir()
        for path in [Path(__file__), DEV_OOF / "development_candidate.joblib", DEV_OOF / "state.json",
                     DEV / "locked_protocol.json", real.POSE_CSV, real.MASK_JSON,
                     MILD_INPUTS / "frame_manifest.csv", HARD_INPUTS / "frame_manifest.csv",
                     MILD_INPUTS / "real_inputs_validation_trials.csv", HARD_INPUTS / "real_inputs_validation_trials.csv"]:
            shutil.copy2(path, snapshots / path.name)
        state["source_hashes"] = {str(path): digest(path) for path in [Path(__file__), DEV_OOF / "development_candidate.joblib", DEV_OOF / "state.json",
                     DEV / "locked_protocol.json", real.POSE_CSV, real.MASK_JSON,
                     MILD_INPUTS / "frame_manifest.csv", HARD_INPUTS / "frame_manifest.csv",
                     MILD_INPUTS / "real_inputs_validation_trials.csv", HARD_INPUTS / "real_inputs_validation_trials.csv"]}
        save(args.output / "state.json", state)
        generate_candidates(args, args.output, manifest, state)
        evaluate_bop(args, args.output, state)
        apply_frozen_model(args.output, manifest, model, model_meta, state)
        print("FROZEN HOLDOUT VALIDATION COMPLETE", flush=True)
    except Exception:
        state.update(status="FAILED_PRESERVED_NO_AUTOMATIC_RETRY", error=traceback.format_exc())
        save(args.output / "state.json", state)
        raise


if __name__ == "__main__":
    import shutil

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dataset", type=Path, default=real.DATASET_ROOT)
    parser.add_argument("--toolkit", type=Path, default=Path(r"D:\6D_pose\tools\bop_toolkit"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--reuse-checkpoints", type=Path,
                        help="Reuse completed object checkpoints from a preserved candidate run")
    parser.add_argument("--reuse-bop", type=Path,
                        help="Reuse completed BOP output from a preserved frozen-validation run")
    parser.add_argument("--holdout-objects", default=','.join(map(str, HOLDOUT_OBJECTS)))
    parser.add_argument("--mild-inputs", type=Path, default=MILD_INPUTS)
    parser.add_argument("--hard-inputs", type=Path, default=HARD_INPUTS)
    parsed = parser.parse_args()
    HOLDOUT_OBJECTS = [int(value.strip()) for value in parsed.holdout_objects.split(',') if value.strip()]
    MILD_INPUTS = parsed.mild_inputs
    HARD_INPUTS = parsed.hard_inputs
    run(parsed)
