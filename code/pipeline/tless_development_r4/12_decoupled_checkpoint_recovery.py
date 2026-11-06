"""Validate author-stopped complete objects and repair only initial update metadata."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import prepare_winner_instance_expansion as p
import run_frozen_real_inputs_validation as real
import run_se3_trust_region_geometric_development as trust
import train_bop_aware_group_oof as base
from pose_perturbation_protocols import CAMERA, DECOUPLED, noise, paired_poses
from protocolG_observation import OBSERVATION_FIELDS, HASH_FIELDS
from winner_training_data_io import read_csv

INITIAL_DIAGNOSTICS = dict(direction="initial", trust_scale=0., full_update_rotation_deg=0.,
    full_update_translation_mm=0., applied_rotation_deg=0., applied_translation_mm=0.,
    rotation_scale=0., translation_scale=0.)
ZERO_FIELDS = [key for key in INITIAL_DIAGNOSTICS if key != "direction"]


def repair_initial_diagnostics(scores):
    result = scores.copy()
    selected = result.method.eq(base.INITIAL)
    for name in ZERO_FIELDS:
        if name in result and result.loc[selected, name].dropna().ne(0).any():
            raise ValueError("Nonzero initial-update diagnostic is not the approved omission")
        if name not in result:
            result[name] = np.nan
        result.loc[selected, name] = 0.
    if "direction" not in result:
        result["direction"] = ""
    result.loc[selected, "direction"] = "initial"
    return result


def recover(source, output, manifest, poses):
    state = json.loads((source/"state.json").read_text())
    if state["status"] != "RUNNING" or state["stage"] != "regenerating_inputs_candidates_and_tails" or (source/"bop").exists():
        raise ValueError("Only the author-stopped pre-BOP checkpoint batch is reusable")
    checks, inventory = [], {}

    def check(name, passed):
        checks.append(dict(check=name, passes=bool(passed)))
        if not passed:
            raise ValueError("Checkpoint validation failed: "+name)

    old_manifest = read_csv(source/"locked_development_instances.csv")
    pd.testing.assert_frame_equal(manifest, old_manifest, check_dtype=False)
    check("same_locked_manifest", True)
    mutable = {"run_decoupled_development.py", "test_decoupled_development.py"}
    for filename, expected in state["source_hashes"].items():
        path = Path(filename)
        if path.name in mutable:
            snapshots = list((source/"source_snapshot").glob("*_"+path.name))
            check("preserved_original_source_"+path.name, len(snapshots)==1 and p.digest(snapshots[0])==expected)
        else:
            check("unchanged_original_source_"+path.name, p.digest(path)==expected)
    evaluation = json.loads((source/"evaluation_sources_pre_BOP.json").read_text())
    changed = [path for path, expected in evaluation["file_hashes"].items()
               if Path(path).name not in mutable and p.digest(Path(path))!=expected]
    check("recorded_evaluation_sources_unchanged", not changed)
    tables = {name:[] for name in ["inputs", "trials", "scores", "tails"]}
    geometry, observations, reused = [], [], set()
    object_dirs = sorted((source/"generation_checkpoints").glob("object_*"))
    if not object_dirs:
        raise ValueError("No complete object checkpoint")
    for directory in object_dirs:
        oid = int(directory.name.rsplit("_", 1)[1])
        frames = manifest.loc[manifest.object_id.eq(oid)&manifest.matched_prediction.eq(1)]
        expected_cases = set(map(tuple, frames[["cohort", "frame_id"]].to_numpy()))
        local = {}
        for name in tables:
            path = directory/(name+".csv")
            if not path.is_file():
                raise ValueError("Partial object checkpoint cannot enter recovery: "+str(directory))
            inventory[str(path)] = p.digest(path)
            table = read_csv(path)
            methods = set(real.METHODS) if name=="inputs" else {base.INITIAL, *base.ACTIONS}
            check(f"object_{oid}_{name}_full_case_coverage",
                len(table)==len(frames)*2*len(methods) and set(map(tuple, table[["cohort", "frame_id"]].to_numpy()))==expected_cases)
            check(f"object_{oid}_{name}_methods_per_condition",
                table.groupby(["cohort", *base.KEYS]).method.agg(set).map(lambda x:x==methods).all()
                and not table.duplicated(["cohort", *base.KEYS, "method"]).any())
            check(f"object_{oid}_{name}_both_depths", set(table.input_depth_condition)=={name for name, _ in real.INPUT_CONDITIONS})
            local[name] = table
        initial = local["trials"].loc[local["trials"].method.eq(base.INITIAL)]
        for row in frames.itertuples(index=False):
            rotation, translation, seed = p.REGIMES[row.init_regime]
            q, u, metadata = noise(rotation, translation, int(oid), int(row.selected_index), seed)
            public = poses.iloc[int(row.initializer_source_index)]
            check("public_identity_"+row.cohort+"_"+row.frame_id,
                  tuple(map(int, [public.scene_id, public.im_id, public.obj_id]))==(int(row.scene_id), int(row.img_id), int(oid)))
            R, t = real.bop_pose_from_row(public)
            reference = real.internal_pose_from_bop(R, t)
            pair = paired_poses(reference, q, u)
            expected = pair[DECOUPLED]
            geometry.append(dict(cohort=row.cohort, frame_id=row.frame_id, object_id=int(oid), init_regime=row.init_regime,
                selected_index=int(row.selected_index), actual_origin_shift_mm=float(np.linalg.norm(expected[:3,3]-reference[:3,3])*1000),
                nominal_increment_mm=translation, original_camera_origin_shift_mm=float(np.linalg.norm(pair[CAMERA][:3,3]-reference[:3,3])*1000),
                noise_rotation_json=json.dumps(q.tolist()), noise_translation_m_json=json.dumps(u.tolist()), **metadata))
            for condition, _ in real.INPUT_CONDITIONS:
                selector = lambda table:table.loc[table.cohort.eq(row.cohort)&table.frame_id.eq(row.frame_id)&table.input_depth_condition.eq(condition)]
                actual = selector(initial).iloc[0]
                check("initial_formula_"+row.cohort+"_"+row.frame_id+"_"+condition,
                    np.allclose(real.support.transform_from_row(actual), expected, rtol=1e-10, atol=1e-10))
                originals = selector(local["inputs"])
                bank = selector(local["trials"])
                tail_rows = selector(local["tails"])
                for name in ["trials", "scores", "tails", "inputs"]:
                    group = selector(local[name])
                    check("observation_"+name+"_"+row.cohort+"_"+row.frame_id+"_"+condition,
                          all(group[field].eq(actual[field]).all() for field in OBSERVATION_FIELDS))
                for direction, trunk in [("p2p", real.TRUNK), ("filtered", real.PROPOSED)]:
                    full = real.support.transform_from_row(originals.loc[originals.method.eq(trunk)].iloc[0])
                    for scale in trust.SCALES:
                        expected_pose, diagnostics = trust.restricted_pose(expected, full, scale)
                        method = trust.method_name(direction, scale)
                        stored = bank.loc[bank.method.eq(method)].iloc[0]
                        score = selector(local["scores"]).loc[lambda table:table.method.eq(method)].iloc[0]
                        check("bounded_pose_"+row.cohort+"_"+row.frame_id+"_"+condition+"_"+method,
                              np.allclose(real.support.transform_from_row(stored), expected_pose, rtol=1e-10, atol=1e-10))
                        check("bounded_diagnostics_"+row.cohort+"_"+row.frame_id+"_"+condition+"_"+method,
                              all(np.isclose(float(score[key]), value, rtol=1e-9, atol=1e-9) for key,value in diagnostics.items()))
                tail_initial = tail_rows.loc[tail_rows.method.eq(base.INITIAL)].iloc[0]
                observations.append(dict(cohort=row.cohort, frame_id=row.frame_id, object_id=int(oid),
                    input_depth_condition=condition, replay_status=tail_initial.replay_status,
                    **{field:tail_initial[field] for field in OBSERVATION_FIELDS}))
            reused.add((row.cohort, row.frame_id))
        original_scores = local["scores"]
        local["scores"] = repair_initial_diagnostics(original_scores)
        check(f"object_{oid}_noninitial_scores_unchanged", original_scores.loc[original_scores.method.ne(base.INITIAL)].equals(
              local["scores"].loc[local["scores"].method.ne(base.INITIAL)]))
        for name, table in local.items():
            tables[name].extend(table.to_dict("records"))
    for path, expected in inventory.items():
        check("sealed_checkpoint_unchanged_"+path, p.digest(Path(path))==expected)
    pd.DataFrame(checks).to_csv(output/"checkpoint_recovery_checks.csv", index=False)
    receipt = dict(status="VERIFIED_AUTHOR_APPROVED_METADATA_REPAIR", source=str(source),
        source_process_stopped_by_author_choice=True, reused_objects=[int(path.name.rsplit("_",1)[1]) for path in object_dirs],
        reused_cases=len(reused), remaining_cases=int(manifest.matched_prediction.sum())-len(reused),
        checks=len(checks), source_checkpoint_hashes=inventory,
        copied_tables="raw scores preserved in source; only initial update diagnostics restored to their known zero values in new output",
        validation="all complete object checkpoints retained; identity/coverage, source hashes, observation contracts, exact initializer and six bounded poses/diagnostics verified; tail scalars reused, not independently recomputed",
        no_outcome_filtering=True, no_labels_or_models_reused=True)
    p.save(output/"approved_checkpoint_reuse.json", receipt)
    print("CHECKPOINT RECOVERY VERIFIED", len(reused), "cases", len(checks), "checks", flush=True)
    return tables, geometry, observations, reused


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous checkpoint inspection")
    args.output.mkdir(parents=True)
    recover(args.source, args.output, read_csv(args.source/"locked_development_instances.csv"), read_csv(real.POSE_CSV))
