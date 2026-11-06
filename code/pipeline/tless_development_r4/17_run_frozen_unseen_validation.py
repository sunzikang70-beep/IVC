"""Frozen unseen-object validation with paired deterministic pose directions."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd

import run_p2p_reliability_filter_development as support
from run_strong_baselines_only import make_record, make_output

ROOT = support.OUTPUT_ROOT / 'frozen_unseen_validation_20260907'
CONFIG = support.OUTPUT_ROOT / 'bop_metrics_p2p_reliability_dual_filter_development' / 'frozen_method_config.json'
OBJECTS = [3, 10, 12, 15, 16, 19, 23, 24, 25, 26, 28, 29]
DEFAULT_SEED_NAMESPACE = 'frozen-unseen-v1:20260907'
LEVELS = {'mild_5deg_5mm': (5.0, 5.0), 'hard_15deg_15mm': (15.0, 15.0)}
FRACTIONS = [0.2, 0.3, 0.4]
INITIAL = 'protocolB_e2e_initial_pose'
PROPOSED = 'protocolB_e2e_gate_p2p_direct_dual_filter_r035'
METHODS = [INITIAL, support.TRUNK_METHOD, PROPOSED]
POSE_COLUMNS = list(support.pose_fields(np.eye(4)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initial_pose(gt, object_id, frame, level, seed_namespace=DEFAULT_SEED_NAMESPACE):
    identity = f"{seed_namespace}:{object_id}:{frame['scene_id']}:{frame['img_id']}:{frame['inst_id']}"
    seed = int.from_bytes(hashlib.sha256(identity.encode('ascii')).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    angle, translation_mm = LEVELS[level]
    rotation_cv = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.deg2rad(angle))
    flip = np.diag([1., -1., 1.])
    transform = gt.copy()
    # Rotate about the object origin; convert camera-CV noise to internal Y-up.
    transform[:3, :3] = flip @ rotation_cv @ flip @ gt[:3, :3]
    transform[:3, 3] += flip @ direction * translation_mm / 1000.
    metadata = {
        'init_seed_hex': f'{seed:016x}', 'init_difficulty': level,
        'init_angle_deg': angle, 'init_translation_norm_mm': translation_mm,
        **{f'init_axis_cv_{c}': axis[i] for i, c in enumerate('xyz')},
        **{f'init_translation_direction_cv_{c}': direction[i] for i, c in enumerate('xyz')},
    }
    return transform, metadata


def fit(ns, source, target, initial):
    if len(source) < 3 or len(target.points) < 3:
        return initial.copy(), None, 'insufficient_points', 0.
    start = time.perf_counter()
    result = support.run_point_to_point(ns, source, target, initial)
    elapsed = (time.perf_counter() - start) * 1000.
    if not np.isfinite(result.transformation).all() or len(result.correspondence_set) < 3:
        return initial.copy(), None, 'invalid_registration', elapsed
    return result.transformation, result, 'completed', elapsed


def run(args):
    config = json.loads(CONFIG.read_text())
    assert config['method'] == PROPOSED and config['reliability_keep_rule'] == 'reliability > 0.35'
    objects = [int(x) for x in args.objects.split(',')]
    if not args.smoke and not args.development and not args.frame_disjoint_validation and not args.main_experiment and not args.external_dataset_validation and objects != OBJECTS:
        raise ValueError('The validation object list is fixed.')
    output = args.output
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    output.mkdir(parents=True)
    snapshots = output / 'source_snapshot'
    snapshots.mkdir()
    paths = [Path(__file__), Path(support.__file__), support.SOURCE, CONFIG,
             Path(__file__).with_name('run_strong_baselines_only.py'),
             Path(__file__).with_name('evaluate_bop_metrics_controlled.py'),
             Path(__file__).with_name('analyze_frozen_unseen_validation.py')]
    is_development = bool(args.development)
    is_frame_disjoint = bool(args.frame_disjoint_validation)
    is_main = bool(args.main_experiment)
    excluded_rgbd = set()
    if args.exclude_rgbd_manifest is not None:
        excluded = pd.read_csv(args.exclude_rgbd_manifest)
        excluded_rgbd = set(zip(excluded.scene_id.astype(int), excluded.img_id.astype(int)))
    manifest = {
        'status': (
            'protocol_locked_external_dataset'
            if args.external_dataset_validation else
            'protocol_locked_before_results_frame_disjoint'
            if is_frame_disjoint else
            'protocol_locked_large_sample_main'
            if is_main else
            'development_candidate_pool' if is_development else 'protocol_locked_before_results'
        ),
        'objects': objects,
        'dataset_name': args.dataset_name,
        'dataset_root': str(args.dataset_root),
        'dataset_split': args.dataset_split,
        'depth_extension': args.depth_extension,
        'planned_frames_per_object': args.frames, 'levels': LEVELS,
        'seed_namespace': args.seed_namespace,
        'seed_rule': f'SHA256({args.seed_namespace}:object:scene:image:instance), first 8 bytes little-endian',
        'initialization': 'Fixed rotation angle and translation norm; isotropic random directions shared between difficulties; rotation about object origin.',
        'fractions': FRACTIONS, 'methods': METHODS, 'frozen_method_config': config,
        'primary_comparisons': 'Frozen minus P2P mean AR per difficulty, all corruptions; Holm family of two tests.',
        'secondary_comparisons': 'Outlier AR, individual AR metrics, ADD-S, initial-pose baseline, per-object and per-fraction results.',
        'sampling': 'Evenly selected eligible frames; pre-method exclusion only if clean valid points < 500. All subsequent fallbacks remain in evaluation.',
        'timing': 'Measured current-run time; excludes synthetic corruption generation and GT evaluation. Reports shared cloud/reliability preparation separately.',
        'scope': (
            f'Frozen-method external-dataset validation on {args.dataset_name}; '
            'controlled local depth corruptions, BOP visible masks, and GT-perturbed initialization. '
            'No method threshold or route was tuned on this dataset.'
            if args.external_dataset_validation else
            'RGB-D-image-disjoint validation relative to acceptance-rule tuning images; '
            'same T-LESS object identities, controlled corruptions, and GT-perturbed initialization.'
            if is_frame_disjoint else
            'Large-sample controlled T-LESS main experiment with the already frozen method; '
            'the selected objects appeared in legacy method development, so this is not an untouched validation.'
            if is_main else
            'Development object identities used only to fit a GT-free candidate acceptance rule; '
            'not an independent validation or official BOP leaderboard test.'
            if is_development else
            'Unseen target object identities within T-LESS; not a scene-disjoint or official BOP leaderboard test; uses BOP visible masks and GT-perturbed initialization.'
        ),
        'hashes': {str(p): digest(p) for p in paths},
    }
    if args.exclude_rgbd_manifest is not None:
        manifest['excluded_rgbd_manifest'] = str(args.exclude_rgbd_manifest)
        manifest['excluded_rgbd_images'] = len(excluded_rgbd)
        manifest['excluded_rgbd_manifest_sha256'] = digest(args.exclude_rgbd_manifest)
    (output / 'protocol_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for p in paths:
        shutil.copy2(p, snapshots / p.name)
    ns = support.load_experiment_namespace(
        objects,
        args.frames,
        1,
        FRACTIONS,
        dataset_root=args.dataset_root,
        dataset_split=args.dataset_split,
        depth_extension=args.depth_extension,
    )
    rows = {level: [] for level in LEVELS}
    frames = []
    for object_id in objects:
        meta = ns['configure_object'](object_id)
        candidates = ns['discover_candidates']()
        if excluded_rgbd:
            candidates = [
                candidate for candidate in candidates
                if (int(candidate['scene_id']), int(candidate['img_id'])) not in excluded_rgbd
            ]
        selected = ns['select_evenly'](candidates, args.frames)
        print(f'Object {object_id}: {len(selected)} planned frames', flush=True)
        for fi, candidate in enumerate(selected):
            frame = ns['load_frame_data'](candidate)
            clean, _ = ns['depth_to_cloud'](frame['depth'], frame['mask_bool'], frame['K'])
            frame_id = f"O{object_id:03d}_S{frame['scene_id']:06d}_I{frame['img_id']:06d}_N{frame['inst_id']:02d}"
            eligible = len(clean.points) >= ns['MIN_VISIBLE_PIXELS']
            frames.append({'object_id': object_id, 'frame_id': frame_id,
                           'scene_id': frame['scene_id'], 'img_id': frame['img_id'],
                           'inst_id': frame['inst_id'], 'selected_index': fi,
                           'clean_points': len(clean.points), 'included': eligible})
            if not eligible:
                print(f'  SKIP {frame_id}: {len(clean.points)} clean points', flush=True)
                continue
            initializations = {
                level: initial_pose(
                    frame['T_gt'], object_id, frame, level, args.seed_namespace
                )
                for level in LEVELS
            }
            for fidx, fraction in enumerate(FRACTIONS):
                patch_seed = 200000 + object_id * 1000000 + fi * 10000 + fidx * 100
                patch, u, v, actual = ns['select_local_patch'](frame['depth'], frame['mask_bool'], fraction, patch_seed)
                for cidx, corruption in enumerate(ns['corruption_types']):
                    value_seed = 500000 + object_id * 1000000 + fi * 10000 + fidx * 1000 + cidx
                    corrupted = ns['make_local_corrupted_target'](frame['depth'], frame['mask_bool'], frame['K'], patch, corruption, value_seed)
                    # Time real-input preparation separately from synthetic corruption.
                    start = time.perf_counter()
                    target, uv = ns['depth_to_cloud'](corrupted['corrupted_depth'], frame['mask_bool'], frame['K'])
                    target_ms = (time.perf_counter() - start) * 1000.
                    start = time.perf_counter()
                    reliability = ns['compute_depth_reliability'](corrupted['corrupted_depth'], frame['mask_bool'], uv)
                    gate = ns['compute_reliability_gate_features'](corrupted['corrupted_depth'], frame['mask_bool'])
                    gate_ms = (time.perf_counter() - start) * 1000.
                    assert np.array_equal(uv, corrupted['target_uv'])
                    assert np.array_equal(reliability, corrupted['depth_reliability'])
                    for level, (initial, init_meta) in initializations.items():
                        start = time.perf_counter()
                        source, _, _ = ns['generate_observation_supported_visible_model'](
                            ns['full_model_points'], initial, corrupted['corrupted_depth'], frame['mask_bool'], frame['K'], target)
                        source_ms = (time.perf_counter() - start) * 1000.
                        trunk_T, trunk_result, trunk_reason, trunk_icp_ms = fit(ns, source, target, initial)
                        trunk_ms = target_ms + source_ms + trunk_icp_ms
                        chosen, chosen_result = trunk_T, trunk_result
                        reason = 'gate_closed:p2p_trunk'
                        source_count, filtered_count, retention = len(source), len(target.points), 1.
                        used = 0
                        start = time.perf_counter()
                        if gate['trigger']:
                            filtered, filtered_count, retention = support.filtered_cloud(target, reliability, 0.35)
                            depth = support.filtered_depth_support(corrupted, 0.35)
                            filtered_source, _, _ = ns['generate_observation_supported_visible_model'](
                                ns['full_model_points'], initial, depth, frame['mask_bool'], frame['K'], filtered)
                            source_count = len(filtered_source)
                            reason = 'filter_not_viable:p2p_trunk'
                            if min(source_count, filtered_count) >= 300 and retention >= 0.25:
                                candidate_T, result, status, _ = fit(ns, filtered_source, filtered, initial)
                                if status == 'completed':
                                    chosen, chosen_result = candidate_T, result
                                    used, reason = 1, 'gate_open:filtered_candidate'
                                else:
                                    reason = status + ':p2p_trunk'
                        filter_ms = (time.perf_counter() - start) * 1000.
                        outputs = [(INITIAL, initial, None, 'no_refinement', 0., len(source)),
                                   (support.TRUNK_METHOD, trunk_T, trunk_result, trunk_reason, trunk_ms, len(source)),
                                   (PROPOSED, chosen, chosen_result, reason, trunk_ms + gate_ms + filter_ms, source_count if used else len(source))]
                        for method, transform, result, status, runtime, count in outputs:
                            evaluated = make_output(ns, transform, frame['T_gt'],
                                                    result.fitness if result is not None else np.nan,
                                                    result.inlier_rmse if result is not None else np.nan, status)
                            rec = make_record(ns, meta, frame, frame_id, corruption, fraction, actual, patch, u, v, 0, method, count, evaluated, runtime)
                            rec.update(init_meta)
                            rec.update({'patch_seed': patch_seed, 'value_seed': value_seed,
                                        'selected_frame_index': fi, 'target_preparation_ms': target_ms,
                                        'trunk_source_ms': source_ms, 'trunk_icp_ms': trunk_icp_ms})
                            if method == PROPOSED:
                                support.annotate_gate(rec, gate)
                                rec.update({'candidate_used': used, 'filtered_source_points': source_count,
                                            'filtered_target_points': filtered_count, 'filtered_retention_ratio': retention,
                                            'gate_ms': gate_ms, 'filter_ms': filter_ms, 'trunk_runtime_ms': trunk_ms})
                            rows[level].append(rec)
            for level in LEVELS:
                folder = output / level
                folder.mkdir(exist_ok=True)
                pd.DataFrame(rows[level]).to_csv(folder / 'trials_checkpoint.csv', index=False)
            pd.DataFrame(frames).to_csv(output / 'frame_manifest.csv', index=False)
            print(f"  {frame_id}: complete; {sum(f['included'] for f in frames)} valid frames", flush=True)
    expected = sum(f['included'] for f in frames) * 9 * len(METHODS)
    for level, records in rows.items():
        data = pd.DataFrame(records)
        assert len(data) == expected
        assert not data.duplicated(support.KEY_COLUMNS + ['method']).any()
        assert np.isfinite(data[POSE_COLUMNS + ['ADD_S_mm']]).all().all()
        data.to_csv(output / level / 'reliability_weighted_icp_hard_validation_trials.csv', index=False)
    (output / 'run_complete.json').write_text(json.dumps({'valid_frames': expected // 27, 'rows_per_difficulty': expected}, indent=2))
    print('COMPLETE', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--objects', default=','.join(map(str, OBJECTS)))
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--frame-disjoint-validation', action='store_true')
    parser.add_argument('--main-experiment', action='store_true')
    parser.add_argument('--external-dataset-validation', action='store_true')
    parser.add_argument('--dataset-name', default='T-LESS')
    parser.add_argument('--dataset-root', type=Path, default=Path(r'D:\6D_pose\bop_datasets\tless'))
    parser.add_argument('--dataset-split', default='test_primesense')
    parser.add_argument('--depth-extension', default='.png')
    parser.add_argument('--seed-namespace', default=DEFAULT_SEED_NAMESPACE)
    parser.add_argument('--exclude-rgbd-manifest', type=Path)
    run(parser.parse_args())
