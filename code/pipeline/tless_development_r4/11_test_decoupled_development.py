"""Protocol/data-boundary checks before expensive new-protocol regeneration."""
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import run_decoupled_development as runner
import train_decoupled_group_oof as train
from decoupled_checkpoint_recovery import INITIAL_DIAGNOSTICS, ZERO_FIELDS, repair_initial_diagnostics
from pose_perturbation_protocols import CAMERA, DECOUPLED, noise, paired_poses


class DecoupledDevelopmentTests(unittest.TestCase):
    def test_full_identity_lock(self):
        frame, sources = runner.load_sources(runner.ROOT)
        self.assertEqual(len(frame), 867)
        self.assertEqual(int(frame.matched_prediction.sum()), 733)
        self.assertEqual(frame.frame_id.nunique(), 743)
        self.assertEqual(set(frame.object_id), set(runner.p.OBJECTS))
        self.assertTrue(all(path.is_file() for path in sources))

    def test_reserved_images_rejected(self):
        frame = pd.DataFrame([dict(object_id=2, scene_id=1, img_id=3, init_regime="native", cohort="x", frame_id="a")])
        with self.assertRaisesRegex(ValueError, "Reserved"):
            runner.check_boundaries(frame, pd.DataFrame([dict(scene_id=1, img_id=3)]))

    def test_excluded_object_rejected(self):
        frame = pd.DataFrame([dict(object_id=5, scene_id=1, img_id=3, init_regime="native", cohort="x", frame_id="a")])
        with self.assertRaisesRegex(ValueError, "Nondevelopment"):
            runner.check_boundaries(frame, pd.DataFrame(columns=["scene_id", "img_id"]))

    def test_original_missing_ledger_is_not_inference(self):
        frame = pd.DataFrame([dict(object_id=2, scene_id=1, img_id=3, init_regime="native", cohort="x", frame_id="a", matched_prediction=0)])
        runner.check_boundaries(frame, pd.DataFrame([dict(scene_id=1, img_id=3)]))

    def test_noise_exact_old_composition(self):
        reference = np.eye(4)
        reference[:3, 3] = [.03, -.1, .8]
        for regime, (angle, shift, seed) in runner.p.REGIMES.items():
            q, u, _ = noise(angle, shift, 2, 17, seed)
            pair = paired_poses(reference, q, u)
            old, _ = runner.real.perturb_initial_pose(reference, angle, shift, 2, 17, seed)
            np.testing.assert_array_equal(pair[CAMERA], old)
            np.testing.assert_array_equal(pair[CAMERA][:3,:3], pair[DECOUPLED][:3,:3])
            self.assertAlmostEqual(np.linalg.norm(pair[DECOUPLED][:3,3]-reference[:3,3])*1000, shift)

    def test_outer_folds_never_split_instances(self):
        frame, _ = runner.load_sources(runner.ROOT)
        frame = frame.loc[frame.matched_prediction.eq(1)]
        for a, b in runner.base.folds(frame, 5):
            training, held = frame.iloc[a], frame.iloc[b]
            train.partition_record(training, held)
            self.assertFalse(set(training.frame_id)&set(held.frame_id))

    def test_partition_rejects_leakage(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            train.partition_record(pd.DataFrame({"object_id":[2,3]}), pd.DataFrame({"object_id":[3,4]}))

    def test_shadow_poison_outcomes(self):
        data = pd.DataFrame({"action":["a"], "object_id":[2], "tail_depth_abs_mm_p95":[3.], "ADD_S_mm":[.1]})
        shadow = train.observable_shadow(data)
        self.assertEqual(shadow.ADD_S_mm.iloc[0], 99999.)
        self.assertEqual(shadow.tail_depth_abs_mm_p95.iloc[0], 3.)

    def test_dict_pose_interface_is_lossless(self):
        pose = np.eye(4)
        pose[:3, 3] = [.01, -.05, .7]
        record = runner.real.support.pose_fields(pose)
        np.testing.assert_array_equal(runner.pose_from_record(record), pose)
        np.testing.assert_array_equal(runner.pose_from_record(pd.Series(record)), pose)

    def test_initial_diagnostics_match_frozen_candidate_semantics(self):
        self.assertTrue(all(INITIAL_DIAGNOSTICS[name]==0 for name in ZERO_FIELDS))
        self.assertEqual(INITIAL_DIAGNOSTICS["direction"], "initial")
        score = pd.DataFrame([dict(method=runner.base.INITIAL, **INITIAL_DIAGNOSTICS)])
        self.assertFalse(score[ZERO_FIELDS].isna().any().any())

    def test_repair_changes_only_known_initial_zeros(self):
        score = pd.DataFrame([dict(method=runner.base.INITIAL, direction="initial", trust_scale=0., **{name:np.nan for name in ZERO_FIELDS if name!="trust_scale"}),
                              dict(method=runner.base.ACTIONS[0], direction="p2p", **{name:.25 for name in ZERO_FIELDS})])
        score["score_bidir"] = [3., 2.]
        score["depth_residual"] = np.nan
        repaired = repair_initial_diagnostics(score)
        pd.testing.assert_frame_equal(score.iloc[1:], repaired.iloc[1:])
        self.assertTrue(repaired.loc[0, ZERO_FIELDS].eq(0).all())
        pd.testing.assert_series_equal(score.score_bidir, repaired.score_bidir)
        self.assertTrue(repaired.depth_residual.isna().all())

    def test_nonzero_initial_diagnostic_cannot_be_repaired(self):
        with self.assertRaisesRegex(ValueError, "Nonzero"):
            repair_initial_diagnostics(pd.DataFrame([dict(method=runner.base.INITIAL, applied_rotation_deg=1.)]))

    def test_repaired_delta_update_features_retain_magnitude(self):
        directory = runner.ROOT/"outputs/decoupled_development_20261020_r2/generation_checkpoints/object_02"
        scores = repair_initial_diagnostics(runner.read_csv(directory/"scores.csv"))
        trials = runner.read_csv(directory/"trials.csv")
        cohort = trials.cohort.iloc[0]
        features, _ = runner.base.build_features(scores.loc[scores.cohort.eq(cohort)], trials.loc[trials.cohort.eq(cohort)], cohort)
        for name in ["full_update_rotation_deg", "full_update_translation_mm", "applied_rotation_deg", "applied_translation_mm"]:
            np.testing.assert_allclose(features["delta_"+name], features["candidate_"+name], equal_nan=True)
            self.assertTrue(features["delta_"+name].notna().any())

    def test_planned_denominator_does_not_drop_missing(self):
        manifest = pd.DataFrame({"init_regime":["hard", "hard"]})
        rows = pd.DataFrame([dict(method="m", init_regime="hard", frame_id="a", object_id=2,
            input_depth_condition=condition, selected_action="m", delta_ADD_S_mm=-1., ADD_S_mm=2., AR_mean_row=.6,
            **{c:.6 for c in train.base.AR}, **{"initial_"+c:.3 for c in train.base.AR})
            for condition in ["native_depth", "local_outlier_30pct"]])
        summary = train.planned_summary(rows, manifest).iloc[0]
        self.assertEqual(summary.planned_conditions, 4)
        self.assertAlmostEqual(summary.AR_matched, .6)
        self.assertAlmostEqual(summary.AR_all_planned_missing_zero, .3)
        self.assertAlmostEqual(summary.AR_gain_all_planned_missing_zero, .15)


if __name__ == "__main__":
    unittest.main()
