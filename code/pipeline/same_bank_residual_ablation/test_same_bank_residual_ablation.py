"""Focused checks for fixed/state/joint post-hoc selection controls."""
import unittest
import numpy as np
import pandas as pd

import run_same_bank_residual_ablation as a


class AblationTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"target_id": [0, 0, 1, 1], "action": ["z", "a", "z", "a"],
                                  "pose_changed": [True] * 4, "features_valid": [True] * 4,
                                  "observation_support": [.5] * 4}, index=[5, 8, 9, 12])
        self.p = pd.DataFrame(index=self.data.index)
        for t in a.base.TARGETS[:3]:
            self.p["pred_" + t] = .03
            self.p["lcb_" + t] = .02
        self.p["pred_delta_ADD_S_mm"] = -.5
        self.p["ucb_delta_ADD_S_mm"] = -.3
        self.p["training_support"] = .5
        self.p["adds_risk_upper"] = .1
        self.p["lcb_delta_AR_mean"] = .02
        self.p["lcb_delta_joint_utility"] = .0203
        self.plan = {"joint_continuous_weight": .005, "continuous_minimum_ADD_gain_mm": .25}
        self.policy = dict(risk_limit=.3, component_loss=.01, continuous_weight=.005, disabled=False)
        self.bundle = {"base_bundle": {"calibration": {"corrections": {s: [.01, .02, .03, .1] for s in ["a", "z"]}}}}

    def test_fixed_and_raw_share_probability_and_support(self):
        raw = a.adjusted(self.data, self.p, self.bundle, "A_delta_rank")
        fixed = a.adjusted(self.data, self.p, self.bundle, "B_fixed_residual")
        np.testing.assert_allclose(raw.lcb_delta_AR_VSD, .03)
        np.testing.assert_allclose(fixed.lcb_delta_AR_VSD, .02)
        np.testing.assert_allclose(fixed.ucb_delta_ADD_S_mm, -.4)
        for c in ["adds_risk_upper", "training_support", "pred_delta_ADD_S_mm"]:
            pd.testing.assert_series_equal(raw[c], self.p[c])
            pd.testing.assert_series_equal(fixed[c], self.p[c])

    def test_legacy_and_joint_replay(self):
        for arm, variant in [("C_state_residual", "conditional_residual_legacy"), ("D_joint_full", "conditional_residual_joint")]:
            self.assertEqual(a.choose(self.data, self.p, self.policy, arm, self.plan),
                             a.res.decisions(self.data, self.p, self.policy, variant, self.plan))
        self.assertEqual(a.choose(self.data, self.p, self.policy, "D_joint_full", self.plan), {0: "a", 1: "a"})

    def test_disabled_falls_back(self):
        for arm in a.ARMS:
            self.assertEqual(a.choose(self.data, self.p, dict(self.policy, disabled=True), arm, self.plan),
                             {0: a.base.INITIAL, 1: a.base.INITIAL})

    def test_joint_gate_is_independent_from_score(self):
        p = self.p.copy()
        for t in a.base.TARGETS[:3]:
            p["lcb_" + t] = -.0001
        p["lcb_delta_AR_mean"] = -.0001
        p["lcb_delta_joint_utility"] = .0001
        p["ucb_delta_ADD_S_mm"] = -.2
        for arm in ["C_state_residual", "D_joint_legacy_gate"]:
            self.assertEqual(a.choose(self.data, p, self.policy, arm, self.plan), {0: "a", 1: "a"})
        for arm in ["C_state_joint_gate", "D_joint_full"]:
            self.assertEqual(a.choose(self.data, p, self.policy, arm, self.plan), {0: a.base.INITIAL, 1: a.base.INITIAL})

    def test_index_mismatch_fails(self):
        with self.assertRaises(ValueError):
            a.choose(self.data, self.p.reset_index(drop=True), self.policy, "C_state_residual", self.plan)

    def test_nonfinite_rejected(self):
        p = self.p.copy()
        p["lcb_delta_AR_VSD"] = np.nan
        self.assertEqual(a.choose(self.data, p, self.policy, "C_state_residual", self.plan), {0: a.base.INITIAL, 1: a.base.INITIAL})

    def test_holm(self):
        np.testing.assert_allclose(a.holm([.01, .04, .02]), [.03, .04, .04])


if __name__ == "__main__":
    unittest.main()
