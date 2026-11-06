"""Tests for GT-blind quotas and development-only risk operating points."""
import unittest
import numpy as np
import pandas as pd

import run_matched_risk_coverage_audit as a


def example():
    rows = pd.DataFrame({"target_id": range(10), "object_id": [1] * 5 + [2] * 5,
        "scene_id": 1, "img_id": range(10), "inst_id": 0, "cohort": "native",
        "input_depth_condition": "native", "base_eligible": True, "utility": np.arange(10) / 10 + .1,
        "predicted_risk": np.arange(10) / 100, "base_proposed_action": "candidate", "selected_action": "candidate",
        "update": 1, "regression": 0, "bop_regression": 0, "ADD_S_mm": 9., "initial_ADD_S_mm": 10.,
        "delta_ADD_S_mm": -1., "AR_mean_row": .6, "AR_gain": .1})
    for metric in a.ref.base.AR:
        rows[metric], rows["initial_" + metric] = .6, .5
    return rows


class MatchedAuditTests(unittest.TestCase):
    def test_confidence_axes(self):
        t = example()
        np.testing.assert_allclose(a.confidence(t, "utility"), t.utility)
        np.testing.assert_allclose(a.confidence(t, "predicted_risk"), -t.predicted_risk)

    def test_noncontiguous_rank_index(self):
        t = example().iloc[[1, 3, 8]]
        self.assertEqual(a.rank_indices(t, "utility").tolist(), [8, 3, 1])

    def test_exact_top_k(self):
        tables = {arm: example() for arm in a.ARMS}
        masks, records = a.quota_masks(tables, "utility", .3, "pooled_exact")
        for mask in masks.values():
            np.testing.assert_array_equal(np.where(mask)[0], [7, 8, 9])

    def test_capped_object_with_zero_candidates(self):
        tables = {arm: example() for arm in a.ARMS}
        tables[a.ARMS[-1]].loc[:4, "base_eligible"] = False
        masks, records = a.quota_masks(tables, "utility", .4, "object_capped")
        for mask in masks.values():
            self.assertEqual(int(mask[:5].sum()), 0)
            self.assertEqual(int(mask[5:].sum()), 2)
        self.assertTrue(records[0]["capped"])

    def test_infeasible_pooled_quota_not_relaxed(self):
        tables = {arm: example() for arm in a.ARMS}
        tables[a.ARMS[-1]].loc[:8, "base_eligible"] = False
        masks, _ = a.quota_masks(tables, "utility", .3, "pooled_exact")
        self.assertIsNone(masks)

    def test_poisoned_labels_do_not_change_quota(self):
        tables = {arm: example() for arm in a.ARMS}
        clean, _ = a.quota_masks(tables, "utility", .3, "pooled_exact")
        fake = {arm: t.assign(AR_gain=-999., regression=1, ADD_S_mm=999.) for arm, t in tables.items()}
        poisoned, _ = a.quota_masks(fake, "utility", .3, "pooled_exact")
        for arm in a.ARMS:
            np.testing.assert_array_equal(clean[arm], poisoned[arm])

    def test_reject_restores_initial_pose_metrics(self):
        t = example()
        out = a.reject(t, np.zeros(10, bool))
        np.testing.assert_allclose(out.AR_mean_row, .5)
        np.testing.assert_allclose(out.delta_ADD_S_mm, 0)
        self.assertTrue(np.isnan(a.totals(out)["accepted_risk"]))

    def test_cannot_accept_ineligible(self):
        t = example()
        t.loc[0, "base_eligible"] = False
        with self.assertRaises(ValueError):
            a.reject(t, np.ones(10, bool))

    def test_threshold_ties_and_disabled(self):
        t = example()
        self.assertEqual(int(a.threshold_mask(t, dict(basis="utility", threshold=float(t.utility.iloc[7]), disabled=False)).sum()), 3)
        self.assertEqual(int(a.threshold_mask(t, dict(disabled=True)).sum()), 0)

    def test_risk_policy_feasibility(self):
        grid = a.calibration_grid(example())
        best = a.choose_risk_policy(grid, .1)
        self.assertFalse(best["disabled"])
        self.assertEqual(best["development_risk"], 0)
        self.assertEqual(best["development_updates"], 10)

    def test_no_feasible_risk_policy_falls_back(self):
        t = example()
        t["regression"], t["delta_ADD_S_mm"], t["ADD_S_mm"] = 1, 1., 11.
        policy = a.choose_risk_policy(a.calibration_grid(t), .1)
        self.assertTrue(policy["disabled"])

    def test_identity_mismatch_not_matched(self):
        tables = {arm: example() for arm in a.ARMS}
        tables[a.ARMS[-1]].loc[0, "img_id"] = 999
        with self.assertRaises(ValueError):
            a.quota_masks(tables, "utility", .3, "pooled_exact")


if __name__ == "__main__":
    unittest.main()
