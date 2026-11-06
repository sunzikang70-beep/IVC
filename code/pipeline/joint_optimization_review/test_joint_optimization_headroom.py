import unittest

import numpy as np
import pandas as pd

from audit_joint_optimization_headroom import best_choices, base


class OracleTests(unittest.TestCase):
    def test_initial_pose_is_always_available(self):
        data = pd.DataFrame(dict(target_id=[0, 0, 1, 1], action=["b", "a", "a", "b"], delta_ADD_S_mm=[0., 0., 0., 0.]))
        result = best_choices(data, [True] * 4, [-1., 0., .1, .2])
        self.assertEqual(result, {0: base.INITIAL, 1: "b"})

    def test_mask_prevents_gt_best_from_being_selected(self):
        data = pd.DataFrame(dict(target_id=[0, 0], action=["a", "b"], delta_ADD_S_mm=[0., 0.]))
        self.assertEqual(best_choices(data, [False, True], [.2, .1]), {0: "b"})

    def test_tie_is_stable_and_prefers_lower_add_delta(self):
        data = pd.DataFrame(dict(target_id=[0, 0, 0], action=["b", "a", "c"], delta_ADD_S_mm=[-1., -1., 0.]))
        self.assertEqual(best_choices(data, np.ones(3, bool), [.1] * 3), {0: "a"})

    def test_nonoracle_ties_do_not_read_labels(self):
        data = pd.DataFrame(dict(target_id=[0, 0], action=["b", "a"], delta_ADD_S_mm=[-10., 5.]))
        expected = best_choices(data, [True, True], [.1, .1], [.2, .2])
        data.delta_ADD_S_mm = [100., -100.]
        self.assertEqual(expected, {0: "a"})
        self.assertEqual(best_choices(data, [True, True], [.1, .1], [.2, .2]), expected)


if __name__ == "__main__":
    unittest.main()
