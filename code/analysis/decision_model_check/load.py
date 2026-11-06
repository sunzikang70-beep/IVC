import pandas as pd, numpy as np, json
D='/mnt/user-data/uploads/codex_paper_workspace/outputs/decoupled_development_20261020_r4/'
STATE_NAMES = [
    "action_step", "action_is_filtered", "actual_translation_change_mm",
    "actual_rotation_change_deg", "translation_over_diameter",
    "model_motion_p95_over_diameter", "projected_motion_p95_640px",
    "candidate_candidate_joint_support_fraction", "candidate_front_free_space_fraction",
    "common_support_fraction", "common_abs_p95_delta_mm",
    "common_pixel_improved_fraction", "common_pixel_worsened_fraction",
    "tail_silhouette_iou", "tail_delta_silhouette_iou",
    "tail_depth_abs_mm_p95", "tail_delta_depth_abs_mm_p95",
    "tail_forward_nn_mm_p95", "tail_delta_forward_nn_mm_p95",
    "tail_reverse_nn_mm_p95", "tail_delta_reverse_nn_mm_p95",
    "tail_boundary_640px_p95", "tail_delta_boundary_640px_p95",
    "initializer_score", "mask_score", "observable_missing_fraction",
]
def load():
    l=pd.read_csv(D+'labeled_candidate_data.csv')
    o=pd.read_csv(D+'group_oof/outer_oof_predictions.csv')
    assert (l.candidate_index.values==o.candidate_index.values).all() if 'candidate_index' in l else True
    names=json.load(open(D+'feature_schema.json'))['feature_names']
    man=pd.read_csv(D+'locked_development_instances.csv')[['cohort','frame_id','init_regime']]
    l=l.merge(man,on=['cohort','frame_id'],how='left',validate='many_to_one')
    return l,o,names
