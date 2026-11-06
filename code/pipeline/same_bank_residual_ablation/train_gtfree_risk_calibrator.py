"""Train and group-CV audit a GT-free route-risk calibrator on development data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


INITIAL = "protocolB_e2e_initial_pose"
ACTIONS = (
    "protocolF_se3_trust_p2p_s025",
    "protocolF_se3_trust_p2p_s050",
    "protocolF_se3_trust_p2p_s100",
    "protocolF_se3_trust_filtered_s025",
    "protocolF_se3_trust_filtered_s050",
    "protocolF_se3_trust_filtered_s100",
)
KEYS = ["object_id", "scene_id", "img_id", "inst_id", "frame_id", "input_depth_condition"]
SCORE_FEATURES = [
    "gate_triggered", "full_update_rotation_deg", "full_update_translation_mm",
    "applied_rotation_deg", "applied_translation_mm", "score_bidir",
    "score_bidir_visibility", "score_bidir_visibility_freespace",
    "forward_robust_cost", "reverse_robust_cost", "candidate_mean_soft_support",
    "candidate_joint_support_fraction", "front_free_space_fraction",
    "behind_surface_fraction", "signed_depth_valid_fraction",
]
TRIAL_FEATURES = [
    "initializer_score", "mask_score", "pose_mask_bbox_iou", "target_points",
    "mask_pixels", "gate_mean_reliability",
    "gate_median_reliability", "gate_q10_reliability", "gate_q25_reliability",
    "gate_strong_low_fraction", "gate_opened_low_fraction",
    "gate_largest_component_fraction", "gate_largest_component_share_of_low",
    "gate_mean_local_diff_mm", "gate_q90_local_diff_mm", "filtered_source_points",
    "filtered_target_points", "filtered_retention_ratio", "candidate_used",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def build_table(scores: pd.DataFrame, trials: pd.DataFrame):
    score_keys = KEYS
    score_rows = scores[scores.method.isin((INITIAL,) + ACTIONS)].copy()
    if score_rows.duplicated(score_keys + ["method"]).any():
        raise ValueError("Duplicate score rows")
    wide_parts = []
    for method in (INITIAL,) + ACTIONS:
        subset = score_rows[score_rows.method.eq(method)].set_index(score_keys)
        values = subset[SCORE_FEATURES].copy()
        values.columns = [f"{method}__{column}" for column in values.columns]
        wide_parts.append(values)
    features = pd.concat(wide_parts, axis=1).reset_index()
    source = trials[trials.method.eq(INITIAL)].copy()
    if source.duplicated(score_keys).any():
        raise ValueError("Duplicate initial trial feature rows")
    source = source.set_index(score_keys)[TRIAL_FEATURES].reset_index()
    features = features.merge(source, on=score_keys, how="inner", validate="one_to_one")
    outcomes = trials[trials.method.isin((INITIAL,) + ACTIONS)].pivot_table(
        index=score_keys, columns="method", values="ADD_S_mm", aggfunc="first"
    ).reset_index()
    outcomes = outcomes.rename(columns={method: f"ADD_S_mm__{method}" for method in (INITIAL,) + ACTIONS})
    table = features.merge(outcomes, on=score_keys, how="inner", validate="one_to_one")
    required = [f"ADD_S_mm__{method}" for method in (INITIAL,) + ACTIONS]
    table = table.dropna(subset=required).copy()
    if table.empty:
        raise ValueError("No complete calibrator rows")
    return table


def feature_matrix(table: pd.DataFrame, mode: str):
    if mode == "wide":
        feature_columns = [
            column for column in table.columns
            if "__" in column and not column.startswith("ADD_S_mm__")
        ] + TRIAL_FEATURES
        numeric = table[feature_columns].apply(pd.to_numeric, errors="coerce")
    elif mode == "differential":
        feature_columns = []
        values = {}
        for action in ACTIONS:
            for column in SCORE_FEATURES:
                name = f"delta_{action}__{column}"
                feature_columns.append(name)
                values[name] = table[f"{action}__{column}"] - table[f"{INITIAL}__{column}"]
        for column in TRIAL_FEATURES:
            feature_columns.append(column)
            values[column] = table[column]
        numeric = pd.DataFrame(values, index=table.index).apply(pd.to_numeric, errors="coerce")
    else:
        raise ValueError(f"Unknown feature mode: {mode}")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
    return numeric.to_numpy(dtype=np.float64), feature_columns


def make_models(X, y_delta, train_index, action):
    delta = y_delta[train_index]
    good = (delta <= 0.0).astype(int)
    if len(np.unique(good)) < 2:
        classifier = None
        prior = float(np.mean(good))
    else:
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=20261016),
        )
        classifier.fit(X[train_index], good)
        prior = None
    regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    regressor.fit(X[train_index], delta)
    return classifier, regressor, prior


def predict_action(models, X):
    classifier, regressor, prior = models
    probability = np.full(len(X), prior if prior is not None else 0.5, dtype=np.float64)
    if classifier is not None:
        probability = classifier.predict_proba(X)[:, 1]
    prediction = regressor.predict(X)
    return probability, prediction


def required_improvement(action: str, args) -> float:
    if args.acceptance_policy == "step_aware":
        if action.endswith("s025"):
            return args.min_predicted_improvement_s025_mm
        if action.endswith("s050"):
            return args.min_predicted_improvement_s050_mm
        if action.endswith("s100"):
            return args.min_predicted_improvement_s100_mm
    return args.min_predicted_improvement_mm


def summarize(oof: pd.DataFrame):
    fixed_columns = {
        action: f"{action}__ADD_S_mm"
        for action in ACTIONS
    }
    rows = []
    for condition, group in oof.groupby("input_depth_condition", sort=False):
        initial = group["initial_ADD_S_mm"].to_numpy()
        selected = group["selected_ADD_S_mm"].to_numpy()
        row = {
            "input_depth_condition": condition,
            "targets": len(group),
            "initial_mean_ADD_S_mm": float(np.mean(initial)),
            "selected_mean_ADD_S_mm": float(np.mean(selected)),
            "selected_delta_vs_initial_mm": float(np.mean(selected - initial)),
            "selected_regression_rate": float(np.mean(selected > initial)),
            "route_coverage": float(np.mean(group.selected_action.ne(INITIAL))),
        }
        row.update({
            f"{action}_fixed_mean_ADD_S_mm": float(group[column].mean())
            for action, column in fixed_columns.items()
        })
        rows.append(row)
    pooled = {
        "input_depth_condition": "pooled",
        "targets": len(oof),
        "initial_mean_ADD_S_mm": float(oof.initial_ADD_S_mm.mean()),
        "selected_mean_ADD_S_mm": float(oof.selected_ADD_S_mm.mean()),
        "selected_delta_vs_initial_mm": float((oof.selected_ADD_S_mm - oof.initial_ADD_S_mm).mean()),
        "selected_regression_rate": float((oof.selected_ADD_S_mm > oof.initial_ADD_S_mm).mean()),
        "route_coverage": float(oof.selected_action.ne(INITIAL).mean()),
    }
    pooled.update({
        f"{action}_fixed_mean_ADD_S_mm": float(oof[column].mean())
        for action, column in fixed_columns.items()
    })
    rows.append(pooled)
    return pd.DataFrame(rows)


def run(args):
    output = args.output
    if output.exists():
        raise FileExistsError(f"Preserve existing output: {output}")
    output.mkdir(parents=True)
    scores = pd.read_csv(args.scores)
    trials = pd.read_csv(args.trials)
    table = build_table(scores, trials)
    X, feature_columns = feature_matrix(table, args.feature_mode)
    y_initial = table[f"ADD_S_mm__{INITIAL}"].to_numpy(dtype=np.float64)
    y_actions = {action: table[f"ADD_S_mm__{action}"].to_numpy(dtype=np.float64) - y_initial for action in ACTIONS}
    groups = table.object_id.to_numpy()
    oof = table[KEYS].copy()
    oof["initial_ADD_S_mm"] = y_initial
    for action in ACTIONS:
        oof[f"{action}__ADD_S_mm"] = table[f"ADD_S_mm__{action}"].to_numpy(dtype=np.float64)
        oof[f"{action}__p_good"] = np.nan
        oof[f"{action}__predicted_delta_mm"] = np.nan
    oof["selected_action"] = INITIAL

    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    fold_records = []
    for fold, (train_index, test_index) in enumerate(splitter.split(X, groups=groups)):
        fold_models = {}
        for action in ACTIONS:
            models = make_models(X, y_actions[action], train_index, action)
            fold_models[action] = models
            probability, prediction = predict_action(models, X[test_index])
            oof.loc[test_index, f"{action}__p_good"] = probability
            oof.loc[test_index, f"{action}__predicted_delta_mm"] = prediction
        for row_index in test_index:
            candidates = []
            for action in ACTIONS:
                probability = float(oof.loc[row_index, f"{action}__p_good"])
                prediction = float(oof.loc[row_index, f"{action}__predicted_delta_mm"])
                if probability >= args.min_good_probability and prediction <= -required_improvement(action, args):
                    candidates.append((prediction, -probability, action))
            selected = min(candidates)[2] if candidates else INITIAL
            oof.loc[row_index, "selected_action"] = selected
        fold_records.append({"fold": fold, "train_objects": sorted(set(groups[train_index].tolist())),
                             "test_objects": sorted(set(groups[test_index].tolist())),
                             "train_rows": len(train_index), "test_rows": len(test_index)})

    oof["selected_ADD_S_mm"] = oof["initial_ADD_S_mm"]
    for action in ACTIONS:
        selected = oof.selected_action.eq(action)
        oof.loc[selected, "selected_ADD_S_mm"] = oof.loc[selected, f"{action}__ADD_S_mm"]
    summary = summarize(oof)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    summary.to_csv(output / "oof_summary.csv", index=False)
    pd.DataFrame(fold_records).to_json(output / "group_folds.json", orient="records", indent=2)

    full_models = {action: make_models(X, y_actions[action], np.arange(len(X)), action) for action in ACTIONS}
    artifact = {
        "schema": "gtfree-risk-calibrator-v2",
        "status": "DEVELOPMENT_ONLY_NOT_FROZEN",
        "features": feature_columns,
        "feature_mode": args.feature_mode,
        "actions": ACTIONS,
        "initial_action": INITIAL,
        "min_good_probability": args.min_good_probability,
        "min_predicted_improvement_mm": args.min_predicted_improvement_mm,
        "acceptance_policy": args.acceptance_policy,
        "step_aware_min_predicted_improvement_mm": {
            "s025": args.min_predicted_improvement_s025_mm,
            "s050": args.min_predicted_improvement_s050_mm,
            "s100": args.min_predicted_improvement_s100_mm,
        },
        "group_column": "object_id",
        "source_scores_sha256": digest(args.scores),
        "source_trials_sha256": digest(args.trials),
        "gt_usage": "labels_only_for_offline_development; inference_features_exclude_GT_fields",
        "real_platform_validation": "cancelled_by_author",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump({"models": full_models, "artifact": artifact}, output / "risk_calibrator.joblib")
    (output / "calibrator_manifest.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    report = [
        "# GT-free Risk Calibrator Development Report",
        "",
        "- Status: DEVELOPMENT_ONLY_NOT_FROZEN",
        "- Training labels use ADD-S only offline; route features contain no GT fields.",
        "- Validation: object-group 5-fold out-of-fold evaluation.",
        "- Actions: initializer plus P2P/reliability-filtered candidates at s025, s050, and s100.",
        f"- Feature mode: {args.feature_mode}.",
        f"- Acceptance policy: {args.acceptance_policy}; p_good >= {args.min_good_probability:.2f}.",
        f"- Minimum predicted improvement: uniform={args.min_predicted_improvement_mm:.3f} mm; step-aware s025/s050/s100={args.min_predicted_improvement_s025_mm:.3f}/{args.min_predicted_improvement_s050_mm:.3f}/{args.min_predicted_improvement_s100_mm:.3f} mm.",
        "- Proposal generation is not used by this first calibrator prototype.",
        "- Real-platform validation: cancelled by author.",
        "",
        "## OOF Summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Guardrail",
        "",
        "This artifact is a development calibrator. It must not be applied to the official T-LESS test results or frozen as a method until a separate object-disjoint validation protocol is approved.",
    ]
    (output / "report_zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print("GT-free risk calibrator development complete", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path(r"D:\6D_pose\codex_paper_workspace\outputs\se3_trust_region_geometric_development_20260916\se3_trust_region_scores.csv"))
    parser.add_argument("--trials", type=Path, default=Path(r"D:\6D_pose\codex_paper_workspace\outputs\se3_trust_region_geometric_development_20260916\se3_trust_region_trials.csv"))
    parser.add_argument("--output", type=Path, default=Path(r"D:\6D_pose\codex_paper_workspace\outputs\gtfree_risk_calibrator_20261016_v3"))
    parser.add_argument("--min-good-probability", type=float, default=0.90)
    parser.add_argument("--min-predicted-improvement-mm", type=float, default=0.20)
    parser.add_argument("--acceptance-policy", choices=("uniform", "step_aware"), default="uniform")
    parser.add_argument("--min-predicted-improvement-s025-mm", type=float, default=0.20)
    parser.add_argument("--min-predicted-improvement-s050-mm", type=float, default=0.60)
    parser.add_argument("--min-predicted-improvement-s100-mm", type=float, default=1.00)
    parser.add_argument("--feature-mode", choices=("wide", "differential"), default="wide")
    run(parser.parse_args())
