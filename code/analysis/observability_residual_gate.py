"""Pre-specified development gate for the observability-conditioned residual correction.

Question
    Does the state-conditioned adverse-residual correction raise AR beyond ranking by the
    same base predictions, when the base model and the correction may both use the new
    local-observability state?

Stages (run in this order)
    --stage lock       write the protocol and hash the scripts and inputs, BEFORE the
                       observability features exist; refuses if they already exist.
    (run extract_observability_state_features.py)
    --stage evaluate   verify the lock, run all arms on the 22 development objects with the
                       stored 5 outer object folds, and write decision.json + report_zh.md.
    --stage selftest   pipeline check with random placeholder state; never writes a decision.

Only development data are read. No held-out T-LESS object and no YCB-V file is opened.
"""
from __future__ import annotations

import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_name, "2")

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(r"D:\6D_pose\codex_paper_workspace")
BANK = ROOT / "outputs" / "decoupled_development_20261020_r4"
FEATURES = ROOT / "outputs" / "observability_state_dev_20261105"
OUTPUT = ROOT / "outputs" / "observability_gate_20261105"
EXTRACTOR = Path(__file__).resolve().with_name("extract_observability_state_features.py")

AR_TARGETS = ["delta_AR_VSD", "delta_AR_MSSD", "delta_AR_MSPD"]
REG_TARGETS = AR_TARGETS + ["delta_ADD_S_mm"]
CLASSES = ["adds_regression", "vsd_regression", "mssd_regression", "mspd_regression", "mean_ar_gain"]
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
POSE_FEATURES = ["inliers", "inlier_fraction", "p2plane_rms_mm", "inv_cond", "log10_lmin", "n_weak",
                 "rot_block_min", "trans_block_min", "std_weak_mm", "std_trace_mm"]
CANDIDATE_OBS = (["obs_" + f for f in POSE_FEATURES]
                 + ["upd_rayleigh", "upd_weak_fraction", "upd_std_along_mm", "upd_snr"]
                 + ["ms_valid_runs", "ms_spread_norm", "ms_to_trunk_norm", "ms_frac_agree", "ms_rmse_std_mm"]
                 + ["ms_cand_to_medoid_norm", "ms_cand_toward_medoid_norm"])
INITIAL_OBS = ["obs0_" + f for f in POSE_FEATURES]
DELTA_OBS = ["dobs_" + f for f in POSE_FEATURES]
OBS_NAMES = CANDIDATE_OBS + INITIAL_OBS + DELTA_OBS

BASE_PARAMS = dict(max_iter=60, max_leaf_nodes=7, min_samples_leaf=25, learning_rate=0.07,
                   l2_regularization=5.0, random_state=20261019)
QUANTILE_PARAMS = dict(max_iter=60, max_leaf_nodes=5, min_samples_leaf=40, learning_rate=0.06,
                       l2_regularization=5.0, max_bins=64, random_state=20261019)
UTILITY_WEIGHT = 0.005
RESIDUAL_QUANTILE = 0.75
COVERAGES = np.round(np.linspace(0.05, 1.0, 20), 2)

ARMS = {
    "A":         dict(base="X191",     residual=None,  role="ranking by base predicted utility, no correction"),
    "A_obs":     dict(base="X191+OBS", residual=None,  role="same, base model also given the observability state"),
    "D":         dict(base="X191",     residual="Z26", role="current SARef correction (26 state features)"),
    "D_obs":     dict(base="X191",     residual="Z26+OBS", role="correction given the observability state, base without it"),
    "SARef_obs": dict(base="X191+OBS", residual="Z26+OBS", role="candidate method: base and correction both given the state"),
}
PROTOCOL = dict(
    version="observability-residual-gate-v1",
    data="22 T-LESS development objects of decoupled_development_20261020_r4 only; 1466 conditions x 6 candidates",
    outer_folds="the 5 object folds stored in group_oof/outer_oof_predictions.csv (outer_fold column)",
    nested=("inside each outer training set: base models refit; 3 GroupKFold object folds give the out-of-object "
            "base predictions and classifier scores used to fit the residual model; 3 further object cross-fits "
            "give the nonnegative offset b; held-out objects are never used for fitting"),
    base_model=dict(regressors=REG_TARGETS, classifiers=CLASSES, params=BASE_PARAMS,
                    preprocessing="median imputation with indicators, standardisation, clip to [-20, 20]",
                    weights="each object total weight 1, split equally over its physical instances, then rows"),
    utility="u = mean(dAR_VSD, dAR_MSSD, dAR_MSPD) - 0.005 * dADD-S / 5 mm",
    residual=dict(target="adverse utility residual r = u_hat - u (out-of-object)", quantile=RESIDUAL_QUANTILE,
                  params=QUANTILE_PARAMS, inputs="state features + 4 predicted changes + 5 classifier scores",
                  width="w = max(0, q_hat(z) + b); score L = u_hat - w"),
    arms=ARMS,
    state_sets={"Z26": "26 STATE_NAMES of the paper", "OBS": OBS_NAMES},
    selection=("per condition: argmax score over candidates with features_valid, pose_changed and "
               "observation_support >= 0.01; conditions ranked globally by that score; the top c accepted"),
    metric=("object-equal mean dAR (pp, mean of VSD/MSSD/MSPD AR changes, 0 if not accepted) averaged over "
            "coverage c = 0.05, 0.10, ..., 1.00 (gain AUC)"),
    bootstrap=dict(unit="object", draws=20000, seed=20261104, interval="percentile 95%"),
    primary="SARef_obs - A_obs",
    co_primary="SARef_obs - A",
    decision="PASS iff the 95% lower bound of BOTH differences in gain AUC is > 0; otherwise FAIL and stop",
    after_pass="freeze SARef_obs; confirm on a dataset never used before, under a protocol fixed before running",
    after_fail="no held-out evaluation; the contribution statement must be revised to what the data support",
    reported_not_decisive=["D - A (reproduction of the known result)", "D_obs - A_obs", "A_obs - A",
                           "max gain, AR-drop and ADD-S-increase rates at matched coverage, gain at matched AR-drop rate"],
    omitted_for_all_arms=["training-support (domain) gate", "risk / ADD-S upper / component / branch gates",
                          "low-support fallback width", "probability calibration and the 0.064 bias guard"],
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------------- data

def load_development(bank: Path):
    data = pd.read_csv(bank / "labeled_candidate_data.csv", float_precision="round_trip")
    names = json.loads((bank / "feature_schema.json").read_text())["feature_names"]
    oof = pd.read_csv(bank / "group_oof" / "outer_oof_predictions.csv", usecols=["candidate_index", "object_id", "outer_fold"])
    fold = oof.drop_duplicates("object_id").set_index("object_id").outer_fold
    if oof.groupby("object_id").outer_fold.nunique().max() != 1:
        raise ValueError("Object split across outer folds")
    data = data.copy()
    data["outer_fold"] = data.object_id.map(fold).astype(int)
    if len(names) != 191 or len(data) != 8796 or data.object_id.nunique() != 22 or data.target_id.nunique() != 1466:
        raise ValueError("Development bank differs from the locked r4 bank")
    data["true_mean"] = data[AR_TARGETS].mean(axis=1)
    data["true_utility"] = data.true_mean - UTILITY_WEIGHT * data.delta_ADD_S_mm / 5.
    return data, names


def attach_observability(data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    keys = ["cohort", "frame_id", "input_depth_condition"]
    initial = features.loc[features.method.eq("protocolB_e2e_initial_pose"), keys + ["obs_" + f for f in POSE_FEATURES]]
    initial = initial.rename(columns={"obs_" + f: "obs0_" + f for f in POSE_FEATURES})
    candidate = features.loc[~features.method.eq("protocolB_e2e_initial_pose"), keys + ["method"] + CANDIDATE_OBS]
    candidate = candidate.rename(columns={"method": "action"})
    merged = data.merge(candidate, on=keys + ["action"], how="left", validate="one_to_one")
    merged = merged.merge(initial, on=keys, how="left", validate="many_to_one")
    deltas = pd.DataFrame({"dobs_" + f: merged["obs_" + f] - merged["obs0_" + f] for f in POSE_FEATURES}, index=merged.index)
    merged = pd.concat([merged, deltas], axis=1)
    if merged[CANDIDATE_OBS[0]].isna().all() or len(merged) != len(data):
        raise ValueError("Observability features did not attach")
    return merged


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    """Each object weighs 1 in total, split equally over its physical instances, then over rows."""
    per_object_instances = frame.groupby("object_id").frame_id.transform("nunique").to_numpy()
    rows_in_instance = frame.groupby(["object_id", "frame_id"]).frame_id.transform("size").to_numpy()
    return 1. / per_object_instances / rows_in_instance


# ----------------------------------------------------------------------------- models

class Prep:
    """Median imputation + missing indicators + standardisation + clipping, fitted on training rows."""

    def fit(self, X: np.ndarray):
        X = np.where(np.isfinite(X), X, np.nan)
        self.median = np.nanmedian(np.where(np.isnan(X).all(axis=0), 0., X), axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.)
        self.indicator = np.isnan(X).any(axis=0)
        Z = self._raw(X)
        self.mean = Z.mean(axis=0)
        self.scale = Z.std(axis=0)
        self.keep = self.scale > 1e-12
        self.scale = np.where(self.keep, self.scale, 1.)
        return self

    def _raw(self, X):
        filled = np.where(np.isnan(X), self.median, X)
        return np.hstack([filled, np.isnan(X[:, self.indicator]).astype(float)])

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.where(np.isfinite(X), X, np.nan)
        Z = np.clip((self._raw(X) - self.mean) / self.scale, -20., 20.)
        return Z[:, self.keep]


def fit_base(train: pd.DataFrame, names):
    fit = train.loc[train.pose_changed.astype(bool) & train.features_valid.astype(bool)]
    prep = Prep().fit(fit[names].to_numpy(dtype=float))
    X = prep.transform(fit[names].to_numpy(dtype=float))
    w = sample_weights(fit)
    regressors = {t: HistGradientBoostingRegressor(**BASE_PARAMS).fit(X, fit[t].to_numpy(), sample_weight=w)
                  for t in REG_TARGETS}
    classifiers = {}
    for c in CLASSES:
        y = fit[c].to_numpy().astype(int)
        classifiers[c] = (HistGradientBoostingClassifier(**BASE_PARAMS).fit(X, y, sample_weight=w)
                          if len(np.unique(y)) > 1 else float(y.mean()))
    return dict(prep=prep, names=list(names), regressors=regressors, classifiers=classifiers)


def predict_base(model, frame: pd.DataFrame) -> pd.DataFrame:
    X = model["prep"].transform(frame[model["names"]].to_numpy(dtype=float))
    out = {"pred_" + t: m.predict(X) for t, m in model["regressors"].items()}
    for c, m in model["classifiers"].items():
        out["prob_" + c] = m.predict_proba(X)[:, 1] if not isinstance(m, float) else np.full(len(frame), m)
    out = pd.DataFrame(out, index=frame.index)
    out["u_hat"] = out[["pred_" + t for t in AR_TARGETS]].mean(axis=1) - UTILITY_WEIGHT * out.pred_delta_ADD_S_mm / 5.
    return out


def inner_oof(train: pd.DataFrame, names) -> pd.DataFrame:
    parts = []
    for a, b in GroupKFold(n_splits=3).split(train, groups=train.object_id.to_numpy()):
        fitting, held = train.iloc[a], train.iloc[b]
        if set(fitting.object_id) & set(held.object_id):
            raise ValueError("Object leakage in inner folds")
        parts.append(predict_base(fit_base(fitting, names), held))
    return pd.concat(parts).loc[train.index]


def residual_matrix(frame: pd.DataFrame, predictions: pd.DataFrame, state) -> np.ndarray:
    columns = [frame[state].to_numpy(dtype=float),
               predictions[["pred_" + t for t in REG_TARGETS]].to_numpy(),
               predictions[["prob_" + c for c in CLASSES]].to_numpy()]
    return np.hstack(columns)


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return float(values[np.searchsorted(cumulative, q, side="left").clip(0, len(values) - 1)])


def fit_residual(train: pd.DataFrame, oof: pd.DataFrame, state):
    mask = (train.pose_changed.astype(bool) & train.features_valid.astype(bool)).to_numpy()
    fit, pred = train.loc[mask], oof.loc[mask]
    residual = (pred.u_hat - fit.true_utility).to_numpy()
    w = sample_weights(fit)

    def quantile_model(rows):
        prep = Prep().fit(residual_matrix(fit.iloc[rows], pred.iloc[rows], state))
        model = HistGradientBoostingRegressor(loss="quantile", quantile=RESIDUAL_QUANTILE, **QUANTILE_PARAMS)
        model.fit(prep.transform(residual_matrix(fit.iloc[rows], pred.iloc[rows], state)), residual[rows], sample_weight=w[rows])
        return prep, model

    crossfit = np.full(len(fit), np.nan)
    for a, b in GroupKFold(n_splits=3).split(fit, groups=fit.object_id.to_numpy()):
        prep, model = quantile_model(a)
        crossfit[b] = model.predict(prep.transform(residual_matrix(fit.iloc[b], pred.iloc[b], state)))
    offset = max(0., weighted_quantile(residual - crossfit, w, RESIDUAL_QUANTILE))
    prep, model = quantile_model(np.arange(len(fit)))
    return dict(prep=prep, model=model, offset=offset, state=list(state))


def apply_residual(residual, frame: pd.DataFrame, predictions: pd.DataFrame) -> np.ndarray:
    q = residual["model"].predict(residual["prep"].transform(residual_matrix(frame, predictions, residual["state"])))
    width = np.maximum(0., q + residual["offset"])
    return predictions.u_hat.to_numpy() - width


def run_arms(data: pd.DataFrame, names, log=lambda m: print(m, flush=True)) -> pd.DataFrame:
    base_sets = {"X191": list(names), "X191+OBS": list(names) + OBS_NAMES}
    state_sets = {"Z26": STATE_NAMES, "Z26+OBS": STATE_NAMES + OBS_NAMES}
    scores = pd.DataFrame(index=data.index, columns=list(ARMS), dtype=float)
    offsets = []
    start = time.monotonic()
    for fold in sorted(data.outer_fold.unique()):
        train, held = data.loc[data.outer_fold.ne(fold)], data.loc[data.outer_fold.eq(fold)]
        if set(train.object_id) & set(held.object_id):
            raise ValueError("Outer object leakage")
        for base_name, features in base_sets.items():
            arms = {arm: spec for arm, spec in ARMS.items() if spec["base"] == base_name}
            outer = predict_base(fit_base(train, features), held)
            oof = inner_oof(train, features) if any(spec["residual"] for spec in arms.values()) else None
            for arm, spec in arms.items():
                if spec["residual"] is None:
                    scores.loc[held.index, arm] = outer.u_hat.to_numpy()
                else:
                    residual = fit_residual(train, oof, state_sets[spec["residual"]])
                    scores.loc[held.index, arm] = apply_residual(residual, held, outer)
                    offsets.append(dict(outer_fold=int(fold), arm=arm, offset=residual["offset"]))
            log(f"outer fold {fold} base {base_name}: {', '.join(arms)}; elapsed {time.monotonic()-start:.0f}s")
    scores.attrs["offsets"] = offsets
    return scores


# ----------------------------------------------------------------------------- evaluation

def selection_table(data: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    valid = (data.features_valid.astype(bool) & data.pose_changed.astype(bool) & data.observation_support.ge(.01)).to_numpy()
    frame = data[["target_id", "object_id", "true_mean", "delta_ADD_S_mm"]].assign(_s=np.where(valid, score, -np.inf))
    best = frame.loc[frame.groupby("target_id")._s.idxmax()].set_index("target_id")
    targets = data.drop_duplicates("target_id").set_index("target_id")[["object_id"]]
    return targets.join(best[["_s", "true_mean", "delta_ADD_S_mm"]])


def per_object_curves(table: pd.DataFrame, objects):
    finite = np.isfinite(table._s.to_numpy())
    order = np.argsort(-np.where(finite, table._s.to_numpy(), -np.inf), kind="mergesort")
    n = len(table)
    gains = np.zeros((len(COVERAGES), len(objects)))
    rows = []
    for j, c in enumerate(COVERAGES):
        accept = np.zeros(n, bool)
        accept[order[:int(round(c * n))]] = True
        accept &= finite
        g = np.where(accept, table.true_mean.to_numpy(), 0.)
        gains[j] = pd.Series(g).groupby(table.object_id.to_numpy()).mean().reindex(objects).to_numpy() * 100.
        accepted = table.loc[accept]
        rows.append(dict(coverage=c, accepted=int(accept.sum()), gain_pp=gains[j].mean(),
                         ar_drop_rate=float((accepted.true_mean < 0).mean()) if len(accepted) else np.nan,
                         adds_increase_rate=float((accepted.delta_ADD_S_mm > 1e-9).mean()) if len(accepted) else np.nan))
    return gains, pd.DataFrame(rows)


def evaluate_scores(data: pd.DataFrame, scores: pd.DataFrame):
    objects = np.sort(data.object_id.unique())
    rng = np.random.default_rng(PROTOCOL["bootstrap"]["seed"])
    draws = rng.integers(0, len(objects), (PROTOCOL["bootstrap"]["draws"], len(objects)))
    per_object, curves = {}, []
    for arm in scores.columns:
        gains, curve = per_object_curves(selection_table(data, scores[arm].to_numpy()), objects)
        per_object[arm] = gains.mean(axis=0)            # per-object gain AUC
        curves.append(curve.assign(arm=arm))
    curves = pd.concat(curves, ignore_index=True)
    comparisons = [("SARef_obs", "A_obs"), ("SARef_obs", "A"), ("D", "A"), ("D_obs", "A_obs"), ("D_obs", "A"), ("A_obs", "A")]
    rows = []
    for first, second in comparisons:
        diff = per_object[first] - per_object[second]
        boot = diff[draws].mean(axis=1)
        rows.append(dict(comparison=f"{first} - {second}", auc_difference_pp=diff.mean(),
                         ci95_low=float(np.quantile(boot, .025)), ci95_high=float(np.quantile(boot, .975)),
                         objects_higher=int((diff > 1e-12).sum()), objects=len(objects)))
    summary = pd.DataFrame([dict(arm=arm, gain_auc_pp=per_object[arm].mean(),
                                 max_gain_pp=curves.loc[curves.arm.eq(arm), "gain_pp"].max())
                            for arm in scores.columns])
    return summary, pd.DataFrame(rows), curves, pd.DataFrame(per_object, index=objects)


def matched_risk_table(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, curve in curves.groupby("arm", sort=False):
        for limit in [0.03, 0.05, 0.08, 0.10]:
            feasible = curve.loc[curve.ar_drop_rate <= limit]
            rows.append(dict(arm=arm, ar_drop_limit=limit, best_gain_pp=feasible.gain_pp.max() if len(feasible) else np.nan))
    return pd.DataFrame(rows)


def decide(comparisons: pd.DataFrame) -> dict:
    table = comparisons.set_index("comparison")
    primary, co = table.loc["SARef_obs - A_obs"], table.loc["SARef_obs - A"]
    passed = bool(primary.ci95_low > 0 and co.ci95_low > 0)
    return dict(decision="PASS" if passed else "FAIL",
                primary=dict(comparison="SARef_obs - A_obs", **{k: float(primary[k]) for k in ["auc_difference_pp", "ci95_low", "ci95_high"]}),
                co_primary=dict(comparison="SARef_obs - A", **{k: float(co[k]) for k in ["auc_difference_pp", "ci95_low", "ci95_high"]}),
                rule=PROTOCOL["decision"])


def write_report(path: Path, summary, comparisons, curves, matched, decision, selftest=False):
    def fmt(frame):
        return frame.to_string(index=False, float_format=lambda v: f"{v:.3f}")
    picks = curves.loc[curves.coverage.isin([0.3, 0.5, 0.7, 1.0])]
    lines = ["# 可观测性状态残差修正：开发集门槛" + ("（SELFTEST，随机占位特征，无效）" if selftest else ""), "",
             f"- 判定：**{decision['decision']}**（规则：两项差值的 95% 下界都 > 0）",
             f"- 主比较 SARef_obs − A_obs：{decision['primary']['auc_difference_pp']:+.3f} pp "
             f"[{decision['primary']['ci95_low']:+.3f}, {decision['primary']['ci95_high']:+.3f}]",
             f"- 共同主比较 SARef_obs − A：{decision['co_primary']['auc_difference_pp']:+.3f} pp "
             f"[{decision['co_primary']['ci95_low']:+.3f}, {decision['co_primary']['ci95_high']:+.3f}]", "",
             "## 各臂增益 AUC（物体等权，pp）", "```text", fmt(summary), "```", "",
             "## 配对差值（物体聚类 bootstrap，20000 次）", "```text", fmt(comparisons), "```", "",
             "## 匹配覆盖率", "```text", fmt(picks[["arm", "coverage", "accepted", "gain_pp", "ar_drop_rate", "adds_increase_rate"]]), "```", "",
             "## 在 AR 下降率上限内的最大增益", "```text", fmt(matched.pivot(index="arm", columns="ar_drop_limit", values="best_gain_pp").reset_index()), "```", "",
             "## 范围", "- 只用 22 个开发物体；所有臂共享同一基学习器实现、权重、外层折和选择规则。",
             "- 所有臂都省略了训练支持度门、风险/ADD-S/分量/分支门、低支持回退和概率校准，因此 D 的数值与论文中的 D 不完全相同。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- stages

def lock(args):
    if (args.features / "observability_features.csv").exists():
        raise FileExistsError("Observability features already exist; the gate must be locked before they are computed")
    args.output.mkdir(parents=True, exist_ok=False)
    sys_path = Path(__file__).resolve()
    record = dict(status="LOCKED_BEFORE_OBSERVABILITY_FEATURES", locked_at_utc=now(), protocol=PROTOCOL,
                  hashes={str(path): sha256(path) for path in [
                      sys_path, EXTRACTOR, args.bank / "labeled_candidate_data.csv", args.bank / "feature_schema.json",
                      args.bank / "group_oof" / "outer_oof_predictions.csv", args.bank / "inputs.csv", args.bank / "trials.csv"]})
    import extract_observability_state_features as extractor
    record["extractor_config"] = extractor.CONFIG
    record["extractor_config_sha256"] = extractor.config_digest()
    (args.output / "lock.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print("LOCKED", args.output / "lock.json")


def evaluate(args):
    record = json.loads((args.output / "lock.json").read_text(encoding="utf-8"))
    if (args.output / "decision.json").exists():
        raise FileExistsError("Decision already written; the gate is evaluated once")
    for path, expected in record["hashes"].items():
        if sha256(Path(path)) != expected:
            raise ValueError("Changed after lock: " + path)
    if record["protocol"] != json.loads(json.dumps(PROTOCOL)):
        raise ValueError("Protocol in the script differs from the locked protocol")
    state = json.loads((args.features / "state.json").read_text(encoding="utf-8"))
    if state.get("status") != "COMPLETE" or state.get("smoke"):
        raise ValueError("Observability extraction is not a complete full run")
    if state["config_sha256"] != record["extractor_config_sha256"] or state["created_at_utc"] <= record["locked_at_utc"]:
        raise ValueError("Features were not produced by the locked extractor after the lock")
    if sha256(args.features / "observability_features.csv") != state["features_sha256"]:
        raise ValueError("Feature file changed after extraction")
    data, names = load_development(args.bank)
    data = attach_observability(data, pd.read_csv(args.features / "observability_features.csv"))
    log_path = args.output / "evaluate.log"
    with log_path.open("w", encoding="utf-8") as log:
        def echo(message):
            print(message, flush=True)
            log.write(message + "\n")
        scores = run_arms(data, names, echo)
    finish(args.output, data, scores, selftest=False)


def selftest(args):
    data, names = load_development(args.bank)
    rng = np.random.default_rng(1)
    placeholder = data[["cohort", "frame_id", "input_depth_condition", "action"]].rename(columns={"action": "method"}).copy()
    for column in CANDIDATE_OBS:
        placeholder[column] = rng.normal(size=len(placeholder))
    initial = placeholder.drop_duplicates(["cohort", "frame_id", "input_depth_condition"]).assign(method="protocolB_e2e_initial_pose")
    for f in POSE_FEATURES:
        initial["obs_" + f] = rng.normal(size=len(initial))
    data = attach_observability(data, pd.concat([placeholder, initial], ignore_index=True))
    output = args.output.with_name(args.output.name + "_selftest")
    output.mkdir(parents=True, exist_ok=True)
    scores = run_arms(data, names)
    finish(output, data, scores, selftest=True)


def finish(output: Path, data, scores, selftest):
    summary, comparisons, curves, per_object = evaluate_scores(data, scores)
    matched = matched_risk_table(curves)
    decision = decide(comparisons)
    if selftest:
        decision["decision"] = "SELFTEST_ONLY"
    keys = data[["candidate_index", "target_id", "object_id", "cohort", "frame_id", "input_depth_condition", "action", "outer_fold"]]
    pd.concat([keys, scores], axis=1).to_csv(output / "arm_scores.csv.gz", index=False)
    summary.to_csv(output / "arm_summary.csv", index=False)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    curves.to_csv(output / "coverage_curves.csv", index=False)
    matched.to_csv(output / "gain_at_matched_ar_drop.csv", index=False)
    per_object.to_csv(output / "per_object_gain_auc.csv", index_label="object_id")
    pd.DataFrame(scores.attrs.get("offsets", [])).to_csv(output / "residual_offsets.csv", index=False)
    decision.update(evaluated_at_utc=now(), selftest=selftest)
    (output / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output / "report_zh.md", summary, comparisons, curves, matched, decision, selftest)
    print(summary.to_string(index=False))
    print(comparisons.to_string(index=False))
    print("DECISION", decision["decision"])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["lock", "evaluate", "selftest"], required=True)
    parser.add_argument("--bank", type=Path, default=BANK)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    {"lock": lock, "evaluate": evaluate, "selftest": selftest}[arguments.stage](arguments)
