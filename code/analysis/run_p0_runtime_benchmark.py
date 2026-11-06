"""P0-2 end-to-end runtime benchmark of the frozen r4 method (read-only instrumentation).

Replays a small subset of the original 5 held-out objects through the unchanged
candidate-generation code of run_decoupled_frozen_unseen5.py, wrapping each stage with a
timer, then times frozen-model inference + acceptance decision per target condition.
Nothing in the method, model, thresholds or candidates is changed. New output directory only.

Run on the Windows machine with the project environment:
    python work\\run_p0_runtime_benchmark.py
Optional: --frames-per-object-regime 4 --inference-targets 300
"""
from __future__ import annotations
import argparse, json, os, platform, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))
import run_decoupled_frozen_unseen5 as runner  # noqa: E402

real, trust, tail, development, base, residual = (runner.real, runner.trust, runner.tail,
                                                  runner.development, runner.base, runner.residual)
TIMES: list[dict] = []
RECORDS: list[dict] = []
CUR = {"condition": -1, "depth": 0}


def timed(stage, fn, method=False):
    def wrapper(*args, **kwargs):
        if CUR["depth"] > 0:            # nested call: counted by the enclosing stage
            return fn(*args, **kwargs)
        CUR["depth"] += 1
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            CUR["depth"] -= 1
            TIMES.append(dict(condition=CUR["condition"], stage=stage, ms=(time.perf_counter() - start) * 1000.0))
    return wrapper


def patch():
    replay = development.replay

    def replay_wrapper(*args, **kwargs):
        CUR["condition"] += 1
        return timed("input_replay_EXCLUDED", replay)(*args, **kwargs)
    development.replay = replay_wrapper

    run_methods = real.run_methods

    def run_methods_wrapper(*args, **kwargs):
        out = timed("registration_p2p_and_filtered", run_methods)(*args, **kwargs)
        for r in out:
            RECORDS.append(dict(condition=CUR["condition"], method=r.get("method"),
                                **{k: r.get(k) for k in ["runtime_ms", "target_preparation_ms", "trunk_source_ms", "trunk_icp_ms"]}))
        return out
    real.run_methods = run_methods_wrapper

    loader = real.support.load_experiment_namespace

    def loader_wrapper(*args, **kwargs):
        ns = loader(*args, **kwargs)
        for name in ["depth_to_cloud", "compute_depth_reliability",
                     "generate_observation_supported_visible_model", "compute_reliability_gate_features"]:
            ns[name] = timed("feature_preparation", ns[name])
        ns["load_frame_data"] = timed("frame_loading_EXCLUDED", ns["load_frame_data"])
        return ns
    real.support.load_experiment_namespace = loader_wrapper

    runner.cKDTree = timed("feature_preparation", runner.cKDTree)
    trust.restricted_pose = timed("candidate_step_clipping", trust.restricted_pose)
    trust.geometry_scores = timed("geometric_scores", trust.geometry_scores)
    real.support.update_pose_metrics = timed("gt_evaluation_EXCLUDED", real.support.update_pose_metrics)
    tail.Open3DRaycastRenderer.render_object = timed("render_depth", tail.Open3DRaycastRenderer.render_object)
    tail.Open3DRaycastRenderer.add_object = timed("model_loading_one_time_EXCLUDED", tail.Open3DRaycastRenderer.add_object)
    for name in ["geometry_features", "common_support", "model_motion"]:
        setattr(tail, name, timed("tail_features", getattr(tail, name)))


def inference_benchmark(n_targets: int, out: Path) -> pd.DataFrame:
    import joblib
    model = joblib.load(runner.DEV_OOF / "development_candidate.joblib")
    plan = json.loads((WORK / "bop_aware_state_residual_plan_20261019.json").read_text(encoding="utf-8"))
    frozen = runner.ROOT / "outputs" / "decoupled_frozen_unseen5_20261021_r4"
    features = runner.read_csv(frozen / "observable_candidate_features.csv")
    manifest = pd.read_csv(frozen / "locked_holdout_instances.csv")
    regimes = manifest[["cohort", "frame_id", "init_regime"]].drop_duplicates()
    features = features.merge(regimes, on=["cohort", "frame_id"], validate="many_to_one")
    data = base.attach_labels(features, runner.read_csv(frozen / "holdout_bop_labels.csv")).reset_index(drop=True)
    policy = model["policies"]["conditional_residual_joint"]
    targets = data.target_id.drop_duplicates().sample(n=min(n_targets, data.target_id.nunique()), random_state=20261104)
    rows = []
    for i, target in enumerate(targets):
        part = data[data.target_id.eq(target)]
        start = time.perf_counter()
        pred = residual.predict_bundle(model["bundle"], part)
        mid = time.perf_counter()
        choice = residual.decisions(part, pred, policy, "conditional_residual_joint", plan)
        end = time.perf_counter()
        rows.append(dict(target_id=target, candidates=len(part), predict_ms=(mid - start) * 1000.0,
                         decision_ms=(end - mid) * 1000.0, warmup=i < 5, choice=choice[target]))
    start = time.perf_counter()
    pred_all = residual.predict_bundle(model["bundle"], data)
    batch_ms = (time.perf_counter() - start) * 1000.0
    table = pd.DataFrame(rows)
    table.attrs["batch_ms_per_target"] = batch_ms / data.target_id.nunique()
    return table


def describe(series: pd.Series) -> dict:
    s = series.dropna()
    return dict(n=int(len(s)), mean=float(s.mean()), median=float(s.median()), p95=float(s.quantile(.95)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=runner.ROOT / "outputs" / "p0_runtime_benchmark_20261104_run")
    ap.add_argument("--dataset", type=Path, default=real.DATASET_ROOT)
    ap.add_argument("--frames-per-object-regime", type=int, default=4)
    ap.add_argument("--inference-targets", type=int, default=300)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve prior output: {args.output}")
    args.output.mkdir(parents=True)
    env = dict(platform=platform.platform(), processor=platform.processor(), python=sys.version,
               cpu_count=os.cpu_count(), started=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        import open3d, sklearn
        env.update(open3d=open3d.__version__, sklearn=sklearn.__version__)
    except Exception:
        pass
    (args.output / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    patch()
    manifest = runner.build_holdout_manifest(args.output)
    subset = (manifest.sort_values(["object_id", "init_regime", "frame_id"])
              .groupby(["object_id", "init_regime"], group_keys=False).head(args.frames_per_object_regime)
              .reset_index(drop=True))
    subset.to_csv(args.output / "benchmark_subset.csv", index=False)
    run_args = argparse.Namespace(dataset=args.dataset, reuse_checkpoints=None)
    state = {}
    wall = time.perf_counter()
    try:
        runner.generate_candidates(run_args, args.output, subset, state)
    except ValueError as error:   # expected: coverage checks are hard-coded to the full 220-frame holdout
        if "coverage" not in str(error) and "Incomplete" not in str(error):
            raise
    wall_ms = (time.perf_counter() - wall) * 1000.0
    times = pd.DataFrame(TIMES)
    times.to_csv(args.output / "stage_call_timings.csv", index=False)
    pd.DataFrame(RECORDS).to_csv(args.output / "registration_records.csv", index=False)
    per = times.pivot_table(index="condition", columns="stage", values="ms", aggfunc="sum").fillna(0.0)
    first = per.index.min()
    per = per.drop(index=[c for c in per.index if c < first + 3])   # drop warm-up conditions
    inf = inference_benchmark(args.inference_targets, args.output)
    inf.to_csv(args.output / "inference_timings.csv", index=False)
    steady = inf[~inf.warmup]
    method_stages = ["registration_p2p_and_filtered", "feature_preparation", "candidate_step_clipping",
                     "geometric_scores", "render_depth", "tail_features"]
    per["inference_and_decision"] = float((steady.predict_ms + steady.decision_ms).median())
    per["method_total_ms"] = per[[c for c in method_stages if c in per] + ["inference_and_decision"]].sum(axis=1)
    rec = pd.DataFrame(RECORDS)
    p2p = rec[rec.method.eq(real.TRUNK)].groupby("condition").runtime_ms.sum()
    per["p2p_only_ms"] = p2p.reindex(per.index)
    per.to_csv(args.output / "per_condition_stage_ms.csv")
    summary = {c: describe(per[c]) for c in per.columns}
    summary["inference_predict_ms"] = describe(steady.predict_ms)
    summary["inference_decision_ms"] = describe(steady.decision_ms)
    summary["inference_batch_ms_per_target"] = inf.attrs["batch_ms_per_target"]
    summary["conditions_timed"] = int(len(per))
    summary["generate_candidates_wall_ms_per_condition_including_bookkeeping"] = wall_ms / max(1, times.condition.nunique())
    summary["note"] = ("Stages ending in _EXCLUDED are input simulation, GT evaluation or one-time setup and are not part "
                       "of the deployed method cost. p2p_only_ms is the plain Point-to-Point trunk alone.")
    (args.output / "runtime_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k in ["method_total_ms", "p2p_only_ms", "conditions_timed"]}, indent=2))
    print("P0 RUNTIME BENCHMARK COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
