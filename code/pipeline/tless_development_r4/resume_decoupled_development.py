"""Continue a nonfailed interrupted label stage, preserving the completed candidate bank."""
from __future__ import annotations

import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_name] = "1"

import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import pandas as pd

import run_decoupled_development as parent
import evaluate_bop_metrics_controlled as evaluator
from decoupled_checkpoint_recovery import ZERO_FIELDS
from winner_training_data_io import read_csv

p, base = parent.p, parent.base


def split_trials(path, output, objects):
    """Use CSV fields verbatim so splitting cannot round pose coordinates."""
    output.mkdir()
    counts = {int(oid):0 for oid in objects}
    with ExitStack() as stack:
        source = stack.enter_context(path.open(newline="", encoding="utf-8"))
        reader = csv.DictReader(source)
        writers = {}
        for oid in counts:
            stream = stack.enter_context((output/f"object_{oid:02d}.csv").open("w", newline="", encoding="utf-8"))
            writers[oid] = csv.DictWriter(stream, fieldnames=reader.fieldnames)
            writers[oid].writeheader()
        for row in reader:
            oid = int(float(row["object_id"]))
            if oid not in writers:
                raise ValueError("Unexpected object while partitioning frozen trials")
            writers[oid].writerow(row)
            counts[oid] += 1
    return counts


def verify_sources(state, evaluation):
    for path, expected in {**state["source_hashes"], **state.get("dataset_hashes", {}), **evaluation["file_hashes"]}.items():
        if p.digest(Path(path)) != expected:
            raise ValueError("Frozen source or evaluation file changed: "+path)


def run(args):
    if args.output.exists():
        raise FileExistsError("Preserve previous recovery")
    old = json.loads((args.source/"state.json").read_text())
    if old["status"] != "RUNNING" or old["stage"] != "bop" or old.get("error") or old["completed_cases"] != 733:
        raise ValueError("Only the nonfailed interrupted fully generated bank may be continued")
    if (args.source/"group_oof").exists() or list((args.source/"bop").glob("*.csv")):
        raise ValueError("This recovery is specifically for no completed labels or models")
    evaluation = json.loads((args.source/"evaluation_sources_pre_BOP.json").read_text())
    verify_sources(old, evaluation)
    trials = read_csv(args.source/"trials.csv")
    features = read_csv(args.source/"observable_candidate_features.csv")
    scores = read_csv(args.source/"scores.csv")
    schema = json.loads((args.source/"feature_schema.json").read_text())
    if len(trials)!=10262 or len(features)!=8796 or features.frame_id.nunique()!=629 or len(schema["feature_names"])!=191:
        raise ValueError("Completed bank coverage changed")
    if not scores.loc[scores.method.eq(base.INITIAL), ZERO_FIELDS].eq(0).all().all():
        raise ValueError("Initial update diagnostic omission was not corrected")
    for name in ["full_update_rotation_deg", "full_update_translation_mm", "applied_rotation_deg", "applied_translation_mm"]:
        np.testing.assert_allclose(features["delta_"+name], features["candidate_"+name], equal_nan=True)
    args.output.mkdir(parents=True)
    inventory = {}
    for child in args.source.iterdir():
        if child.name in {"state.json", "bop", "bop.log"}:
            continue
        target = args.output/child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)
        for path in ([child] if child.is_file() else child.rglob("*")):
            if path.is_file():
                relative = path.relative_to(args.source)
                expected = p.digest(path)
                if p.digest(args.output/relative)!=expected:
                    raise ValueError("Recovery copy differs: "+str(relative))
                inventory[str(path)] = expected
    p.save(args.output/"resume_input_inventory.json", dict(source=str(args.source), file_hashes=inventory,
        note="New user continue after lost process/session; parent RUNNING with no error; complete generation reused, no completed labels or models"))
    shutil.copy2(Path(__file__), args.output/"source_snapshot/resume_decoupled_development.py")
    state = dict(old)
    state.update(status="RUNNING", stage="interruption_recovery_verified", pid=os.getpid(), started_at_utc=parent.now(),
        commands=[], objects_complete=[], previous_execution_interrupted=True,
        previous_output=str(args.source), previous_state_sha256=p.digest(args.source/"state.json"),
        recovery="user-authorized continuation; exact original evaluator per object; same parameters; no candidate regeneration")
    state["source_hashes"] = {**old["source_hashes"], str(Path(__file__)):p.digest(Path(__file__)),
        str(args.source/"state.json"):p.digest(args.source/"state.json")}
    p.save(args.output/"state.json", state)
    (args.output/"interruption_recovery_zh.md").write_text("\n".join([
        "# 执行中断后的继续", "", "## Material Passport",
        "- 作者已再次要求继续。原执行会话已丢失，未发现 Python 进程；原状态为 RUNNING，无异常记录。",
        "- 原目录保持不变，完整候选、特征、观测、源码与评价来源哈希已核验，并逐文件复制到本目录。",
        "- 不重算候选，不复用任何完成的 BOP 标签或校准模型。",
        "- BOP 调度改为逐物体独立落盘，仍调用同一原评价脚本：相同位姿、阈值、对称性、传感器深度和 CUDA batch size=1。",
        "- CSV 分片按字段文本原样保存，避免位姿二次舍入；全量合并后对账10262行，再执行同一对象分组 OOF 训练。",
        "- 不启动新科学方案，不自动重试失败任务，不读取预留实例。", ""]), encoding="utf-8")
    try:
        counts = split_trials(args.output/"trials.csv", args.output/"bop_trials", p.OBJECTS)
        if sum(counts.values())!=10262 or not all(counts.values()):
            raise ValueError("BOP object partition lost trials")
        work = Path(__file__).parent
        labels = []
        for oid in p.OBJECTS:
            output = args.output/"bop_by_object"/f"object_{oid:02d}"
            parent.run_command(args.output, f"bop_object_{oid:02d}", [sys.executable, "-u", str(work/"evaluate_bop_metrics_controlled.py"),
                "--trials-csv", str(args.output/"bop_trials"/f"object_{oid:02d}.csv"), "--output-dir", str(output),
                "--dataset-root", str(args.dataset), "--toolkit-root", str(args.toolkit), "--methods", ",".join(parent.METHODS),
                "--extra-key-columns", "cohort,input_depth_condition", "--device", "cuda", "--gpu-batch-size", "1", "--checkpoint-every", "50"], state)
            table = read_csv(output/"bop_pose_errors.csv")
            keys = ["cohort", *base.KEYS, "method"]
            if len(table)!=counts[oid] or table.duplicated(keys).any() or set(table.object_id)!={oid}:
                raise ValueError("Incomplete per-object BOP labels")
            labels.append(table)
            state["objects_complete"].append(dict(object_id=oid, pose_labels=len(table), labels_sha256=p.digest(output/"bop_pose_errors.csv")))
            p.save(args.output/"state.json", state)
            print("BOP OBJECT COMPLETE", oid, len(state["objects_complete"]), "/22", flush=True)
        table = pd.concat(labels, ignore_index=True)
        directory = args.output/"bop"
        directory.mkdir()
        table.to_csv(directory/"bop_pose_errors.csv", index=False)
        metrics = {"vsd", "mssd", "mspd"}
        evaluator.aggregate_scores(table, metrics).to_csv(directory/"table_bop_ar_summary.csv", index=False)
        evaluator.per_object_scores(table, metrics).to_csv(directory/"table_bop_ar_per_object.csv", index=False)
        evaluator.threshold_recalls(table, metrics).to_csv(directory/"table_bop_threshold_recalls.csv", index=False)
        parent.finalize_bop(args)
        state.update(stage="labels_complete", labels_sha256=p.digest(args.output/"development_bop_labels.csv"))
        p.save(args.output/"state.json", state)
        parent.run_command(args.output, "group_oof", [sys.executable, "-u", str(work/"train_decoupled_group_oof.py"),
            "--data", str(args.output), "--output", str(args.output/"group_oof")], state)
        verify_sources(old, evaluation)
        for path, expected in inventory.items():
            if p.digest(Path(path))!=expected:
                raise ValueError("Recovery input changed during execution: "+path)
        if parent.verify_prior_delivery(args.output.parent):
            raise ValueError("Prior delivery changed")
        state.update(status="COMPLETE_DECOUPLED_DEVELOPMENT_NOT_PROMOTED", completed_at_utc=parent.now())
        state["artifact_hashes"] = {str(path.relative_to(args.output)):p.digest(path) for path in args.output.rglob("*")
            if path.is_file() and path!=args.output/"state.json"}
        p.save(args.output/"state.json", state)
        print("DECOUPLED DEVELOPMENT RECOVERY COMPLETE", flush=True)
    except Exception:
        state.update(status="FAILED_PRESERVED_NO_AUTOMATIC_RETRY", error=traceback.format_exc())
        p.save(args.output/"state.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=parent.ROOT/"outputs/decoupled_development_20261020_r3")
    parser.add_argument("--output", type=Path, default=parent.ROOT/"outputs/decoupled_development_20261020_r4")
    parser.add_argument("--dataset", type=Path, default=parent.real.DATASET_ROOT)
    parser.add_argument("--toolkit", type=Path, default=Path(r"D:\6D_pose\tools\bop_toolkit"))
    run(parser.parse_args())
