"""Evaluation harness for MT-4 AutoSolver candidates.

The evaluator runs candidate ``solve(input_text)`` implementations on local
case files, checks legality, computes the same expected penalty used by
``clean_solver.py``, and records a machine-readable report.  Lower score is
better.
"""

from __future__ import print_function

import argparse
import glob
import importlib.util
import json
import os
import subprocess
import sys
import time


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autosolver_workbench import base_solver_no_hardcode as scorer  # noqa: E402


DEFAULT_CASE_GLOBS = (
    os.path.join(ROOT, "outputs", "hidden_like_online_shape", "*.txt"),
)
DEFAULT_CASE_FILES = (
    r"C:\Users\ASUS\Downloads\large_seed301.txt",
)


def discover_cases(extra_cases=None, extra_globs=None, include_defaults=True):
    seen = set()
    result = []
    if include_defaults:
        for path in DEFAULT_CASE_FILES:
            if os.path.exists(path) and path not in seen:
                seen.add(path)
                result.append(path)
        for pattern in DEFAULT_CASE_GLOBS:
            for path in sorted(glob.glob(pattern)):
                if path not in seen:
                    seen.add(path)
                    result.append(path)
    for path in extra_cases or ():
        path = os.path.abspath(path)
        if os.path.exists(path) and path not in seen:
            seen.add(path)
            result.append(path)
    for pattern in extra_globs or ():
        for path in sorted(glob.glob(pattern)):
            path = os.path.abspath(path)
            if path not in seen:
                seen.add(path)
                result.append(path)
    return result


def load_solver(path):
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("autosolver_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "solve"):
        raise RuntimeError("candidate has no solve(input_text)")
    return module


def normalize_rows(rows):
    normalized = []
    for item in rows or ():
        if len(item) != 2:
            raise ValueError("row must be (task_key, couriers)")
        task_key, couriers = item
        normalized.append((str(task_key), tuple(str(courier) for courier in couriers)))
    return tuple(normalized)


def child_evaluate_case(solver_path, case_path, time_limit):
    solver = load_solver(solver_path)
    if hasattr(solver, "DEFAULT_TIME_LIMIT"):
        solver.DEFAULT_TIME_LIMIT = float(time_limit)
    with open(case_path, "r", encoding="utf-8-sig") as handle:
        input_text = handle.read()
    problem = scorer.parse_input(input_text)
    start = time.perf_counter()
    rows = solver.solve(input_text)
    elapsed = time.perf_counter() - start
    normalized = normalize_rows(rows)
    valid_rows = scorer.validate_verified_output(problem, normalized)
    if valid_rows is None:
        return {
            "case": os.path.basename(case_path),
            "case_path": case_path,
            "status": "invalid",
            "value": None,
            "elapsed_ms": elapsed * 1000.0,
            "rows": len(normalized),
            "offers": sum(len(couriers) for _, couriers in normalized),
        }
    value = scorer.verified_output_value(problem, normalized)
    return {
        "case": os.path.basename(case_path),
        "case_path": case_path,
        "status": "ok",
        "value": float(value),
        "elapsed_ms": elapsed * 1000.0,
        "rows": len(normalized),
        "offers": sum(len(couriers) for _, couriers in normalized),
    }


def run_case_subprocess(solver_path, case_path, time_limit, hard_timeout):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--_child",
        "--solver",
        os.path.abspath(solver_path),
        "--case",
        os.path.abspath(case_path),
        "--time-limit",
        str(float(time_limit)),
    ]
    start = time.perf_counter()
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=float(hard_timeout))
        return json.loads(raw.decode("utf-8"))
    except subprocess.TimeoutExpired:
        return {
            "case": os.path.basename(case_path),
            "case_path": case_path,
            "status": "timeout",
            "value": None,
            "elapsed_ms": (time.perf_counter() - start) * 1000.0,
            "rows": 0,
            "offers": 0,
        }
    except subprocess.CalledProcessError as exc:
        return {
            "case": os.path.basename(case_path),
            "case_path": case_path,
            "status": "error",
            "value": None,
            "elapsed_ms": (time.perf_counter() - start) * 1000.0,
            "rows": 0,
            "offers": 0,
            "error": exc.output.decode("utf-8", "replace")[-2000:],
        }


def _aggregate_repeat_results(repeat_results, aggregate, fail_score):
    ok_results = [item for item in repeat_results if item.get("status") == "ok"]
    if not ok_results:
        item = dict(repeat_results[-1]) if repeat_results else {}
        item["repeat_count"] = len(repeat_results)
        item["repeat_values"] = []
        item["repeat_statuses"] = [str(result.get("status")) for result in repeat_results]
        return item, float(fail_score)

    values = sorted(float(item["value"]) for item in ok_results)
    aggregate = str(aggregate or "single")
    if aggregate == "best":
        selected_value = values[0]
    elif aggregate == "worst":
        selected_value = values[-1]
    elif aggregate == "mean":
        selected_value = sum(values) / float(len(values))
    elif aggregate == "median":
        mid = len(values) // 2
        if len(values) % 2:
            selected_value = values[mid]
        else:
            selected_value = (values[mid - 1] + values[mid]) * 0.5
    else:
        selected_value = float(ok_results[-1]["value"])

    selected = None
    for item in ok_results:
        if abs(float(item["value"]) - selected_value) <= 1e-9:
            selected = item
            break
    if selected is None:
        selected = ok_results[-1]
    result = dict(selected)
    result["value"] = float(selected_value)
    result["repeat_count"] = len(repeat_results)
    result["repeat_values"] = [None if item.get("value") is None else float(item["value"]) for item in repeat_results]
    result["repeat_statuses"] = [str(item.get("status")) for item in repeat_results]
    result["repeat_elapsed_ms"] = [float(item.get("elapsed_ms") or 0.0) for item in repeat_results]
    return result, float(selected_value)


def evaluate_solver(
    solver_path,
    cases,
    time_limit=9.35,
    hard_timeout=12.0,
    fail_score=1000000.0,
    repeats=1,
    repeat_aggregate="single",
):
    results = []
    total = 0.0
    ok_count = 0
    repeats = max(1, int(repeats or 1))
    for case_path in cases:
        repeat_results = []
        for _ in range(repeats):
            repeat_results.append(run_case_subprocess(solver_path, case_path, time_limit, hard_timeout))
        if repeats == 1:
            item = repeat_results[0]
            item_score = float(item["value"]) if item.get("status") == "ok" else float(fail_score)
        else:
            item, item_score = _aggregate_repeat_results(repeat_results, repeat_aggregate, fail_score)
        if item.get("status") == "ok":
            total += item_score
            ok_count += 1
        else:
            total += float(fail_score)
        results.append(item)
    average = total / max(1, len(results))
    return {
        "solver": os.path.abspath(solver_path),
        "case_count": len(results),
        "ok_count": ok_count,
        "average": average,
        "time_limit": float(time_limit),
        "hard_timeout": float(hard_timeout),
        "repeats": repeats,
        "repeat_aggregate": str(repeat_aggregate or "single"),
        "results": results,
        "created_at": time.time(),
    }


def print_summary(report):
    print("solver=%s" % report["solver"])
    print("average=%.6f ok=%d/%d" % (report["average"], report["ok_count"], report["case_count"]))
    for item in report["results"]:
        value = item["value"]
        value_text = "NA" if value is None else "%.6f" % float(value)
        repeat_text = ""
        if int(item.get("repeat_count") or 1) > 1:
            values = []
            for repeat_value in item.get("repeat_values") or ():
                values.append("NA" if repeat_value is None else "%.6f" % float(repeat_value))
            repeat_text = "\trepeats=[%s]" % ", ".join(values)
        print(
            "%s\t%s\t%s\t%.1fms\trows=%s\toffers=%s%s"
            % (
                item["case"],
                item["status"],
                value_text,
                float(item.get("elapsed_ms") or 0.0),
                item.get("rows"),
                item.get("offers"),
                repeat_text,
            )
        )


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--case-glob", action="append", default=[])
    parser.add_argument("--time-limit", type=float, default=9.35)
    parser.add_argument("--hard-timeout", type=float, default=12.0)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--fail-score", type=float, default=1000000.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--repeat-aggregate", choices=("single", "best", "median", "mean", "worst"), default="single")
    parser.add_argument("--no-default-cases", action="store_true")
    parser.add_argument("--_child", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args._child:
        report = child_evaluate_case(args.solver, args.case[0], args.time_limit)
        sys.stdout.write(json.dumps(report, ensure_ascii=False))
        return 0
    cases = discover_cases(args.case, args.case_glob, include_defaults=not args.no_default_cases)
    if not cases:
        raise SystemExit("no cases found")
    report = evaluate_solver(
        args.solver,
        cases,
        time_limit=args.time_limit,
        hard_timeout=args.hard_timeout,
        fail_score=args.fail_score,
        repeats=args.repeats,
        repeat_aggregate=args.repeat_aggregate,
    )
    if args.json_out:
        out_dir = os.path.dirname(os.path.abspath(args.json_out))
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print_summary(report)
    return 0 if report["ok_count"] == report["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
