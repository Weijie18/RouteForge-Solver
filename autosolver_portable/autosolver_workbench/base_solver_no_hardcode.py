"""No-hardcode MT-4 solver kernel used by the AutoSolver workbench.

This module deliberately reuses the current independent ``clean_solver.py``
algorithmic core while disabling literal verified-output shortcuts.  It gives
the AutoSolver a stable baseline to mutate without relying on sampled online
rows or hidden-case constants.

The offline AutoSolver loop is allowed to take minutes or hours.  Candidate
files generated from this module still expose a normal ``solve(input_text)``
entrypoint, and each individual call is controlled by ``DEFAULT_TIME_LIMIT``.
"""

from __future__ import print_function

import importlib.util
import os
import sys
import time


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CLEAN_SOLVER_PATH = os.path.join(ROOT, "clean_solver.py")


def _load_clean_solver():
    spec = importlib.util.spec_from_file_location("mt4_clean_solver_kernel", CLEAN_SOLVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kernel = _load_clean_solver()


def _disabled_hardcoded_case_output(problem):
    return None


def install_no_hardcode(module=None):
    """Disable literal case-output shortcuts on a clean_solver-like module."""

    target = module if module is not None else kernel
    target.hardcoded_case_output = _disabled_hardcoded_case_output
    return target


install_no_hardcode(kernel)


DEFAULT_TIME_LIMIT = kernel.DEFAULT_TIME_LIMIT
FAIL_PENALTY = kernel.FAIL_PENALTY
EPS = kernel.EPS
INF = kernel.INF


def solve(input_text):
    """Solve one MT-4 TSV instance without using hardcoded verified outputs."""

    install_no_hardcode(kernel)
    old_limit = kernel.DEFAULT_TIME_LIMIT
    kernel.DEFAULT_TIME_LIMIT = DEFAULT_TIME_LIMIT
    try:
        return kernel.solve(input_text)
    finally:
        kernel.DEFAULT_TIME_LIMIT = old_limit


def parse_input(input_text):
    return kernel.parse_input(input_text)


def validate_verified_output(problem, rows):
    return kernel.validate_verified_output(problem, rows)


def verified_output_value(problem, rows):
    return kernel.verified_output_value(problem, rows)


def evaluate_state(problem, state):
    return kernel.evaluate_state(problem, state)


def state_to_output(state):
    return kernel.state_to_output(state)


def all_single_grouping(problem):
    return kernel.all_single_grouping(problem)


def make_expected_grouping(problem, mode, threshold):
    return kernel.make_expected_grouping(problem, mode, threshold)


def make_score_grouping(problem, alpha, threshold):
    return kernel.make_score_grouping(problem, alpha, threshold)


def make_potential_grouping(problem, mode, top_k, threshold):
    return kernel.make_potential_grouping(problem, mode, top_k, threshold)


def default_deadline(seconds=0.25):
    return time.perf_counter() + float(seconds)


def greedy_assignment(problem, groups, deadline=None, ensure_initial=True, allowed_couriers=None):
    if deadline is None:
        deadline = default_deadline()
    return kernel.greedy_assignment(
        problem,
        groups,
        deadline,
        ensure_initial=ensure_initial,
        allowed_couriers=allowed_couriers,
    )


def improve_fixed_groups(problem, state, deadline=None):
    if deadline is None:
        deadline = default_deadline()
    return kernel.improve_fixed_groups(problem, state, deadline)


def local_repartition(problem, state, best_value, deadline):
    return kernel.local_repartition(problem, state, best_value, deadline)


def random_repartition(problem, state, best_value, deadline, seed):
    return kernel.random_repartition(problem, state, best_value, deadline, seed)


def pair_repartition_sweep(problem, state, best_value, deadline):
    return kernel.pair_repartition_sweep(problem, state, best_value, deadline)


def best_offer_subset(candidates):
    return kernel.best_offer_subset(candidates)


def group_value(offers, task_count):
    return kernel.group_value(offers, task_count)


def bit_count(value):
    return kernel.bit_count(value)


def normalize_rows(rows):
    result = []
    for task_key, couriers in rows:
        result.append((str(task_key), tuple(str(courier) for courier in couriers)))
    return tuple(result)


def score_rows(input_text, rows):
    """Return (valid, value) for output rows on an input text."""

    problem = parse_input(input_text)
    normalized = normalize_rows(rows)
    if validate_verified_output(problem, normalized) is None:
        return False, INF
    return True, verified_output_value(problem, normalized)


def run_text(input_text, time_limit=None):
    """Run ``solve`` and return a compact evaluation dictionary."""

    global DEFAULT_TIME_LIMIT
    old_limit = DEFAULT_TIME_LIMIT
    if time_limit is not None:
        DEFAULT_TIME_LIMIT = float(time_limit)
    start = time.perf_counter()
    rows = solve(input_text)
    elapsed = time.perf_counter() - start
    valid, value = score_rows(input_text, rows)
    DEFAULT_TIME_LIMIT = old_limit
    return {
        "valid": bool(valid),
        "value": float(value),
        "elapsed_ms": elapsed * 1000.0,
        "rows": len(rows) if rows is not None else 0,
    }


def make_wrapper_source(config):
    """Build a small candidate solver that mutates selected kernel knobs.

    The generated file is not meant to be the final contest submission yet; it
    is a search artifact for the AutoSolver loop.  A later distillation step can
    fold retained changes into a standalone solver.
    """

    name = config.get("name", "candidate")
    time_limit = float(config.get("time_limit", DEFAULT_TIME_LIMIT))
    fixed_penalty = config.get("fixed_penalty")
    profile_penalty = config.get("profile_penalty", "")
    lines = [
        "from __future__ import print_function",
        "import os, sys",
        "ROOT = %r" % ROOT,
        "if ROOT not in sys.path: sys.path.insert(0, ROOT)",
        "from autosolver_workbench import base_solver_no_hardcode as base",
        "base.DEFAULT_TIME_LIMIT = %.12g" % time_limit,
        "base.install_no_hardcode(base.kernel)",
        "NAME = %r" % name,
    ]
    if fixed_penalty is not None:
        lines.extend(
            [
                "def _candidate_penalty(problem):",
                "    return %.12g" % float(fixed_penalty),
                "base.kernel.choose_model_penalty = _candidate_penalty",
            ]
        )
    elif profile_penalty == "scarce_high":
        lines.extend(
            [
                "def _candidate_penalty(problem):",
                "    c = len(problem.all_couriers)",
                "    if problem.n_tasks >= 25 and c <= problem.n_tasks + 3:",
                "        return 110.0",
                "    return base.kernel.choose_model_penalty(problem)",
                "base.kernel.choose_model_penalty = _candidate_penalty",
            ]
        )
    elif profile_penalty == "dense_low":
        lines.extend(
            [
                "def _candidate_penalty(problem):",
                "    c = len(problem.all_couriers)",
                "    if 25 <= problem.n_tasks <= 32 and c >= int(problem.n_tasks * 1.8):",
                "        return 85.0",
                "    return base.kernel.choose_model_penalty(problem)",
                "base.kernel.choose_model_penalty = _candidate_penalty",
            ]
        )
    lines.extend(
        [
            "def solve(input_text):",
            "    return base.solve(input_text)",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    data = sys.stdin.read()
    if not data:
        raise SystemExit("expected TSV on stdin")
    print(run_text(data))
