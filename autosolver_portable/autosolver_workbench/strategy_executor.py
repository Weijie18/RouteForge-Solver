"""Constrained strategy executor for DeepSeek AutoSolver candidates.

DeepSeek should not directly manipulate ``clean_solver`` internal state.  This
module gives it a small action DSL: choose starts, improvement operators, and a
few numeric knobs; the executor owns parsing, legality, state shape, timing,
and fallback handling.
"""

from __future__ import print_function

import ast
import copy
import json
import os
import sys
import time


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autosolver_workbench import base_solver_no_hardcode as base  # noqa: E402


INF = base.INF
EPS = base.EPS


DEFAULT_STRATEGY = {
    "time_limit": 9.35,
    "include_base": True,
    "base_time": 8.50,
    "polish_base": True,
    "per_start_budget": 0.28,
    "selection_fixed_budget": 0.08,
    "starts": [
        {"kind": "sweep"},
        {"kind": "all_single"},
        {"kind": "expected", "mode": "pair_gain", "threshold": 0.0},
        {"kind": "potential", "mode": "potential_gain", "top_k": 3, "threshold": 0.0},
        {"kind": "score", "alpha": 30.0, "threshold": 0.0},
    ],
    "improvements": [
        {"kind": "improve_fixed", "budget": 0.16},
        {"kind": "local_repartition", "budget": 0.45},
        {"kind": "random_repartition", "seed": 97, "budget": 0.45},
        {"kind": "pair_sweep", "budget": 0.24},
    ],
    "penalty": {"kind": "default"},
}


ALLOWED_STARTS = set(["all_single", "expected", "score", "potential", "sweep"])
ALLOWED_EXPECTED_MODES = set(["pair_gain", "pair_half", "pair_raw", "raw"])
ALLOWED_POTENTIAL_MODES = set(["potential_gain", "potential_half", "potential_raw", "raw"])
ALLOWED_IMPROVEMENTS = set([
    "improve_fixed",
    "local_repartition",
    "random_repartition",
    "pair_sweep",
    "pair_injection",
    "pair_seed_search",
    "ejection_chain",
    "repair_chain",
    "learned_operator",
])
ALLOWED_PENALTIES = set(["default", "fixed", "profile"])


def _as_float(value, default, low=None, high=None):
    try:
        result = float(value)
    except Exception:
        result = float(default)
    if low is not None and result < low:
        result = float(low)
    if high is not None and result > high:
        result = float(high)
    return result


def _as_int(value, default, low=None, high=None):
    try:
        result = int(value)
    except Exception:
        result = int(default)
    if low is not None and result < low:
        result = int(low)
    if high is not None and result > high:
        result = int(high)
    return result


def _as_bool(value, default):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "y", "on"):
            return True
        if text in ("0", "false", "no", "n", "off"):
            return False
    return bool(default)


def _clean_param_value(value, depth=0):
    if depth > 3:
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        cleaned = []
        for item in list(value)[:32]:
            child = _clean_param_value(item, depth + 1)
            if child is not None:
                cleaned.append(child)
        return cleaned
    if isinstance(value, dict):
        cleaned_dict = {}
        for key, item in list(value.items())[:32]:
            if not isinstance(key, str):
                continue
            child = _clean_param_value(item, depth + 1)
            if child is not None:
                cleaned_dict[str(key)[:64]] = child
        return cleaned_dict
    return None


def _clean_params(value):
    cleaned = _clean_param_value(value if isinstance(value, dict) else {}, 0)
    return cleaned if isinstance(cleaned, dict) else {}


def learned_operator_issues(source):
    """Return static safety issues for a generated operator body."""

    issues = []
    source = str(source or "")
    if not source.strip():
        return ["operator source is empty"]
    if len(source) > 14000:
        issues.append("operator source is too large")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ["operator source syntax error: %s" % exc]

    has_apply = False
    forbidden_nodes = (
        ast.Import,
        ast.ImportFrom,
        ast.ClassDef,
        ast.Global,
        ast.Nonlocal,
        ast.With,
        ast.AsyncWith,
        ast.AsyncFunctionDef,
    )
    forbidden_calls = set([
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
    ])
    forbidden_names = set(["os", "sys", "subprocess", "socket", "pathlib", "shutil"])

    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            issues.append("operator source uses forbidden syntax %s" % node.__class__.__name__)
        if isinstance(node, ast.FunctionDef) and node.name == "apply":
            has_apply = True
        if isinstance(node, ast.Name):
            if node.id in forbidden_names or node.id.startswith("__"):
                issues.append("operator source uses forbidden name %s" % node.id)
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                issues.append("operator source uses forbidden dunder attribute")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in forbidden_calls:
                issues.append("operator source calls forbidden builtin %s" % func.id)
    if not has_apply:
        issues.append("operator source must define apply(problem, state, value, deadline, params)")
    return sorted(set(issues))


def _clean_operators(value):
    if not isinstance(value, dict):
        return {}
    cleaned = {}
    for name, source in list(value.items())[:12]:
        name = safe_operator_name(name)
        if not name:
            continue
        source = str(source or "")
        if learned_operator_issues(source):
            continue
        cleaned[name] = source
    return cleaned


def safe_operator_name(value):
    cleaned = []
    for ch in str(value or ""):
        if ch.isalnum() or ch == "_":
            cleaned.append(ch)
    result = "".join(cleaned)[:48]
    if not result:
        return ""
    if result[0].isdigit():
        result = "op_" + result
    return result


def _mode(value, allowed, default):
    value = str(value or default)
    if value == "raw":
        if "pair_raw" in allowed:
            value = "pair_raw"
        elif "potential_raw" in allowed:
            value = "potential_raw"
    return value if value in allowed else default


def _accept_mode(value):
    value = str(value or "bounded")
    if value in ("improve", "bounded", "always_legal"):
        return value
    return "bounded"


def normalize_strategy(strategy):
    """Return a sanitized strategy dictionary from model-provided input."""

    if not isinstance(strategy, dict):
        strategy = {}
    result = copy.deepcopy(DEFAULT_STRATEGY)
    result["time_limit"] = _as_float(strategy.get("time_limit", result["time_limit"]), 9.35, 1.0, 9.60)
    result["include_base"] = _as_bool(strategy.get("include_base", result["include_base"]), True)
    result["base_time"] = _as_float(strategy.get("base_time", result["base_time"]), 2.0, 0.0, 9.20)
    result["polish_base"] = _as_bool(strategy.get("polish_base", result["polish_base"]), True)
    result["per_start_budget"] = _as_float(strategy.get("per_start_budget", result["per_start_budget"]), 0.28, 0.02, 2.0)
    result["selection_fixed_budget"] = _as_float(
        strategy.get("selection_fixed_budget", result["selection_fixed_budget"]),
        0.08,
        0.0,
        0.5,
    )

    penalty = strategy.get("penalty", result["penalty"])
    if not isinstance(penalty, dict):
        penalty = {}
    penalty_kind = str(penalty.get("kind", "default"))
    if penalty_kind not in ALLOWED_PENALTIES:
        penalty_kind = "default"
    clean_penalty = {"kind": penalty_kind}
    if penalty_kind == "fixed":
        clean_penalty["value"] = _as_float(penalty.get("value", 100.0), 100.0, 60.0, 140.0)
    elif penalty_kind == "profile":
        clean_penalty["dense"] = _as_float(penalty.get("dense", 100.0), 100.0, 60.0, 140.0)
        clean_penalty["scarce"] = _as_float(penalty.get("scarce", 100.0), 100.0, 60.0, 140.0)
    result["penalty"] = clean_penalty
    result["operators"] = _clean_operators(strategy.get("operators", {}))

    starts = []
    raw_starts = strategy.get("starts", result["starts"])
    if not isinstance(raw_starts, list):
        raw_starts = result["starts"]
    for item in raw_starts[:32]:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", ""))
        if kind not in ALLOWED_STARTS:
            continue
        clean = {"kind": kind}
        if kind == "sweep":
            clean["expected_thresholds"] = _clean_float_list(item.get("expected_thresholds"), (-80.0, -15.0, 0.0, 20.0), -300.0, 300.0, 12)
            clean["potential_thresholds"] = _clean_float_list(item.get("potential_thresholds"), (-100.0, -10.0, 0.0, 15.0), -400.0, 400.0, 12)
            clean["score_alphas"] = _clean_float_list(item.get("score_alphas"), (-5.0, 15.0, 30.0, 60.0), -300.0, 300.0, 12)
            clean["score_thresholds"] = _clean_float_list(item.get("score_thresholds"), (-10.0, 0.0, 10.0), -300.0, 300.0, 12)
            clean["top_ks"] = _clean_int_list(item.get("top_ks"), (2, 3, 4), 1, 8, 8)
        elif kind == "expected":
            clean["mode"] = _mode(item.get("mode"), ALLOWED_EXPECTED_MODES, "pair_gain")
            clean["threshold"] = _as_float(item.get("threshold", 0.0), 0.0, -300.0, 300.0)
        elif kind == "score":
            clean["alpha"] = _as_float(item.get("alpha", 30.0), 30.0, -300.0, 300.0)
            clean["threshold"] = _as_float(item.get("threshold", 0.0), 0.0, -300.0, 300.0)
        elif kind == "potential":
            clean["mode"] = _mode(item.get("mode"), ALLOWED_POTENTIAL_MODES, "potential_gain")
            clean["top_k"] = _as_int(item.get("top_k", 3), 3, 1, 8)
            clean["threshold"] = _as_float(item.get("threshold", 0.0), 0.0, -400.0, 400.0)
        starts.append(clean)
    if not starts:
        starts = copy.deepcopy(DEFAULT_STRATEGY["starts"])
    result["starts"] = starts

    improvements = []
    raw_improvements = strategy.get("improvements", result["improvements"])
    if not isinstance(raw_improvements, list):
        raw_improvements = result["improvements"]
    for item in raw_improvements[:24]:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", ""))
        if kind not in ALLOWED_IMPROVEMENTS:
            continue
        max_budget = 8.5 if kind == "pair_seed_search" else 3.0
        clean = {"kind": kind, "budget": _as_float(item.get("budget", 0.25), 0.25, 0.0, max_budget)}
        if kind == "random_repartition":
            clean["seed"] = _as_int(item.get("seed", 97), 97, 0, 10000000)
        elif kind == "pair_injection":
            clean["top_candidates"] = _as_int(item.get("top_candidates", 6), 6, 1, 16)
            clean["max_subset"] = _as_int(item.get("max_subset", 6), 6, 2, 8)
        elif kind == "pair_seed_search":
            clean["scan_pairs"] = _as_int(item.get("scan_pairs", 13), 13, 4, 30)
            clean["top_pairs"] = _as_int(item.get("top_pairs", 7), 7, 3, 12)
            clean["deep_trials"] = _as_int(item.get("deep_trials", 3), 3, 1, 8)
        elif kind == "ejection_chain":
            clean["max_depth"] = _as_int(item.get("max_depth", 5), 5, 1, 8)
            clean["width"] = _as_int(item.get("width", 80), 80, 8, 240)
            clean["branch"] = _as_int(item.get("branch", 80), 80, 8, 200)
        elif kind == "repair_chain":
            clean["max_depth"] = _as_int(item.get("max_depth", 7), 7, 1, 10)
            clean["width"] = _as_int(item.get("width", 180), 180, 8, 260)
            clean["start_branch"] = _as_int(item.get("start_branch", 400), 400, 16, 600)
            clean["fill_branch"] = _as_int(item.get("fill_branch", 120), 120, 8, 240)
        elif kind == "learned_operator":
            clean["name"] = safe_operator_name(item.get("name", ""))
            clean["params"] = _clean_params(item.get("params", {}))
            clean["accept"] = _accept_mode(item.get("accept", "bounded"))
            clean["max_worsen"] = _as_float(item.get("max_worsen", 5.0), 5.0, 0.0, 80.0)
            clean["max_worsen_ratio"] = _as_float(item.get("max_worsen_ratio", 0.01), 0.01, 0.0, 0.20)
        improvements.append(clean)
    result["improvements"] = improvements
    return result


def apply_strategy_guard(strategy, guard):
    """Apply harness-side limits that the model cannot override."""

    strategy = normalize_strategy(strategy or {})
    guard = str(guard or "none")
    if guard in ("none", "", "default"):
        return strategy
    if guard == "tail-polish":
        strategy["include_base"] = True
        strategy["polish_base"] = True
        strategy["base_time"] = _as_float(strategy.get("base_time", 8.35), 8.35, 8.30, 8.55)
        strategy["penalty"] = {"kind": "default"}
        strategy["per_start_budget"] = min(_as_float(strategy.get("per_start_budget", 0.08), 0.08, 0.02, 0.20), 0.12)
        strategy["selection_fixed_budget"] = min(
            _as_float(strategy.get("selection_fixed_budget", 0.04), 0.04, 0.0, 0.08),
            0.05,
        )
        strategy["starts"] = [{"kind": "all_single"}]
        cleaned = []
        for action in strategy.get("improvements") or ():
            kind = action.get("kind")
            if kind not in ALLOWED_IMPROVEMENTS:
                continue
            item = {"kind": kind, "budget": _as_float(action.get("budget", 0.20), 0.20, 0.0, 0.85)}
            if kind == "random_repartition":
                item["seed"] = _as_int(action.get("seed", 97), 97, 0, 10000000)
            elif kind == "pair_injection":
                item["top_candidates"] = _as_int(action.get("top_candidates", 6), 6, 1, 16)
                item["max_subset"] = _as_int(action.get("max_subset", 6), 6, 2, 8)
            elif kind == "pair_seed_search":
                item["scan_pairs"] = _as_int(action.get("scan_pairs", 13), 13, 4, 30)
                item["top_pairs"] = _as_int(action.get("top_pairs", 7), 7, 3, 12)
                item["deep_trials"] = _as_int(action.get("deep_trials", 3), 3, 1, 8)
            elif kind == "ejection_chain":
                item["max_depth"] = _as_int(action.get("max_depth", 5), 5, 1, 8)
                item["width"] = _as_int(action.get("width", 80), 80, 8, 240)
                item["branch"] = _as_int(action.get("branch", 80), 80, 8, 200)
            elif kind == "repair_chain":
                item["max_depth"] = _as_int(action.get("max_depth", 7), 7, 1, 10)
                item["width"] = _as_int(action.get("width", 180), 180, 8, 260)
                item["start_branch"] = _as_int(action.get("start_branch", 400), 400, 16, 600)
                item["fill_branch"] = _as_int(action.get("fill_branch", 120), 120, 8, 240)
            elif kind == "learned_operator":
                item["name"] = safe_operator_name(action.get("name", ""))
                item["params"] = _clean_params(action.get("params", {}))
                item["accept"] = _accept_mode(action.get("accept", "bounded"))
                item["max_worsen"] = _as_float(action.get("max_worsen", 5.0), 5.0, 0.0, 80.0)
                item["max_worsen_ratio"] = _as_float(action.get("max_worsen_ratio", 0.01), 0.01, 0.0, 0.20)
            cleaned.append(item)
        if not cleaned:
            cleaned = [
                {"kind": "improve_fixed", "budget": 0.08},
                {"kind": "pair_injection", "budget": 0.34, "top_candidates": 6, "max_subset": 6},
                {"kind": "random_repartition", "seed": 97, "budget": 0.34},
                {"kind": "pair_sweep", "budget": 0.24},
            ]
        strategy["improvements"] = cleaned[:8]
        return strategy
    return strategy


def strategy_issues(strategy):
    """Return human-readable issues for clearly invalid model output."""

    issues = []
    if not isinstance(strategy, dict):
        return ["STRATEGY must be a dictionary"]
    if "starts" not in strategy:
        issues.append("missing starts list")
    elif not isinstance(strategy.get("starts"), list):
        issues.append("starts must be a list")
    if "improvements" in strategy and not isinstance(strategy.get("improvements"), list):
        issues.append("improvements must be a list")
    operators = strategy.get("operators", {})
    if operators and not isinstance(operators, dict):
        issues.append("operators must be a dictionary of name -> source")
        operators = {}
    clean_operator_names = set()
    for name, source in (operators or {}).items():
        clean_name = safe_operator_name(name)
        if not clean_name:
            issues.append("operator name %r is not allowed" % (name,))
            continue
        clean_operator_names.add(clean_name)
        for issue in learned_operator_issues(source):
            issues.append("operator %s: %s" % (clean_name, issue))
    for index, start in enumerate(strategy.get("starts") or []):
        if not isinstance(start, dict):
            issues.append("starts[%d] must be a dictionary" % index)
            continue
        if str(start.get("kind", "")) not in ALLOWED_STARTS:
            issues.append("starts[%d].kind is not allowed" % index)
    for index, action in enumerate(strategy.get("improvements") or []):
        if not isinstance(action, dict):
            issues.append("improvements[%d] must be a dictionary" % index)
            continue
        if str(action.get("kind", "")) not in ALLOWED_IMPROVEMENTS:
            issues.append("improvements[%d].kind is not allowed" % index)
        elif str(action.get("kind", "")) == "learned_operator":
            name = safe_operator_name(action.get("name", ""))
            if not name:
                issues.append("improvements[%d].name is required for learned_operator" % index)
            elif name not in clean_operator_names:
                issues.append("improvements[%d] references unknown learned operator %s" % (index, name))
    return issues


def _clean_float_list(value, default, low, high, limit):
    if not isinstance(value, list):
        value = list(default)
    result = []
    for item in value[:limit]:
        number = _as_float(item, default[0], low, high)
        if number not in result:
            result.append(number)
    return result or list(default)


def _clean_int_list(value, default, low, high, limit):
    if not isinstance(value, list):
        value = list(default)
    result = []
    for item in value[:limit]:
        number = _as_int(item, default[0], low, high)
        if number not in result:
            result.append(number)
    return result or list(default)


def _groups_key(groups):
    return tuple(sorted(int(mask) for mask in groups))


def _valid_groups(problem, groups):
    used = 0
    cleaned = []
    for mask in groups or ():
        try:
            mask = int(mask)
        except Exception:
            return None
        if mask <= 0 or mask not in problem.by_mask:
            return None
        if used & mask:
            return None
        used |= mask
        cleaned.append(mask)
    if used != problem.all_task_mask:
        return None
    return tuple(sorted(cleaned))


def make_groups(problem, start_spec):
    kind = start_spec.get("kind")
    if kind == "all_single":
        return base.all_single_grouping(problem)
    if kind == "expected":
        return base.make_expected_grouping(
            problem,
            _mode(start_spec.get("mode"), ALLOWED_EXPECTED_MODES, "pair_gain"),
            _as_float(start_spec.get("threshold", 0.0), 0.0, -300.0, 300.0),
        )
    if kind == "score":
        return base.make_score_grouping(
            problem,
            _as_float(start_spec.get("alpha", 30.0), 30.0, -300.0, 300.0),
            _as_float(start_spec.get("threshold", 0.0), 0.0, -300.0, 300.0),
        )
    if kind == "potential":
        return base.make_potential_grouping(
            problem,
            _mode(start_spec.get("mode"), ALLOWED_POTENTIAL_MODES, "potential_gain"),
            _as_int(start_spec.get("top_k", 3), 3, 1, 8),
            _as_float(start_spec.get("threshold", 0.0), 0.0, -400.0, 400.0),
        )
    return ()


def make_group_sets(problem, start_spec):
    kind = start_spec.get("kind")
    if kind != "sweep":
        groups = make_groups(problem, start_spec)
        return [groups] if groups else []

    group_sets = [base.all_single_grouping(problem)]
    for mode in ("pair_gain", "pair_raw", "pair_half"):
        for threshold in start_spec.get("expected_thresholds") or ():
            group_sets.append(base.make_expected_grouping(problem, mode, threshold))
    for top_k in start_spec.get("top_ks") or ():
        for mode in ("potential_gain", "potential_half", "potential_raw"):
            for threshold in start_spec.get("potential_thresholds") or ():
                group_sets.append(base.make_potential_grouping(problem, mode, top_k, threshold))
    for alpha in start_spec.get("score_alphas") or ():
        for threshold in start_spec.get("score_thresholds") or ():
            group_sets.append(base.make_score_grouping(problem, alpha, threshold))
    return group_sets


def _time_slice(deadline, budget):
    now = time.perf_counter()
    if now >= deadline:
        return now
    budget = _as_float(budget, 0.0, 0.0, 8.5)
    if budget <= 0.0:
        return now
    return min(deadline, now + budget)


def _safe_builtins():
    names = (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "range",
        "reversed",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    )
    builtins_obj = __builtins__
    if isinstance(builtins_obj, dict):
        return dict((name, builtins_obj[name]) for name in names if name in builtins_obj)
    return dict((name, getattr(builtins_obj, name)) for name in names if hasattr(builtins_obj, name))


def _load_learned_apply(source):
    issues = learned_operator_issues(source)
    if issues:
        return None
    namespace = {}
    globals_dict = {
        "__builtins__": _safe_builtins(),
        "base": base,
        "time": time,
        "EPS": EPS,
        "INF": INF,
    }
    try:
        exec(str(source), globals_dict, namespace)
    except Exception:
        return None
    apply = namespace.get("apply")
    if apply is None:
        apply = globals_dict.get("apply")
    return apply if callable(apply) else None


def _accept_learned_value(old_value, trial_value, action):
    if trial_value >= INF * 0.5:
        return False
    if trial_value + EPS < old_value:
        return True
    mode = _accept_mode(action.get("accept", "bounded"))
    if mode == "improve":
        return False
    if mode == "always_legal":
        return True
    max_worsen = _as_float(action.get("max_worsen", 5.0), 5.0, 0.0, 80.0)
    max_worsen_ratio = _as_float(action.get("max_worsen_ratio", 0.01), 0.01, 0.0, 0.20)
    allowed_worsen = max(max_worsen, abs(old_value) * max_worsen_ratio)
    return trial_value <= old_value + allowed_worsen + EPS


def apply_learned_operator(problem, state, value, deadline, source, action):
    """Run a generated action and accept legal improving or bounded exploratory states."""

    apply = _load_learned_apply(source)
    if apply is None:
        return state, value
    try:
        start_state = base.kernel.normalize_state([list(offers) for offers in state if offers])
        result = apply(problem, [list(offers) for offers in start_state], value, deadline, _clean_params(action.get("params", {})))
    except Exception:
        return state, value

    trial_state = None
    if isinstance(result, tuple) and len(result) >= 1:
        trial_state = result[0]
    elif isinstance(result, list):
        trial_state = result
    if trial_state is None:
        return state, value

    try:
        trial_state = base.kernel.normalize_state([list(offers) for offers in trial_state if offers])
        rows = base.state_to_output(trial_state)
        if base.validate_verified_output(problem, base.normalize_rows(rows)) is None:
            return state, value
        trial_value = base.evaluate_state(problem, trial_state)
    except Exception:
        return state, value
    if _accept_learned_value(value, trial_value, action):
        return trial_state, trial_value
    return state, value


def improve_state(problem, state, value, improvements, deadline, learned_operators=None):
    learned_operators = learned_operators or {}
    best_state = base.kernel.normalize_state([list(offers) for offers in state if offers])
    state = best_state
    best_value = value
    for action in improvements:
        if time.perf_counter() >= deadline:
            break
        budget = action.get("budget", 0.0)
        action_deadline = _time_slice(deadline, budget)
        if action_deadline <= time.perf_counter():
            continue
        kind = action.get("kind")
        if kind == "improve_fixed":
            trial = base.improve_fixed_groups(problem, state, action_deadline)
            trial_value = base.evaluate_state(problem, trial)
            if trial_value + EPS < value:
                state, value = trial, trial_value
        elif kind == "local_repartition":
            trial, trial_value = base.local_repartition(problem, state, value, action_deadline)
            if trial_value + EPS < value:
                state, value = trial, trial_value
        elif kind == "random_repartition":
            seed = _as_int(action.get("seed", 97), 97, 0, 10000000)
            trial, trial_value = base.random_repartition(problem, state, value, action_deadline, seed)
            if trial_value + EPS < value:
                state, value = trial, trial_value
        elif kind == "pair_sweep":
            trial, trial_value = base.pair_repartition_sweep(problem, state, value, action_deadline)
            if trial_value + EPS < value:
                state, value = trial, trial_value
        elif kind == "pair_injection":
            trial, trial_value = pair_injection(
                problem,
                state,
                value,
                action_deadline,
                _as_int(action.get("top_candidates", 6), 6, 1, 16),
                _as_int(action.get("max_subset", 6), 6, 2, 8),
            )
            if trial_value + EPS < value:
                state, value = trial, trial_value
        elif kind == "pair_seed_search":
            trial, trial_value = pair_seed_search(
                problem,
                action_deadline,
                _as_int(action.get("scan_pairs", 13), 13, 4, 30),
                _as_int(action.get("top_pairs", 7), 7, 3, 12),
                _as_int(action.get("deep_trials", 3), 3, 1, 8),
            )
            if trial is not None and trial_value + EPS < value:
                state, value = trial, trial_value
        elif kind == "ejection_chain":
            trial, trial_value = ejection_chain_search(
                problem,
                state,
                value,
                action_deadline,
                _as_int(action.get("max_depth", 5), 5, 1, 8),
                _as_int(action.get("width", 80), 80, 8, 240),
                _as_int(action.get("branch", 80), 80, 8, 200),
            )
            if trial_value + EPS < value:
                state, value = trial, trial_value
        elif kind == "repair_chain":
            trial, trial_value = repair_chain_search(
                problem,
                state,
                value,
                action_deadline,
                _as_int(action.get("max_depth", 7), 7, 1, 10),
                _as_int(action.get("width", 180), 180, 8, 260),
                _as_int(action.get("start_branch", 400), 400, 16, 600),
                _as_int(action.get("fill_branch", 120), 120, 8, 240),
            )
            if trial_value + EPS < value:
                state, value = trial, trial_value
        elif kind == "learned_operator":
            name = safe_operator_name(action.get("name", ""))
            source = learned_operators.get(name, "")
            trial, trial_value = apply_learned_operator(
                problem,
                state,
                value,
                action_deadline,
                source,
                action,
            )
            state, value = trial, trial_value
        if value + EPS < best_value:
            best_state = base.kernel.normalize_state([list(offers) for offers in state if offers])
            best_value = value
    return best_state, best_value


def ejection_chain_search(problem, state, value, deadline, max_depth, width, branch):
    """Beam over legal courier moves, allowing temporary uphill prefixes."""

    normalize_state = base.kernel.normalize_state
    group_value = base.group_value
    evaluate_state = base.evaluate_state
    by_mask_courier = problem.by_mask_courier

    best_state = normalize_state([list(offers) for offers in state if offers])
    best_value = value
    beam = [(value, best_state)]

    for _ in range(max_depth):
        if time.perf_counter() >= deadline:
            break

        next_states = []
        seen = set()
        for _, current in beam:
            if time.perf_counter() >= deadline:
                break

            values = [group_value(offers, offers[0].task_count) for offers in current]
            moves = []
            for source_index, source in enumerate(current):
                if len(source) <= 1:
                    continue
                source_value = values[source_index]
                for offer_index, old_candidate in enumerate(source):
                    source_without = source[:offer_index] + source[offer_index + 1 :]
                    source_without_value = group_value(source_without, source[0].task_count)
                    for target_index, target in enumerate(current):
                        if target_index == source_index:
                            continue
                        moved = by_mask_courier.get(target[0].mask, {}).get(old_candidate.courier)
                        if moved is None:
                            continue
                        delta = (
                            source_without_value
                            + group_value(target + [moved], target[0].task_count)
                            - source_value
                            - values[target_index]
                        )
                        moves.append((delta, source_index, offer_index, target_index, moved))

            moves.sort(key=lambda item: item[0])
            for _, source_index, offer_index, target_index, moved in moves[:branch]:
                if time.perf_counter() >= deadline:
                    break
                trial = [list(offers) for offers in current]
                del trial[source_index][offer_index]
                trial[target_index].append(moved)
                trial = normalize_state(trial)
                key = tuple(tuple(candidate.courier for candidate in offers) for offers in trial)
                if key in seen:
                    continue
                seen.add(key)
                next_states.append((evaluate_state(problem, trial), trial))

        if not next_states:
            break

        next_states.sort(key=lambda item: item[0])
        beam = next_states[:width]
        if beam[0][0] + EPS < best_value:
            best_value, best_state = beam[0]

    return best_state, best_value


def repair_chain_search(problem, state, value, deadline, max_depth, width, start_branch, fill_branch):
    """Start from high target-gain moves, then repair the source group."""

    normalize_state = base.kernel.normalize_state
    group_value = base.group_value
    evaluate_state = base.evaluate_state
    by_mask_courier = problem.by_mask_courier

    start_state = normalize_state([list(offers) for offers in state if offers])
    best_state = start_state
    best_value = value
    values = [group_value(offers, offers[0].task_count) for offers in start_state]
    starts = []

    for source_index, source in enumerate(start_state):
        if len(source) <= 1:
            continue
        source_value = values[source_index]
        for offer_index, old_candidate in enumerate(source):
            source_without = source[:offer_index] + source[offer_index + 1 :]
            source_without_value = group_value(source_without, source[0].task_count)
            for target_index, target in enumerate(start_state):
                if target_index == source_index:
                    continue
                moved = by_mask_courier.get(target[0].mask, {}).get(old_candidate.courier)
                if moved is None:
                    continue
                target_with_value = group_value(target + [moved], target[0].task_count)
                target_delta = target_with_value - values[target_index]
                delta = source_without_value + target_with_value - source_value - values[target_index]
                starts.append((target_delta, delta, source_index, offer_index, target_index, moved))

    starts.sort(key=lambda item: item[0])
    beam = []
    for _, _, source_index, offer_index, target_index, moved in starts[:start_branch]:
        if time.perf_counter() >= deadline:
            break
        source_mask = start_state[source_index][0].mask
        trial = [list(offers) for offers in start_state]
        del trial[source_index][offer_index]
        trial[target_index].append(moved)
        target_value = group_value(start_state[target_index] + [moved], start_state[target_index][0].task_count)
        trial = normalize_state(trial)
        trial_value = evaluate_state(problem, trial)
        focus = target_value - values[target_index]
        active_index = _mask_index(trial, source_mask)
        if active_index >= 0:
            moved_key = (start_state[source_index][offer_index].courier,)
            beam.append((trial_value, trial, active_index, focus, moved_key))
        if trial_value + EPS < best_value:
            best_value, best_state = trial_value, trial

    beam = _select_repair_beam(beam, width)

    for _ in range(1, max_depth):
        if time.perf_counter() >= deadline:
            break

        next_states = []
        seen = set()
        for _, current, active_index, _, moved_key in beam:
            if time.perf_counter() >= deadline:
                break

            values = [group_value(offers, offers[0].task_count) for offers in current]
            active = current[active_index]
            active_value = values[active_index]
            fills = []
            for source_index, source in enumerate(current):
                if source_index == active_index or len(source) <= 1:
                    continue
                source_value = values[source_index]
                for offer_index, old_candidate in enumerate(source):
                    if old_candidate.courier in moved_key:
                        continue
                    moved = by_mask_courier.get(active[0].mask, {}).get(old_candidate.courier)
                    if moved is None:
                        continue
                    source_without = source[:offer_index] + source[offer_index + 1 :]
                    source_without_value = group_value(source_without, source[0].task_count)
                    active_with_value = group_value(active + [moved], active[0].task_count)
                    target_delta = active_with_value - active_value
                    delta = active_with_value + source_without_value - active_value - source_value
                    fills.append((target_delta, delta, source_index, offer_index, moved))

            fills.sort(key=lambda item: item[0])
            for _, _, source_index, offer_index, moved in fills[:fill_branch]:
                if time.perf_counter() >= deadline:
                    break
                source_mask = current[source_index][0].mask
                trial = [list(offers) for offers in current]
                del trial[source_index][offer_index]
                trial[active_index].append(moved)
                active_value_after = group_value(active + [moved], active[0].task_count)
                trial = normalize_state(trial)
                key = tuple(tuple(candidate.courier for candidate in offers) for offers in trial)
                if key in seen:
                    continue
                seen.add(key)
                trial_value = evaluate_state(problem, trial)
                focus = active_value_after - active_value
                new_active_index = _mask_index(trial, source_mask)
                if new_active_index >= 0:
                    next_states.append(
                        (trial_value, trial, new_active_index, focus, moved_key + (current[source_index][offer_index].courier,))
                    )
                if trial_value + EPS < best_value:
                    best_value, best_state = trial_value, trial

        if not next_states:
            break

        beam = _select_repair_beam(next_states, width)

    return best_state, best_value


def _mask_index(state, mask):
    for index, offers in enumerate(state):
        if offers and offers[0].mask == mask:
            return index
    return -1


def _select_repair_beam(entries, width):
    """Keep both good prefixes and high-target-gain uphill prefixes."""

    if len(entries) <= width:
        return list(entries)

    selected = []
    seen = set()

    def add(entry):
        key = (entry[2], tuple(tuple(candidate.courier for candidate in offers) for offers in entry[1]))
        if key in seen:
            return
        seen.add(key)
        selected.append(entry)

    value_quota = max(1, width // 2)
    for entry in sorted(entries, key=lambda item: item[0])[:value_quota]:
        add(entry)
    by_active = {}
    for entry in entries:
        active_offers = entry[1][entry[2]]
        active_mask = active_offers[0].mask if active_offers else 0
        bucket = by_active.setdefault(active_mask, [])
        bucket.append(entry)
    for bucket in by_active.values():
        bucket.sort(key=lambda item: item[3])
        for entry in bucket[:3]:
            add(entry)
            if len(selected) >= width:
                return selected[:width]
    for entry in sorted(entries, key=lambda item: item[3])[:width - len(selected)]:
        add(entry)
    if len(selected) < width:
        for entry in sorted(entries, key=lambda item: item[0]):
            add(entry)
            if len(selected) >= width:
                break
    return selected[:width]


def _groups_for_forced_pairs(problem, pair_masks):
    used = 0
    groups = []
    for mask in pair_masks:
        if used & mask:
            return ()
        used |= mask
        groups.append(mask)
    for index in range(problem.n_tasks):
        bit = 1 << index
        if not (used & bit) and bit in problem.by_mask:
            groups.append(bit)
    return tuple(sorted(groups))


def _forced_pair_state(problem, pair_masks, deadline, anneal_budget):
    groups = _groups_for_forced_pairs(problem, pair_masks)
    if not groups:
        return None, INF
    state = base.greedy_assignment(problem, groups, deadline)
    if not state:
        return None, INF
    if time.perf_counter() < deadline:
        state = base.kernel.reassign_used_couriers(problem, state, min(deadline, time.perf_counter() + 0.14))
    if time.perf_counter() < deadline:
        state = base.kernel.chain_reassign_used_couriers(problem, state, min(deadline, time.perf_counter() + 0.12))
    if anneal_budget > 0.0 and time.perf_counter() < deadline:
        state = base.kernel.anneal_used_couriers(problem, state, min(deadline, time.perf_counter() + anneal_budget))
    value = base.evaluate_state(problem, state)
    return state, value


def _disjoint_combo(combo):
    used = 0
    for mask in combo:
        if used & mask:
            return False
        used |= mask
    return True


def pair_seed_search(problem, deadline, scan_pairs, top_pairs, deep_trials):
    """Build sparse pair starts, then solve them with stronger courier anneal."""

    pair_groups = [
        mask
        for mask in base.make_potential_grouping(problem, "potential_raw", 3, 0.0)
        if base.bit_count(mask) == 2
    ][:scan_pairs]
    if len(pair_groups) < 2:
        return None, INF

    singles = []
    for mask in pair_groups:
        if time.perf_counter() >= deadline - 0.40:
            break
        _, value = _forced_pair_state(problem, (mask,), min(deadline, time.perf_counter() + 0.18), 0.0)
        singles.append((value, mask))
    singles.sort(key=lambda item: item[0])
    top = [mask for _, mask in singles[:top_pairs]]
    if len(top) < 2:
        return None, INF

    combos = []
    base_pair = tuple(top[:2])
    if _disjoint_combo(base_pair):
        combos.append(base_pair)
    for mask in top[2:]:
        combo = tuple(top[:2] + [mask])
        if _disjoint_combo(combo):
            combos.append(combo)
    if len(top) >= 4:
        for index in range(2, min(len(top), 6)):
            combo = (top[0], top[index])
            if _disjoint_combo(combo):
                combos.append(combo)
    seen = set()
    quick = []
    for combo in combos:
        if time.perf_counter() >= deadline - 0.75:
            break
        combo = tuple(sorted(combo))
        if combo in seen:
            continue
        seen.add(combo)
        _, value = _forced_pair_state(problem, combo, min(deadline, time.perf_counter() + 0.22), 0.0)
        quick.append((value, combo))
    quick.sort(key=lambda item: item[0])

    best_state = None
    best_value = INF
    for _, combo in quick[:deep_trials]:
        if time.perf_counter() >= deadline - 0.35:
            break
        state, value = _forced_pair_state(problem, combo, min(deadline, time.perf_counter() + 1.75), 1.30)
        if state is not None and value + EPS < best_value:
            best_state, best_value = state, value
    return best_state, best_value


def _state_owners(state):
    task_owner = {}
    courier_owner = {}
    for index, offers in enumerate(state):
        if not offers:
            continue
        mask = offers[0].mask
        bit = 1
        while bit <= mask:
            if mask & bit:
                task_owner[bit] = index
            bit <<= 1
        for candidate in offers:
            courier_owner[candidate.courier] = index
    return task_owner, courier_owner


def pair_injection(problem, state, best_value, deadline, top_candidates, max_subset):
    """Force promising pair masks and include groups holding their best couriers."""

    current = base.kernel.normalize_state(state)
    current_value = best_value
    if len(current) < 2:
        return current, current_value

    improved = True
    while improved and time.perf_counter() < deadline:
        improved = False
        task_owner, courier_owner = _state_owners(current)
        moves = []
        for pair_mask, left, right in problem.pair_items:
            if time.perf_counter() >= deadline:
                break
            left_owner = task_owner.get(left)
            right_owner = task_owner.get(right)
            if left_owner is None or right_owner is None:
                continue
            subset = set([left_owner, right_owner])
            candidates = list(problem.by_mask.get(pair_mask, ())[:top_candidates])
            preferred = base.best_offer_subset(candidates) if candidates else []
            for candidate in preferred:
                owner = courier_owner.get(candidate.courier)
                if owner is not None:
                    subset.add(owner)
            if len(subset) > max_subset:
                subset = set([left_owner, right_owner])
                for candidate in preferred:
                    owner = courier_owner.get(candidate.courier)
                    if owner is not None:
                        subset.add(owner)
                    if len(subset) >= max_subset:
                        break
            if 2 <= len(subset) <= max_subset:
                moves.append(tuple(sorted(subset)))

        seen = set()
        for subset in moves:
            if time.perf_counter() >= deadline:
                break
            if subset in seen:
                continue
            seen.add(subset)
            trial_deadline = min(deadline, time.perf_counter() + 0.055 + 0.018 * len(subset))
            trial = base.kernel.best_repartition_for_subset(problem, current, subset, trial_deadline)
            if trial is None:
                continue
            value = base.evaluate_state(problem, trial)
            if value + EPS < current_value:
                current = trial
                current_value = value
                improved = True
                break

    return current, current_value


def run_groups(problem, groups, improvements, deadline):
    groups = _valid_groups(problem, groups)
    if not groups:
        return None, INF
    state = base.greedy_assignment(problem, groups, deadline)
    if not state:
        return None, INF
    value = base.evaluate_state(problem, state)
    if value >= INF / 2:
        return None, INF
    state, value = improve_state(problem, state, value, improvements, deadline)
    return state, value


def initial_state_for_groups(problem, groups, deadline, fixed_budget):
    groups = _valid_groups(problem, groups)
    if not groups:
        return None, INF
    state = base.greedy_assignment(problem, groups, deadline)
    if not state:
        return None, INF
    if fixed_budget > 0.0 and time.perf_counter() < deadline:
        state = base.improve_fixed_groups(problem, state, min(deadline, time.perf_counter() + fixed_budget))
    value = base.evaluate_state(problem, state)
    if value >= INF / 2:
        return None, INF
    return state, value


def _score_rows(problem, rows):
    normalized = base.normalize_rows(rows)
    if base.validate_verified_output(problem, normalized) is None:
        return INF
    return base.verified_output_value(problem, normalized)


def _mask_from_task_key(problem, task_key):
    mask = 0
    for task in str(task_key).split(","):
        task = task.strip()
        if task not in problem.task_to_idx:
            return None
        mask |= 1 << problem.task_to_idx[task]
    return mask


def rows_to_state(problem, rows):
    """Convert legal output rows back to Candidate state rows."""

    normalized = base.normalize_rows(rows)
    if base.validate_verified_output(problem, normalized) is None:
        return None
    state = []
    for task_key, couriers in normalized:
        mask = _mask_from_task_key(problem, task_key)
        if mask is None:
            return None
        lookup = problem.by_mask_courier.get(mask, {})
        offers = []
        for courier in couriers:
            candidate = lookup.get(courier)
            if candidate is None or candidate.task_key != task_key:
                return None
            offers.append(candidate)
        if not offers:
            return None
        state.append(offers)
    return base.kernel.normalize_state(state)


def _apply_penalty(problem, penalty):
    old_penalty = base.kernel.FAIL_PENALTY
    kind = penalty.get("kind")
    new_penalty = old_penalty
    if kind == "fixed":
        new_penalty = _as_float(penalty.get("value", old_penalty), old_penalty, 60.0, 140.0)
    elif kind == "profile":
        courier_count = len(problem.all_couriers)
        if courier_count <= max(problem.n_tasks + 3, int(problem.n_tasks * 1.15)):
            new_penalty = _as_float(penalty.get("scarce", old_penalty), old_penalty, 60.0, 140.0)
        elif courier_count >= int(problem.n_tasks * 1.8):
            new_penalty = _as_float(penalty.get("dense", old_penalty), old_penalty, 60.0, 140.0)
    if abs(new_penalty - old_penalty) > EPS:
        base.kernel.FAIL_PENALTY = new_penalty
        base.kernel.refresh_candidate_order(problem)
    return old_penalty


def solve_with_strategy(input_text, strategy=None):
    """Solve one case using a constrained strategy dictionary."""

    strategy = normalize_strategy(strategy or {})
    start_time = time.perf_counter()
    deadline = start_time + strategy["time_limit"]
    problem = base.parse_input(input_text)
    if problem.n_tasks <= 0:
        return []

    old_penalty = _apply_penalty(problem, strategy.get("penalty") or {})
    learned_operators = strategy.get("operators") or {}
    old_limit = base.DEFAULT_TIME_LIMIT
    best_rows = None
    best_value = INF
    try:
        if strategy.get("include_base"):
            base_budget = min(strategy["base_time"], max(0.0, deadline - time.perf_counter() - 0.05))
            if base_budget > 0.02:
                base.DEFAULT_TIME_LIMIT = base_budget
                rows = base.solve(input_text)
                value = _score_rows(problem, rows)
                if value + EPS < best_value:
                    best_rows, best_value = rows, value
                if strategy.get("polish_base") and time.perf_counter() < deadline - 0.05:
                    state = rows_to_state(problem, rows)
                    if state is not None:
                        polished_state, polished_value = improve_state(
                            problem,
                            state,
                            value,
                            strategy.get("improvements") or (),
                            deadline - 0.03,
                            learned_operators,
                        )
                        polished_rows = base.state_to_output(polished_state)
                        polished_row_value = _score_rows(problem, polished_rows)
                        if polished_row_value + EPS < best_value:
                            best_rows, best_value = polished_rows, polished_row_value

        seen = set()
        best_state = None
        best_state_value = INF
        for start_spec in strategy.get("starts") or ():
            if time.perf_counter() >= deadline - 0.04:
                break
            for groups in make_group_sets(problem, start_spec):
                if time.perf_counter() >= deadline - 0.04:
                    break
                groups = _valid_groups(problem, groups)
                if not groups:
                    continue
                key = _groups_key(groups)
                if key in seen:
                    continue
                seen.add(key)
                state_deadline = min(deadline - 0.02, time.perf_counter() + strategy.get("per_start_budget", 0.28))
                state, value = initial_state_for_groups(
                    problem,
                    groups,
                    state_deadline,
                    strategy.get("selection_fixed_budget", 0.08),
                )
                if state is None:
                    continue
                if value + EPS < best_state_value:
                    best_state, best_state_value = state, value

        if best_state is not None and time.perf_counter() < deadline - 0.03:
            best_state, best_state_value = improve_state(
                problem,
                best_state,
                best_state_value,
                strategy.get("improvements") or (),
                deadline - 0.02,
                learned_operators,
            )
            rows = base.state_to_output(best_state)
            row_value = _score_rows(problem, rows)
            if row_value + EPS < best_value:
                best_rows, best_value = rows, row_value

        if best_rows is not None:
            return best_rows
        base.DEFAULT_TIME_LIMIT = max(0.20, min(strategy["time_limit"], deadline - start_time))
        return base.solve(input_text)
    finally:
        base.DEFAULT_TIME_LIMIT = old_limit
        if abs(base.kernel.FAIL_PENALTY - old_penalty) > EPS:
            base.kernel.FAIL_PENALTY = old_penalty


def make_solver_source(strategy, notes=None):
    """Build a tiny candidate file from strategy + audit notes."""

    strategy = normalize_strategy(strategy or {})
    if not isinstance(notes, dict):
        notes = {}
    cleaned_notes = {
        "hypothesis": str(notes.get("hypothesis", "constrained strategy search")),
        "changed_parts": list(notes.get("changed_parts", ["strategy parameters"]))
        if isinstance(notes.get("changed_parts", []), list)
        else [str(notes.get("changed_parts"))],
        "expected_effect": str(notes.get("expected_effect", "test a bounded strategy without touching state internals")),
        "risk": str(notes.get("risk", "may be baseline-equivalent or worse on local cases")),
        "next_if_failed": str(notes.get("next_if_failed", "mutate starts, budgets, or operator order")),
    }
    return "\n".join(
        [
            "from __future__ import print_function",
            "import os, sys",
            "ROOT = %r" % ROOT,
            "if ROOT not in sys.path: sys.path.insert(0, ROOT)",
            "from autosolver_workbench import strategy_executor",
            "STRATEGY = %s" % repr(strategy),
            "CANDIDATE_NOTES = %s" % repr(cleaned_notes),
            "DEFAULT_TIME_LIMIT = float(STRATEGY.get('time_limit', 9.35))",
            "",
            "def solve(input_text):",
            "    STRATEGY['time_limit'] = float(DEFAULT_TIME_LIMIT)",
            "    return strategy_executor.solve_with_strategy(input_text, STRATEGY)",
            "",
        ]
    )


def strategy_json(strategy):
    return json.dumps(normalize_strategy(strategy or {}), ensure_ascii=False, indent=2, sort_keys=True)
