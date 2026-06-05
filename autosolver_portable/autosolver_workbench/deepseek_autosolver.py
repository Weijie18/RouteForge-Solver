"""DeepSeek-driven AutoSolver loop for MT-4.

This is an offline search harness.  It asks an LLM to propose a complete
candidate ``solve(input_text)`` implementation, evaluates that candidate on
local cases, then feeds the measured result back into the next prompt.

The generated contest candidate is still required to be Python 3.6 compatible
and standard-library-only.  LLM connection settings can be edited in
``LLM_SETTINGS`` below, then optionally overridden by CLI flags or environment
variables.  API keys are never written to history.
"""

from __future__ import print_function

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autosolver_workbench import evaluator  # noqa: E402
from autosolver_workbench import strategy_executor  # noqa: E402


RUN_DIR = os.path.join(ROOT, "outputs", "autosolver_workbench", "deepseek_runs")

# Edit this block when sharing/running the AutoSolver with a different
# OpenAI-compatible LLM endpoint.  Keep api_key empty in committed copies; use a
# local edit, --api-key-file, or the provider's environment variable for real
# credentials.
LLM_SETTINGS = {
    "provider": "deepseek",
    "api_key": "",
    "model": "",
    "base_url": "",
    "providers": {
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-pro",
            "api_key_env": "DEEPSEEK_API_KEY",
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4.1",
            "api_key_env": "OPENAI_API_KEY",
        },
        "custom": {
            "base_url": "https://your-openai-compatible-host.example/v1",
            "model": "your-model-name",
            "api_key_env": "LLM_API_KEY",
        },
    },
}

try:
    from autosolver_workbench.llm_private_config import LLM_SETTINGS_OVERRIDE
except Exception:
    LLM_SETTINGS_OVERRIDE = {}


def merge_llm_settings(base, override):
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if key == "providers" and isinstance(value, dict):
            providers = dict(merged.get("providers") or {})
            for provider_name, provider_config_value in value.items():
                existing = dict(providers.get(provider_name) or {})
                if isinstance(provider_config_value, dict):
                    existing.update(provider_config_value)
                    providers[provider_name] = existing
            merged["providers"] = providers
        else:
            merged[key] = value
    return merged


LLM_SETTINGS = merge_llm_settings(LLM_SETTINGS, LLM_SETTINGS_OVERRIDE)

DEFAULT_PROVIDER = str(LLM_SETTINGS.get("provider") or "deepseek")
_DEFAULT_PROVIDER_CONFIG = (LLM_SETTINGS.get("providers") or {}).get(DEFAULT_PROVIDER, {})
DEFAULT_MODEL = str(LLM_SETTINGS.get("model") or _DEFAULT_PROVIDER_CONFIG.get("model") or "deepseek-v4-pro")
DEFAULT_BASE_URL = str(
    LLM_SETTINGS.get("base_url") or _DEFAULT_PROVIDER_CONFIG.get("base_url") or "https://api.deepseek.com"
)
MAX_SOURCE_CHARS = 82000


SYSTEM_PROMPT = """You are an expert algorithm engineer for an MT-4 delivery assignment optimization contest.

You are not allowed to write a full solver in this loop.  You may only propose
a constrained strategy dictionary for the local executor.  The generated
candidate file will be assembled by the harness and will define solve(input_text).

Return one Python code block containing these dictionaries:

    STRATEGY = {
        "time_limit": 9.35,
        "include_base": True,
        "base_time": 8.5,
        "penalty": {"kind": "default"},
        "starts": [...],
        "improvements": [...]
    }

and:

    CANDIDATE_NOTES = {
        "hypothesis": "...",
        "changed_parts": ["..."],
        "expected_effect": "...",
        "risk": "...",
        "next_if_failed": "..."
    }

When the user prompt explicitly says OPERATOR INVENTION MODE, you may also
return:

    LEARNED_OPERATORS = {
        "operator_name": r'''
def apply(problem, state, value, deadline, params):
    # Return a trial state or (trial_state, trial_value).
    return state, value
'''
    }

Then reference it from STRATEGY with:

    {"kind": "learned_operator", "name": "operator_name", "budget": 0.2,
     "accept": "bounded", "max_worsen": 5.0, "max_worsen_ratio": 0.01,
     "params": {...}}

These notes should summarize the strategy decision without revealing private chain-of-thought.

Problem summary:
- Input is TSV rows: task_id_list, courier_id, score, willingness.
- The first row is usually a header; robust parsers must skip it instead of parsing "total_score".
- A row may represent one task or a two-task bundle.
- A legal output covers every task exactly once.
- A courier may appear in at most one output row.
- Every emitted (task_id_list_str, courier_id) must exist in the input exactly.
- Multiple couriers may be offered the same selected task or pair bundle.
- Expected value for one selected bundle with offers S:
  miss = product(1 - willingness_j)
  accepted_score = sum(w_j * score_j) / sum(w_j)
  value = (1 - miss) * accepted_score + miss * 100 * bundle_size
- Objective is to minimize the sum of selected bundle values.

Search guidance:
- Do not hardcode task IDs, courier IDs, literal hidden solutions, filenames, or online detail rows.
- Improve algorithmic behavior only through the strategy DSL: initialization, pair grouping thresholds,
  adaptive penalty, local-search operator order, randomized repartition seeds, and budget allocation.
- A strategy that simply calls the base solver for the full time is not useful.  Keep some budget for
  non-baseline starts or operators.
- The harness owns all parsing, legality, Candidate state objects, exact task keys, courier IDs, and output format.
- Do not invent Python functions, state rows, or Problem/Candidate fields.
- In ordinary strategy mode, do not add LEARNED_OPERATORS.
- In OPERATOR INVENTION MODE, any learned operator must be a small bounded
  improvement action.  It must not import modules, read/write files, call eval/exec,
  hardcode task/courier IDs, invent task keys, or return illegal states.
- Learned operators are safety wrapped: invalid, exception-throwing, or illegal
  states are discarded by the harness.  A learned operator action may accept a
  legal bounded non-improving intermediate state when STRATEGY sets
  "accept": "bounded" or "always_legal"; the harness still returns the best
  state seen after the full improvement chain.
- Use bounded acceptance only when the temporary state is a plausible bridge for
  a later operator, and set conservative max_worsen / max_worsen_ratio values.
  Still write defensive code and check time.perf_counter() < deadline inside loops.
- Return only one fenced python code block.  No prose outside the code block.

Allowed STRATEGY schema:
- time_limit: float from 1.0 to 9.6
- include_base: bool
- base_time: float from 0.0 to 9.2
- polish_base: bool, whether to convert the base output back to state and run the improvement list on it
- per_start_budget: float from 0.02 to 2.0, used while scoring each start before final improvement
- selection_fixed_budget: float from 0.0 to 0.5, quick fixed-group polish during start scoring
- penalty:
  - {"kind": "default"}
  - {"kind": "fixed", "value": 60.0..140.0}
  - {"kind": "profile", "dense": 60.0..140.0, "scarce": 60.0..140.0}
- starts: list, max 32 items:
  - {"kind": "sweep", "expected_thresholds": [...], "potential_thresholds": [...],
     "score_alphas": [...], "score_thresholds": [...], "top_ks": [...]}
  - {"kind": "all_single"}
  - {"kind": "expected", "mode": "pair_gain"|"pair_half"|"pair_raw", "threshold": -300.0..300.0}
  - {"kind": "score", "alpha": -300.0..300.0, "threshold": -300.0..300.0}
  - {"kind": "potential", "mode": "potential_gain"|"potential_half"|"potential_raw", "top_k": 1..8, "threshold": -400.0..400.0}
- improvements: list, max 24 items:
  - {"kind": "improve_fixed", "budget": 0.0..3.0}
  - {"kind": "local_repartition", "budget": 0.0..3.0}
  - {"kind": "random_repartition", "seed": integer, "budget": 0.0..3.0}
  - {"kind": "pair_sweep", "budget": 0.0..3.0}
  - {"kind": "pair_injection", "budget": 0.0..3.0, "top_candidates": 1..16, "max_subset": 2..8}
  - {"kind": "pair_seed_search", "budget": 0.0..8.5, "scan_pairs": 4..30, "top_pairs": 3..12, "deep_trials": 1..8}
  - {"kind": "ejection_chain", "budget": 0.0..3.0, "max_depth": 1..8, "width": 8..240, "branch": 8..200}
  - {"kind": "repair_chain", "budget": 0.0..3.0, "max_depth": 1..10, "width": 8..260,
     "start_branch": 16..600, "fill_branch": 8..240}
  - {"kind": "learned_operator", "budget": 0.0..3.0, "name": "<LEARNED_OPERATORS key>",
     "accept": "improve"|"bounded"|"always_legal", "max_worsen": 0.0..80.0,
     "max_worsen_ratio": 0.0..0.20, "params": {"small_json_like": "values"}}
"""


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def read_text(path, limit=None):
    with open(path, "r", encoding="utf-8") as handle:
        data = handle.read()
    if limit is not None and len(data) > limit:
        return data[:limit] + "\n\n# ... truncated ...\n"
    return data


def write_text(path, text):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def read_jsonl(path):
    items = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items


def append_jsonl(path, item):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def case_label(path):
    return os.path.basename(path)


def result_summary(report):
    lines = []
    lines.append("average=%.6f ok=%d/%d" % (report["average"], report["ok_count"], report["case_count"]))
    for item in report.get("results", []):
        value = item.get("value")
        value_text = "NA" if value is None else "%.6f" % float(value)
        lines.append(
            "%s status=%s value=%s elapsed=%.1fms rows=%s offers=%s"
            % (
                item.get("case"),
                item.get("status"),
                value_text,
                float(item.get("elapsed_ms") or 0.0),
                item.get("rows"),
                item.get("offers"),
            )
        )
    return "\n".join(lines)


def extract_candidate_notes(code):
    try:
        tree = ast.parse(code)
    except Exception:
        return {}
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "CANDIDATE_NOTES":
                try:
                    value = ast.literal_eval(node.value)
                except Exception:
                    return {}
                if isinstance(value, dict):
                    cleaned = {}
                    for key, item in value.items():
                        if isinstance(key, str):
                            cleaned[key] = item
                    return cleaned
    return {}


def format_candidate_notes(notes):
    if not notes:
        return "No CANDIDATE_NOTES found."
    order = ("hypothesis", "changed_parts", "expected_effect", "risk", "next_if_failed")
    lines = []
    for key in order:
        if key not in notes:
            continue
        value = notes.get(key)
        if isinstance(value, (list, tuple)):
            value = "; ".join(str(item) for item in value)
        lines.append("%s: %s" % (key, value))
    for key in sorted(notes):
        if key in order:
            continue
        lines.append("%s: %s" % (key, notes.get(key)))
    return "\n".join(lines) if lines else "No readable CANDIDATE_NOTES found."


def best_average(history):
    best = None
    for item in history:
        report = item.get("report") or {}
        if report.get("ok_count") != report.get("case_count"):
            continue
        value = report.get("average")
        if value is None:
            continue
        value = float(value)
        if best is None or value < best:
            best = value
    return best


def no_improvement_streak(history):
    """Count completed rounds since the last strict incumbent improvement."""

    best = None
    streak = 0
    for item in history:
        report = item.get("report") or {}
        if report.get("ok_count") != report.get("case_count"):
            continue
        value = report.get("average")
        if value is None:
            continue
        value = float(value)
        if best is None or value < best - 1e-9:
            best = value
            streak = 0
        else:
            streak += 1
    return streak


def operator_mode_enabled(args, history):
    if bool(getattr(args, "force_operator_mode", False)):
        return True
    threshold = int(getattr(args, "operator_stagnation_rounds", 0) or 0)
    if threshold <= 0:
        return False
    return no_improvement_streak(history) >= threshold


def learned_operator_requested(strategy):
    if not isinstance(strategy, dict):
        return False
    if strategy.get("operators"):
        return True
    for action in strategy.get("improvements") or ():
        if isinstance(action, dict) and str(action.get("kind", "")) == "learned_operator":
            return True
    return False


def attempt_summary(attempts):
    if not attempts:
        return "No attempts recorded."
    lines = []
    for item in attempts:
        label = "attempt %s" % item.get("attempt")
        stage = item.get("stage", "?")
        if item.get("issues"):
            lines.append("%s: %s rejected: %s" % (label, stage, "; ".join(item.get("issues") or [])))
        elif item.get("error"):
            lines.append("%s: %s error: %s" % (label, stage, shorten(item.get("error"), 500).replace("\n", " | ")))
        elif item.get("report"):
            report = item.get("report")
            lines.append(
                "%s: %s average=%.6f ok=%s/%s"
                % (
                    label,
                    stage,
                    float(report.get("average", 1e100)),
                    report.get("ok_count"),
                    report.get("case_count"),
                )
            )
        else:
            lines.append("%s: %s" % (label, stage))
    return "\n".join(lines)


def write_round_report(args, record, baseline_report, history):
    report = record.get("report") or {}
    notes = record.get("candidate_notes") or {}
    baseline_avg = float(baseline_report.get("average", 1e100))
    current_avg = float(report.get("average", 1e100))
    previous_best = best_average(history)
    if previous_best is None:
        previous_best = baseline_avg
    delta_baseline = current_avg - baseline_avg
    delta_best = current_avg - previous_best
    verdict = "kept"
    if report.get("ok_count") != report.get("case_count"):
        verdict = "rejected: invalid/error"
    elif current_avg > previous_best + 1e-9:
        verdict = "rejected: worse than previous best"
    elif abs(delta_best) <= 1e-9:
        verdict = "neutral: tied previous best"
    elif current_avg < previous_best - 1e-9:
        verdict = "kept: improved previous best"

    path = os.path.join(args.out_dir, "round_report_%04d.md" % int(record.get("round", -1)))
    lines = []
    lines.append("# Round %s Report" % record.get("round"))
    lines.append("")
    if record.get("operator_mode"):
        lines.append("Operator invention mode: active after %s non-improving rounds." % record.get("stagnation_count", 0))
        lines.append("")
    lines.append("## Strategy Notes")
    lines.append("")
    lines.append(format_candidate_notes(notes))
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append("- Candidate: `%s`" % record.get("candidate"))
    lines.append("- Verdict: %s" % verdict)
    lines.append("- Baseline average: %.6f" % baseline_avg)
    lines.append("- Candidate average: %.6f" % current_avg)
    lines.append("- Delta vs baseline: %.6f" % delta_baseline)
    lines.append("- Delta vs previous best: %.6f" % delta_best)
    lines.append("")
    lines.append("## Attempts")
    lines.append("")
    lines.append(attempt_summary(record.get("attempts") or []))
    lines.append("")
    lines.append("## Case Breakdown")
    lines.append("")
    lines.append(result_summary(report))
    lines.append("")
    lines.append("## Experience Summary")
    lines.append("")
    if verdict.startswith("kept: improved"):
        lines.append("This strategy should be used as a parent for later mutation.")
    elif verdict.startswith("neutral"):
        lines.append("The strategy is legal but did not change the objective; future rounds should alter grouping or assignment decisions more directly.")
    elif "worse" in verdict:
        lines.append("The strategy changed the search surface but hurt the objective; keep the failure as negative evidence and avoid repeating the same operator mix.")
    else:
        lines.append("The candidate failed validation or execution; repair prompts should focus on API/schema correctness before algorithm changes.")
    write_text(path, "\n".join(lines) + "\n")
    return path, verdict


def compact_history(history, limit=10):
    if not history:
        return "No prior DeepSeek candidates in this run directory."
    scored = []
    for item in history:
        report = item.get("report") or {}
        scored.append((float(report.get("average", 1e100)), item))
    scored.sort(key=lambda pair: pair[0])
    chosen = [item for _, item in scored[: min(4, len(scored))]]
    if len(history) > len(chosen):
        chosen.extend(history[-max(0, limit - len(chosen)) :])
    seen = set()
    lines = []
    for item in chosen:
        key = item.get("candidate")
        if key in seen:
            continue
        seen.add(key)
        report = item.get("report") or {}
        lines.append(
            "round=%s candidate=%s average=%.6f ok=%s/%s note=%s"
            % (
                item.get("round"),
                os.path.basename(key or ""),
                float(report.get("average", 1e100)),
                report.get("ok_count"),
                report.get("case_count"),
                item.get("note", ""),
            )
        )
        for result in report.get("results", [])[:8]:
            value = result.get("value")
            lines.append(
                "  - %s %s %s %.1fms"
                % (
                    result.get("case"),
                    result.get("status"),
                    "NA" if value is None else "%.6f" % float(value),
                    float(result.get("elapsed_ms") or 0.0),
                )
            )
            if result.get("error"):
                lines.append("    error: %s" % shorten(result.get("error"), 1200).replace("\n", " | "))
    return "\n".join(lines)


def shorten(text, limit):
    text = str(text or "")
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return text[:half] + " ... " + text[-half:]


def safe_name(value):
    cleaned = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return ("".join(cleaned) or "candidate")[:80]


def make_prompt(round_index, cases, baseline_report, history, reference_source, strategy_guard, operator_mode=False, stagnation_count=0):
    case_lines = []
    for path in cases:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        case_lines.append("- %s (%d bytes)" % (case_label(path), size))

    prompt = []
    prompt.append("We are optimizing MT-4 candidates. Generate a constrained strategy for round %d." % round_index)
    if operator_mode:
        prompt.append("")
        prompt.append("OPERATOR INVENTION MODE is active.")
        prompt.append("The search has gone %d completed candidate rounds without a strict incumbent improvement." % stagnation_count)
        prompt.append("First summarize the repeated failure pattern in CANDIDATE_NOTES, then add exactly one small LEARNED_OPERATORS entry and call it from STRATEGY.")
        prompt.append("The new operator should be a bounded local action that changes the neighborhood structure, not just another seed or budget tweak.")
    prompt.append("")
    prompt.append("Local cases:")
    prompt.extend(case_lines)
    prompt.append("")
    prompt.append("Current baseline report:")
    prompt.append(result_summary(baseline_report))
    prompt.append("")
    prompt.append("Previous LLM candidate history:")
    prompt.append(compact_history(history))
    prompt.append("")
    prompt.append("Reference strategy executor source.  You may only choose values from its DSL; do not write a solver:")
    prompt.append("```python")
    prompt.append(reference_source)
    prompt.append("```")
    prompt.append("")
    prompt.append("Task:")
    prompt.append("- Propose one STRATEGY dictionary and one CANDIDATE_NOTES dictionary.")
    prompt.append("- Focus on a narrow hypothesis: different starts, thresholds, penalty profile, operator order, seeds, or budgets.")
    if operator_mode:
        prompt.append("- You may write one operator function only inside a LEARNED_OPERATORS string as def apply(problem, state, value, deadline, params).")
        prompt.append("- The apply function may use base, time, EPS, INF, problem.by_mask/by_mask_courier/pair_items, Candidate fields, and standard containers.")
        prompt.append("- It must preserve exact Candidate objects and return a state or (state, value); never emit output rows itself.")
        prompt.append("- Do not import modules, access files, call eval/exec/open/getattr/setattr, or use hardcoded task/courier literals.")
        prompt.append("- learned_operator actions may use accept='bounded' with conservative max_worsen/max_worsen_ratio to keep legal non-improving bridge states for later operators.")
        prompt.append("- The harness records the best state seen and returns that final best, so an exploratory bridge should unlock later improvement rather than just wander.")
    else:
        prompt.append("- Do not write def solve, helper functions, imports, monkey patches, direct state manipulation, or LEARNED_OPERATORS.")
    prompt.append("- Do not spend the whole budget on include_base/base_time; leave meaningful time for the proposed action.")
    prompt.append("- Prefer deterministic fixed seeds.")
    if strategy_guard == "tail-polish":
        prompt.append("- Harness guard is active: include_base and polish_base will be forced true; penalty will be forced default; base_time will be clamped to 8.30..8.55; starts will be reduced to all_single.")
        prompt.append("- Therefore your useful degrees of freedom are mainly improvement operator order, pair_injection settings, random seeds, and small budgets for the final tail polish.")
    if operator_mode:
        prompt.append("- Return only a fenced python code block with STRATEGY, LEARNED_OPERATORS, and CANDIDATE_NOTES.")
    else:
        prompt.append("- Return only a fenced python code block with STRATEGY and CANDIDATE_NOTES.")
    return "\n".join(prompt)


def provider_config(provider):
    providers = LLM_SETTINGS.get("providers") or {}
    provider = str(provider or DEFAULT_PROVIDER)
    return providers.get(provider, providers.get(DEFAULT_PROVIDER, {}))


def resolve_llm_settings(args):
    config = provider_config(getattr(args, "provider", DEFAULT_PROVIDER))
    args.provider = str(getattr(args, "provider", "") or DEFAULT_PROVIDER)
    if not getattr(args, "model", ""):
        args.model = str(LLM_SETTINGS.get("model") or config.get("model") or DEFAULT_MODEL)
    if not getattr(args, "base_url", ""):
        args.base_url = str(LLM_SETTINGS.get("base_url") or config.get("base_url") or DEFAULT_BASE_URL)
    if not getattr(args, "api_key_env", ""):
        args.api_key_env = str(config.get("api_key_env") or "LLM_API_KEY")
    return args


def load_api_key(args):
    if args.api_key:
        return args.api_key.strip()
    if args.api_key_file:
        path = os.path.abspath(args.api_key_file)
        if os.path.exists(path):
            return read_text(path).strip()
    configured_key = str(LLM_SETTINGS.get("api_key") or "").strip()
    if configured_key:
        return configured_key
    config = provider_config(getattr(args, "provider", DEFAULT_PROVIDER))
    provider_key = str(config.get("api_key") or "").strip()
    if provider_key:
        return provider_key
    env_names = [
        str(getattr(args, "api_key_env", "") or ""),
        str(config.get("api_key_env") or ""),
        "LLM_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    ]
    seen = set()
    for env_name in env_names:
        if not env_name or env_name in seen:
            continue
        seen.add(env_name)
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return ""


def deepseek_chat(api_key, model, base_url, messages, timeout, max_tokens):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.55,
        "top_p": 0.95,
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    raw = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=raw,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
        method="POST",
    )
    print(
        "calling LLM model=%s endpoint=%s timeout=%.0fs max_tokens=%d ..."
        % (model, base_url.rstrip("/"), float(timeout), int(max_tokens)),
        flush=True,
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            data = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError("LLM HTTP %s: %s" % (exc.code, detail[-2000:]))
    parsed = json.loads(data)
    choices = parsed.get("choices") or []
    if not choices:
        raise RuntimeError("LLM returned no choices: %s" % data[:1000])
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if not content:
        raise RuntimeError("DeepSeek returned empty content: %s" % data[:1000])
    print("DeepSeek response received in %.1fs" % (time.perf_counter() - start), flush=True)
    return content, parsed.get("usage")


def extract_python_code(text):
    fence = re.search(r"```(?:python|py)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1).strip() + "\n"
    stripped = text.strip()
    if stripped.startswith("from ") or stripped.startswith("import ") or "def solve" in stripped:
        return stripped + "\n"
    raise RuntimeError("could not find python code block in model response")


def extract_strategy_payload(text):
    """Extract STRATEGY and CANDIDATE_NOTES from a model response."""

    stripped = text.strip()
    if stripped.startswith("{"):
        parsed = json.loads(stripped)
        strategy = parsed.get("STRATEGY", parsed.get("strategy"))
        notes = parsed.get("CANDIDATE_NOTES", parsed.get("candidate_notes", {}))
        operators = parsed.get("LEARNED_OPERATORS", parsed.get("learned_operators", {}))
        if not isinstance(strategy, dict):
            raise RuntimeError("JSON response has no strategy dictionary")
        if isinstance(operators, dict) and operators:
            strategy = dict(strategy)
            strategy["operators"] = operators
        return strategy, notes if isinstance(notes, dict) else {}

    code = extract_python_code(text)
    tree = ast.parse(code)
    strategy = None
    notes = {}
    operators = {}
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == "STRATEGY":
                strategy = ast.literal_eval(node.value)
            elif target.id == "CANDIDATE_NOTES":
                value = ast.literal_eval(node.value)
                if isinstance(value, dict):
                    notes = value
            elif target.id == "LEARNED_OPERATORS":
                value = ast.literal_eval(node.value)
                if isinstance(value, dict):
                    operators = value
    if not isinstance(strategy, dict):
        raise RuntimeError("could not find STRATEGY dictionary in model response")
    if operators:
        strategy = dict(strategy)
        strategy["operators"] = operators
    return strategy, notes


def static_candidate_issues(code):
    issues = []
    if "def solve" not in code:
        issues.append("missing def solve(input_text)")
    bad_fields = (
        "problem.offers",
        "problem.candidates",
        "problem.tasks",
        "problem.couriers",
    )
    for field in bad_fields:
        if field in code:
            issues.append("uses nonexistent field %s" % field)
    if "base.kernel.greedy_assignment" in code:
        issues.append("calls base.kernel.greedy_assignment directly; use base.greedy_assignment wrapper with safe deadline")
    if "base.kernel.improve_fixed_groups" in code:
        issues.append("calls base.kernel.improve_fixed_groups directly; use base.improve_fixed_groups wrapper with safe deadline")
    if ".split('|')" in code or "\"|\"" in code or "'|'" in code:
        issues.append("uses pipe-separated task keys; output must preserve exact comma task_key strings")
    if "int(x) for x in task_str.split" in code or "int(" in code and "task_str" in code:
        issues.append("appears to coerce task IDs to int; task IDs are strings such as T0001")
    if ":=" in code:
        issues.append("uses walrus operator, not Python 3.6 compatible")
    if "from __future__ import annotations" in code:
        issues.append("uses future annotations, not Python 3.6 compatible")
    if "dataclasses" in code:
        issues.append("uses dataclasses, not available in stock Python 3.6")
    return issues


def build_strategy_repair_prompt(round_index, bad_response, issues, baseline_report, history, eval_report=None):
    prompt = []
    prompt.append("Repair this MT-4 strategy proposal for round %d." % round_index)
    prompt.append("")
    prompt.append("Hard requirements:")
    prompt.append("- Return one fenced python code block.")
    prompt.append("- The code block must contain STRATEGY = {...} and CANDIDATE_NOTES = {...}.")
    prompt.append("- Do not write def solve, top-level imports, top-level helper functions, monkey patches, or literal output rows.")
    prompt.append("- If repairing a learned_operator issue, put def apply only inside a LEARNED_OPERATORS string.")
    prompt.append("- Use only the DSL documented in the system prompt.")
    prompt.append("- Keep it Python literal compatible with ast.literal_eval.")
    prompt.append("")
    prompt.append("Current baseline:")
    prompt.append(result_summary(baseline_report))
    prompt.append("")
    prompt.append("History summary:")
    prompt.append(compact_history(history))
    prompt.append("")
    prompt.append("Detected issues:")
    for issue in issues:
        prompt.append("- " + issue)
    if eval_report:
        prompt.append("")
        prompt.append("Evaluation report:")
        prompt.append(result_summary(eval_report))
        for item in eval_report.get("results", []):
            if item.get("error"):
                prompt.append("Error detail:")
                prompt.append(shorten(item.get("error"), 3000))
                break
    prompt.append("")
    prompt.append("Bad response:")
    prompt.append(shorten(bad_response, 6000))
    return "\n".join(prompt)


def request_strategy_repair(args, round_index, bad_response, issues, baseline_report, history, api_key, attempt, eval_report=None):
    print("requesting strategy repair attempt %d for round %d ..." % (attempt, round_index), flush=True)
    prompt = build_strategy_repair_prompt(
        round_index,
        bad_response,
        issues,
        baseline_report,
        history,
        eval_report=eval_report,
    )
    repair_prompt_path = os.path.join(args.out_dir, "repair_prompt_round_%04d_%02d.md" % (round_index, attempt))
    write_text(repair_prompt_path, prompt)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response_text, usage = deepseek_chat(
        api_key,
        args.model,
        args.base_url,
        messages,
        timeout=args.api_timeout,
        max_tokens=args.max_tokens,
    )
    repair_response_path = os.path.join(args.out_dir, "repair_response_round_%04d_%02d.md" % (round_index, attempt))
    write_text(repair_response_path, response_text)
    strategy, notes = extract_strategy_payload(response_text)
    return strategy, notes, response_text, repair_prompt_path, repair_response_path, usage


def build_repair_prompt(round_index, bad_code, issues, baseline_report, history, compile_log="", eval_report=None):
    prompt = []
    prompt.append("Repair this MT-4 candidate for round %d." % round_index)
    prompt.append("")
    prompt.append("Hard requirements:")
    prompt.append("- Return one complete Python 3.6-compatible code block.")
    prompt.append("- Keep def solve(input_text).")
    prompt.append("- Keep or update CANDIDATE_NOTES with hypothesis, changed_parts, expected_effect, risk, next_if_failed.")
    prompt.append("- Do not hardcode task IDs, courier IDs, filenames, or online rows.")
    prompt.append("- Prefer reusing autosolver_workbench.base_solver_no_hardcode as base.")
    prompt.append("- Valid Problem fields: task_to_idx, idx_to_task, by_mask, by_mask_courier, single_masks, pair_masks, pair_items, all_couriers, n_tasks, all_task_mask, avg_willingness.")
    prompt.append("- Valid Candidate fields: mask, task_key, courier, score, p, q, weighted_score, task_count.")
    prompt.append("- Iterate offers with: for cands in problem.by_mask.values(): for cand in cands: ...")
    prompt.append("- Output exact Candidate.task_key strings and exact Candidate.courier strings.")
    prompt.append("- If using existing states, call base.state_to_output(state).")
    prompt.append("")
    prompt.append("Current baseline:")
    prompt.append(result_summary(baseline_report))
    prompt.append("")
    prompt.append("History summary:")
    prompt.append(compact_history(history))
    prompt.append("")
    prompt.append("Detected issues:")
    for issue in issues:
        prompt.append("- " + issue)
    if compile_log:
        prompt.append("")
        prompt.append("Compile error:")
        prompt.append(shorten(compile_log, 2500))
    if eval_report:
        prompt.append("")
        prompt.append("Evaluation report:")
        prompt.append(result_summary(eval_report))
        for item in eval_report.get("results", []):
            if item.get("error"):
                prompt.append("Error detail:")
                prompt.append(shorten(item.get("error"), 3000))
                break
    prompt.append("")
    prompt.append("Bad candidate:")
    prompt.append("```python")
    prompt.append(bad_code)
    prompt.append("```")
    return "\n".join(prompt)


def request_repair(args, round_index, bad_code, issues, baseline_report, history, api_key, attempt, compile_log="", eval_report=None):
    print("requesting repair attempt %d for round %d ..." % (attempt, round_index), flush=True)
    prompt = build_repair_prompt(
        round_index,
        bad_code,
        issues,
        baseline_report,
        history,
        compile_log=compile_log,
        eval_report=eval_report,
    )
    repair_prompt_path = os.path.join(args.out_dir, "repair_prompt_round_%04d_%02d.md" % (round_index, attempt))
    write_text(repair_prompt_path, prompt)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response_text, usage = deepseek_chat(
        api_key,
        args.model,
        args.base_url,
        messages,
        timeout=args.api_timeout,
        max_tokens=args.max_tokens,
    )
    response_path = os.path.join(args.out_dir, "repair_response_round_%04d_%02d.md" % (round_index, attempt))
    write_text(response_path, response_text)
    return extract_python_code(response_text), repair_prompt_path, response_path, usage


def compile_candidate(path):
    cmd = [sys.executable, "-m", "py_compile", path]
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=20)
        return True, ""
    except subprocess.CalledProcessError as exc:
        return False, exc.output.decode("utf-8", "replace")[-4000:]
    except subprocess.TimeoutExpired:
        return False, "py_compile timed out"


def evaluate_candidate(path, cases, args):
    return evaluator.evaluate_solver(
        path,
        cases,
        time_limit=args.time_limit,
        hard_timeout=args.hard_timeout,
        fail_score=args.fail_score,
        repeats=getattr(args, "eval_repeats", 1),
        repeat_aggregate=getattr(args, "eval_aggregate", "single"),
    )


def best_history_record(history):
    best = None
    for item in history:
        report = item.get("report") or {}
        if report.get("ok_count") != report.get("case_count"):
            continue
        if report.get("average") is None:
            continue
        if best is None or float(report["average"]) < float(best.get("report", {}).get("average", 1e100)):
            best = item
    return best


def copy_best(history, out_dir):
    best = best_history_record(history)
    if not best:
        return None
    source = best.get("candidate")
    if source and os.path.exists(source):
        target = os.path.join(out_dir, "best_deepseek_candidate.py")
        shutil.copyfile(source, target)
        return target
    return None


def make_initial_reference():
    path = os.path.join(ROOT, "autosolver_workbench", "strategy_executor.py")
    return read_text(path, limit=MAX_SOURCE_CHARS)


def run_round(args, round_index, cases, baseline_report, history, api_key):
    reference = make_initial_reference()
    stagnation_count = no_improvement_streak(history)
    operator_mode = operator_mode_enabled(args, history)
    prompt = make_prompt(
        round_index,
        cases,
        baseline_report,
        history,
        reference,
        args.strategy_guard,
        operator_mode=operator_mode,
        stagnation_count=stagnation_count,
    )
    prompt_path = os.path.join(args.out_dir, "prompt_round_%04d.md" % round_index)
    write_text(prompt_path, prompt)
    if args.dry_run:
        print("dry-run prompt written: %s" % prompt_path)
        return None

    print("round=%d prompt written: %s" % (round_index, prompt_path), flush=True)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response_text, usage = deepseek_chat(
        api_key,
        args.model,
        args.base_url,
        messages,
        timeout=args.api_timeout,
        max_tokens=args.max_tokens,
    )
    response_path = os.path.join(args.out_dir, "response_round_%04d.md" % round_index)
    write_text(response_path, response_text)
    print("round=%d response written: %s" % (round_index, response_path), flush=True)
    record = {
        "round": round_index,
        "prompt": prompt_path,
        "response": response_path,
        "created_at": time.time(),
        "model": args.model,
        "usage": usage,
        "note": args.tag,
        "operator_mode": bool(operator_mode),
        "stagnation_count": int(stagnation_count),
        "attempts": [],
    }
    current_response = response_text
    current_strategy = None
    current_notes = {}
    final_report = None
    final_candidate_path = None
    final_compile_log = ""
    final_code = ""

    max_attempts = 1 + max(0, int(args.repair_attempts))
    for attempt in range(max_attempts):
        try:
            if current_strategy is None:
                current_strategy, current_notes = extract_strategy_payload(current_response)
        except Exception as exc:
            issues = ["strategy parse failed: %s" % exc]
            print("round=%d attempt=%d strategy parse failed: %s" % (round_index, attempt, exc), flush=True)
            record["attempts"].append(
                {"attempt": attempt, "candidate": None, "stage": "strategy_parse", "issues": issues}
            )
            if attempt + 1 < max_attempts:
                current_strategy, current_notes, current_response, repair_prompt, repair_response, repair_usage = request_strategy_repair(
                    args,
                    round_index,
                    current_response,
                    issues,
                    baseline_report,
                    history,
                    api_key,
                    attempt + 1,
                )
                record["attempts"][-1]["repair_prompt"] = repair_prompt
                record["attempts"][-1]["repair_response"] = repair_response
                record["attempts"][-1]["repair_usage"] = repair_usage
                continue
            final_report = {
                "average": float(args.fail_score),
                "ok_count": 0,
                "case_count": len(cases),
                "results": [{"case": os.path.basename(path), "status": "strategy_parse", "value": None} for path in cases],
            }
            break

        issues = strategy_executor.strategy_issues(current_strategy)
        if bool(getattr(args, "forbid_learned_operators", False)) and learned_operator_requested(current_strategy):
            issues.append("learned operators are disabled for this controlled-action phase")
        if issues:
            print("round=%d attempt=%d strategy reject: %s" % (round_index, attempt, "; ".join(issues)), flush=True)
            record["attempts"].append(
                {"attempt": attempt, "candidate": None, "stage": "strategy", "issues": issues}
            )
            if attempt + 1 < max_attempts:
                current_strategy, current_notes, current_response, repair_prompt, repair_response, repair_usage = request_strategy_repair(
                    args,
                    round_index,
                    current_response,
                    issues,
                    baseline_report,
                    history,
                    api_key,
                    attempt + 1,
                )
                record["attempts"][-1]["repair_prompt"] = repair_prompt
                record["attempts"][-1]["repair_response"] = repair_response
                record["attempts"][-1]["repair_usage"] = repair_usage
                continue
            final_report = {
                "average": float(args.fail_score),
                "ok_count": 0,
                "case_count": len(cases),
                "results": [{"case": os.path.basename(path), "status": "strategy_reject", "value": None} for path in cases],
            }
            break

        normalized_strategy = strategy_executor.apply_strategy_guard(current_strategy, args.strategy_guard)
        code = strategy_executor.make_solver_source(normalized_strategy, current_notes)
        final_code = code
        candidate_name = "deepseek_r%04d_%s.py" % (round_index, safe_name(args.tag))
        if attempt:
            candidate_name = "deepseek_r%04d_%s_repair%02d.py" % (round_index, safe_name(args.tag), attempt)
        candidate_path = os.path.join(args.out_dir, candidate_name)
        write_text(candidate_path, code)
        final_candidate_path = candidate_path
        candidate_notes = extract_candidate_notes(code)
        if candidate_notes:
            print("round=%d attempt=%d notes: %s" % (round_index, attempt, candidate_notes.get("hypothesis", "")), flush=True)
        print("round=%d attempt=%d candidate written: %s" % (round_index, attempt, candidate_path), flush=True)

        ok, compile_log = compile_candidate(candidate_path)
        final_compile_log = compile_log
        if not ok:
            print("round=%d attempt=%d compile failed" % (round_index, attempt), flush=True)
            record["attempts"].append(
                {"attempt": attempt, "candidate": candidate_path, "stage": "compile", "error": compile_log}
            )
            if attempt + 1 < max_attempts:
                current_strategy, current_notes, current_response, repair_prompt, repair_response, repair_usage = request_strategy_repair(
                    args,
                    round_index,
                    current_response,
                    ["generated strategy candidate failed py_compile"],
                    baseline_report,
                    history,
                    api_key,
                    attempt + 1,
                )
                record["attempts"][-1]["repair_prompt"] = repair_prompt
                record["attempts"][-1]["repair_response"] = repair_response
                record["attempts"][-1]["repair_usage"] = repair_usage
                continue
            final_report = {
                "average": float(args.fail_score),
                "ok_count": 0,
                "case_count": len(cases),
                "results": [{"case": os.path.basename(path), "status": "compile_error", "value": None} for path in cases],
            }
            break

        print("round=%d attempt=%d evaluating candidate ..." % (round_index, attempt), flush=True)
        report = evaluate_candidate(candidate_path, cases, args)
        final_report = report
        record["attempts"].append({"attempt": attempt, "candidate": candidate_path, "stage": "evaluate", "report": report})
        if report.get("ok_count") == report.get("case_count"):
            break
        if attempt + 1 < max_attempts:
            issues = ["evaluation failed or returned invalid output"]
            current_strategy, current_notes, current_response, repair_prompt, repair_response, repair_usage = request_strategy_repair(
                args,
                round_index,
                current_response,
                issues,
                baseline_report,
                history,
                api_key,
                attempt + 1,
                eval_report=report,
            )
            record["attempts"][-1]["repair_prompt"] = repair_prompt
            record["attempts"][-1]["repair_response"] = repair_response
            record["attempts"][-1]["repair_usage"] = repair_usage

    if final_report is None:
        final_report = {
            "average": float(args.fail_score),
            "ok_count": 0,
            "case_count": len(cases),
            "results": [],
        }
    record["candidate"] = final_candidate_path
    record["raw_strategy"] = strategy_executor.normalize_strategy(current_strategy or {})
    record["strategy"] = strategy_executor.apply_strategy_guard(current_strategy or {}, args.strategy_guard)
    record["strategy_guard"] = args.strategy_guard
    record["candidate_notes"] = extract_candidate_notes(final_code) if final_code else current_notes
    record["report"] = final_report
    if final_compile_log:
        record["compile_error"] = final_compile_log
    append_jsonl(args.history, record)
    report_path, verdict = write_round_report(args, record, baseline_report, history)

    print(
        "round=%d candidate=%s average=%.6f ok=%d/%d"
        % (
            round_index,
            os.path.basename(final_candidate_path or ""),
            float(final_report["average"]),
            int(final_report["ok_count"]),
            int(final_report["case_count"]),
        )
    )
    print("round=%d verdict=%s" % (round_index, verdict), flush=True)
    print("round=%d report written: %s" % (round_index, report_path), flush=True)
    if record.get("candidate_notes"):
        print("round=%d strategy notes:" % round_index, flush=True)
        for line in format_candidate_notes(record.get("candidate_notes")).splitlines():
            print("  " + line, flush=True)
    for item in final_report.get("results", []):
        value = item.get("value")
        print(
            "  %s %s %s %.1fms"
            % (
                item.get("case"),
                item.get("status"),
                "NA" if value is None else "%.6f" % float(value),
                float(item.get("elapsed_ms") or 0.0),
            )
        )
    return record


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--api-key-file", default="")
    parser.add_argument("--api-timeout", type=float, default=240.0)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--time-limit", type=float, default=9.35)
    parser.add_argument("--hard-timeout", type=float, default=12.0)
    parser.add_argument("--fail-score", type=float, default=1000000.0)
    parser.add_argument("--eval-repeats", type=int, default=1)
    parser.add_argument("--eval-aggregate", choices=("single", "best", "median", "mean", "worst"), default="single")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--case-glob", action="append", default=[])
    parser.add_argument("--no-default-cases", action="store_true")
    parser.add_argument("--out-dir", default=RUN_DIR)
    parser.add_argument("--history", default=os.path.join(RUN_DIR, "history.jsonl"))
    parser.add_argument("--baseline-solver", default=os.path.join(ROOT, "autosolver_workbench", "base_solver_no_hardcode.py"))
    parser.add_argument("--tag", default="candidate")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--strategy-guard", default="tail-polish")
    parser.add_argument("--force-operator-mode", action="store_true")
    parser.add_argument("--forbid-learned-operators", action="store_true")
    parser.add_argument(
        "--operator-stagnation-rounds",
        type=int,
        default=6,
        help="Enable learned-operator invention after this many full candidate rounds without improvement; use 0 to disable.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args = resolve_llm_settings(args)
    ensure_dir(args.out_dir)
    cases = evaluator.discover_cases(args.case, args.case_glob, include_defaults=not args.no_default_cases)
    if not cases:
        raise SystemExit("no cases found")

    baseline_report = evaluate_candidate(args.baseline_solver, cases, args)
    print("baseline:")
    print(result_summary(baseline_report))

    history = read_jsonl(args.history)
    if not args.dry_run:
        api_key = load_api_key(args)
        if not api_key:
            raise SystemExit("missing API key: set DEEPSEEK_API_KEY or pass --api-key-file")
    else:
        api_key = ""

    start_round = 0
    if history:
        rounds = [int(item.get("round", -1)) for item in history if str(item.get("round", "")).lstrip("-").isdigit()]
        if rounds:
            start_round = max(rounds) + 1
    for round_index in range(start_round, start_round + max(1, int(args.rounds))):
        try:
            record = run_round(args, round_index, cases, baseline_report, history, api_key)
            if record is not None:
                history.append(record)
                best_path = copy_best(history, args.out_dir)
                if best_path:
                    print("best candidate copied to %s" % best_path)
        except Exception as exc:
            failure = {
                "round": round_index,
                "created_at": time.time(),
                "model": args.model,
                "error": str(exc),
                "traceback": traceback.format_exc()[-4000:],
            }
            append_jsonl(args.history, failure)
            print("round=%d failed: %s" % (round_index, exc))
            if args.rounds <= 1:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
