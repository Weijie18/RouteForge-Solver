# MT-4 AutoSolver Workbench

This directory contains the offline AutoSolver Agent harness.  The contest
solver still needs only `solve(input_text)`, but this workbench can spend
offline time generating, evaluating, repairing, and selecting candidate solvers.

## Requirement Mapping

1. Autonomous strategy exploration
   - `deepseek_autosolver.py` asks DeepSeek to generate a constrained
     `STRATEGY` dictionary rather than arbitrary solver code.
   - `strategy_executor.py` turns that strategy into a legal `solve` wrapper
     and owns parsing, state objects, timing, and output formatting.
   - Prompts request changes to starts, thresholds, operator order, seeds,
     budgets, and adaptive penalties.
   - `agent_loop.py` provides deterministic non-LLM parameter/template probes.

2. Automatic evaluation and screening
   - `evaluator.py` runs each candidate in a child process.
   - It checks exact output legality and computes the expected penalty.
   - Histories are stored as JSONL under `outputs/autosolver_workbench/`.

3. Iterative improvement loop
   - Each DeepSeek round receives prior scores, runtime, legality failures, and
     traceback snippets.
   - Strategy checks reject unknown DSL fields before wasting a full benchmark
     run.
   - Broken strategy proposals can be automatically repaired by one or more LLM
     repair attempts.
   - After a configurable number of non-improving rounds, the loop enters
     learned-operator mode: DeepSeek may add one small `apply(...)` action in
     `LEARNED_OPERATORS`, then call it from the strategy.  The executor runs
     that action in a restricted namespace.  Learned operators can be strict
     improvement moves or bounded legal bridge moves (`accept: "bounded"`);
     the chain may walk through a reasonable non-improving state, while the
     final solver output is still the best legal state seen during the chain.
   - Every round writes `round_report_XXXX.md`, including strategy notes,
     attempt-level failures, score deltas, verdict, and a short experience
     summary.

4. Final solver output under per-case time limit
   - Every generated candidate defines `solve(input_text)`.
   - The evaluator enforces a soft solver time limit and a hard subprocess
     timeout.
   - The current best candidate is copied to
     `outputs/autosolver_workbench/deepseek_runs/best_deepseek_candidate.py`.

## Useful Commands

Evaluate a candidate on large301:

```powershell
python autosolver_workbench\evaluator.py `
  --solver outputs\autosolver_workbench\deepseek_runs\best_deepseek_candidate.py `
  --case C:\Users\ASUS\Downloads\large_seed301.txt `
  --no-default-cases `
  --time-limit 9.35 `
  --hard-timeout 12
```

Run one DeepSeek strategy generation round:

```powershell
$env:DEEPSEEK_API_KEY = "your_key"
python autosolver_workbench\deepseek_autosolver.py `
  --rounds 1 `
  --case C:\Users\ASUS\Downloads\large_seed301.txt `
  --no-default-cases `
  --time-limit 9.35 `
  --hard-timeout 12 `
  --repair-attempts 1 `
  --tag large301_try
```

## LLM Provider Settings

The harness uses an OpenAI-compatible `/chat/completions` endpoint.  Edit the
`LLM_SETTINGS` block near the top of
`autosolver_workbench/deepseek_autosolver.py` to set the default provider,
model, base URL, and API-key environment variable:

```python
LLM_SETTINGS = {
    "provider": "deepseek",
    "api_key": "",
    "model": "",
    "base_url": "",
    "providers": {
        "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-v4-pro", "api_key_env": "DEEPSEEK_API_KEY"},
        "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4.1", "api_key_env": "OPENAI_API_KEY"},
        "custom": {"base_url": "https://your-openai-compatible-host.example/v1", "model": "your-model-name", "api_key_env": "LLM_API_KEY"},
    },
}
```

Command-line flags still override the code defaults:

```powershell
python autosolver_workbench\deepseek_autosolver.py `
  --provider custom `
  --base-url https://your-endpoint.example/v1 `
  --model your-model `
  --api-key-env LLM_API_KEY
```

Disable learned-operator invention for pure parameter search:

```powershell
python autosolver_workbench\deepseek_autosolver.py `
  --rounds 5 `
  --case C:\Users\ASUS\Downloads\large_seed301.txt `
  --no-default-cases `
  --operator-stagnation-rounds 0
```

The model response files contain only `STRATEGY` and `CANDIDATE_NOTES`; the
actual executable candidate is assembled by `strategy_executor.py`.  In
learned-operator mode, responses may also contain `LEARNED_OPERATORS`; the
candidate still goes through the same legality and scoring checks.
