# MT-4 AutoSolver Portable Package

Do not commit API keys. Set the key with an environment variable or edit a local ignored config file.

Quick smoke:
python -m py_compile clean_solver.py autosolver_workbench\deepseek_autosolver.py autosolver_workbench\evaluator.py autosolver_workbench\strategy_executor.py autosolver_workbench\base_solver_no_hardcode.py
python autosolver_workbench\evaluator.py --solver clean_solver.py --case cases\large_seed301.txt --no-default-cases --time-limit 9.35 --hard-timeout 12

Run one DeepSeek round:
set DEEPSEEK_API_KEY=your_key_here
python autosolver_workbench\deepseek_autosolver.py --rounds 1 --case cases\large_seed301.txt --no-default-cases --time-limit 9.35 --hard-timeout 12 --repair-attempts 1 --tag portable_try

PowerShell key setting:
$env:DEEPSEEK_API_KEY = "your_key_here"
