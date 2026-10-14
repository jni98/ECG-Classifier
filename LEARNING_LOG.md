# Day 1

- Ruff: A fast Python tool that checks code for errors, style issues and unused imports, and can format it automatically.

- pre-commit: A framework that automatically runs configured checks and tools such as Ruff before every Git commit, so that faulty or inconsistent code doesn't get committed.

- Due to previously using anaconda, this was still activated in my environment and led to uv using anaconda. That's why all anaconda files were scanned asnd things got slow. Deactivating anaconda, deleting .venv and new uv python solved the problem
