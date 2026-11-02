# Decisions

Dated log of every choice made before seeing test results.

## 2026-11-02: environment

- WSL2 `Ubuntu` with uv and Python 3.11. The venv, weights and data live on WSL ext4, whose disk is on N:. C: has only 14.9 GB free and is kept out of it.
- Torch 2.11.0+cu128. DA3 pinned at commit `3d835ec`. Full versions are in `requirements.lock`.
