# gepa — cloud snapshot

Curated snapshot of `C:\Users\25875\gepa` (local workstation) for running on Claude cloud.
Created 2026-08-18.

## What is included

- Top-level `*.py` — GEPA / OpenEvolve router optimization & evaluation scripts
  (`optimize_ollama_router_gepa.py`, `evaluate_router_testdata.py`, `train_softprompt_router.py`, ...)
- Top-level `*.json` / `*.csv` — routing test data, GEPA prompt results, cost comparisons (~66 MB)
- `scripts/` — experiment drivers (computer-QA label-budget runs, active selection, sweeps, ...)
- `gepa_router_run*/`, `openevolve_router_run*/` — per-run logs and candidates
- `sembench_movie_router/` — SemBench movie router experiment results
- `outputs/` — experiment result JSON/MD (~42 MB). **`*.pt` checkpoints (5.5 GB) are excluded**;
  they stay on the workstation and can be regenerated with `train_softprompt_router.py`.
- `paper/` — VLDB paper source (also mirrored on Overleaf)
- `deepscholar/` — working tree snapshot (`.venv` excluded; reinstall via its `requirements.txt`)
- `lotus/` — fork working tree with local router additions
  (`lotus/models/routed_lm.py`, `cascaded_lm.py`, softprompt router; fork: `daofuz/lotus`)

## Not included — restore from upstream at these pinned commits

| repo | upstream | commit (local HEAD, 2026-08-18) |
|---|---|---|
| `sembench/` | https://github.com/SemBench/SemBench.git | `c814e3807e72d4cf876b852b17e77f3cc94575c2` (clean clone, no local changes) |
| `lotus/` upstream base | https://github.com/lotus-data/lotus.git (fork: daofuz/lotus) | `136ae4f4a344a2f75d89f811e516dfcb0de30e46` + local changes included here |
| `deepscholar/` base | https://github.com/guestrin-lab/deepscholar.git | `c95413b3b2f3255b461b90d0ce650f685ae2d1ff` + working tree included here |
| `paper/` base | VLDB template; Overleaf remote | `9aa76de0f5efc95a638af8a5c2413132621da576` |

## Setup on cloud

```bash
pip install -r deepscholar/requirements.txt   # deepscholar deps
pip install -e ./lotus                        # lotus with local router models
git clone https://github.com/SemBench/SemBench.git sembench && git -C sembench checkout c814e380
cp .env.example .env                          # then fill in OPENAI_API_KEY / GEMINI_API_KEY
```

Ollama-dependent runs (`*_ollama_reflect*`) need a local Ollama server and will not work on cloud
without one; OpenAI-reflect (`*_openai_reflect*`) paths run as-is once `OPENAI_API_KEY` is set.
