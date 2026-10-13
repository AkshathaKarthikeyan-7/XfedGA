# XFedGA — standing rules for Claude Code

Read `docs/XFEDGA_SPEC.md` at the start of every session. It is the master spec. This file holds only the rules that apply to every session.

## Environment (do not change)
- Repo: `D:\new repo\xfedga`. Shell: PowerShell on Windows.
- Python: `& "D:\conda_envs\fedga\python.exe"` (Python 3.10, torch 2.11+cu128, monai 1.3.0, flwr 1.7.0).
- GPU: RTX 4050 laptop, 6 GB. No Docker.
- Never upgrade, install or remove packages in this env without asking me first.
- Write files directly with your file tools. After writing, run a verification command (tests, import check or a smoke run) and show me the result.

## Non-negotiables
- Never invent numbers. Every number in a table, slide or report comes from a file in `results/`, produced by a script in `analysis/`.
- Never quote a paper's numbers or author names from memory. The base FedGA paper is cited only from the PDF at `D:\Amrita\Project\Literature review\FedGA_Genetic_Algorithm-Guided_Federated_Learning_for_Medical_Image_Segmentation_with_Non-IID_Features.pdf`.
- Never select rounds, checkpoints or hyperparameters on the test set. Selection uses the validation set only.
- If a result does not support a claim, say so and drop the claim. Do not tune to rescue it.
- Do not rewrite or delete working code (`fl_dist/`, existing tests). Keep all existing tests passing after every change. Report pass/fail counts before and after.
- The algorithm (aggregation, GA, fitness, q_i) lives only in `fedga/` and `xai/`. The custom loop and `fl_dist/` import it.
- Ask me only when a decision cannot be inferred from the repo or the data documentation. Flag every assumption.

## Vocabulary (one name per concept, in code, slides and report)
- **q_i** — client saliency score, computed client-side.
- **w_agg** — aggregated model before the GA.
- **fitness weights (w1, w2, w3)** — weights on DSC, HD95_norm and EC in the GA fitness. Do not call them "lambda".
- **beta** — saliency-weighting exponent in aggregation. It is not a post-GA blend.
- **fitness set / validation set / test set** — the three disjoint server splits.
- **multi-process federation on a single machine over loopback gRPC** — the only allowed description of the `fl_dist/` runs.

## End-of-session report (always, in this format)
1. Files added, changed or deleted, with a one-line reason each.
2. Test suite: pass/fail before and after.
3. Runs completed: run id, task, method, seed, config hash, git commit.
4. Compute used this session vs the costed plan (hours).
5. 3-line summary per experiment (format in the spec, section 10).
6. Open questions for me.
