# Research rules

These rules apply to all work in this repository: code, tables, figures, slides and the report.

## Data
- The server holds three disjoint splits: the **fitness set**, the **validation set** and the **test set**. They never overlap.
- Never select rounds, checkpoints or hyperparameters on the test set. Selection uses the validation set only.

## Numbers and sources
- Never invent numbers. Every number in a table, slide or report comes from a file in `results/`, produced by a script in `analysis/`.
- Never quote a paper's numbers or author names from memory. The base FedGA paper is cited only from its PDF.

## Claims
- If a result does not support a claim, say so and drop the claim. Do not tune to rescue it.

## Code
- The algorithm (aggregation, GA, fitness, q_i) lives only in `fedga/` and `xai/`. The custom loop and `fl_dist/` import it.
- Keep every test in `tests/` passing after each change.
- Do not delete working code without the project owner's approval.

## Vocabulary (one name per concept, in code, slides and report)
- **q_i**: client saliency score, computed client-side.
- **w_agg**: aggregated model before the GA.
- **fitness weights (w1, w2, w3)**: weights on DSC, HD95_norm and EC in the GA fitness. Do not call them "lambda".
- **beta**: saliency-weighting exponent in aggregation. It is not a post-GA blend.
- **fitness set / validation set / test set**: the three disjoint server splits.
- **multi-process federation on a single machine over loopback gRPC**: the only allowed description of `fl_dist/` runs.
