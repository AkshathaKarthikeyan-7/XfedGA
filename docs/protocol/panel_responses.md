# Panel responses

*Facts and file paths only. Numbers are quoted from the files named; nothing here is a new result.*

## 1. Fitness weights (λ1–λ3 on the panel; w1, w2, w3 in the code)

The GA fitness is F = w1·DSC − w2·HD95_norm + w3·EC on the server's fixed fitness subsample (`fedga/evaluation.py::fitness_multi`). The weights (0.5, 0.3, 0.2) were fixed in advance as the pre-registered default (plan.md §0.0), not tuned.
- **Sensitivity (breast, offline re-ranking of the logged GA candidates):** at (1, 0, 0) the round's winner changes in 51 of 75 rounds (68%), but its fitness-subset DSC moves only from 0.6559 to 0.6576 (`reports/review/fitness_weights_sensitivity.md`, from `results/fitness_weights/step0.json`). The re-ranking is within-round only.
- **Tuning dropped:** a weight-tuning stage was considered on 18 Oct and dropped on this evidence before any run (plan.md).
- **Trajectory effect:** a validation-only screen of (1, 0, 0), (0.6, 0.2, 0.2) and (0.4, 0.4, 0.2) on `prostate_ms` is pre-registered (item h). Its decision rule runs the best combination on test only if it beats the default by more than the pooled seed s.d. (`analysis/sensitivity_screens.py`; outputs in `results/sensitivity/`).

## 2. Distributed clients (Flower)

`fl_dist/` runs a **multi-process federation on a single machine over loopback gRPC** with Flower 1.7.0:
- **Processes:** `launch.py` starts one server process (`server.py`, strategy `strategy.py`) and one OS process per client (`client.py`). Each client loads only its own data (`experiments/tasks.py::build_task(part=k)`).
- **Messages:** clients send exactly their weights, n_i, q_i and the client index. The server records every message in `message_audit.json` (keys, shapes, dtypes, bytes received). `tests/test_fl_dist.py` checks the audit and that the run reproduces the custom loop, because both call the same `experiments/run_federated.py::server_step`.
- **Limitation:** on this 6 GB GPU the 5-client `prostate_ms` run fails (CUDA out of memory in one client process, even with the cross-process GPU lock `fl_dist/gpu_lock.py`). The test is marked xfail as a single-GPU limitation, and the Flower demo and parity use breast (2 clients; plan.md, item b).
- **For deployment on several machines:**
  - the server address would change from 127.0.0.1 to a reachable host;
  - the transport would need encryption and authentication (not implemented here);
  - each site would run `client.py` on its own data, as now;
  - the GPU lock would not be needed, since each machine has its own GPU;
  - network cost per round can be read from the logged bytes received.

## 3. Beta (saliency-weighting exponent)

Aggregation weights are alpha_i ∝ (n_i/N)·max(q_i, eps)^β, with β = 0 equal to FedAvg (`fedga/fedavg.py::aggregation_weights`). XFedGA+ uses the same β with its server-verified score s_i instead of q_i.
- **Rule:** β was chosen on validation by a pre-registered rule: keep 0.5 unless another value beats it by more than the pooled seed s.d. (`analysis/beta_decision.py`, `b556334`).
- **Breast:** β = 0.5 was kept; no value beat it (pooled s.d. 0.0048; `reports/review/beta_breast_table.md`).
- **Prostate:** the earlier prostate β screen is superseded by the PROMISE12 → Decathlon leakage (`docs/protocol/leakage_note.md`). A `prostate_ms` β screen {0, 0.2, 0.5, 1.0} is pre-registered (item i, `results/sensitivity/beta_screen_prostate_ms.md` once run).
- **Effect on the weights:** in the breast headline, XFedGA's alpha differs from n_i/N by only 0.0185 per client on average (RC1, `reports/diagnostics/root_cause_verification.md`). So q_i weighting changes the aggregation little at β = 0.5.
