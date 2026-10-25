# Guide Review demo (30 Oct 2026): XFedGA as a working federated system

**Length:** about 8 minutes (live run about 3 min) plus questions. **Marks it supports:** "design demonstrated via architecture / live PoC" and the "working demonstration" TEA.

**What it shows:**
- XFedGA running as a **multi-process federation on a single machine over loopback gRPC** (Flower 1.7.0);
- one server process and two client processes (the two breast clients, BUSI and BUS-UCLM; 5-client Flower on this 6 GB GPU is a known limitation for now);
- each client trains only on its own scans and sends its model update with its Grad-CAM saliency score q_i;
- the server weights clients by q_i, runs the GA, validates and broadcasts.

**What it does not show:** network faults, stragglers, or several machines. These are untested; say so if asked.

## Before the review (the evening before, and again 15 min before)
```powershell
cd "D:\new repo\xfedga"
$py = "D:\conda_envs\fedga\python.exe"
git pull; git log --oneline -1                     # the commit you are demoing
& $py -m pytest -q -m "not slow"                   # all pass (about 25 s)
nvidia-smi --query-gpu=name,clocks.sm,power.draw --format=csv
```
- Windows power mode: **Best performance**. Laptop on mains. Close other GPU programs, and stop the experiment queue: create `results\runs\STOP`, then wait for the current run to finish.
- Do one **rehearsal run**: the command in step 3 with `--tag rehearsal` instead of `--tag demo`. It gets its own folder (`results
uns\*_rehearsal`), which the live run cannot overwrite; keep it as the fallback. Delete any old `*_demo` folder, so the live run starts clean.
- Arrange 3 PowerShell windows: **A** the launcher, **B** the server log, **C** the client logs. Have `ARCHITECTURE.md` open in the IDE preview.

## Script
| # | Time | Window | Do | Say |
|---|---|---|---|---|
| 1 | 0:00 | IDE | Show the diagram in `ARCHITECTURE.md` | "Each hospital trains locally and computes q_i: how well its Grad-CAM agrees with its own lesion masks. The server weights clients by data size and q_i, then a genetic algorithm searches around the aggregate using a fitness that combines Dice, boundary error and explanation consistency. The best candidate, never worse than the plain aggregate on the server's fitness set, becomes the global model." |
| 2 | 1:00 | IDE | Show the table "What fl_dist sends" in `ARCHITECTURE.md` | "This is the whole protocol. Clients send weights, n_i, q_i and their index. The server sends weights and the round number. No images or masks leave a client." |
| 3 | 1:30 | A | `& $py fl_dist/launch.py --task breast --method xfedga --seed 1 --tag demo --set train.rounds=2 train.local_epochs=2` | "One command starts a server and one OS process per hospital, on this laptop, talking over loopback gRPC." |
| 4 | 1:40 | A | Point at `server listening` and `client 1/2 started`, each with a PID | "Three independent processes; each client loads only its own scans." |
| 5 | 1:50 | C | `Get-Content (Get-ChildItem results\runs\*_demo\client_1.log).FullName -Wait -Tail 5` | "Client 1 trains locally, then computes q_i on a fixed set of its own images. Only that number leaves the client." |
| 6 | 2:40 | B | `Get-Content (Get-ChildItem results\runs\*_demo\server.log).FullName -Wait -Tail 8` | "The server aggregates with q_i weights, runs the GA on its own fitness set, and scores the winner on a separate validation set. The test set is used once, at the end." |
| 7 | 4:30 | A | Wait for `all processes finished cleanly` | "Two rounds done. The validation-selected model was evaluated once on held-out test cases." |
| 8 | 4:40 | A | `Get-Content (Get-ChildItem results\runs\*_demo\message_audit.json).FullName -TotalCount 14` | "The server audited every message: 63 float32 weight arrays (6.5 MB per client per round), n_i, q_i and the index. Nothing else is accepted; anything else would stop the run." |
| 9 | 5:10 | A | `Import-Csv (Get-ChildItem results\runs\*_demo\round_log.csv).FullName \| Select round,val_dsc,q_i,alpha_i \| Format-Table` | "Per round: validation Dice, each client's q_i, and the aggregation weight it earned." |
| 10 | 5:40 | A | `& $py -m pytest tests/test_fl_dist.py -q` (about 50 s) | "This test runs the same federation and checks it is bit-identical to our single-process simulation, so every result we report from the simulation also holds for this federated implementation." |
| 11 | 6:40 | IDE | Open the results summary prepared for the review (Session 5; filled from `results/`) | Present the results that exist by then. State each as WIN / TIE / LOSE with its confidence interval. The earlier prostate TIE is superseded (leakage); prostate results come from the multi-site experiment `prostate_ms`. |
| 12 | 7:40 | – | Close | "The algorithm lives in `fedga/` and `xai/`; the simulation and the Flower federation call the same functions. All numbers come from scripts in `analysis/` reading `results/`." |

**If the live run fails:**
- Say "here is the identical run from the rehearsal".
- Show `server.log`, `message_audit.json` and `round_log.csv` from the `*_rehearsal` folder (steps 6–9).
- Then run step 10, the parity test.

**If asked to stop it mid-run:** press Ctrl+C in window A. The launcher stops all three processes and returns.

## Likely questions (answer briefly, accurately)
- **"Is this really distributed?"**
  - It is separate OS processes communicating over gRPC on one machine.
  - The design needs only a server-address change for multiple machines, but that has not been tested, and I don't claim it.
- **"Does q_i leak data?"**
  - q_i is one number per round, the mean Grad-CAM–mask overlap on 40 of the client's own images.
  - The weights themselves can leak information; no formal privacy guarantee is claimed.
  - The DP experiment (clipping + Gaussian noise) reports the accuracy cost.
- **"Why a GA?"** It searches around the aggregate on server-side data; the random-search control (same number of fitness evaluations) tests whether the GA adds anything over random perturbation.
- **"Why should I trust the explanations?"**
  - EC is optimised by the method, so it is not evidence on its own.
  - We report independent metrics: pointing game, deletion/insertion AUC, Grad-CAM++, and a parameter-randomisation sanity check.
- **"How were β and the checkpoint chosen?"**
  - On the validation set only, by a rule fixed before the results (`docs/protocol/plan.md` §0.0).
  - The test set is evaluated once per run.
