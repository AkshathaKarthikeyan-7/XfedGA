# Live demo: XFedGA as a multi-process federation on a single machine over loopback gRPC

**Length:** about 4 minutes. **What it shows:**
- one server process and two client processes (PROMISE12 clients 1 and 2);
- each client trains only on its own data and sends **weights, n_i and its Grad-CAM saliency score q_i**;
- the server weights the clients by q_i, runs the GA, validates, and broadcasts.

**What it does not show:** network faults, stragglers, or several machines. These are untested, so say so if asked.

## Before recording (once)
```powershell
cd "D:\new repo\xfedga"
$py = "D:\conda_envs\fedga\python.exe"
# Windows power mode: Best performance. Close other GPU programs.
nvidia-smi --query-gpu=name,clocks.sm,power.draw --format=csv     # expect RTX 4050, ~2.7 GHz under load
```
Arrange 3 windows side by side: **A** the launcher, **B** the server log, **C** the two client logs.

## Recording script
| Time | Window | Action / command | Say |
|---|---|---|---|
| 0:00 | A | `& $py fl_dist/launch.py --task prostate --method xfedga --seed 1 --tag demo --set train.rounds=2 train.local_epochs=2` | "One command starts a server and one OS process per hospital, all on this laptop, talking over loopback gRPC." |
| 0:10 | A | read the lines `server listening`, `client 1 started`, `client 2 started` (each with a PID) | "Three separate processes. Each client loads only its own scans." |
| 0:20 | C | `Get-Content results\runs\*_demo\client_1.log -Wait -Tail 5` | "Client 1 trains locally, then computes q_i: how well its Grad-CAM matches its own masks. Only the score leaves the client." |
| 1:30 | B | `Get-Content results\runs\*_demo\server.log -Wait -Tail 8` | "The server weights the two clients by data size and q_i, then runs the GA on its own fitness set and validates the winner." |
| 3:00 | A | wait for `all processes finished cleanly` | "Two rounds done; the validation-selected model was evaluated once on the held-out test set." |
| 3:10 | A | `Get-Content results\runs\*_demo\message_audit.json -TotalCount 12` | "The audit lists everything the server received: float32 weight arrays (6.5 MB per client per round), n_i, q_i and the client's index. No images, no masks." |
| 3:30 | A | `Import-Csv results\runs\*_demo\round_log.csv \| Select round,val_dsc,q_i,alpha_i \| Format-Table` | "Per round: validation Dice, each client's q_i and the aggregation weight it earned." |
| 3:50 | A | (optional) `& $py -m pytest tests/test_fl_dist.py -q` | "A test checks that this federation gives bit-identical results to our single-process simulation." |

If Ctrl+C is pressed during the run, the launcher stops all three processes (show this only if asked).

## Talking points (accurate; do not overstate)
- **Claim:** "multi-process federation on a single machine over loopback gRPC (Flower 1.7.0)." Moving it to several machines would only change the server address in the config, but that has **not** been tested.
- **The protocol, as audited:** clients send weights, n_i, q_i and an index; the server sends the global weights and the round number.
- **No formal privacy guarantee is claimed.** The weights themselves can leak information; the DP experiment reports the accuracy cost of clipping plus noise.
- **Same code as the simulation:** no algorithm code is duplicated in `fl_dist/`, and the 1-round parity test is bit-identical.
