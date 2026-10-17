"""
Multi-process federation on a single machine over loopback gRPC (Flower 1.7.0).

One server process (aggregation + GA + validation, `server.py` with `strategy.py`) and one OS
process per client (local training + q_i, `client.py`), started together by `launch.py`. No
algorithm code lives here: every step calls the same functions as the custom loop
(fedga/, xai/, experiments/run_federated.server_step). Only `server_address` in the config would
change for a multi-machine deployment; that has NOT been tested.
"""
