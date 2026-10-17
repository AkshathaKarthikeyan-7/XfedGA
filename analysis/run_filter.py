"""
Which run folders analysis may use. A run is analysable only if it used the current XFedGA CAM definition
(fitness.cam_layer = fullres, fitness.cam_target = pred; decision 17 Oct 2026, plan.md section 0.0).
Runs made with the old definition (beta screen of 17 Oct before 13:49, kept in results/runs/_oldcam_definition/)
are excluded everywhere, even if a glob would match them.
"""
import os

import yaml

CURRENT_CAM = {"cam_layer": "fullres", "cam_target": "pred"}


def uses_current_cam(run_dir):
    path = os.path.join(run_dir, "config_resolved.yaml")
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        fit = (yaml.safe_load(f) or {}).get("fitness", {})
    return all(fit.get(k) == v for k, v in CURRENT_CAM.items())
