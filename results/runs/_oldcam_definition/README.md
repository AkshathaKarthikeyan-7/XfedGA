# Beta-screen runs with the OLD Grad-CAM definition: excluded from all analysis

These 4 runs (17 Oct 2026, 11:55–13:55, frozen code `bd6baeb`) used the pre-17-Oct CAM definition:
the half-resolution decoder conv (`unet_cam_layer`) with the all-pixel target. That layer failed a
known-answer localisation test (`tests/test_xai_known_answer.py`), so the definition was replaced by
decision (b) on 17 Oct (plan.md section 0.0; re-audit item 6). The screen was stopped after its 4th run
and rerun with the new definition (full-resolution decoder layer, predicted-foreground target).

They are kept for the record only. `analysis/run_filter.py` excludes them from every table and from the
beta decision, even if a glob matched them. The one place they are read is
`analysis/cam_change_sanity.py`, which compares the old and new CAM on one checkpoint (validation and
client data only).
