#!/usr/bin/env bash
# Second chain (user order 20 Oct 2026): a (running in chain 1) -> b (chain 1) -> g -> g2 -> h -> i -> c -> d -> e -> f.
# Chain 1 (analysis/post_headline_chain.sh) is NOT edited. Handover: once item b's queue has finished, results/runs/STOP
# is kept present until chain 1 logs "chain finished"; chain 1's items c-f therefore stop before their first run.
# Then: full suite -> (if clean) commit of the new task code + frozen worktree -> g, g2, h, i -> h / i decisions
# (committed; the conditional h follow-up queue is committed before it runs) -> XAI metrics on g / g2 -> c, d, e, f.
# Safety per item: a FAILED run or a run with median SM clock < 1,500 MHz -> STOP after the current run, recorded,
# next item. If the suite fails, nothing is committed and g (which needs the new code) is skipped.
REPO="D:/new repo/xfedga"
PY="/d/conda_envs/fedga/python.exe"
Q="$REPO/results/runs/_queues"
LOG="$Q/post_chain2.log"
FROZEN="D:/new repo/xfedga_frozen"
cd "$REPO" || exit 1
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
TR=$'\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'

run_item() {   # $1 = queue name (queues/post_$1.yaml), $2 = code root
  local item="$1" W="$2" L="$Q/post_$1.log" stopped=""
  rm -f results/runs/STOP
  say "start item $item (code $W)"
  "$PY" "$W/analysis/run_queue.py" "$W/queues/post_$item.yaml" --repo "$REPO" > "$Q/post_$item.stdout" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    if [ -z "$stopped" ] && [ -f "$L" ] && { grep -q FAILED "$L" || grep -qE "GPU median (1[0-4][0-9][0-9]|[0-9]{3})\.0 MHz" "$L"; }; then
      touch results/runs/STOP; stopped=1
      say "item $item: FAILED or throttled run -> STOP after the current run: $(grep -E 'FAILED|GPU median (1[0-4][0-9][0-9]|[0-9]{3})\.0' "$L" | tail -1)"
    fi
  done
  wait "$pid"
  say "end item $item done=$(grep -c ' done ' "$L" 2>/dev/null) failed=$(grep -c FAILED "$L" 2>/dev/null)${stopped:+ (stopped early)}"
  rm -f results/runs/STOP
}

worktree() {   # $1 = commit -> prints the frozen path
  local c; c=$(git rev-parse --short "$1")
  [ -d "$FROZEN/$c" ] || git worktree add -q --detach "$FROZEN/$c" "$c" >> "$LOG" 2>&1
  echo "$FROZEN/$c"
}

# 1. handover from chain 1 (between items; nothing running is interrupted)
until grep -q "queue post_b_flower finished" "$Q/post_b_flower.log" 2>/dev/null; do sleep 30; done
say "item b finished; holding STOP until chain 1 ends (its c-f stop before their first run)"
until grep -q "chain finished" "$Q/post_chain.log" 2>/dev/null; do touch results/runs/STOP; sleep 0.2; done
rm -f results/runs/STOP
say "chain 1 finished; its items c-f: $(grep -h 'STOP file found' "$Q"/post_[cdef]_*.log 2>/dev/null | wc -l) stopped before their first run"

# 2. full suite, then commit the new task code
"$PY" -m pytest -q -p no:warnings > "$Q/post_chain2_suite.txt" 2>&1
rc=$?
say "full suite rc=$rc: $(tail -1 "$Q/post_chain2_suite.txt")"
G_OK=""
if [ "$rc" -eq 0 ]; then
  git add configs/prostate_ms_runmc_clients.yaml experiments/config.py experiments/tasks.py experiments/run_federated.py \
    fl_dist/launch.py analysis/fov_crop_size.py analysis/protocol_sets.py analysis/make_ghi_queues.py \
    analysis/sensitivity_screens.py analysis/recost_20oct_ghi.py analysis/post_headline_chain2.sh tests/test_prostate_ms.py \
    reports/review/fitness_weights_sensitivity.md reports/review/beta_breast_table.md reports/review/panel_responses.md \
    results/timing/recost_20oct_ghi.json results/timing/recost_20oct_ghi.md >> "$LOG" 2>&1
  git commit -q -m "prostate_ms_runmc task (item g), sensitivity-screen decisions, panel responses, recost with g-i; chain 2${TR}
Full suite: $(tail -1 "$Q/post_chain2_suite.txt")" >> "$LOG" 2>&1 && git push -q origin main >> "$LOG" 2>&1
  say "code commit $(git rev-parse --short HEAD)"
  W=$(worktree HEAD); G_OK=1
else
  say "SUITE FAILED: nothing committed; item g (needs the new code) SKIPPED; g2, h, i, c-f run from the last pre-registration commit"
  W=$(worktree 9f0225f)
fi

# 3. items g, g2, h, i
[ -n "$G_OK" ] && run_item g_site_runmc "$W"
run_item g2_breast_replication "$W"
run_item h_fw_screen "$W"
run_item i_beta_screen "$W"

# 4. decisions of h and i (validation only); conditional h follow-up, committed before it runs
CUDA_VISIBLE_DEVICES=-1 "$PY" analysis/sensitivity_screens.py > "$Q/post_chain2_decisions.stdout" 2>&1
say "h / i decisions rc=$?: $(grep -h 'Decision' results/sensitivity/*_prostate_ms.md | tr '\n' ' ')"
"$PY" analysis/make_ghi_queues.py --h_followup >> "$Q/post_chain2_decisions.stdout" 2>&1
hf=$?
if [ "$hf" -eq 0 ]; then
  git add results/sensitivity queues/post_h_followup.yaml >> "$LOG" 2>&1
  git commit -q -m "h decision: follow-up queue (declared secondary, 5 seeds, test once) committed before its runs${TR}" >> "$LOG" 2>&1
  git push -q origin main >> "$LOG" 2>&1
  say "h follow-up queue committed $(git rev-parse --short HEAD)"
  run_item h_followup "$(worktree HEAD)"
else
  git add results/sensitivity >> "$LOG" 2>&1
  git commit -q -m "h / i decisions (validation only)${TR}" >> "$LOG" 2>&1 && git push -q origin main >> "$LOG" 2>&1
  say "h: no follow-up (exit $hf); decisions committed $(git rev-parse --short HEAD)"
fi

# 5. XAI metrics on the g and g2 runs (for reporting)
for d in results/runs/*_runmc results/runs/*_rep; do
  [ -d "$d" ] || continue
  rid=$(basename "$d")
  [ -f "results/xai/$rid/summary.json" ] && continue
  "$PY" analysis/xai_metrics.py --runs "$d" >> "$Q/post_chain2_xai.stdout" 2>&1
  say "xai $rid rc=$?"
done

# 6. items c-f (pre-registered 19 Oct; queues unchanged)
for item in c_paperga d_hrob2 e_ablations f_fedprox_bounds; do
  run_item "$item" "$W"
done
say "chain 2 finished"
