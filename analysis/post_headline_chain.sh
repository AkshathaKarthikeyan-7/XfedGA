#!/usr/bin/env bash
# Post-headline chain (user order 19 Oct 2026). Started in the background; never touches the running final headline.
#   1. wait for the final headline to finish
#   2. full test suite (for the uncommitted analysis code; logged, does not gate the frozen queues)
#   3. XAI metrics on the 35 final checkpoints (results/xai/<run_id>/; existing outputs are not overwritten)
#   4. final tables: make_tables --group main,final --comparisons final --tasks breast,prostate_ms
#   5. queues a-f, one after another, from the frozen worktree at the last pre-registration commit (fcf54ee).
#      Safety: if an item logs a FAILED run or a run with median SM clock < 1,500 MHz, results/runs/STOP is created so
#      that item stops after its current run; the event is recorded; STOP is removed and the next item starts.
# Every event is appended to results/runs/_queues/post_chain.log.
REPO="D:/new repo/xfedga"
W="D:/new repo/xfedga_frozen/fcf54ee"
PY="/d/conda_envs/fedga/python.exe"
Q="$REPO/results/runs/_queues"
LOG="$Q/post_chain.log"
cd "$REPO" || exit 1
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

until [ "$(grep -c 'queue final_headline finished' "$Q/final_headline.log")" -ge 2 ]; do sleep 120; done
say "final headline finished: $(grep -c ' done ' "$Q/final_headline.log") done, $(grep -c FAILED "$Q/final_headline.log") failed"

"$PY" -m pytest -q -p no:warnings > "$Q/post_chain_suite.txt" 2>&1
say "full suite rc=$?: $(tail -1 "$Q/post_chain_suite.txt")"

for d in results/runs/*_final; do
  rid=$(basename "$d")
  if [ -f "results/xai/$rid/summary.json" ]; then continue; fi
  "$PY" analysis/xai_metrics.py --runs "$d" >> "$Q/post_chain_xai.stdout" 2>&1
  say "xai $rid rc=$?"
done

"$PY" analysis/make_tables.py --group main,final --comparisons final --tasks breast,prostate_ms --out results/tables/final \
  > "$Q/post_chain_tables.stdout" 2>&1
say "final tables rc=$?"

for item in a_dose_response b_flower c_paperga d_hrob2 e_ablations f_fedprox_bounds; do
  rm -f results/runs/STOP
  say "start item $item"
  "$PY" "$W/analysis/run_queue.py" "$W/queues/post_$item.yaml" --repo "$REPO" > "$Q/post_$item.stdout" 2>&1 &
  pid=$!
  L="$Q/post_$item.log"
  stopped=""
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    if [ -z "$stopped" ] && [ -f "$L" ] && { grep -q FAILED "$L" || grep -qE "GPU median (1[0-4][0-9][0-9]|[0-9]{3})\.0 MHz" "$L"; }; then
      touch results/runs/STOP
      stopped=1
      say "item $item: FAILED or throttled run detected -> STOP after the current run: $(grep -E 'FAILED|GPU median (1[0-4][0-9][0-9]|[0-9]{3})\.0' "$L" | tail -1)"
    fi
  done
  wait "$pid"
  say "end item $item rc=$? done=$(grep -c ' done ' "$L" 2>/dev/null) failed=$(grep -c FAILED "$L" 2>/dev/null)${stopped:+ (stopped early)}"
  rm -f results/runs/STOP
done
say "chain finished"
