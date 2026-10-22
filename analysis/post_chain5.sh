#!/usr/bin/env bash
# Chain 5 (22 Oct 2026, 07:00): the ONLY queue on the GPU. Replaces chains 3 / 4, killed after the Flower run hung
# (its server process hit a host-RAM MemoryError at round 7 on Wed ~21:15 and never exited; GPU idle until Thu 07:00).
# Order (user, 22 Oct): suite (the g code is uncommitted) -> code commit -> g -> XAI on g -> d (remaining runs) ->
# b (Flower; the interrupted seed-1 run was moved to _interrupted/) -> e -> f.
# WATCHDOG (every queue from now on): if the current run's directory has no file written for 90 minutes, every process
# whose command line contains that run id is killed; the event is recorded and the queue continues with its next run.
# A FAILED run that was not a watchdog kill, or a throttled run (< 1,500 MHz), stops that item after the current run.
REPO="D:/new repo/xfedga"
PY="/d/conda_envs/fedga/python.exe"
Q="$REPO/results/runs/_queues"
LOG="$Q/post_chain5.log"
FROZEN="D:/new repo/xfedga_frozen"
WATCHDOG_MIN=90
cd "$REPO" || exit 1
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
TR=$'\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'

current_run() {   # last started run id of queue log $1
  grep -oE "start [^ ]+$" "$1" 2>/dev/null | tail -1 | cut -d' ' -f2
}

stale_minutes() {  # minutes since the newest file in results/runs/$1 was written (-1 if the dir is missing)
  "$PY" - "$REPO/results/runs/$1" <<'PYEOF'
import os, sys, time
d = sys.argv[1]
if not os.path.isdir(d):
    print(-1); sys.exit()
t = max([os.path.getmtime(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs] + [os.path.getmtime(d)])
print(int((time.time() - t) / 60))
PYEOF
}

kill_run() {       # kill every process whose command line contains the run id
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match '$1' -and \$_.Name -eq 'python.exe' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >> "$LOG" 2>&1
}

run_item() {   # $1 = queue name (queues/post_$1.yaml), $2 = code root
  local item="$1" W="$2" L="$Q/post_$1.log" stopped="" killed=""
  rm -f results/runs/STOP
  local start_lines; start_lines=$(wc -l < "$L" 2>/dev/null || echo 0)
  say "start item $item (code $W)"
  "$PY" "$W/analysis/run_queue.py" "$W/queues/post_$item.yaml" --repo "$REPO" > "$Q/post_$item.stdout" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    local rid; rid=$(tail -n +"$((start_lines + 1))" "$L" 2>/dev/null | grep -oE "start [^ ]+$" | tail -1 | cut -d' ' -f2)
    if [ -n "$rid" ]; then
      local m; m=$(stale_minutes "$rid")
      if [ "$m" -ge "$WATCHDOG_MIN" ]; then
        say "WATCHDOG: $rid wrote nothing for $m min -> killed; queue continues"
        kill_run "$rid"; killed="$killed $rid"
        sleep 30
      fi
    fi
    local new; new=$(tail -n +"$((start_lines + 1))" "$L" 2>/dev/null)
    local bad; bad=$(echo "$new" | grep -E "FAILED|GPU median (1[0-4][0-9][0-9]|[0-9]{3})\.0 MHz" | grep -vF -e "${killed:- __none__}" | head -1)
    if [ -z "$stopped" ] && [ -n "$bad" ]; then
      for k in $killed; do bad=$(echo "$bad" | grep -v "$k"); done
      if [ -n "$bad" ]; then
        touch results/runs/STOP; stopped=1
        say "item $item: FAILED or throttled run -> STOP after the current run: $bad"
      fi
    fi
  done
  wait "$pid"
  local new; new=$(tail -n +"$((start_lines + 1))" "$L" 2>/dev/null)
  say "end item $item done=$(echo "$new" | grep -c ' done ') failed=$(echo "$new" | grep -c FAILED) watchdog_kills=$(echo $killed | wc -w)${stopped:+ (stopped early)}"
  rm -f results/runs/STOP
}

worktree() {
  local c; c=$(git rev-parse --short "$1")
  [ -d "$FROZEN/$c" ] || git worktree add -q --detach "$FROZEN/$c" "$c" >> "$LOG" 2>&1
  echo "$FROZEN/$c"
}

say "chain 5 started (single queue)"
"$PY" -m pytest -q -p no:warnings > "$Q/post_chain5_suite.txt" 2>&1
rc=$?
say "full suite rc=$rc: $(tail -1 "$Q/post_chain5_suite.txt")"
G_OK=""
if [ "$rc" -eq 0 ]; then
  git add configs/prostate_ms_runmc_clients.yaml experiments/config.py experiments/tasks.py experiments/run_federated.py \
    fl_dist/launch.py analysis/fov_crop_size.py analysis/protocol_sets.py analysis/make_ghi_queues.py \
    analysis/sensitivity_screens.py analysis/recost_20oct_ghi.py analysis/recost_22oct.py analysis/post_headline_chain2.sh \
    analysis/post_headline_chain3.sh analysis/post_headline_chain4.sh analysis/post_chain5.sh tests/test_prostate_ms.py \
    reports/review/fitness_weights_sensitivity.md reports/review/beta_breast_table.md reports/review/panel_responses.md \
    results/timing/recost_20oct_ghi.json results/timing/recost_20oct_ghi.md results/timing/recost_22oct.json \
    results/timing/recost_22oct.md >> "$LOG" 2>&1
  git commit -q -m "prostate_ms_runmc task (item g), sensitivity screens, panel responses, recosts, chains 2-5 (watchdog)${TR}
Full suite: $(tail -1 "$Q/post_chain5_suite.txt")" >> "$LOG" 2>&1 && git push -q origin main >> "$LOG" 2>&1
  say "code commit $(git rev-parse --short HEAD)"
  W=$(worktree HEAD); G_OK=1
else
  say "SUITE NOT CLEAN: nothing committed; item g SKIPPED (needs the new code); d, b, e, f run from 9f0225f"
  W=$(worktree 9f0225f)
fi

if [ -n "$G_OK" ]; then
  run_item g_site_runmc "$W"
  for d in results/runs/*_runmc; do
    [ -d "$d" ] || continue
    rid=$(basename "$d"); [ -f "results/xai/$rid/summary.json" ] && continue
    "$PY" analysis/xai_metrics.py --runs "$d" >> "$Q/post_chain5_xai.stdout" 2>&1
    say "xai $rid rc=$?"
  done
fi
run_item d_hrob2 "$W"
run_item b_flower "$W"
run_item e_ablations "$W"
run_item f_fedprox_bounds "$W"
say "chain 5 finished"
