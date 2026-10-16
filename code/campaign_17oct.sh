#!/bin/bash
# Campaign of 17 Oct 2026 (PLAN: Phases 2, 5, 6) in one command, on 25 cores of the shared machine.
#   1. check the load: if 25 cores (and memory) are not free, stop here and start NOTHING (user's rule);
#   2. smoke stage: one 1-2 packet job of every type (scripts/campaign_jobs_smoke.py), ~10 min;
#   3. only if every smoke job succeeded: check the load again and run the full campaign (scripts/campaign_jobs.py,
#      179 jobs, ~100 core-hours, ~4-4.5 h).
# Runs in the foreground of whatever starts it: launch it inside tmux, e.g.
#   tmux new-window -d -t doppler -n camp17 "bash code/campaign_17oct.sh > code/results/campaign/camp17.log 2>&1"
# Monitor: python scripts/campaign.py status ; tail code/results/campaign/scheduler.log
set -u
cd "$(dirname "$0")"
PY=/home/dario/Underwater/dario/bin/python
mkdir -p results/campaign results/campaign_smoke
echo "$(date '+%F %H:%M') step 1: load check"
$PY scripts/campaign.py check --workers 25 || { echo "not enough free cores: nothing started"; exit 1; }
echo "$(date '+%F %H:%M') step 2: smoke stage"
CAMPAIGN_JOBS=campaign_jobs_smoke CAMPAIGN_STATE=results/campaign_smoke \
    $PY scripts/campaign.py run --workers 25 > results/campaign_smoke/scheduler.log 2>&1
if ls results/campaign_smoke/*.failed >/dev/null 2>&1 || \
   CAMPAIGN_JOBS=campaign_jobs_smoke CAMPAIGN_STATE=results/campaign_smoke $PY scripts/campaign.py status | grep -q "pending"; then
    echo "smoke stage NOT clean (see results/campaign_smoke/logs): full campaign not started"
    CAMPAIGN_JOBS=campaign_jobs_smoke CAMPAIGN_STATE=results/campaign_smoke $PY scripts/campaign.py status | grep -v "  done"
    exit 2
fi
echo "$(date '+%F %H:%M') smoke clean; step 3: full campaign"
$PY scripts/campaign.py run --workers 25 > results/campaign/scheduler.log 2>&1
echo "$(date '+%F %H:%M') exit $?"
