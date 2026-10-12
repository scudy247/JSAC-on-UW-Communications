#!/bin/bash
# Run one experiment as a window of the tmux session "doppler", light on the shared machine: single thread, lowest
# priority (nice 19), log in results/logs/<name>.log ending with "exit <code> <date>". Survives closing the terminal
# or Claude. Refuses to start when MAX_JOBS of our experiments already run (the machine is shared).
# Usage: bash run_job.sh <name> <script.py> [args...]
# Monitor: tmux ls; tmux list-windows -t doppler; tail -f code/results/logs/<name>.log; attach: tmux attach -t doppler
set -e
MAX_JOBS=${MAX_JOBS:-4}
cd "$(dirname "$0")"
here=$(pwd)
name=$1
shift
pattern="^/home/dario/Underwater/dario/bin/python scripts/"
running=$(pgrep -u "$USER" -fc "$pattern" || true)
wait_cmd=""
if [ "$running" -ge "$MAX_JOBS" ]; then
    if [ "${WAIT:-0}" != "1" ]; then
        echo "not started: $running experiments already running (MAX_JOBS=$MAX_JOBS); WAIT=1 queues it" >&2
        exit 1
    fi
    # queued: the tmux window polls every 2 min and starts the job when a slot is free
    wait_cmd="while [ \$(pgrep -u $USER -fc '$pattern') -ge $MAX_JOBS ]; do sleep 120; done; "
    echo "queued: $running experiments running, $name starts when one finishes"
fi
mkdir -p results/logs
log=results/logs/$name.log
cmd="cd $here && echo \"queued \$(date)\" > $log && ${wait_cmd}echo \"started \$(date)\" >> $log && \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$here/src:$here/scripts \
nice -n 19 /home/dario/Underwater/dario/bin/python scripts/$* >> $log 2>&1; echo \"exit \$? \$(date)\" >> $log"
tmux has-session -t doppler 2>/dev/null || tmux new-session -d -s doppler -n idle
tmux new-window -d -t doppler -n "$name" "$cmd"
echo "started $name -> code/$log"
