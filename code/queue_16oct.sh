#!/bin/bash
# Runs prepared on 16 Oct 2026 (PLAN: Phases 2, 5, 6). Two sequential lanes, so at most two of our jobs at a time;
# each job single-thread, nice 19, 6 GB address-space cap. Launch:  bash code/queue_16oct.sh
# Monitor: tmux list-windows -t doppler; tail -f code/results/logs/<name>.log
# Lane A, multi-user (~4 h): 1) PER floor without guards with the pilot-residual noise; 2-4) sensor-seeded pilot
#   tracking with 8 guards, industrial sensors / consumer sensors / 2 deg direction errors.
# Lane B (~6.5 h): 5) B with the Viterbi frequency-line tracker (lower-threshold signal-only baseline);
#   6) multipath 16-QAM, 100 packets, per-packet errors stored (paired McNemar test).
cd "$(dirname "$0")"
here=$(pwd)
mkdir -p results/logs
RUN="cd $here && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$here/src:$here/scripts prlimit --as=6442450944 nice -n 19 /home/dario/Underwater/dario/bin/python"
job() {  # name, script + args: one step of a lane, with its own log ending in "exit <code> <date>"
    local name=$1; shift
    echo "echo \"started \$(date)\" > results/logs/$name.log; $RUN scripts/$* >> results/logs/$name.log 2>&1; echo \"exit \$? \$(date)\" >> results/logs/$name.log"
}
MU="b22_multiuser.py 60 --mod 16qam --cases 'buoy users' --no-alone"
laneA="$(job mu_noise_max $MU --ks 4,8 --guards 0 --snrs 20,30 --noise-source max --out noise_max_16qam_buoy)"
laneA="$laneA; $(job mu_sp_industrial $MU --ks 4,8 --guards 8 --snrs 30 --out sp_industrial)"
laneA="$laneA; $(job mu_sp_consumer $MU --ks 4,8 --guards 8 --snrs 30 --grade consumer --out sp_consumer)"
laneA="$laneA; $(job mu_sp_direrr2 $MU --ks 4,8 --guards 8 --snrs 30 --dir-err-deg 2 --out sp_direrr2)"
laneB="$(job b21_hmm b21_waveform_b.py 40 hmm)"
laneB="$laneB; $(job mp_16qam_paired b13_multipath.py bb_iso_1s_16qam 100 _paired)"
tmux has-session -t doppler 2>/dev/null || tmux new-session -d -s doppler -n idle
tmux new-window -d -t doppler -n laneA "$laneA"
tmux new-window -d -t doppler -n laneB "$laneB"
echo "launched lanes A and B in tmux session doppler"
