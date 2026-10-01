#!/usr/bin/env bash
# Keep a training run backed up on Hugging Face while it trains, and keep the instance disk from
# filling: every INTERVAL seconds, upload every step_* checkpoint whose step is a multiple of
# EVERY (plus the run's config, statistics and log) with train/upload.py, which skips files already
# identical in the repo; then delete local checkpoints that are not multiples of EVERY, except the
# newest KEEP of them, so a resume always has a recent local one. Stops itself once the run's
# final/ checkpoint has been uploaded. Meant for rented instances that can vanish with the credit
# balance and have 64 GB disks: an adapter checkpoint is 465 MB.
#
#   nohup train/upload_watch.sh <run dir> <repo> [EVERY=5000] [INTERVAL=600] [KEEP=2] > train/runs/upload_watch.log 2>&1 &
set -u
RUN=$1
REPO=$2
EVERY=${3:-5000}
INTERVAL=${4:-600}
KEEP=${5:-2}
PY=${PYTHON:-python}
cd "$(dirname "$0")/.."
while true; do
    only=""
    others=()
    for d in $(ls -d "$RUN"/step_* 2>/dev/null | sort); do
        [ -d "$d" ] || continue
        step=$((10#${d##*step_}))
        if [ $((step % EVERY)) -eq 0 ]; then only="$only,${d##*/}"; else others+=("$d"); fi
    done
    [ -d "$RUN/final" ] && only="$only,final"
    if [ -n "$only" ]; then
        echo "$(date -u +%FT%TZ) uploading ${only#,}"
        $PY train/upload.py --repo "$REPO" "$RUN" --only "${only#,}" 2>&1 | grep -E "uploaded|already|error|Error|refus" || true
    fi
    n=${#others[@]}
    if [ "$n" -gt "$KEEP" ]; then
        for d in "${others[@]:0:$((n - KEEP))}"; do
            echo "$(date -u +%FT%TZ) pruning ${d##*/} (not a multiple of $EVERY, newer ones exist)"
            rm -rf "$d"
        done
    fi
    echo "$(date -u +%FT%TZ) disk: $(df -h "$RUN" | awk 'NR==2 {print $4 " free of " $2}')"
    if [ -d "$RUN/final" ]; then echo "final uploaded, done"; break; fi
    sleep "$INTERVAL"
done
