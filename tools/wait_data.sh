#!/bin/bash
# Waits for the iNaturalist download and detection to finish, then stops the detectors.
# The download runs on the login node and its process is not visible from the compute node, so we look at the files:
# done when there have been no new photos for 15 minutes and both models have processed everything there is.
# Arguments: deadline (e.g. "2026-11-03 03:00"), then the detector job ids.
# After the deadline we move on with what we have: training matters more than the last frames.

DEADLINE=$(date -d "$1" +%s); shift
R=$HOME/DiffusionLearn_data
prev=-1; stable=0
while true; do
    n=$(find "$R/inat/images" -type f ! -name "*.tmp" | wc -l)
    y=$(wc -l < "$R/detections/yolo26x_inat.jsonl")
    d=$(wc -l < "$R/detections/rtdetr-x_inat.jsonl")
    if [ "$n" -eq "$prev" ]; then stable=$((stable + 1)); else stable=0; fi
    prev=$n
    echo "$(date +%H:%M) photos $n, yolo $y, rtdetr $d, no new for $stable min"
    if [ "$stable" -ge 15 ] && [ "$y" -ge "$n" ] && [ "$d" -ge "$n" ]; then echo "data ready"; break; fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "deadline, moving on with what we have"; break; fi
    sleep 60
done
scancel "$@"
