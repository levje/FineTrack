#!/bin/bash

# print the usage message if no arguments are provided
if [ $# -ne 4 ]; then
    echo "Usage: $0 <subjects_list> <dir with results/> <tracking_dir> <sub_tract_dir>"
    echo "Example: $0 /path/part_0 /path/to/results/ /path/to/tracking/ PFT_Tracking"
    exit 1
fi

PART=${1: -2}
SUBJECTS_LIST="$1"
INPUT_DIR="$2"
TRACKING_DIR="$3"
SUBTRACT_DIR="$4"

sed -e "s|{{PART}}|$PART|g" \
    -e "s|{{SUBJECTS_LIST}}|$SUBJECTS_LIST|g" \
    -e "s|{{INPUT_DIR}}|$INPUT_DIR|g" \
    -e "s|{{TRACKING_DIR}}|$TRACKING_DIR|g" \
    -e "s|{{SUBTRACT_DIR}}|$SUBTRACT_DIR|g" \
    ~/FineTrack/postproc_verifyber_slurm.sh > postproc_verifyber_part"$PART".sh