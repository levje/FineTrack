#!/bin/bash
set -e
source ~/Documents/FineTrack/venv/bin/activate

checkpoint=$1
algo=$2
out_dir=$3
rng_seed=$4

# If there's an argument missing, print usage and exit.
if [ -z ${checkpoint} ] || [ -z ${out_dir} ] || [ -z ${algo} ]; then
	echo "Usage: finetrack_on_ismrm.sh <checkpoint> <algo> <out_dir>"
	exit 1
fi

SUB_DATA_DIR=~/Documents/FineTrack/data/datasets/ismrm2015_2mm/

if [ ! -d ${SUB_DATA_DIR} ]; then
	echo "No such directory: ${SUB_DATA_DIR}"
	exit 1
fi

if [ ! -f ${checkpoint} ]; then
	echo "No such file: ${checkpoint}"
	exit 1
fi

echo "Tracking on HCP data for subject ${subid}"

OUT_DIR=${out_dir}

mkdir -p $OUT_DIR


python ~/Documents/FineTrack/FineTrack/runners/ttl_track.py \
	${algo} \
	${SUB_DATA_DIR}/fodfs/ismrm2015_fodf.nii.gz \
	${SUB_DATA_DIR}/maps/interface.nii.gz \
	${SUB_DATA_DIR}/masks/ismrm2015_wm.nii.gz \
	$OUT_DIR/ismrm_ft_tracking.tck \
	--npv 19 \
	--agent_checkpoint ${checkpoint} \
	--rng_seed ${rng_seed} \
	-f

scil_tractogram_compress.py $OUT_DIR/ismrm_ft_tracking.tck $OUT_DIR/ismrm_ftc_tracking.tck -e 0.1 -f
rm $OUT_DIR/ismrm_ft_tracking.tck

echo "Tracking done, scoring the tractogram"

bash ~/Documents/FineTrack/scripts/scil_score_ismrm_Renauld2023.sh $OUT_DIR/ismrm_ftc_tracking.tck $OUT_DIR/scoring ${SUB_DATA_DIR}/scoring_data

echo "Done."

