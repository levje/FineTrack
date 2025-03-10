#!/bin/bash

set -e

# We need to make sure that the right output path is set.
# This is where the tractograms and the results will be saved.
PREDICTION_PATH=~/Documents/FineTrack/data/predictions/

# EDIT ====
# The path to the trained agent checkpoint.
AGENTCHECKPOINT=/home/local/USHERBROOKE/levj1404/Documents/FineTrack/data/experiments/SACAuto-3000/SACAuto-3000_2025-03-04_09-43-46/1111/model/best_model_state.ckpt
# AGENTCHECKPOINT=/home/local/USHERBROOKE/levj1404/Documents/FineTrack/data/experiments/CrossQ_RLHF_2/CrossQ_RLHF_2_2025-03-04_09-34-48/1111/model/best_model_state.ckpt

# Now, based on the agent checkpoint, we can get the experiment's ID.
# In the path we have **/experiments/EXPERIMENT_NAME/EXPERIMENT_ID/**.
# We can extract the experiment name and ID from the path.
# This is useful to save the results in the right directory.
EXPERIMENT_NAME=$(echo ${AGENTCHECKPOINT} | awk -F 'experiments/' '{print $2}' | awk -F '/' '{print $1}')
EXPERIMENT_ID=$(echo ${AGENTCHECKPOINT} | awk -F 'experiments/' '{print $2}' | awk -F '/' '{print $2}')

# Ask to the user if the experiment name and ID are correct.
echo "Experiment name: ${EXPERIMENT_NAME}"
echo "Experiment ID: ${EXPERIMENT_ID}"
read -p "Are the experiment name and ID correct? (y/n) " -n 1 -r
echo
# if the user does not confirm, ask for the experiment name and ID.
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    read -p "Enter the experiment name: " EXPERIMENT_NAME
    read -p "Enter the experiment ID: " EXPERIMENT_ID
fi

# Create the output directory
OUTPUT_DIR=data/predictions/${EXPERIMENT_NAME}/${EXPERIMENT_ID}
mkdir -p ${OUTPUT_DIR}

# EDIT ====
# Arguments
# MC_ORACLE_CHECKPOINT=custom_models/ismrm_paper_oracle/ismrm_paper_oracle.ckpt
MC_ORACLE_CHECKPOINT=custom_models/ismrm_oracles_nb_points/OracleNet-Transformer-Crit-32-Dense/_best_vc_epoch.ckpt
NPV=1

# Fixed parameters
DATASETDIR=data/datasets/ismrm2015_2mm
IN_ODF=$DATASETDIR/fodfs/ismrm2015_fodf.nii.gz
IN_SEED=/home/local/USHERBROOKE/levj1404/Documents/FineTrack/dipy_segment_tissues_new_b02/t1_interface.nii.gz
IN_MASK=/home/local/USHERBROOKE/levj1404/Documents/FineTrack/dipy_segment_tissues_new_b02/t1_wm_dilated.nii.gz
IN_GM_MASK=/home/local/USHERBROOKE/levj1404/Documents/FineTrack/dipy_segment_tissues_new_b02/t1_gm_eroded.nii.gz
OUT_TRACTOGRAM=${OUTPUT_DIR}/tractogram.trk

python FineTrack/runners/ttl_track.py \
    SACAuto \
    ${IN_ODF} \
    ${IN_SEED} \
    ${IN_MASK} \
    ${OUT_TRACTOGRAM} \
    --agent_checkpoint ${AGENTCHECKPOINT} \
    --npv ${NPV} \
    -f \
    --gm_mask ${IN_GM_MASK} \
    --mc_oracle_checkpoint ${MC_ORACLE_CHECKPOINT}

# Once the tractogram is generated, we can compute the metrics.
# We will save the results in the same directory as the tractogram.
bash scripts/tractogram_post_processing.sh ${OUTPUT_DIR} ${DATASETDIR} -f tck

