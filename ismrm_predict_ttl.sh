
# Arguments
AGENTCHECKPOINT=custom_models/sac_checkpoint/model/last_model_state.ckpt
# AGENTCHECKPOINT=custom_models/ismrm_1mm_sac_checkpoint/last_model_state.ckpt
MC_ORACLE_CHECKPOINT=custom_models/ismrm_paper_oracle/ismrm_paper_oracle.ckpt
OUT_TRACTOGRAM=out_tractograms/ttl_test.tck
NPV=10

# Fixed parameters
IN_ODF=data/datasets/ismrm2015_2mm/fodfs/ismrm2015_fodf.nii.gz
IN_SEED=data/datasets/ismrm2015_2mm/masks/ismrm2015_interface.nii.gz
IN_MASK=data/datasets/ismrm2015_2mm/masks/ismrm2015_wm.nii.gz
IN_GM_MASK=data/datasets/ismrm2015_2mm/masks/t1_gm_resampled.nii.gz
python FineTrack/runners/ttl_track.py \
    ${IN_ODF} \
    ${IN_SEED} \
    ${IN_MASK} \
    ${OUT_TRACTOGRAM} \
    --min_length 20 \
    --max_length 200 \
    --agent_checkpoint ${AGENTCHECKPOINT} \
    --mc_oracle_checkpoint ${MC_ORACLE_CHECKPOINT} \
    --gm_mask ${IN_GM_MASK} \
    --npv ${NPV} \
    -f
