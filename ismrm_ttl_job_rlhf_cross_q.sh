#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=30000M
#SBATCH --time=0-25:00:00
#SBATCH --mail-user=jeremi.levesque@usherbrooke.ca
#SBATCH --mail-type=ALL

# The above comments are used by SLURM to set the job parameters.
set -e

# Experiment parameters
USECOMET=1
EXPNAME="RLHF-CrossQ"
COMETPROJECT="RLHF-CrossQ"
NB_STREAMLINES_POINTS=32
EXPID="test-logging-"_$(date +"%F-%H_%M_%S")

MAXEP=16               # Number of RLHF iterations
PRETRAINSTEPS=0        # Number of steps for pretraining if no agent checkpoint is provided.
WARMUPSTEPS=200        # Number of steps to warmup the agent before generating new streamlines.
AGENTNBSTEPS=50        # Number of steps for the agent
ORACLENBSTEPS=4        # Number of steps for the oracle
ALG="CrossQ"

NPV=20 # Number of points per tractogram for training
RLHFINTERNPV=30         # Number of seeds per tractogram generated during the RLHF pipeline
SEEDS=(1111)
BATCHSIZE=4096
N_ACTORS=4096
GAMMA=0.95 # Reward discounting (could also be 0.95)
LR=0.0005 # 1e-5
THETA=30

# Oracle training params
ORACLE_LR=0.00001 # This will override the LR within the checkpoint.
TOTAL_BATCH_SIZE=2048
ORACLE_MICRO_BATCH_SIZE=1024
GRAD_ACCUM_STEPS=$((TOTAL_BATCH_SIZE / ORACLE_MICRO_BATCH_SIZE))
DISABLE_ORACLE_TRAINING=0

# Check if the script is ran locally or on a cluster node.
if [ -z $SLURM_JOB_ID ]; then
    islocal=1
else
    islocal=0
fi

if [ $islocal -eq 1 ]; then
    # This script should be ran from the root of the project is ran locally.
    echo "Running training locally..."
    SOURCEDIR=.
    DATADIR=data/datasets/ismrm2015_2mm
    EXPDIR=data/experiments
    # BACKUPDIR=data/backups
    DATASETDIR=$DATADIR
    
    # Oracle Antoine with partial streamlines (dense).
    ORACLE_CRIT_CHECKPOINT=custom_models/ismrm_oracles_nb_points/OracleNet-Transformer-Crit-32-Dense/_best_vc_epoch.ckpt

    # Oracle trained on full streamlines (not dense).
    ORACLE_REWARD_CHECKPOINT=custom_models/ismrm_oracles_nb_points/OracleNet-Transformer-Crit-32-Classif/_best_vc_epoch.ckpt

    # Checkpoint used to load the initial weights and optimizers of the agent.
    # AGENTCHECKPOINT=custom_models/sac_checkpoint/model/last_model_state.ckpt
    # AGENTCHECKPOINT=/home/local/USHERBROOKE/levj1404/Documents/FineTrack/data/experiments/TrackToLearn-ISMRM/fullckpt-32pts-_2024-12-10-13_29_48/1111/model/best_model_state.ckpt

    # Setup dataset to augment
    #DATASET_TO_AUGMENT=data/datasets/ismrm2015_1mm/streamlines/stable/train_test_classical_tracts_antoine.hdf5
else
    echo "Running training on a cluster node..."
    module load python/3.10 cuda cudnn httpproxy
    SOURCEDIR=~/FineTrack
    DATADIR=$SLURM_TMPDIR/data
    EXPDIR=$SLURM_TMPDIR/experiments
    PYTHONEXEC=python
    PROJECTS_DIR=~/projects/def-pmjodoin/levje
    BACKUPDIR=~/scratch/
    export COMET_API_KEY=$(cat ~/.comet_api_key)

    # Prepare virtualenv
    echo "Sourcing ENV-TTL-2 virtual environment..."
    source ~/ENV-TTL-2/bin/activate

    # Prepare datasets
    mkdir -p $DATADIR
    mkdir -p $EXPDIR

    echo "Unpacking datasets..."
    tar xf ${PROJECTS_DIR}/datasets/ismrm2015_2mm_ttl.tar.gz -C $DATADIR
    DATASETDIR=$DATADIR/ismrm2015_2mm

    echo "Copying oracle checkpoint..."
    cp ${PROJECTS_DIR}/oracles/ismrm_oracles_nb_points/OracleNet-Transformer-Crit-32-Classif/_best_vc_epoch.ckpt $DATADIR/oracle_reward.ckpt
    cp ${PROJECTS_DIR}/oracles/ismrm_oracles_nb_points/OracleNet-Transformer-Crit-32-Dense/_best_vc_epoch.ckpt $DATADIR/oracle_crit.ckpt

    ORACLE_REWARD_CHECKPOINT=$DATADIR/oracle_reward.ckpt
    ORACLE_CRIT_CHECKPOINT=$DATADIR/oracle_crit.ckpt

    echo "Copying agent checkpoint..."
    # cp ${PROJECTS_DIR}/agents/sac_checkpoint/* $DATADIR/sac_checkpoint
    # AGENTCHECKPOINT=$DATADIR/sac_checkpoint/last_model_state.ckpt

    # Setup dataset to augment
    # cp ${PROJECTS_DIR}/datasets/train_test_classical_tracts_antoine_valid.hdf5 $DATADIR/tracts_dataset.hdf5
    # DATASET_TO_AUGMENT=$DATADIR/tracts_dataset.hdf5
fi

for RNGSEED in "${SEEDS[@]}"
do
    DEST_FOLDER="${EXPDIR}/${EXPNAME}/${EXPID}/${RNGSEED}"

    # Append the current seed to the EXPID
    EXPID_W_SEED="${RNGSEED}-${EXPID}"

    additionnal_args=()

    if [ -n "$AGENTCHECKPOINT" ]; then
        additionnal_args+=('--agent_checkpoint' "${AGENTCHECKPOINT}")
    else
        additionnal_args+=('--pretrain_max_ep' "${PRETRAINSTEPS}")
    fi

    # If ORACLE_LR is set AND is higher than zero, add it to the arguments.
    if [[ -n "${ORACLE_LR}" && $(echo "${ORACLE_LR} > 0" | bc -l) -eq 1 ]]; then
        additionnal_args+=('--oracle_lr' "${ORACLE_LR}")
    fi

    if [[ -n "${DATASET_TO_AUGMENT}" ]]; then
        additionnal_args+=('--dataset_to_augment' "${DATASET_TO_AUGMENT}")
    fi

    if [[ -n "${BACKUPDIR}" ]]; then
        additionnal_args+=('--backup_dir' "${BACKUPDIR}")
    fi

    if [ $DISABLE_ORACLE_TRAINING -eq 1 ]; then
        echo "DISABLE_ORACLE_TRAINING is set to 1. Disabling oracle training."
        additionnal_args+=('--disable_oracle_training')
    fi

    # Add use_comet flag is USECOMET is set to 1
    if [ $USECOMET -eq 1 ]; then
        additionnal_args+=('--use_comet')
    fi

    # Start training
    python -O $SOURCEDIR/FineTrack/trainers/rlhf_train_simple.py \
        ${DEST_FOLDER} \
        "${COMETPROJECT}" \
        "${EXPID}" \
        "${DATASETDIR}/ismrm2015.hdf5" \
        --alg ${ALG} \
        --max_ep ${MAXEP} \
        --warmup_agent_steps ${WARMUPSTEPS} \
        --agent_train_steps ${AGENTNBSTEPS} \
        --oracle_train_steps ${ORACLENBSTEPS} \
        --workspace "mrzarfir" \
        --hidden_dims "1024-1024-1024" \
        --n_actor ${N_ACTORS} \
        --min_length 20 \
        --max_length 200 \
        --noise 0.0 \
        --replay_size 1000000 \
        --alignment_weighting 1.0 \
        --binary_stopping_threshold 0.1 \
        --oracle_crit_checkpoint ${ORACLE_CRIT_CHECKPOINT} \
        --oracle_reward_checkpoint ${ORACLE_REWARD_CHECKPOINT} \
        --oracle_validator \
        --oracle_stopping_criterion \
        --oracle_bonus 10.0 \
        --scoring_data "${DATASETDIR}/scoring_data" \
        --tractometer_reference "${DATASETDIR}/scoring_data/t1.nii.gz" \
        --tractometer_validator \
        --rng_seed ${RNGSEED} \
        --npv ${NPV} \
        --rlhf_inter_npv ${RLHFINTERNPV} \
        --batch_size ${BATCHSIZE} \
        --lr ${LR} \
        --gamma ${GAMMA} \
        --theta ${THETA} \
        --n_dirs 100 \
        --flatten_state \
        --neighborhood_type "axes" \
        --neighborhood_radius 1 \
        --log_interval 100 \
        --utd 1 \
        --oracle_batch_size ${ORACLE_MICRO_BATCH_SIZE} \
        --grad_accumulation_steps ${GRAD_ACCUM_STEPS} \
        --max_dataset_size 5000000 \
        --nb_new_streamlines_per_iter 500 \
        "${additionnal_args[@]}"

    # POST-PROCESSING
    bash scripts/tractogram_post_processing.sh ${DEST_FOLDER} ${DATASETDIR}
done

if [ $islocal -eq 1 ]; then
    echo "Done."
    echo "Experiment results are saved in ${EXPDIR}."
else
    # Archive and save everything
    OUTNAME=${EXPID}$(date -d "today" +"%Y%m%d%H%M").tar

    echo "Archiving experiment..."
    tar -cvf ${DATADIR}/${OUTNAME} $EXPDIR
    echo "Copying archive to scratch..."
    cp ${DATADIR}/${OUTNAME} ~/scratch/${OUTNAME}

    echo "Done."
    echo "Experiment results are saved in ~/scratch/${OUTNAME}."
fi
