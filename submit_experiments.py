import yaml
import os
import subprocess
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description="Submit experiments to SLURM.")
parser.add_argument("config", type=str, default="config.yaml", help="Path to the YAML configuration file.")
parser.add_argument("--dry-run", action="store_true", help="Print the SLURM script without submitting it.")
parser.add_argument("--local", action="store_true", help="Run the experiments locally.")
args = parser.parse_args()

# Load the YAML config file
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

# Extract global settings
data_config = config["data"]
global_config = config["global"]
experiments = config["experiments"]

# Extract paths
SOURCEDIR = os.path.expanduser(data_config["SOURCEDIR"])
PROJECTS_DIR = os.path.expanduser(data_config["PROJECTS_DIR"])
EXPDIR = os.path.expanduser(data_config["EXPDIR"])

def get_dataset_type(dataset_path):
    """Determine dataset type based on file extension."""
    if dataset_path.endswith(".tar.gz") or dataset_path.endswith(".tar"):
        return "tar"
    elif dataset_path.endswith(".hdf5"):
        return "hdf5"
    else:
        raise ValueError(f"Unknown dataset format: {dataset_path}")
    
def get_prepare_dataset_command(dataset_name, data_config):
    dataset_path = os.path.join(PROJECTS_DIR, data_config[dataset_name].get("location", None))
    dataset_type = get_dataset_type(dataset_path)
    if dataset_type == "tar":
        return f"tar xf {dataset_path} -C $SLURM_TMPDIR/data", "$SLURM_TMPDIR/data/ismrm2015_2mm/"
    elif dataset_type == "hdf5":
        return f"cp {dataset_path} $SLURM_TMPDIR/data/{dataset_name}.hdf5", "$SLURM_TMPDIR/data/"
    else:
        raise ValueError(f"Unknown dataset format: {dataset_path}")

######################################################
# Make sure everything is in order with each experiment
# before generating the SLURM jobs.
######################################################
already_created_exp_ids = []
for i, exp in enumerate(experiments):
    exp_config = {**global_config}
    exp_config.update(exp)

    exp_name = exp["exp_name"]
    exp_id = f"{exp_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    if exp_id in already_created_exp_ids:
        raise ValueError(f"Experiment ID {exp_id} already exists.")
    
    already_created_exp_ids.append(exp_id)

    # Also, make sure that the dataset exist for every experiment
    dataset_name = exp_config["dataset"]
    dataset_path = os.path.join(PROJECTS_DIR, data_config[dataset_name].get("location", None))
    if not dataset_path:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration.")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found at location '{dataset_path}'.")
    
    # Make sure the oracle checkpoints exist
    if not os.path.exists(os.path.join(PROJECTS_DIR, exp_config["reward_ckpt"])):
        raise FileNotFoundError(f"Oracle reward checkpoint '{os.path.join(PROJECTS_DIR, exp_config['reward_ckpt'])}' not found.")
    if not os.path.exists(os.path.join(PROJECTS_DIR, exp_config["crit_ckpt"])):
        raise FileNotFoundError(f"Oracle critic checkpoint '{os.path.join(PROJECTS_DIR, exp_config['crit_ckpt'])}' not found.")
    
    # Make sure the launch script exists
    launch_script_path = os.path.join(SOURCEDIR, exp_config["launch_script"])
    if not os.path.exists(launch_script_path):
        raise FileNotFoundError(f"Launch script '{launch_script_path}' not found.")
    
    # If there's a FODF encoder checkpoint, make sure it exists
    if exp_config.get("fodf_encoder_ckpt", None) is not None:
        if not os.path.exists(os.path.join(PROJECTS_DIR, exp_config["fodf_encoder_ckpt"])):
            raise FileNotFoundError(f"FODF encoder checkpoint '{os.path.join(PROJECTS_DIR, exp_config['fodf_encoder_ckpt'])}' not found.")
        
    # Check the mutually exclusive flags
    state_state_specified = False
    if exp_config.get("flatten_state", False):
        state_state_specified = True
    if exp_config.get("fodf_encoder_ckpt", None) is not None:
        assert not state_state_specified, f"Can only specify one of flatten_state, fodf_encoder_ckpt or conv_state for experiment {i}"
        state_state_specified = True
    if exp_config.get("conv_state", False):
        assert not state_state_specified, f"Can only specify one of flatten_state, fodf_encoder_ckpt or conv_state for experiment {i}"
        state_state_specified = True
    if not state_state_specified:
        assert False, f"Must specify one of flatten_state, fodf_encoder_ckpt or conv_state for experiment {i}"
    
######################################################
# Generate SLURM Jobs for each experiment.
######################################################
all_jobs = []
for i, exp in enumerate(experiments):
    # Merge the global and experiment-specific configurations
    # NB: The experiment-specific configuration takes precedence
    exp_config = {**global_config}
    exp_config.update(exp)

    exp_name = exp_config["exp_name"]
    dataset_name = exp_config["dataset"]
    dataset_path = data_config[dataset_name].get("location", None)

    # Generate a experiment ID
    exp_id = f"{exp_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Prepare extra flags
    extra_flags = ""
    if exp_config.get("use_comet", False):
        extra_flags += "--use_comet "

    # Those are mutually exclusive flags
    state_state_specified = False
    if exp_config.get("flatten_state", False):
        extra_flags += "--flatten_state "
        state_state_specified = True
    if exp_config.get("fodf_encoder_ckpt", None) is not None:
        assert not state_state_specified, f"Can only specify one of flatten_state, fodf_encoder_ckpt or conv_state for experiment {i}"
        extra_flags += f"--fodf_encoder_ckpt {os.path.join(PROJECTS_DIR, exp_config['fodf_encoder_ckpt'])} "
        state_state_specified = True
    if exp_config.get("conv_state", False):
        assert not state_state_specified, f"Can only specify one of flatten_state, fodf_encoder_ckpt or conv_state for experiment {i}"
        extra_flags += "--conv_state "
        state_state_specified = True
    if not state_state_specified:
        assert False, f"Must specify one of flatten_state, fodf_encoder_ckpt or conv_state for experiment {i}"

    # Paths to the dataset
    prepare_ds_cmd, slurm_ds_dir = get_prepare_dataset_command(dataset_name, data_config)

    # If the dataset has a field "tractometer_reference", add it to the extra flags
    if data_config[dataset_name].get("tractometer_reference", None) is not None:
        extra_flags += f"--tractometer_reference {slurm_ds_dir}/{data_config[dataset_name]['tractometer_reference']} "
        extra_flags += "--tractometer_validator "
    if data_config[dataset_name].get("scoring_data", None) is not None:
        extra_flags += f"--scoring_data {slurm_ds_dir}/{data_config[dataset_name]['scoring_data']} "


    # SLURM script content
    slurm_script = f"""#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem={exp_config["mem"]}
#SBATCH --time={exp_config["time"]}
#SBATCH --job-name={exp_name}
#SBATCH --mail-user=jeremi.levesque@usherbrooke.ca
#SBATCH --mail-type=ALL

# THIS SCRIPT IS AUTOMATICALLY GENERATED BY submit_experiments.py
# DO NOT EDIT MANUALLY. INSTEAD, MODIFY THE CONFIGURATION FILE.

set -e

echo "Loading modules and virtual env..."
module load python/3.10 cuda cudnn httpproxy
source ~/FineTrack/venv/bin/activate

# Prepare directories
mkdir -p $SLURM_TMPDIR/data
mkdir -p $SLURM_TMPDIR/experiments

# Extract dataset
echo "Preparing {dataset_name} dataset..."
{prepare_ds_cmd}

# Define paths
ORACLE_REWARD_CHECKPOINT=$SLURM_TMPDIR/data/oracle_reward.ckpt
ORACLE_CRIT_CHECKPOINT=$SLURM_TMPDIR/data/oracle_crit.ckpt

# Prepare oracle checkpoints
echo "Preparing oracle checkpoints..."
cp {os.path.join(PROJECTS_DIR, exp_config["reward_ckpt"])} $ORACLE_REWARD_CHECKPOINT
cp {os.path.join(PROJECTS_DIR, exp_config["crit_ckpt"])} $ORACLE_CRIT_CHECKPOINT

DEST_FOLDER="{EXPDIR}/{exp_name}/{exp_id}/{exp_config['seed']}"

# Run training script
echo "Running experiment..."
python -O {SOURCEDIR}/{exp_config["launch_script"]} \\
    $DEST_FOLDER \\
    "{exp_config['project_name']}" \\
    "{exp_id}" \\
    "{slurm_ds_dir}/{dataset_name}.hdf5" \\
    --max_ep {exp_config["max_ep"]} \\
    --hidden_dims "{exp_config['hidden_dims']}" \\
    --oracle_reward_checkpoint $ORACLE_REWARD_CHECKPOINT \\
    --oracle_crit_checkpoint $ORACLE_CRIT_CHECKPOINT \\
    --oracle_validator \\
    --oracle_stopping_criterion \\
    --oracle_bonus 10.0 \\
    --workspace {exp_config["workspace"]} \\
    --rng_seed {exp_config["seed"]} \\
    --n_actor {exp_config["n_actors"]} \\
    --npv {exp_config["npv"]} \\
    --min_length {exp_config["min_length"]} \\
    --max_length {exp_config["max_length"]} \\
    --noise {exp_config["noise"]} \\
    --batch_size {exp_config["batch_size"]} \\
    --replay_size {exp_config["replay_size"]} \\
    --lr {exp_config["lr"]} \\
    --gamma {exp_config["gamma"]} \\
    --theta {exp_config["theta"]} \\
    --binary_stopping_threshold {exp_config["binary_stopping_threshold"]} \\
    --n_dirs {exp_config["n_dirs"]} \\
    --neighborhood_type "{exp_config['neighborhood_type']}" \\
    --neighborhood_radius {exp_config["neighborhood_radius"]} \\
    --log_interval {exp_config["log_interval"]} \\
    --utd {exp_config["utd"]} {extra_flags}
    
# Archive and save everything
OUTNAME={exp_id}$(date -d "today" +"%Y%m%d%H%M").tar

echo "Archiving experiment..."
tar -cvf $SLURM_TMPDIR/data/$OUTNAME {EXPDIR}
echo "Copying archive to scratch..."
cp $SLURM_TMPDIR/data/$OUTNAME ~/scratch/$OUTNAME
"""

    slurm_script_path = f"{SOURCEDIR}/slurm_scripts/{exp_id}.sh"
    os.makedirs(os.path.dirname(slurm_script_path), exist_ok=True)

    # Save and submit the SLURM script
    with open(slurm_script_path, "w") as f:
        f.write(slurm_script)

    all_jobs.append((exp_name, exp_id, slurm_script_path))
    print(f"Generated SLURM script for experiment {exp_name}: {slurm_script_path}")

######################################################
# Submit the jobs to SLURM.
######################################################
if not args.dry_run:
    for exp_name, exp_id, slurm_script_path in all_jobs:
        if args.local:
            print(f"Running experiment {exp_name} locally with job script: {slurm_script_path}")
            subprocess.run(["qc", "bash", slurm_script_path], check=True)
        else:
            print(f"Submitted experiment {exp_name} with job script: {slurm_script_path}")
            subprocess.run(["sbatch", slurm_script_path], check=True)
