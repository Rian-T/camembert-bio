#!/bin/bash

#SBATCH --job-name=camembert_bio_experiment    # create a short name for your job
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=rian.touchent@inria.fr   # Where to send mail
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --partition=gpu          # Name of the partition
#SBATCH --gres=gpu:1     # GPU nodes are only available in gpu partition
#SBATCH --mem=10G                # Total memory allocated
#SBATCH --time=20:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=experiment_logs/exp_1%j.out   # output file name

echo "### Launching Experiment - CamemBERT-bio ###"

source /home/$USER/.bashrc
cd $HOME/camembert-bio

conda activate py310

# Define the combinations of configurations
datasets=("quaero_medline_bigbio_kb" "quaero_emea_bigbio_kb")
models=("per_depth" "per_class")
pretrained_models=("camembert-base" "almanach/camembert-bio-base" "Dr-BERT/DrBERT-7GB")

# Calculate the index for each configuration
dataset_index=$(( (SLURM_ARRAY_TASK_ID - 1) / 6))
model_index=$(( (SLURM_ARRAY_TASK_ID - 1) % 2))
pretrained_model_index=$(( (SLURM_ARRAY_TASK_ID - 1) % 3))

# Run the experiment
python camembert_bio/main.py model=${models[$model_index]} dataset=${datasets[$dataset_index]} model.pretrained_model_name=${pretrained_models[$pretrained_model_index]} --multirun