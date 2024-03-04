#!/bin/bash

#SBATCH --job-name=llm_a100    # create a short name for your job
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=rian.touchent@inria.fr   # Where to send mail
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --partition=gpu          # Name of the partition
#SBATCH --gres=gpu:a100:2     # GPU nodes are only available in gpu partition
#SBATCH --mem=64G                # Total memory allocated
#SBATCH --time=20:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=experiment_logs/llm_1%j.out   # output file name

echo "### Launching Experiment - LLM ###"

source /home/$USER/.bashrc
cd $HOME/camembert-bio

conda activate py310


# Run the experiment
python run_llm_qa_benchmark.py --dtype bfloat16 --tensor_parallel_size 2 --additional_models meta-llama/Llama-2-70b-hf epfl-llm/meditron-70b mistralai/Mixtral-8x7B-Instruct-v0.1 mistralai/Mixtral-8x7B-v0.1 m42-health/med42-70b BiMediX/BiMediX-Eng