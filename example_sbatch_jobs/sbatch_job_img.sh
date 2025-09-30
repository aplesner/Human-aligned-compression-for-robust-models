#!/bin/bash

#SBATCH --job-name=adv_img
#SBATCH -n 1
#SBATCH --time=1-00:00:00                         #days-hours:minutes:seconds
#SBATCH --mem-per-cpu=6000
#SBATCH --tmp=4000                           #per node!!
#SBATCH --output=sbatch_log/%j_adv_img.out
#SBATCH --error=sbatch_log/%j_adv_img.err
#SBATCH --gpus=1

#module spider load cuda/11.8.0

start_time=$(date +%s)

echo "GPU & CUDA info"
nvcc --version
nvidia-smi
echo "=============================================================="
source /itet-stor/sraeber/net_scratch/conda/etc/profile.d/conda.sh
conda activate semester_project

cd /itet-stor/sraeber/net_scratch/semester_project
python save_image.py --use_config --config=configs/imagenette_FGSM_2025-01-13-15-42-11.json

end_time=$(date +%s)
execution_time=$((end_time - start_time))

echo "Execution time: $execution_time seconds"
echo "finished"