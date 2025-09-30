#!/bin/bash

#SBATCH --job-name=adv_exp2
#SBATCH -n 1
#SBATCH --time=2-00:00:00                         #days-hours:minutes:seconds
#SBATCH --mem-per-cpu=6000
#SBATCH --tmp=4000                           #per node!!
#SBATCH --output=sbatch_log/%j_adv_experiment2.out
#SBATCH --error=sbatch_log/%j_adv_experiment2.err
#SBATCH --gpus=1
#SBATCH --exclude=tikgpu[04-05],tikgpu[08-10]

#module spider load cuda/11.8.0

start_time=$(date +%s)

echo "GPU & CUDA info"
nvcc --version
nvidia-smi
echo "=============================================================="
source /itet-stor/sraeber/net_scratch/conda/etc/profile.d/conda.sh
conda activate semester_project

cd /itet-stor/sraeber/net_scratch/semester_project
for n in {1..5}
do
    python main_script.py --epsilons 4/255 8/255 12/255 --attack=iFGSM  --output='results/HiFiC_seq/ResNet_hific_low_'$n'_' --save_config --model_attack=ResNet50 --defense=HiFiC_seq --defense_param=$n:'hific_low.pt' --attack_through --get_baseline --dataset=imagenet_1000
done



end_time=$(date +%s)
execution_time=$((end_time - start_time))

echo "Execution time: $execution_time seconds"
echo "finished"