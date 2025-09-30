#!/bin/bash

#SBATCH --job-name=adv_exp
#SBATCH -n 1
#SBATCH --time=2-00:00:00                         #days-hours:minutes:seconds
#SBATCH --mem-per-cpu=6000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --tmp=4000                           #per node!!
#SBATCH --output=sbatch_log/%j_adv_experiment.out
#SBATCH --error=sbatch_log/%j_adv_experiment.err
#SBATCH --gpus=1
#SBATCH --exclude=tikgpu04,tikgpu[08-10]

#module spider load cuda/11.8.0

start_time=$(date +%s)

echo "GPU & CUDA info"
nvcc --version
nvidia-smi
echo "=============================================================="
source /itet-stor/sraeber/net_scratch/conda/etc/profile.d/conda.sh
conda activate semester_project

cd /itet-stor/sraeber/net_scratch/semester_project

for d in {jpeg,ELIC,HiFiC}
do
    for m in {Vit,ResNet50}
    do
        python main_script.py --epsilons 2/255 4/255 6/255 8/255 10/255 12/255 --attack=$1  --output='results/imagenet_1000/'$m'_'$d'_' --save_config --model_attack=$m --defense=$d --get_baseline --dataset=imagenet_1000
        python main_script.py --epsilons 2/255 4/255 6/255 8/255 10/255 12/255 --attack=$1  --output='results/imagenet_1000/'$m'_'$d'_T_' --save_config --model_attack=$m --defense=$d --get_baseline --attack_through --dataset=imagenet_1000
    done
done
for m in {Vit,ResNet50}
do
    python main_script.py --epsilons 2/255 4/255 6/255 8/255 10/255 12/255 --attack=$1  --output='results/imagenet_1000/'$m'_' --save_config --model_attack=$m --get_baseline --dataset=imagenet_1000
done



end_time=$(date +%s)
execution_time=$((end_time - start_time))

echo "Execution time: $execution_time seconds"
echo "finished"