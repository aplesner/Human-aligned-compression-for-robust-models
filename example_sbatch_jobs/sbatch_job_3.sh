#!/bin/bash

#SBATCH --job-name=adv_exp3
#SBATCH -n 1
#SBATCH --time=1-00:00:00                         #days-hours:minutes:seconds
#SBATCH --mem-per-cpu=6000
#SBATCH --tmp=4000                           #per node!!
#SBATCH --output=sbatch_log/%j_adv_experiment3.out
#SBATCH --error=sbatch_log/%j_adv_experiment3.err
#SBATCH --gpus=1
#SBATCH --exclude=tikgpu[04-10]
#module spider load cuda/11.8.0

start_time=$(date +%s)

echo "GPU & CUDA info"
nvcc --version
nvidia-smi
echo "=============================================================="
source /itet-stor/sraeber/net_scratch/conda/etc/profile.d/conda.sh
conda activate semester_project

cd /itet-stor/sraeber/net_scratch/semester_project

for p in {'0004','0008','0016','0032','0150','0450'}
do
    for m in {Vit,ResNet50}
    do
        python main_script.py --epsilons 2/255 4/255 8/255 12/255 --attack=iFGSM  --output='results/elic_abl/'$m'_elic_'$p'_' --save_config --model_attack=$m --defense=ELIC --defense_param=$p --get_baseline
        python main_script.py --epsilons 2/255 4/255 8/255 12/255 --attack=iFGSM  --output='results/elic_abl/'$m'_elic_'$p'_' --save_config --model_attack=$m --defense=ELIC --defense_param=$p --get_baseline --attack_through
    done
done



end_time=$(date +%s)
execution_time=$((end_time - start_time))

echo "Execution time: $execution_time seconds"
echo "finished"