Source code for the paper [https://arxiv.org/abs/2504.12255](Human Aligned Compression for Robust Models) presented at the Workshop AdvML at CVPR 2025 



installation:
conda create --name compression_project python=3.11

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

HiFiC:
git clone https://github.com/Justin-Tan/high-fidelity-generative-compression.git

move compress.py, default_config.py, train.py and src from high-fidelity-generative-compression to parent directory.

add Experiment.py, main_script.py, net_training.py and utility.py to parent directory

pip install kornia

changed from skimage.measure import compare_ssim to from skimage.metrics import structural_similarity
