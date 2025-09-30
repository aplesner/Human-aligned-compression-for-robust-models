Source code for the paper Human Aligned Compression for Robust Models (https://arxiv.org/abs/2504.12255) presented at the Workshop AdvML at CVPR 2025.
```
@InProceedings{raeber2025human,
	author=	{Samuel Räber and Andreas Plesner and Till Aczel and Roger Wattenhofer},
	title=	{{Human Aligned Compression for Robust Models}},
	booktitle=	{{The 5th Workshop of Adversarial Machine Learning on Computer Vision: Foundation Models + X  (AdvML@CVPR 2025), Nashville, Tennessee, USA}},
	month=	{June},
	year=	{2025},
}
```



installation:
conda create --name compression_project python=3.11

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

HiFiC:
git clone https://github.com/Justin-Tan/high-fidelity-generative-compression.git

move compress.py, default_config.py, train.py and src from high-fidelity-generative-compression to parent directory.

add Experiment.py, main_script.py, net_training.py and utility.py to parent directory

pip install kornia

changed from skimage.measure import compare_ssim to from skimage.metrics import structural_similarity
