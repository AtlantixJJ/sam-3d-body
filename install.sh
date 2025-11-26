# better copy this installation script line to line rather than running it directly
# because conda activate might fail

conda create -n sam_3d_body python=3.11 -y
conda activate sam_3d_body

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black pycocotools tensorboard huggingface_hub

# Install detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps

# Install MoGe
pip install git+https://github.com/microsoft/MoGe.git