# Investigating the Mitigation of Shortcut Learning in Image Classification through Object-Centric Representations

This repository includes the relevant code for my bachelor thesis on "Investigating the Mitigation of Shortcut Learning in Image Classification through Object-Centric Representations"

The code itself was run on a cluster of the Center for Scientific Computing (CSC).


## Setup
Python 3.14.7 was used throughout this project.

The following packages need to be installed with `pip install` (https://pypi.org/project/pip/) before running the code:
- numpy
- pandas
- matplotlib
- scikit-learn
- pillow
- opencv-python
- grad-cam (https://github.com/jacobgil/pytorch-grad-cam)


Additionally PyTorch (https://pytorch.org/get-started/locally/) needs to be installed:

`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132`

SAM2 (Segment Anything 2) was used for segmenting the images. The installation guide and overview of the model can be found here:
https://github.com/facebookresearch/sam2#sam-2-segment-anything-in-images-and-videos

**Note**: 
- Before running the code all paths need to be adjusted as they mostly reference absolute paths from the cluster.
- The data (ImageNet, MAPS) is also not included.
- After creating the evaluation benchmark dataset with `evaluation_dataset/dataset_creation.py`, the script `evaluation_dataset/fix_bg_only_t.py` must be run afterwards, as the created images for the category "Only-BG-T" are not created correctly.
- For training models with specific MAPS ratios (script trains only one model at a time):
    - use the corresponding training script (ResNet-50, ConvNeXt-Tiny)
    - adjust `RATIO_MAPS` in the main()-function accordingly