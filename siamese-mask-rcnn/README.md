# Siamese Mask R-CNN

This is a fork of the official implementation of Siamese Mask R-CNN from [One-Shot Instance Segmentation](https://arxiv.org/abs/1811.11507). It is based on the [Mask R-CNN](https://arxiv.org/abs/1703.06870) implementation by [Matterport](https://github.com/matterport/Mask_RCNN).

This fork contains extra code for evaluating Siamese Mask R-CNN within the scope of Package Buddy, an autonomous intelligent robotic delivery system.  Package Buddy is currently under development by a project team at New York University.  Siamese Mask R-CNN is implemented on Package Buddy to aid in object detection and avoidance.  This work contributes to [Matthew Avallone](https://github.com/mattavallone)'s MS Thesis, "Package Buddy: Few-Shot Object Detection for a Mobile Robotic System".  For further information about the project, please visit [here](https://github.com/mattavallone/packagebuddy).  The query set for test images used in Package Buddy evaluation can be found [here](https://drive.google.com/open?id=1bcwJxa3VePWThfVx3oMp108eI4EDoRKt) through Google Drive.  To use pycocotools with this new dataset, rename the dataset and [annotation file](https://github.com/mattavallone/siamese-mask-rcnn/blob/master/data/packagebuddy_queryset.json) following MS COCO conventions for either a training or validation set (e.g. "val2014").  Evaluation was done through Google Colab with a GPU machine.

<p align="center">
 <img src="figures/teaser_web.jpg" width=80%>
</p>

The repository includes:
- [x] Source code of Siamese Mask R-CNN
- [x] Training code for MS COCO
- [x] Evaluation on MS COCO metrics (AP)
- [x] Training and evaluation of one-shot splits of MS COCO
- [x] Training code to reproduce the results from the paper
- [x] Pre-trained weights for ImageNet
- [x] Pre-trained weights for all models from the paper
- [x] Code to evaluate all models from the paper
- [x] Code to generate result figures
- [x] NEW: Code for Package Buddy evaluation
## One-Shot Instance Segmentation

One-shot instance segmentation can be summed up as: Given a query image and a reference image showing an object of a novel category, we seek to detect and segment all instances of the corresponding category (in the image above ‘person’ on the left, ‘car’ on the right). Note that no ground truth annotations of reference categories are used during training.
This type of visual search task creates new challenges for computer vision algorithms, as methods from metric and few-shot learning have to be incorporated into the notoriously hard tasks ofobject identification and segmentation. 
Siamese Mask R-CNN extends Mask R-CNN - a state-of-the-art object detection and segmentation system - with a Siamese backbone and a matching procedure to perform this type of visual search.

## Installation

1. Clone this repository
2. Prepare COCO dataset as described below
3. Run the [install_requirements.ipynb](install_requirements.ipynb) notebook to install all relevant dependencies.

### Requirements

Linux, Python 3.4+, Tensorflow, Keras 2.1.6, cython, scikit_image 0.13.1, h5py, imgaug and opencv_python

### Prepare COCO dataset

The model requires [MS COCO](http://cocodataset.org/#home) and the [CocoAPI](https://github.com/waleedka/coco) to be added to `/data`.
```
cd data
git clone https://github.com/cocodataset/cocoapi.git
```
It is recommended to symlink the dataset root of MS COCO. 
```
ln -s $PATH_TO_COCO$/coco coco
```
If unsure follow the instructions of the [Matterport Mask R-CNN implementation](https://github.com/matterport/Mask_RCNN#ms-coco-requirements).

### Get pretrained weights

Get the pretrained weights from the [releases menu](https://github.com/bethgelab/siamese-mask-rcnn/releases) and save them to `/checkpoints`.

## Training

To train siamese mask r-cnn on MS COCO simply follow the instructions in the [train.ipynb](train.ipynb) notebook. 
There are two model configs available, a small one which runs on a single GPU with 12GB memory and a large one which needs 4 GPUs with 12GB memory each.
The second model config is the same as used in our experiments.

To reproduce our results and train the models reported in the paper run the notebooks provided in [experiments](experiments). 
Those models need 4 GPUs with 12GB memory each.

Our models are trained on the coco 2017 training set, of which we remove the last 3000 images for validation.

## Evaluation

To evaluate and visualize a models results run the [evaluate.ipynb](evaluate.ipynb) notebook. Make sure to use the same config as used for training the model.

To evaluate the models reported in the paper run the evaluation notebook provided in [experiments](experiments). 
Each model will be evaluated 5 times to compensate for the stochastic effects introduced by randomly choosing the reference instances. 
The final result is the mean of those five runs.

We use the coco 2017 val set for testing and the last 3000 images from the training set for validation.

## Model description

Siamese Mask R-CNN is designed as a minimal variation of Mask R-CNN which can perform the visual search task described above. For more details please read the [paper](https://arxiv.org/abs/1811.11507).


<p align="center">
 <img src="figures/siamese-mask-rcnn-sketch.png" width=50%>
</p>

## Citation

If you use this repository or want to reference our work please cite our paper:
```
@article{michaelis_one-shot_2018,
    title = {One-Shot Instance Segmentation},
    author = {Michaelis, Claudio and Ustyuzhaninov, Ivan and Bethge, Matthias and Ecker, Alexander S.},
    year = {2018},
    journal = {arXiv},
    url = {http://arxiv.org/abs/1811.11507}
}
```
