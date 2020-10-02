# % load_ext
# autoreload
# % autoreload
# 2
# % matplotlib
# inline
# %load_ext line_profiler

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
sess_config = tf.ConfigProto()

import sys
import os

COCO_DATA = 'data/coco'
MASK_RCNN_MODEL_PATH = 'lib/Mask_RCNN/'

if MASK_RCNN_MODEL_PATH not in sys.path:
    sys.path.append(MASK_RCNN_MODEL_PATH)

from samples.coco import coco
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

from lib import utils as siamese_utils
from lib import model as siamese_model
from lib import config as siamese_config

import time
import datetime
import random
import numpy as np
import skimage.io
import imgaug
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# train_classes = coco_nopascal_classes
train_classes = np.array(range(1,81))


# Load COCO/train dataset
coco_train = siamese_utils.IndexedCocoDataset()
coco_train.load_coco(COCO_DATA, subset="train", subsubset="train", year="2017")
coco_train.prepare()
coco_train.build_indices()
coco_train.ACTIVE_CLASSES = train_classes

# Load COCO/val dataset
coco_val = siamese_utils.IndexedCocoDataset()
coco_val.load_coco(COCO_DATA, subset="train", subsubset="val", year="2017")
coco_val.prepare()
coco_val.build_indices()
coco_val.ACTIVE_CLASSES = train_classes


class SmallTrainConfig(siamese_config.Config):
    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 12  # A 16GB GPU is required for a batch_size of 12
    NUM_CLASSES = 1 + 1
    NAME = 'small_coco'
    EXPERIMENT = 'example'
    CHECKPOINT_DIR = 'checkpoints/'
    # Adapt loss weights
    LOSS_WEIGHTS = {'rpn_class_loss': 2.0,
                    'rpn_bbox_loss': 0.1,
                    'mrcnn_class_loss': 2.0,
                    'mrcnn_bbox_loss': 0.5,
                    'mrcnn_mask_loss': 1.0}


class LargeTrainConfig(siamese_config.Config):
    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 4
    IMAGES_PER_GPU = 3  # 4 16GB GPUs are required for a batch_size of 12
    NUM_CLASSES = 1 + 1
    NAME = 'large_coco'
    EXPERIMENT = 'example'
    CHECKPOINT_DIR = 'checkpoints/'
    # Reduced image sizes
    TARGET_MAX_DIM = 192
    TARGET_MIN_DIM = 150
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # Reduce model size
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    FPN_FEATUREMAPS = 256
    # Reduce number of rois at all stages
    RPN_ANCHOR_STRIDE = 1
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000
    TRAIN_ROIS_PER_IMAGE = 200
    DETECTION_MAX_INSTANCES = 100
    MAX_GT_INSTANCES = 100
    # Adapt NMS Threshold
    DETECTION_NMS_THRESHOLD = 0.5
    # Adapt loss weights
    LOSS_WEIGHTS = {'rpn_class_loss': 2.0,
                    'rpn_bbox_loss': 0.1,
                    'mrcnn_class_loss': 2.0,
                    'mrcnn_bbox_loss': 0.5,
                    'mrcnn_mask_loss': 1.0}

# The small model trains on a single GPU and runs much faster.
# The large model is the same we used in our experiments but needs multiple GPUs and more time for training.
model_size = 'small' # or 'large'

if model_size == 'small':
    config = SmallTrainConfig()
elif model_size == 'large':
    config = LargeTrainConfig()

config.display()

# Create model object in inference mode.
model = siamese_model.SiameseMaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)
train_schedule = OrderedDict()
train_schedule[1] = {"learning_rate": config.LEARNING_RATE, "layers": "heads"}
train_schedule[120] = {"learning_rate": config.LEARNING_RATE, "layers": "all"}
train_schedule[160] = {"learning_rate": config.LEARNING_RATE/10, "layers": "all"}


# Load weights trained on Imagenet
try:
    model.load_latest_checkpoint(training_schedule=train_schedule)
except:
    model.load_imagenet_weights(pretraining='imagenet-687')

for epochs, parameters in train_schedule.items():
    print("")
    print("training layers {} until epoch {} with learning_rate {}".format(parameters["layers"],
                                                                          epochs,
                                                                          parameters["learning_rate"]))
    model.train(coco_train, coco_val,
                learning_rate=parameters["learning_rate"],
                epochs=epochs,
                layers=parameters["layers"])