


import sys
import os

#developpath
#root='..'
#sys.path.append(root)
# Root directory of the project
cwd=os.getcwd()
#TX2 path
ROOT_DIR = cwd+'/src/package_buddy/'
sys.path.append(ROOT_DIR)
ROOT_DIR = ROOT_DIR+'package_buddy/'
sys.path.append(ROOT_DIR)
print(ROOT_DIR)
#developpath
#ROOT_DIR = cwd+'/'
import package_buddy.hello
import lib.Mask_RCNN.mrcnn.utils as utils
import lib.Mask_RCNN.mrcnn.model as modellib
import lib.Mask_RCNN.mrcnn.visualize as visualize

import lib.utils as siamese_utils
import lib.model as siamese_model
import lib.config as siamese_config

import time
import datetime
import random
import numpy as np
import skimage.io
#import imgaug
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import OrderedDict

import cv2
from skimage import io
from skimage import img_as_float
import time



def OD_camera(args=None):
	print("hello world")
	# Directory to save logs and trained model
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")


	class SmallEvalConfig(siamese_config.Config):
		# Set batch size to 1 since we'll be running inference on
		# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1
		NUM_CLASSES = 1 + 1
		NAME = 'coco'
		EXPERIMENT = 'evaluation'
		CHECKPOINT_DIR = ROOT_DIR + 'checkpoints/'
		NUM_TARGETS = 1

	class LargeEvalConfig(siamese_config.Config):
		# Set batch size to 1 since we'll be running inference on
		# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1
		NUM_CLASSES = 1 + 1
		NAME = 'coco'
		EXPERIMENT = 'evaluation'
		CHECKPOINT_DIR = ROOT_DIR + 'checkpoints/'
		NUM_TARGETS = 1

		# Large image sizes
		TARGET_MAX_DIM = 192
		TARGET_MIN_DIM = 150
		IMAGE_MIN_DIM = 800
		IMAGE_MAX_DIM = 1024
		# Large model size
		FPN_CLASSIF_FC_LAYERS_SIZE = 1024
		FPN_FEATUREMAPS = 256
		# Large number of rois at all stages
		RPN_ANCHOR_STRIDE = 1
		RPN_TRAIN_ANCHORS_PER_IMAGE = 256
		POST_NMS_ROIS_TRAINING = 2000
		POST_NMS_ROIS_INFERENCE = 1000
		TRAIN_ROIS_PER_IMAGE = 200
		DETECTION_MAX_INSTANCES = 100
		MAX_GT_INSTANCES = 100

	# The small model trains on a single GPU and runs much faster.
	# The large model is the same we used in our experiments but needs multiple GPUs and more time for training.
	model_size = 'large'

	if model_size == 'small':
		config = SmallEvalConfig()
	elif model_size == 'large':
		config = LargeEvalConfig()

	config.display()

	# Provide training schedule of the model
	# When evaluationg intermediate steps the tranining schedule must be provided
	train_schedule = OrderedDict()
	if model_size == 'small':
		train_schedule[1] = {"learning_rate": config.LEARNING_RATE, "layers": "heads"}
		train_schedule[120] = {"learning_rate": config.LEARNING_RATE, "layers": "4+"}
		train_schedule[160] = {"learning_rate": config.LEARNING_RATE / 10, "layers": "all"}
	elif model_size == 'large':
		train_schedule[1] = {"learning_rate": config.LEARNING_RATE, "layers": "heads"}
		train_schedule[240] = {"learning_rate": config.LEARNING_RATE, "layers": "all"}
		train_schedule[320] = {"learning_rate": config.LEARNING_RATE / 10, "layers": "all"}
	# Select checkpoint
	if model_size == 'small':
		checkpoint = ROOT_DIR + 'checkpoints/small_siamese_mrcnn_0160.h5'
	elif model_size == 'large':
		checkpoint = ROOT_DIR + 'checkpoints/large_siamese_mrcnn_coco_full_0320.h5'

	config.NUM_TARGETS = 1



	print("test begin")


	# Create model object in inference mode.
	model = siamese_model.SiameseMaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
	model.load_checkpoint(checkpoint, training_schedule=train_schedule)

	category = 1
	ref_images = []
	for i in range(1, config.NUM_TARGETS + 1):
		ref_image = io.imread(ROOT_DIR + "testimage/train/train%d.jpg" % i)
		ref_images.append(ref_image)



	cap = cv2.VideoCapture(1)
	cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	# check camera availabe
	print("Camera available ？ {}".format(cap.isOpened()))

	# plot size
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


	import matplotlib.pyplot as plt
	from matplotlib.gridspec import GridSpec

	# Use GridSpec to show target smaller than image
	img_count = 1
	while (True):
		start = time.time()
		ret, frame = cap.read()
		## opencv image to skimage BGR->RGB
		# image = img_as_float(frame)
		# cv2.imshow('image_win', frame)
		image = frame[:, :, ::-1]
		query_image=image
		results = model.detect_category(category=category, targets=[ref_images], images=[query_image], verbose=1)
		# Run detection
		r = results[0]
		if max(r['scores'])>0.9:
			yield("Door detected!")
		else:
			yield ("Door NOT detected!")
		key = cv2.waitKey(1)

	cap.release()




if __name__ == '__main__':
	OD_camera()
