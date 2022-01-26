import torch
import os

# Path to dataset
TRAIN_IMG_PATH = '../data/training/images/'
TRAIN_LABEL_PATH = '../data/training/groundtruth/'
TEST_IMG_PATH = '../data/test_set_images/'

# Base path for trained models and output predictions evaluated
OUTPUT_PATH = "output"

# Dataset size parameter
DATA_SIZE = 100
TRAINING_SIZE = 95
VALIDATION_SIZE = DATA_SIZE - TRAINING_SIZE
TESTING_SIZE = 50


# Parameter to generate dataset
PATCH_SIZE = 80
OVERLAP = True
OVERLAP_AMOUNT = 40

# UNet parameters
NUM_CLASSES = 1
ENC_CHANNELS = (3, 16, 32, 64, 128, 256)
DEC_CHANNELS = (256, 128, 64, 32, 16)
RESPATHS_LENGTHS = (4, 3, 2, 1)
PADDING_MODE = 'replicate'

# Training parameters
LR = 0.001
NUM_EPOCHS = 50
BATCH_SIZE = 64

# Seed for reproductibility
SEED = 42


