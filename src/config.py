import numpy as np

TRAIN_IMG_PATH = '../data/training/images/'
TRAIN_LABEL_PATH = '../data/training/groundtruth/'
TEST_IMG_PATH = '../data/test/images/'
TEST_LABEL_DATA = '../data/test/predictions'

BATCH_SIZE = 64
NUM_EPOCHS = 15

PATCH_SIZE = 120
OVERLAP = True
OVERLAP_AMOUNT = 60

TESTING_SIZE = 50
DATA_SIZE = 100
TRAINING_SIZE = 80

SEED = 428796
np.random.seed(SEED)

