############################# model buliding ###################################
FILTER_BASE_LINE = 64
############################## train ##########################
BETAS = (0.5,0.999)
LAMBDA_GP = 10
EPOCHS = 100_00
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
SAMPLE_ON_STEP = 32
SAVE_ON_STEP = 50
VERBOSE_ON_STEP = 4
CRITIC_ITERATIONS = 5
GENERATOR_ITERATIONS = 1
LATENT_SINGLE_SHAPE = [100,1,1]
IMAGE_CROP_SIZE = 64
############################## dir ###########################
import os
_ROOT_PATH = os.path.split(os.path.realpath(__file__))[0]
_SAVE_PATH = _ROOT_PATH + "\\checkpoint\\"
_LOG_PATH = _ROOT_PATH + '\\logs\\'
MODEL_CHECKPOINT_PATH = _SAVE_PATH + f'model.pt' # _STAGES_FILTERBASELINE_
OPTIMIZER_CHECKPOINT_PATH = _SAVE_PATH + f'optimizer.pt'
# DATASET_PATH = 'X:\Machine_Learning\dataset\AnimeFaces\\fullset'
DATASET_PATH = 'X:\\Machine_Learning\\dataset\\AnimeFaces\\subset'