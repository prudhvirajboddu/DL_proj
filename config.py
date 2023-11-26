import torch

BATCH_SIZE = 8 # increase / decrease according to GPU memeory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_EPOCHS = 40 # number of epochs to train for

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training images and XML files directory
TRAIN_DIR = 'dataset/train'
# validation images and XML files directory
VALID_DIR = 'dataset/valid'

TEST_DIR = 'dataset/test'

# classes: 0 index is reserved for background
CLASSES = [
    'student','Security', 'Staff', 'Facility Worker','Food Service worker'
]

# number of classes
NUM_CLASSES = 5

# # location to save model and plots
OUT_DIR = 'outputs'