import torch

BATCH_SIZE = 8 # increase / decrease according to GPU memeory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_EPOCHS = 3 # number of epochs to train for

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training images and XML files directory
TRAIN_DIR = 'D:\\DL_proj\\dataset\\train'
# validation images and XML files directory
VALID_DIR = 'D:\\DL_proj\\dataset\\valid'

# classes: 0 index is reserved for background
CLASSES = [
    'student','Security', 'Staff', 'Facility Worker','Food Service worker'
]

# number of classes
NUM_CLASSES = 5

# # location to save model and plots
# OUT_DIR = '../outputs'
# SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
# SAVE_MODEL_EPOCH = 2 # save model after these many epochs