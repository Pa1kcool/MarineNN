import torch

# TRAINING CONFIG
batch_size = 1
lr = 0.0002
num_epochs = 50
step_size = 400
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training Image Directories(input / label) & training image size
raw_image_path = './data/paired/trainA/'
clear_image_path = './data/paired/trainB/'
train_img_size = 256

# Saving Training Checkpoints
snapshots_folder = './snapshots/unetSSIM/'
snapshot_freq = 5
model_name = 'unetSSIM'

# Testing Image Directories(input / output) & test image size
test_image_path = './data/test_imgs/'
output_images_path = './data/test_output/unetssim/'
test_img_size = 512

# Enter checkpoint filepath if you're resuming training (DO NOT ENTER MODEL.CKPT FILES!)
ckpt_path = './snapshots/unetSSIM/model_epoch_49_unetSSIM.ckpt'

# Enter model path for TESTING (ENTER MODEL.CKPT FILES!)
test_model_path = './model_ckpt/marinenn_FINALMODEL.ckpt'

# Validation Image Directory
VAL_DATA_PATH = './data/paired/validation/'
