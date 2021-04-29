import torch

# constant paths
root_path   = '../input/facialKeypointDetection'
output_path = '../outputs'

# learning params
BATCH_SIZE = 256
LR = 0.0001
EPOCHS = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train/test split
TEST_SPLIT = 0.2

# show dataset keypoint plot
SHOW_DATASET_PLOT = True