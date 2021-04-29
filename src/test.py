import torch
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import cv2 
import utils 
import config 
from tqdm import tqdm 
from model import FaceKeypointModel
resize = 96 

model = FaceKeypointModel().to(config.DEVICE)
checkpoint = torch.load(f"{config.output_path}/model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# prepraring the data 
csv_file = f"{config.root_path}/test/test.csv"
data = pd.read_csv(csv_file)
pixel_col = data.Image 
image_pixels = []
for i in tqdm(range(len(pixel_col))):
    img = pixel_col[i].split(' ')
    image_pixels.append(img)
images = np.array(image_pixels,dtype='float32')

images_list,outputs_list = [],[]
for i in range(9):
    with torch.no_grad():
        image = images[i]
        image = image.reshape(96,96,1)
        image = cv2.resize(image,(resize,resize))
        image = image.reshape(resize,resize,1)
        orig_image = image.copy()
        image = image/255.0
        image = np.transpose(image,(2,0,1))
        image = torch.tensor(image,dtype=torch.float)
        image = image.unsqueeze(0).to(config.DEVICE)

        outputs = model(image)
        images_list.append(orig_image)
        outputs_list.append(outputs)

utils.test_keypoints_plot(images_list,outputs_list)