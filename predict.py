import os
import h5py

import numpy as np
from PIL import Image

import matplotlib as plt
from matplotlib import cm as CM

from model import CSRNet

import torch
from torchvision import transforms

# if: testing a new image
# root = 'path_to_image_to_predict'

# else if: testing on shanghaitech images
root = '...input/shanghaitech_with_people_density_map/ShanghaiTech'
img_path = os.path.join(root, 'part_B/test_data/images', 'IMG_1.jpg')
h5_path = os.path.join(root, 'part_B/test_data/ground-truth-h5', 'IMG_1.h5')

# Load model with trained weights
model = CSRNet()
model_path = '...input/models/model_best.pth.tar'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

transform = transforms.Compose([transforms.ToTensor()])
img = Image.open(img_path).convert('RGB')
img = transform(img)
img = img.unsqueeze(0)

output = model(img)
print("Predicted Count : ",int(output.detach().cpu().sum().numpy()))
temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))

norm = temp / temp.max() # normalize 0 to 1
im = Image.fromarray(np.uint8(CM.jet(norm)*255)) # apply colormap
if im.mode != 'RGB': # prevent oserror
    im = im.convert('RGB')
im.save('prediction.png')

# if: using ShanghaiTech for testing
temp = h5py.File(h5_path, 'r')
temp_1 = np.asarray(temp['density'])
print("Original Count : ",int(np.sum(temp_1)) + 1)

