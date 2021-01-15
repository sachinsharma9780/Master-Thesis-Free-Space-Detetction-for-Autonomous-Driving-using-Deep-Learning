import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision
import os
from glob import glob
from PIL import Image
from utils import AverageMeter, iou
from tqdm import tqdm
import numpy as np
import importlib
import json
from sklearn.metrics import f1_score
import os
from models import erfnet_road
import numpy as np
import cv2 as cv
from PIL import Image
from skimage import feature

#loading models
model = erfnet_road.Net()

#lets try random input
inp1 = torch.randn(1, 3, 360, 640)
inp2 = torch.randn(1, 1, 360, 640)

#inp2 = np.zeros((1, 1, 360, 640))

out = model(inp1, inp2) 
sem, rc = out
print(sem.shape, rc.shape)
exit()
print(inp2.shape)

#imgs = [feature.canny(file) for file in inp2]

#print(np.array(imgs).shape) 

# gives gray image
img = cv.imread('c1.jpg', 0)
print('img', img.shape, type(img))

#---- Apply automatic Canny edge detection using the computed median----
#lower = int(max(0, (1.0 - sigma) * v))
#upper = int(min(255, (1.0 + sigma) * v))
#edges = cv.Canny(img, lower, upper)
edges = feature.canny(img)
print(edges.shape)
plt.imsave('/home/sachin/Desktop/edges3.jpg', edges, cmap = 'gray')