
from PIL import Image, ImageOps
#from ld_lsi.msg import CnnOutput
#import rospkg
#from rospy.numpy_msg import numpy_msg
#from cv_bridge import CvBridge

import torch
import importlib
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import cv2

import sys
import os
import time
from datetime import datetime
from skimage import feature

from models import erfnet_road

### Colors for visualization
# Ego: red, other: blue
COLORS_DEBUG = [(255,0,0), (0,0,255)]
if torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")

# Road name map
# Class Count for Train data {0: 43516, 1: 17379, 3: 8074, 2: 894}
# label_mappings = {"city street":0, "highway":1, "others":2, "residential":3}
ROAD_MAP = ['city street', 'highway', 'others', 'residential']

model_checkpoint = "/home/sachin/Desktop/free_space_detection_script/scripts/experiments/uncertainity/fusion_with_edge_maps/exp6/best_chkpt.pth.tar"


def evaluate(image, edge_maps):
    resize_factor = 5
    debug = True
    with_road = True
    queue_size = 10
    
    model = erfnet_road.Net()
    model_w = torch.load(model_checkpoint)
    model_w = model_w["state_dict"]
    print(datetime.now() - startTime)
    #model = torch.nn.DataParallel(cnn).cuda()
    model.load_state_dict(model_w)
    model = torch.nn.DataParallel(model).to(device)
    model.eval()
    input_tensor = torch.from_numpy(image)
    input_tensor = torch.div(input_tensor.float(), 255)
    print('inp tensor', input_tensor.shape)
    input_tensor = input_tensor.permute(2,0,1).unsqueeze(0)
    print('inp tensor size', input_tensor.size())
    edge_maps = edge_maps.to(device=device, dtype=torch.float)
    print('edge_maps size', edge_maps.size())

    with torch.no_grad():
        input_tensor = Variable(input_tensor).to(device)
        output = model(input_tensor, edge_maps)
    

    if with_road:
        output, output_road = output
        road_type = output_road.max(dim=1)[1][0]
        print("road_type", road_type)
        
    #classification
    #Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor 
    #in the given dimension dim. And indices is the index location of each maximum value found (argmax).
    
    # without road
    #output, _ = output

    output = output.max(dim=1)[1]
    output = output.float().unsqueeze(0)
    ### Resize to desired scale for easier clustering
    output = F.interpolate(output, size=(output.size(2) // resize_factor, output.size(3) // resize_factor) , mode='nearest')
    ### Obtaining actual output
    ego_lane_points = torch.nonzero(output.squeeze() == 1)
    other_lanes_points = torch.nonzero(output.squeeze() == 2)

    ego_lane_points = ego_lane_points.view(-1).cpu().numpy()
    print('elp', ego_lane_points.shape)
    other_lanes_points = other_lanes_points.view(-1).cpu().numpy()

    print("-ego: {}".format(ego_lane_points))
            ### Debug visualization options
    if debug:
        try:
            # Convert the image and substitute the colors for egolane and other lane
            output = output.squeeze().unsqueeze(2).data.cpu().numpy()
            output = output.astype(np.uint8)

            output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
            output[np.where((output == [1, 1, 1]).all(axis=2))] = COLORS_DEBUG[0]
            output[np.where((output == [2, 2, 2]).all(axis=2))] = COLORS_DEBUG[1]

            # Blend the original image and the output of the CNN
            output = cv2.resize(output, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            image = cv2.addWeighted(image, 1, output, 0.4, 0)
            if with_road:
                cv2.putText(image, ROAD_MAP[road_type], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Visualization
            print("Visualizing output")
            cv2.imwrite("/home/sachin/Desktop/free_space_detection_script/test_imgs/erfnet_op_test2.jpg", cv2.resize(image, (480, 360), cv2.INTER_NEAREST))
            #cv2.waitKey(1)
        except Exception as e:
            print("Visualization error. Exception: %s" % e)


global startTime


if __name__ == '__main__':
    startTime = datetime.now()
    image = '/home/sachin/Desktop/free_space_detection_script/test_imgs/test2.jpg'
    img = Image.open(image)
    img = img.resize((640, 360))
    grayscale_image = ImageOps.grayscale(img)
    grayscale_image = np.asarray(grayscale_image)
    edge_maps = feature.canny(grayscale_image)
    edge_maps = torch.from_numpy(edge_maps)
    edge_maps = torch.unsqueeze(edge_maps, 0)
    edge_maps = torch.unsqueeze(edge_maps, 0)
    print("--- Processing {} ---".format(str(image)))
    oriimg = cv2.imread(str(image),cv2.IMREAD_COLOR)
    print(oriimg.shape)

    rgb_img = cv2.resize(oriimg,(640,360))
    evaluate(rgb_img, edge_maps)
    #print(datetime.now() - startTime)

    print("---------------------------\n") 







