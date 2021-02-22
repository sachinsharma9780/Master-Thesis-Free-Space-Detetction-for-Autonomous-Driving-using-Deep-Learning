
from PIL import Image
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


from models import erfnet_road

### Colors for visualization
# Ego: red, other: blue
COLORS_DEBUG = [(255,0,0), (0,0,255)]

# Road name map
ROAD_MAP = ['Residential', 'Highway', 'City Street', 'Other']

def evaluate(image):
    resize_factor = 5
    debug = True
    with_road = True
    queue_size = 10
    cnn = erfnet_road.Net()
    model_w = torch.load('/home/sachin/Desktop/free_space_detection_script/res/weights/weights_erfnet_road.pth')
    cnn = torch.nn.DataParallel(cnn).cuda()
    cnn.load_state_dict(model_w)
    cnn.eval()
    input_tensor = torch.from_numpy(image)
    input_tensor = torch.div(input_tensor.float(), 255)
    input_tensor = input_tensor.permute(2,0,1).unsqueeze(0)
    print('inp tensor size', input_tensor.size())

    with torch.no_grad():
        input_tensor = Variable(input_tensor).cuda()
        output = cnn(input_tensor)
    

    if with_road:
        output, output_road = output
        road_type = output_road.max(dim=1)[1][0]
        print("road_type", road_type)
        
    #classification
    #Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor 
    #in the given dimension dim. And indices is the index location of each maximum value found (argmax).
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
            cv2.imwrite("/home/sachin/Desktop/free_space_detection_script/test_imgs/wts_from_paper/erfnet_op_test9.jpg", cv2.resize(image, (480, 360), cv2.INTER_NEAREST))
            #cv2.waitKey(1)
        except Exception as e:
            print("Visualization error. Exception: %s" % e)




if __name__ == '__main__':
    image = '/home/sachin/Desktop/free_space_detection_script/test_imgs/wts_from_paper/test9.jpg'
    print("--- Processing {} ---".format(str(image)))
    oriimg = cv2.imread(str(image),cv2.IMREAD_COLOR)
    print(oriimg.shape)

    img = cv2.resize(oriimg,(640, 480))
    evaluate(img)
    print("---------------------------\n") 







