import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import cv2
import torch.nn.functional as F
#import torch.functional as F
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from sklearn.metrics import jaccard_score as jsc
from shapely.geometry import Polygon, MultiPolygon

from concurrent.futures import ThreadPoolExecutor, as_completed


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def iou_numpy(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6
    outputs = outputs.squeeze(1)
    
    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded  # Or thresholded.mean()

def get_score(out, target):
    #intersection = np.logical_and(target, out)
    #print("inter", intersection.shape)
    #union = np.logical_or(target, out)
    #iou_score = np.sum(intersection) / np.sum(union)
    # jaccard similarity
    y_true = target.reshape(-1)
    #print("y_true", y_true.shape)
    y_pred = out.reshape(-1)
    jacc_sim = jsc(y_true, y_pred, average=None)
    mean_jacc_sim = np.mean(jacc_sim)

    return mean_jacc_sim

def iou(output, target):
    #doing interpolation and other stuff
    #output, x = output
    ### Classification
    #output = output.max(dim=1)[1]
    #print("out shape", output.shape, "target", target.shape)
    output = output
    output = torch.argmax(output, 1)
 
    #print("out shape", output.shape)
    #output = output.float().unsqueeze(1)
    #print("out shape", output.shape)

    ### get and store the IOU metrics
    output = output.cpu().numpy()
    return get_score(output, target.cpu().numpy())

