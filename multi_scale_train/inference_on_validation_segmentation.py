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
import os
from models import erfnet


# check cuda is available
if torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")

prefix = '/home/sachin/Desktop/free_space_detection_script/scripts/experiments'
model_checkpoint = "multi_scale/exp8/best_chkpt.pth.tar"


"""model = erfnet.Net(num_classes=3)
model_w = torch.load(os.path.join(prefix, model_checkpoint))
iou_score = model_w["best_iou"]
epoch = model_w['epoch']
print('Best Val Iou_Score is: ', iou_score, 'at epoch: ', epoch)"""
#model_w = model_w["state_dict"]



def validation(data_loader, epoch):
    val_epoch_acc = 0
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')

    with torch.no_grad():
        for data in tqdm(data_loader):
            #model.to(device)
            image, target = data['image'].to(device), data['label'].squeeze(1)
            target = torch.round(target*255) #converting to range 0-255
            target = target.type(torch.int64).to(device)
            output = model(image)
            acc1 = iou(output, target)
            top1.update(acc1, image.size(0))  
           
    print('Validation Iou_Acc: * Acc@1 {top1.avg:.3f}'
                            .format(top1=top1))


    return top1.avg

class berkely_driving_dataset(torch.utils.data.Dataset):
    def __init__(self, path, type='train', transform=None, color = True):
    # dataloader for bdd100k segmentation dataset
    # path should contain the address to bdd100k folder
    # it generally has the following diretory structure
        """
        - bdd100k
          - drivable_maps
            - color_labels
            - labels
          - images
            - 100k
            - 10k
        """
        # type can either be 'train' or 'val'
        self.path = path
        self.type = type
        self.transform = transform
        self.imgs = glob(os.path.join(self.path, 'images/100k/' + self.type + '/*.jpg'))
        if color is True:
            self.labels = [os.path.join(self.path, 'drivable_maps/color_labels/' + self.type,\
                        x.split('/')[-1][:-4] + '_drivable_color.png') for x in self.imgs]
        else:
            print('Color False')
            self.labels = [os.path.join(self.path, 'drivable_maps/labels/' + self.type,\
                                     x.split('/')[-1][:-4]+'_drivable_id.png') for x in self.imgs]
            #print('labels', self.labels)
        self.length = len(self.imgs)

    # so that len(dataset) returns the size of the dataset.
    def __len__(self):
        return self.length

   #  to support the indexing such that dataset[i] can be used to get ith sample 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.imgs[idx]

        if self.type == "train":
            image = Image.open(img_name)
            label = Image.open(self.labels[idx])
            if self.transform:
                image = self.transform(image)
                label = self.transform(label)
            #print('img', image.shape, 'label', label.shape)
            return {'image':image, 'label':label}
        elif self.type == "val":
            image = Image.open(img_name)
            label = Image.open(self.labels[idx])
            if self.transform:
                image = self.transform(image)
                label = self.transform(label)
     
            return {'image':image, 'label':label}



#DEFINING SOME IMPORTANT VARIABLES
PATH_TO_BERKELY_DATASET = '/home/sachin/Desktop/bdd100k'

#loading libraries
#from models import erfnet


#loading models
model = erfnet.Net(num_classes=3)
model_w = torch.load(os.path.join(prefix, model_checkpoint))
model_w = model_w["state_dict"]
model.load_state_dict(model_w)
model.to(device)

bdd_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((544, 960)),
    torchvision.transforms.ToTensor()
])

#bdd_train = berkely_driving_dataset(PATH_TO_BERKELY_DATASET, transform=bdd_transforms,  type='train', color = False)
bdd_val = berkely_driving_dataset(PATH_TO_BERKELY_DATASET, transform=bdd_transforms,  type='val', color = False)

#sampler_train = torch.utils.data.RandomSampler(bdd_train)
sampler_val = torch.utils.data.SequentialSampler(bdd_val)


#for data in dl_train:
 #   print('k', data['image'].shape) 
# the valiation only works with a batchsize of 1
dl_val = torch.utils.data.DataLoader(
    bdd_val, batch_size=32,
    sampler=sampler_val, num_workers = 4, pin_memory=True)

val_iou = validation(dl_val, 1)
print('val_road_score', val_iou)