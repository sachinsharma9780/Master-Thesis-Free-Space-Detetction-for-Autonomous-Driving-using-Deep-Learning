import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision
import os
from glob import glob
from PIL import Image, ImageOps
from utils import AverageMeter, iou
from tqdm import tqdm
import numpy as np
import importlib
import json
from skimage import feature
''' TORCH.UTILS.BOTTLENECK : '''

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"]= '1', '2'

if torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")

print(device)

# Class Count for Train data {0: 43516, 1: 17379, 3: 8074, 2: 894}
# label_mappings = {"city street":0, "highway":1, "others":2, "residential":3}
# 
weights_road = [0.020, 0.051, 0.110, 1]
weights_road = torch.FloatTensor(weights_road).to(device)
# loss fn definition
class CrossEntropyLoss2d(torch.nn.Module):
    #NLLLoss2d is negative log-likelihood loss
    #it returns the semantic segmentation cross-entropy loss2d
    def __init__(self, weight=None):
        super().__init__()
        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        outputs, _ = outputs
       # print("op", outputs.shape, targets.shape)
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets) 

criterion2d = CrossEntropyLoss2d()
criterion1d = torch.nn.CrossEntropyLoss(weight=weights_road)

# class CrossEntropyLoss(torch.nn.Module):
#     #NLLLoss2d is negative log-likelihood loss
#     #it returns the semantic segmentation cross-entropy loss2d
#     def __init__(self, weight=weights_road):
#         super().__init__()
#         self.loss = torch.nn.NLLLoss(weight)

#     def forward(self, outputs, targets):
#         _, road_output = outputs
#         print("op", road_output.shape, targets.shape)
#         print(torch.nn.functional.log_softmax(road_output, dim=1).shape)

#         return self.loss(torch.nn.functional.log_softmax(road_output, dim=1), targets) 

# criterion1d = CrossEntropyLoss()

# Multi task loss

class MultiTaskLossWrapper(torch.nn.Module):
    def __init__(self, task_num, model):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        self.task_num = task_num
        self.log_vars = torch.nn.Parameter(torch.zeros((task_num)))

    def forward(self, input_3d, input_2d, target, road_gt):
        
        output = self.model(input_3d, input_2d)
        _, road_pred = output

        loss2d_0 = criterion2d(output, target)
        loss1d_1 = criterion1d (road_pred, road_gt)

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss2d_0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1d_1 + self.log_vars[1]
        
        return (loss0+loss1), road_pred, output




def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)   
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum().item() / len(correct_pred)
    return acc

def create_edge_maps(inputs):
    inputs = inputs.detach().cpu().numpy()
    edge_maps = [feature.canny(np.squeeze(img)) for img in inputs]
    edge_maps = np.array(edge_maps)
    edge_maps = torch.from_numpy(edge_maps)
    print('em', edge_maps.shape)
    edge_maps = torch.unsqueeze(edge_maps, 1)
    print('em', edge_maps.shape)
    return edge_maps


def train(c2d, c1d, optimizer, data_loader, device, epoch):
    train_epoch_acc = 0
    cummulative_loss = 0

    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    for data in tqdm(data_loader):
        image, grayscale_image, target, road_gt = data['image'].to(device), data['gray_image'] , data['label'].squeeze(1) , data["road_label"].to(device)
        edge_maps = create_edge_maps(grayscale_image)
        edge_maps = edge_maps.to(device=device, dtype=torch.float)
        target = torch.round(target*255) #converting to range 0-255
        target = target.type(torch.int64).to(device)

        cummulative_loss, road_pred, output = mtl(image, edge_maps, target, road_gt)

        train_acc = multi_acc(road_pred, road_gt)
        train_epoch_acc += train_acc

        #  clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).
        optimizer.zero_grad()
        # computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation
        cummulative_loss.backward()
        # performs a parameter update based on the current gradient and update rule (for ex sgd)
        optimizer.step()

        acc1 = iou(output, target)
        top1.update(acc1, image.size(0))
        avgloss.update(cummulative_loss, image.size(0))

    road_classification_avg = train_epoch_acc/len(data_loader)

    print('Training Loss', avgloss.avg.item())
    print('Training Segmentation Iou_Acc: * Acc@1 {top1.avg:.3f}'
                        .format(top1=top1))
    print("Training road classification acc: ", road_classification_avg)


    return top1.avg, avgloss.avg.item(), road_classification_avg

def validation(data_loader, epoch):
    val_epoch_acc = 0
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')

    with torch.no_grad():
        for data in tqdm(data_loader):
            #model.to(device)
            image, grayscale_image, target, road_gt = data['image'].to(device), data['gray_image'], data['label'].squeeze(1), data["road_label"].to(device)
            edge_maps = create_edge_maps(grayscale_image)
            edge_maps = edge_maps.to(device=device, dtype=torch.float)
            target = torch.round(target*255) #converting to range 0-255
            target = target.type(torch.int64).to(device)
            output = model(image, edge_maps)
            _, road_pred = output
            val_acc = multi_acc(road_pred, road_gt)
            val_epoch_acc += val_acc
            acc1 = iou(output, target)
            top1.update(acc1, image.size(0))  
    
    val_road_classification_avg =  val_epoch_acc/len(data_loader)          
    print('Validation Iou_Acc: * Acc@1 {top1.avg:.3f}'
                            .format(top1=top1))
    print("Validation road classification acc: ", val_road_classification_avg)

    return top1.avg, val_road_classification_avg

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
            img_id = img_name.split("/")[-1]
            scene_gt = train_label[img_id]
            image = Image.open(img_name)
            label = Image.open(self.labels[idx])
            grayscale_image = ImageOps.grayscale(image)
            if self.transform:
                image = self.transform(image)
                label = self.transform(label)
                grayscale_image = self.transform(grayscale_image)
            #print('img', image.shape, 'label', label.shape)
            return {'image':image, 'gray_image':grayscale_image, 'label':label, 'road_label':scene_gt}

        elif self.type == "val":
            img_id = img_name.split("/")[-1]
            scene_gt = val_label[img_id]
            image = Image.open(img_name)
            label = Image.open(self.labels[idx])
            grayscale_image = ImageOps.grayscale(image)
            if self.transform:
                image = self.transform(image)
                label = self.transform(label)
                grayscale_image = self.transform(grayscale_image)
     
            return {'image':image, 'gray_image':grayscale_image, 'label':label, 'road_label':scene_gt}




"""
Here is an example of how to use these functions for training
This script is designed to train on berkely driving dataset. Therefore, the 
PATH_TO_BERKELY_DATASET variable points to the root of that dataset. You might
have to edit it.
"""

#DEFINING SOME IMPORTANT VARIABLES
PATH_TO_BERKELY_DATASET = '/home/sachin/Desktop/bdd100k'

#loading libraries
#from models import erfnet

from models import erfnet_road

#loading models
model = erfnet_road.Net()

# starting from checkpoint
# model_checkpoint = "/home/sachin/Desktop/free_space_detection_script/scripts/experiments/erfnet_on_full_data_rc_withLoss_wts/erfnet_on_full_data_rc.pth.tar"
# model_w = torch.load(model_checkpoint)
# model_w = model_w["state_dict"]
# model.load_state_dict(model_w)

'''if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = torch.nn.DataParallel(model, device_ids=[1])'''

model.to(device)
mtl = MultiTaskLossWrapper(2, model)
mtl.to(device)
# model_w = torch.load('/home/sachin/Desktop/ld-lsi/res/weights/weights_erfnet_road.pth')

# new_mw = {}
# for k,w in model_w.items():
#     new_mw[k[7:]] = w

# model.load_state_dict(new_mw)

# model_file = importlib.import_module("erfnet_road")
# model = model_file.Net(num_classes=3)
# model = torch.nn.DataParallel(model).to(device)
# print("Loading encoder pretrained in imagenet")
# from erfnet_imagenet import ERFNet as ERFNet_imagenet
# pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
# pretrainedEnc.load_state_dict(torch.load("/home/sachin/Desktop/erfnet_pytorch/trained_models/erfnet_encoder_pretrained.pth.tar")['state_dict'])
# pretrainedEnc = next(pretrainedEnc.children()).features.encoder
# model = model_file.Net(num_classes=3, encoder=pretrainedEnc)
# model = torch.nn.DataParallel(model).to(device)


#   torchvision.transforms.RandomHorizontalFlip(),
#    torchvision.transforms.RandomAffine(degrees=(-2, 2))
# Making Dataloaders
bdd_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((360, 640)),
    torchvision.transforms.ToTensor()
])

bdd_train = berkely_driving_dataset(PATH_TO_BERKELY_DATASET, transform=bdd_transforms,  type='train', color = False)
bdd_val = berkely_driving_dataset(PATH_TO_BERKELY_DATASET, transform=bdd_transforms,  type='val', color = False)

sampler_train = torch.utils.data.RandomSampler(bdd_train)
sampler_val = torch.utils.data.SequentialSampler(bdd_val)

# num_workers allows batches to load in parallel
dl_train = torch.utils.data.DataLoader(
    bdd_train, batch_size=32,
    sampler=sampler_train, num_workers = 4)

#for data in dl_train:
 #   print('k', data['image'].shape) 
# the valiation only works with a batchsize of 1
dl_val = torch.utils.data.DataLoader(
    bdd_val, batch_size=32,
    sampler=sampler_val, num_workers = 4)


#load labels as json file
scene_train = "/home/sachin/Desktop/scene_labels/encoded_train_labels.json"
with open(scene_train, "r") as file:
    train_label = json.load(file)


# validation
scene_val = "/home/sachin/Desktop/scene_labels/encoded_val_labels.json"
with open(scene_val, "r") as file:
    val_label = json.load(file)


#defining losses
optimizer = torch.optim.Adam(mtl.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)

save_path_prefix = '/home/sachin/Desktop/free_space_detection_script/scripts/experiments/uncertainity/fusion_with_edge_maps/exp7'
filename = save_path_prefix + '/best_chkpt.pth.tar'
# save checkpoints
def save_checkpoint(state, is_best, epoch, filename=filename):
    "Save check if new best is achieved"
    if is_best:
       print("=> Saving a new best iou:")
       torch.save(state, filename) # save checkpoint
       #torch.save(model, "/home/sachin/Desktop/free_space_detection_script/scripts/experiments/uncertainity/best_wts.pth") # save model
    else:
       print("=> Best Iou_Acc did not improve and saving checkpoints")
       checkpoint_epoch = "epoch_" + str(epoch) + ".pth.tar"
       checkpoint_path = save_path_prefix
       torch.save(state, os.path.join(checkpoint_path, checkpoint_epoch))

#training an epoch for 100 batches
train_iou_score = []
loss_plot = []
val_iou_score = []
train_road_plot = []
val_road_plot = []
epoch_plot = []
last_epoch = 60
best_acc = 0.0

for epoch in range(0, last_epoch):
    print("epoch:", epoch)
    train_iou, loss, train_road_acc = train(criterion2d, criterion1d, optimizer, dl_train, device, epoch)
    epoch_plot.append(epoch)
    loss_plot.append(loss)
    train_iou_score.append(train_iou)
    val_iou, val_road_score = validation(dl_val, epoch)
    val_iou_score.append(val_iou)
    train_road_plot.append(train_road_acc)
    val_road_plot.append(val_road_score)
    is_best = bool(val_iou>best_acc)
    if val_iou > best_acc:
        best_acc = val_iou
        print("new best validation iou", best_acc)
    save_checkpoint({
        "val_road_score": val_road_score,
        "epoch":epoch,
        "state_dict": model.state_dict(),
        "best_iou": best_acc},
        is_best, epoch)


# save graphs
def plot_graphs(x_axis, y_axis, y_name):
  plt.plot(x_axis, y_axis, '-o')
  plt.xlabel('Epoch')
  plt.ylabel(y_name)
  plt.savefig('{}.png'.format(y_name))
  plt.clf()

plot_graphs(epoch_plot, loss_plot, 'save_path_prefix' + '/Train_Loss')
plot_graphs(epoch_plot, val_iou_score, 'save_path_prefix' + '/val_iou_score')
plot_graphs(epoch_plot, train_iou_score, 'save_path_prefix' + '/train_iou_score')
plot_graphs(epoch_plot, train_road_plot,'save_path_prefix'+'/train_road_score')
plot_graphs(epoch_plot, val_road_plot,'save_path_prefix'+'/val_road_score')


