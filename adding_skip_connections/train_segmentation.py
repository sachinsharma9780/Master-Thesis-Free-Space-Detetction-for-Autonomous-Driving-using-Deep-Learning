import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision
import os
from glob import glob
from PIL import Image
from utils import AverageMeter, iou
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import importlib
from skimage.transform import rescale, resize, downscale_local_mean
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")

# loss fn definition
class CrossEntropyLoss2d(torch.nn.Module):
    #NLLLoss2d is negative log-likelihood loss
    #it returns the semantic segmentation cross-entropy loss2d
    def __init__(self, weight=None):
        super().__init__()
        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        outputs = outputs
        #print("op", outputs.shape, targets.shape)
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets) 


# total training imgs = 69863
def train(criterion, optimizer, data_loader, device, epoch):
    model.train()
    batch_count = 1
    top1 = AverageMeter('Acc@1', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')
    for data in tqdm(data_loader):
        model.to(device)
        #print('.',end='')
        image, target = data['image'].to(device), data['label'].squeeze(1)
        target = target.squeeze(1)
        image = image.to(device)
        target = torch.round(target*255) #converting to range 0-255
        target = target.type(torch.int64).to(device)
        output = model(image)
        
        loss = criterion(output, target)

        #  clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).
        optimizer.zero_grad()
        # computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation
        loss.backward()
        # performs a parameter update based on the current gradient and update rule (for ex sgd)
        optimizer.step()

        acc1 = iou(output, target)
        top1.update(acc1, image.size(0))
        avgloss.update(loss, image.size(0))
        batch_count = batch_count + 1 
    
    print('Training Loss', avgloss.avg.item())
    print('Training: * Acc@1 {top1.avg:.3f}'
                        .format(top1=top1))

    return top1.avg, avgloss.avg.item()

def validation(data_loader, epoch):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')

    with torch.no_grad():
        for data in tqdm(data_loader):
            model.to(device)
            image, target = data['image'].to(device), data['label'].squeeze(1)
            target = torch.round(target*255) #converting to range 0-255
            target = target.type(torch.int64).to(device)
            output = model(image)
            acc1 = iou(output, target)
            top1.update(acc1, image.size(0))

    print('Validation: * Acc@1 {top1.avg:.3f}'
                            .format(top1=top1))
    return top1.avg

class berkely_driving_dataset(torch.utils.data.Dataset):
    def __init__(self, path, type='train', transform=None,  color = True):
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
            print(' True color')
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

        #print('idx', idx)
        img_name = self.imgs[idx]
        image = Image.open(img_name)
        label = Image.open(self.labels[idx])
  
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return {'image':image, 'label':label}


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
from models import erfnet_skip_1

#loading models
model = erfnet_skip_1.Net(num_classes=3)
# starting from checkpoint
model_checkpoint = '/home/sachin/Desktop/free_space_detection_script/scripts/experiments/adding_skip_connections/exp10/epoch_5.pth.tar'
model_w = torch.load(model_checkpoint)
model_w = model_w["state_dict"]
model.load_state_dict(model_w)

model.to(device)


#lets try random input
#input_image = torch.randn(1, 3, 480, 640)
#input_image = input_image.to(device)
#output = model(input_image)
#print(output.shape)


# Making Dataloaders
bdd_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((480, 640)),
    torchvision.transforms.ToTensor()
    ])




bdd_train = berkely_driving_dataset(PATH_TO_BERKELY_DATASET, transform=bdd_transforms, type='train', color = False)
bdd_val = berkely_driving_dataset(PATH_TO_BERKELY_DATASET, transform=bdd_transforms,  type='val', color = False)

sampler_train = torch.utils.data.RandomSampler(bdd_train)
sampler_val = torch.utils.data.SequentialSampler(bdd_val)

batch_size = 32
dl_train = torch.utils.data.DataLoader(
    bdd_train, batch_size=batch_size,
    sampler=sampler_train, num_workers=2)
 
dl_val = torch.utils.data.DataLoader(
    bdd_val, batch_size=batch_size,
    sampler=sampler_val, num_workers=2)



#defining losses
criterion = CrossEntropyLoss2d()
optimizer = torch.optim.Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)

save_path_prefix = '/home/sachin/Desktop/free_space_detection_script/scripts/experiments/adding_skip_connections/exp10'
filename = save_path_prefix + '/best_chkpt.pth.tar'
print('exp:', save_path_prefix)

# save checkpoints
def save_checkpoint(state, is_best, epoch, filename=filename):
    "Save check if new best is achieved"
    if is_best:
       print("=> Saving a new best iou:")
       torch.save(state, filename) # save checkpoint
    else:
       print("=> Best Iou_Acc did not improve and saving checkpoints")
       checkpoint_epoch = "epoch_" + str(epoch) + ".pth.tar"
       checkpoint_path = save_path_prefix
       torch.save(state, os.path.join(checkpoint_path, checkpoint_epoch))

#training an epoch for 100 batches
train_iou_score = []
loss_plot = []
val_iou_score = []
epoch_plot = []
last_epoch = 60
best_acc = 0.0

for epoch in range(6, last_epoch):
    print("epoch:", epoch)
    train_iou, loss = train(criterion, optimizer, dl_train, device, epoch)
    epoch_plot.append(epoch)
    loss_plot.append(loss)
    train_iou_score.append(train_iou)
    val_iou = validation(dl_val, epoch)
    val_iou_score.append(val_iou)
    is_best = bool(val_iou>best_acc)
    if val_iou > best_acc:
        best_acc = val_iou
        print("new best validation iou", best_acc)
    save_checkpoint({
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

plot_graphs(epoch_plot, loss_plot, save_path_prefix +'/Train_Loss_')
plot_graphs(epoch_plot, val_iou_score, save_path_prefix + '/val_iou_score')
plot_graphs(epoch_plot, train_iou_score, save_path_prefix + '/train_iou_score')


