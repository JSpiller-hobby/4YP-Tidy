#original: https://github.com/milesial/Pytorch-UNet
#Licensed under GNU GENERAL PUBLIC LICENSE V3
#MODIFICATION: repurposed sections of 'train.py', 'evaluate.py' and 'predict.py' code for 
# evaluation using the training data.

#NOTE: must be run on the same machine as training, to get same random split for data.

#Imports (Minimal)
import random
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split

from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, OverfitDataset  

#Data locations
dir_img = Path('./data/imgsOverfit/')                                  
dir_mask = Path('./data/masksOverfit/')

#Model location
model = Path('./models/MODEL_OVERFIT.pth')

#Dataloader (replace params with those used in training)
##PARAMS--------------------------------------------------------------
img_scale = 1.0
val_percent = 0.1
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
amp = False
properties = 2
bilinear = False
##--------------------------------------------------------------------

#dataset = BasicDataset(dir_img, dir_mask, img_scale)
dataset = OverfitDataset(dir_img, dir_mask, img_scale)   

n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train_set, _ = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))  

loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
eval_train_loader = DataLoader(train_set, shuffle=False, **loader_args)

#Set up model
net = UNet(n_channels=3, n_properties=properties, bilinear=bilinear)
net.to(device=device)
state_dict = torch.load(model, map_location=device)
net.load_state_dict(state_dict)

#Call evaluate
val_loss = evaluate(net, eval_train_loader, device, amp)
print(val_loss)