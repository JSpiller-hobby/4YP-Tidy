#original: https://github.com/milesial/Pytorch-UNet
#Licensed under GNU GENERAL PUBLIC LICENSE V3
#MODIFICATION: removed us of wandb; removed use of other irrelevant tools; removed use of carvana dataset; updated calls to torch AMP package, using CPU now; 
# replaced cross entropy loss with MSE loss; removed code for handling one mask property only, as this is not releveant to our current data;
# removed addition of dice loss to criterion output; added numpy code to normalise the ground truth and add a ground truth array for the opposite direction, 
#taking the lowest loss as the result (this would be better in the data_logging section, but expedient to add here); modified default arguments; moved global step;
# set zero_grad 'setto_none' to False; added option for overfitting dataset;


#TODO: move the velocity mask processing all into data_loading utility. Put evaluation round outside batch loop?
#      There may be a bug, due to mismatched channel-first and channel-last data. Need to check.
#      Modify normalise_and_reverse() to handle image sizes =/ 256 x 256.

#External modules:
import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

#This REPO:
from evaluate import evaluate, normalise_and_reverse
from unet import UNet
from utils.data_loading import BasicDataset, OverfitDataset   

#Specify the data in this directory for training
dir_img = Path('./data/imgsOverfit/')                                  
dir_mask = Path('./data/masksOverfit/')                                
dir_checkpoint = Path('./checkpoints/')

#Function to run training using torch
def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    #dataset = BasicDataset(dir_img, dir_mask, img_scale) 
    dataset = OverfitDataset(dir_img, dir_mask, img_scale) 

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))  

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    #Log the training parameters
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),                                                                          
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) 
    grad_scaler = torch.amp.GradScaler("cpu", enabled=amp)

    criterion = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    # 5. Begin training
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last) 
                                                                                                           
                #Normalise velocities, add the mask for rotation in the other direction:
                true_masks, true_masks_other_dir = normalise_and_reverse(true_masks)

                true_masks = true_masks.to(device=device, dtype=torch.float)
                true_masks_other_dir = true_masks_other_dir.to(device=device, dtype=torch.float)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred_raw = model(images) 
                    masks_pred, _ = normalise_and_reverse(masks_pred_raw)
                    loss_1 = criterion(masks_pred, true_masks)
                    loss_2 = criterion(masks_pred, true_masks_other_dir)
                    loss = min(loss_1, loss_2)
   
                optimizer.zero_grad(set_to_none=False)
                grad_scaler.scale(loss).backward()      
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0]) 
                global_step += 1                                           
                epoch_loss += loss.item()                                  
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round (each epoch)
                division_step = (n_train // (epochs * batch_size))              
                if division_step > 0:
                    if global_step % division_step == 0:

                        val_loss = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_loss)                              

                        logging.info('Validation loss: {}'.format(val_loss))
                        
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')    

#Set up CLI for starting training
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size') 
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images') 
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--properties', '-p', type=int, default=2, help='Number of properties')

    return parser.parse_args()

if __name__ == '__main__': 
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_properties is the number of properties you want to get per pixel, hence the size of the mask in the non-height/width axis.
    model = UNet(n_channels=3, n_properties=args.properties, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last) # This is for efficiency. It causes the model weights to be stored NHWC, and automatically converts input data to NHWC.

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_properties} output channels\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )