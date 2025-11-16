#original: https://github.com/milesial/Pytorch-UNet
#Licensed under GNU GENERAL PUBLIC LICENSE V3
#MODIFICATION: removed use of dice score, replaced it with just the validation loss

#TODO: There may be a bug, due to mismatched channel-first and channel-last data. Need to check.

import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    val_loss = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            #compute loss 

            #---------------------------------------------------------DEBUG
            print(f"the shape of true mask batch is {mask_true.size()}")
            print(f"the shape of predicted mask batch is {mask_pred.size()}")
            #---------------------------------------------------------DEBUG

            val_loss += F.mse_loss(mask_pred, mask_true, size_average=None, reduce=None, reduction='mean', weight=None).item()

    net.train()
    return val_loss / max(num_val_batches, 1)
