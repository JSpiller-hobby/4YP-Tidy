#original: https://github.com/milesial/Pytorch-UNet
#Licensed under GNU GENERAL PUBLIC LICENSE V3
#MODIFICATION: removed use of dice score, replaced it with just the validation loss

#TODO:

import torch
import torch.nn.functional as F
from tqdm import tqdm

#Function to normalise a batch of masks (velocity fields) and create a mask with reversed gear directions 
def normalise_and_reverse(masks):
    if masks.size() == torch.Size([2, 256, 256]):
        masks = masks.unsqueeze(0)

    mask_norms = torch.linalg.vector_norm(masks, ord = 2, dim = (1,2,3), keepdim = True)

    mask_norms = mask_norms.repeat(1,2,256,256)
    masks = torch.div(masks, mask_norms)

    masks_other_dir = torch.neg(masks)

    return masks, masks_other_dir

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    val_loss_1 = 0
    val_loss_2 = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, true_masks = batch['image'], batch['mask']

            # move images and normalised masks to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

            true_masks, true_masks_other_dir = normalise_and_reverse(true_masks)

            true_masks = true_masks.to(device=device, dtype=torch.float)
            true_masks_other_dir = true_masks_other_dir.to(device=device, dtype=torch.float)

            # predict the mask
            masks_pred_raw = net(image) 
            mask_pred, _ = normalise_and_reverse(masks_pred_raw)

            #compute loss 

            val_loss_1 += F.mse_loss(mask_pred, true_masks, size_average=None, reduce=None, reduction='mean', weight=None).item()
            val_loss_2 += F.mse_loss(mask_pred, true_masks_other_dir, size_average=None, reduce=None, reduction='mean', weight=None).item()
            val_loss = min(val_loss_1, val_loss_2)

    net.train()
    return val_loss / max(num_val_batches, 1)