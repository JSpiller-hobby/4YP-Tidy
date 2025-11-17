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
    val_loss_1 = 0
    val_loss_2 = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, true_masks = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

            true_mask_normed_vectors = torch.linalg.vector_norm(true_masks, ord=2, dim=1, keepdim=False)
            true_mask_norms = torch.linalg.matrix_norm(true_mask_normed_vectors, ord = 'fro', dim = (-2, -1))

            true_mask_norms = true_mask_norms.unsqueeze(1)
            true_mask_norms = true_mask_norms.unsqueeze(2)
            true_mask_norms = true_mask_norms.unsqueeze(3)
            true_mask_norms = true_mask_norms.repeat(1,2,256,256)
            true_masks = torch.div(true_masks, true_mask_norms)

            true_masks_other_dir = torch.neg(true_masks)

            true_masks = true_masks.to(device=device, dtype=torch.float)
            true_masks_other_dir = true_masks_other_dir.to(device=device, dtype=torch.float)

            # predict the mask
            mask_pred = net(image)

            #compute loss 

            #---------------------------------------------------------DEBUG
            #print(f"the shape of true mask batch is {mask_true.size()}")
            #print(f"the shape of predicted mask batch is {mask_pred.size()}")
            #---------------------------------------------------------DEBUG

            val_loss_1 += F.mse_loss(mask_pred, true_masks, size_average=None, reduce=None, reduction='mean', weight=None).item()
            val_loss_2 += F.mse_loss(mask_pred, true_masks_other_dir, size_average=None, reduce=None, reduction='mean', weight=None).item()
            val_loss = min(val_loss_1, val_loss_2)

    net.train()
    return val_loss / max(num_val_batches, 1)
