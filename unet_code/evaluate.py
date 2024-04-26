import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import dice_loss

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    # criterion = torch.nn.CrossEntropyLoss()

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']

        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        # mask_true = mask_true.squeeze(1)

        # predict the mask
        mask_pred = net(image)
        # mask_pred = F.softmax(mask_pred, dim=1)
        # assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
        # convert to one-hot format
        # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        # mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
        # compute the Dice score, ignoring background
        dice_score += dice_loss(mask_pred, mask_true)
        # dice_score += criterion(mask_pred, mask_true)

    net.train()
    return dice_score / max(num_val_batches, 1)