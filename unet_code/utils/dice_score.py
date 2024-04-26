import torch
from torch import Tensor
import torch.nn.functional as F


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


# def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
#     # Average of Dice coefficient for all classes
#     return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def multiclass_dice_coeff(input, target, smooth=1e-6):
    input = F.softmax(input, dim=1)  # Convert to probability distributions
    target = target.squeeze(1).long()
    # Assuming input and target are [N, C, H, W], and target is not one-hot encoded
    target = F.one_hot(target, num_classes=input.shape[1]).permute(0, 3, 1, 2).float()  # One-hot encode target
    intersection = torch.sum(input * target, dim=(0, 2, 3))
    union = torch.sum(input + target, dim=(0, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    # Average over classes
    return dice.mean()

def dice_loss(input: Tensor, target: Tensor, smooth=1e-6):
    input = F.softmax(input, dim=1)  # Convert to probability distributions
    target = target.squeeze(1).long()
    # Assuming input and target are [N, C, H, W], and target is not one-hot encoded
    target = F.one_hot(target, num_classes=input.shape[1]).permute(0, 3, 1, 2).float()  # One-hot encode target
    intersection = torch.sum(input * target, dim=(0, 2, 3))
    union = torch.sum(input + target, dim=(0, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    # Average over classes
    return 1 - dice.mean()