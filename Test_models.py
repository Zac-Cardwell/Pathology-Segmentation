import torch
import numpy as np
from  Deeplab import DeepLabV3
from simple_CNN import SegmentationCNN
from Model_trainer import train_model
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from torch.utils.data import Subset

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        mask = Image.open(self.mask_paths[idx])
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Define a transformation to apply to the images and masks
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
'''
def dice_coefficient(outputs, targets, smooth=1e-6):
    #print(outputs.shape, targets.shape)
    num_classes = outputs.size(1)
    dice = 0.0
    for class_id in range(num_classes):
        output = outputs[:, class_id, ...]vvvvvvvvvvvvvvvvvvvvvvv
        target = (targets == class_id).float()
        intersection = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target)
        dice += (2.0 * intersection + smooth) / (union + smooth)

    return dice / num_classes
  '''
def dice_coefficient(self, y_true, y_pred):

    beta = 0.25
    alpha = 0.25
    gamma = 2
    epsilon = 1e-5
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

class DiceLossWithL2(torch.nn.Module):
    def __init__(self, smooth=1e-6, weight_decay=1e-4, class_weights=None):
        super(DiceLossWithL2, self).__init__()
        self.smooth = smooth
        self.weight_decay = weight_decay
        self.class_weights = class_weights

    def forward(self, outputs, targets):
        num_classes = outputs.size(1)
        dice = 0.0
        for class_id in range(num_classes):
            output = outputs[:, class_id, ...]
            target = (targets == class_id).float()
            intersection = torch.sum(output * target)
            union = torch.sum(output) + torch.sum(target)
            class_dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

            if self.class_weights is not None:
                class_dice *= self.class_weights[class_id]

            dice += class_dice

        dice_loss = 1.0 - dice / num_classes

        # L2 regularization
        l2_regularization = 0.0
        for param in self.parameters():
            l2_regularization += torch.norm(param, p=2)  # L2 norm of the parameters

        return dice_loss + self.weight_decay * l2_regularization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device: %s" % device)
torch.cuda.empty_cache()

#Change the data size ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
x_filterd, y_filterd = torch.load('train_data_augmented.pth')
#indices = range(len(train_data)//5)  # Specify the indices for the subset
#subset_train_data = Subset(train_data, indices)

# Change below line ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
train_dataset = CustomDataset(x_filterd, y_filterd, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0) 
print(f"training on {len(train_dataset)} examples")
del train_dataset

print("Loaded training dataset")

val_data = list(torch.load('val_data.pth'))

# Change below line ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
'''indices = range(len(val_data//2))  # Specify the indices for the subset
subset_val_data = Subset(val_data, indices)'''

val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True, num_workers=0)

del val_data
print("Loaded validation dataset") 
class_weights = torch.tensor([0.24196218826831767,
 0.45524400426776607,
 0.106497442019506,
 0.09951149013708314,
 0.09678487530732706]).to(device)


#Change the model ||||||||||||||||||||||||||||||||||||||||||||||||||
#deeplab_model = DeepLabV3(num_classes=5, in_channels=512)
CNN_model = SegmentationCNN(in_channels=512, num_classes=5)

#Change Loss ||||||||||||||||||||||||||||||||||||||
criterion = DiceLossWithL2()

optimizer = torch.optim.Adam(deeplab_model.parameters(), lr=0.05,  weight_decay=0.05)

print("Begin Training")
# Change below line ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
losses, accuracies, training_times, learning_rates, val_losses, val_accuracies, train_f1s, val_f1 = train_model(CNN_model, criterion, 
optimizer, train_dataloader, val_dataloader, num_epochs=10, device='cuda', scheduler = True,  save_path='best_CNN_model_1.pth', Save_model = True, 
Save_model_path = "CNN_final_state1.pth")

results_save_path = 'Deeplab_results1.pth'
torch.save({'losses': losses,'accuracies': accuracies, 'training_times': training_times, 
'learning_rates': learning_rates, 'val_losses': val_losses, 'val_accuracies': val_accuracies, 'Train_f1': train_f1s, 'val_f1s': val_f1 }, results_save_path)