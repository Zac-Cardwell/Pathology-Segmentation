import torch
from torch import Tensor
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device: %s" % device)
torch.cuda.empty_cache()



# Define the normalization parameters for RGB images (mean and standard deviation)
mean = [0.485, 0.456, 0.406]  
std = [0.229, 0.224, 0.225]  
# Define a transform to normalize RGB images
normalize_rgb = transforms.Normalize(mean=mean, std=std)

# Define the normalization parameters for single-channel images 
mean_gray = [0.5]
std_gray = [0.5]
# Define a transform to normalize single-channel images
normalize_gray = transforms.Normalize(mean=mean_gray, std=std_gray)
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, masks, transform=None, normalize=None):
        self.imgs = imgs
        self.masks = masks
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        if self.normalize:
            if img.shape[0] == 3:  # RGB image
                img = self.normalize(img)
            else:  # Single-channel image (mask)
                img = self.normalize(mask)
        return img, mask

# Create a transform to resize the data
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])



class DiceLossWithL2(torch.nn.Module):
    def __init__(self, smooth=1e-6, weight_decay=1e-4, class_weights=None):
        super(DiceLossWithL2, self).__init__()
        self.smooth = smooth
        self.weight_decay = weight_decay
        self.class_weights = class_weights if class_weights is not None else torch.ones(5)
        self.class_weights = self.class_weights.to(device)


    def forward(self, input: Tensor, target: Tensor):
    
        smooth = self.smooth
        input = F.softmax(input, dim=1)  # Convert to probability distributions
        target = target.squeeze(1).long()
        # Assuming input and target are [N, C, H, W], and target is not one-hot encoded
        target = F.one_hot(target, num_classes=input.shape[1]).permute(0, 3, 1, 2).float()  # One-hot encode target
        intersection = torch.sum(input * target, dim=(0, 2, 3))
        union = torch.sum(input + target, dim=(0, 2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        # Average over classes
        dice_loss =  1 - dice.mean()

        # Compute weighted loss
        weighted_loss = dice_loss * self.class_weights
        weighted_loss = weighted_loss.mean()  # Average over classes

        l2_regularization = 0.0
        for param in self.parameters():
            l2_regularization += torch.norm(param, p=2)  # L2 norm of the parameters

        return weighted_loss + l2_regularization*self.weight_decay



x_filterd, y_filterd = torch.load('augmented_train_data.pth')
train_dataset = CustomDataset(x_filterd, y_filterd, transform=transform, normalize=normalize_rgb)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0) 
print(f"training on {len(train_dataset)} examples")
del train_dataset, x_filterd, y_filterd

print("Loaded training dataset")

val_x, val_y = list(torch.load('val_data.pth'))
val_dataset = CustomDataset(val_x, val_y, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=0)
del val_dataset, val_x, val_y

print("Loaded validation dataset") 


aug_class_weights = torch.tensor([0.00732221649878957,0.45524400426776607,
 0.106497442019506,
 0.09951149013708314,
 0.09678487530732706]).to(device)
class_weights = torch.tensor([0.012452967971797254,0.943406664530095,0.017224022091005885,
 0.014463377435304596,0.012452967971797254]).to(device)
aug_pixel_balenced_weights =  torch.tensor([0.020301930080171213,
 0.6927033298517223,
 0.1268065883428644,
 0.10634791489916325,
 0.053840236826078865]).to(device)
pixel_balenced_weights =  torch.tensor([0.0010834159211355083,
 0.9841470399490003,
 0.006885224040317103,
 0.005360765866233133,
 0.002523554223313973]).to(device)


#Change the model ||||||||||||||||||||||||||||||||||||||||||||||||||
model = DeepLabV3(num_classes=5, in_channels=512)
print("Using Deeplab")
save_path='best_DeepLab_model_3.pth'
Save_model_path = "DeepLab_final_state3.pth"
results_save_path = 'Deeplab_results3.pth'
'''

model = SegmentationCNN(in_channels=3, num_classes=5)
print("Using CNN")
save_path='best_CNN_model_2.pth'
Save_model_path = "CNN_final_state1.pth"
results_save_path = 'CNN_results3.pth'
'''


#Change Loss ||||||||||||||||||||||||||||||||||||||
#criterion = nn.CrossEntropyLoss()
criterion = DiceLossWithL2(smooth=1e-6, weight_decay=1e-5, class_weights=aug_pixel_balenced_weights)


optimizer = torch.optim.Adam(model.parameters(), lr= .05)


print("Begin Training")
losses, accuracies, training_times, learning_rates, val_losses, val_accuracies, train_f1s, val_f1, training_precision, training_recall, val_precision, val_recall, val_iou = train_model(model, criterion, 
optimizer, train_dataloader, val_dataloader, num_epochs=10, device='cuda', scheduler = True,  save_path=save_path, Save_model = True, 
Save_model_path = Save_model_path)


torch.save({'losses': losses,'accuracies': accuracies, 'training_times': training_times, 
'learning_rates': learning_rates, 'val_losses': val_losses, 'val_accuracies': val_accuracies, 'Train_f1': train_f1s, 'val_f1s': val_f1, 'training_precision': training_precision, 'training_recall': training_recall, 'val_precision': val_precision, 'val_recall':val_recall, 'val_iou': val_iou}, results_save_path)
