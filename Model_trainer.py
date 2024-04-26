import time
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

#Add needed Scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

def evaluate_model(model, criterion, val_loader, device='cuda'):
    model.eval()
    model.to(device)

    val_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1).long())
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_pixels += (predicted == masks).sum().item()
            total_pixels += (masks >= 0).sum().item()

            y_true.extend(masks.squeeze(1).cpu().numpy().flatten())  # Move to CPU before converting to numpy
            y_pred.extend(predicted.cpu().numpy().flatten())

    val_accuracy = correct_pixels / total_pixels
    val_loss /= len(val_loader.dataset)
    f1 = f1_score(y_true, y_pred, average='binary')  # Calculate F1 score

    return val_loss, val_accuracy, f1


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10, device='cuda', scheduler = False, save_path='best_model.pth', Save_model = False, Save_model_path = "final_state.pth"):
    model.to(device)
    model.train()

    losses = []
    accuracies = []
    mious = []
    training_times = []
    learning_rates = []
    train_f1s = []
    val_losses = []
    val_accuracies = []
    val_mious = []
    val_f1s = []
    best_val_loss = float('inf')
    best_val_epoch = 0
    model.to(device)

    # Config scheduler
    if scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

    for epoch in range(num_epochs):
       
        epoch_loss = 0.0
        total_pixels = 0
        correct_pixels = 0

        start_time = time.time()
        for images, masks in tqdm(train_loader):

            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,  masks.squeeze(1).long())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_pixels += (predicted == masks).sum().item()
            total_pixels += (masks >= 0).sum().item()
            
           
        end_time = time.time()

        y_true = []
        y_pred = []
        y_true.extend(masks.squeeze(1).cpu().numpy().flatten())  # Move to CPU before converting to numpy
        y_pred.extend(predicted.cpu().numpy().flatten())
        f1 = f1_score(y_true, y_pred, average='binary')

        # Calculate epoch accuracy, IoU, and training time
        epoch_accuracy = correct_pixels / total_pixels

        epoch_loss /= len(train_loader.dataset)
        training_time = end_time - start_time

        # Evaluate on validation set
        with torch.no_grad():
            val_loss, val_accuracy, val_f1 = evaluate_model(model, criterion, val_loader, device=device)

        # Print and store metrics
        print(f"Epoch {epoch+1}/{num_epochs}, Learning Rate {optimizer.param_groups[0]['lr']}\nTraining Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}, Training f1-score: {f1}, Training Time: {training_time:.2f} sec")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation f1-score: {val_f1}")

        # Scheduler Step
        if scheduler:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
                best_val_epoch = epoch

       

        best_val_loss = save_best_model(model, val_loss, best_val_loss, save_path)
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        training_times.append(training_time)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1s.append(val_f1)
        train_f1s.append(f1)

    if Save_model:
        torch.save(model.state_dict(), Save_model_path)

    print(f"Best model was at epoch {best_val_epoch} with a val loss of {best_val_loss}")
    return losses, accuracies, training_times, learning_rates, val_losses, val_accuracies, train_f1s, val_f1


def save_best_model(model, val_loss, best_val_loss, save_path):
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), save_path)
        return val_loss
    else:
        return best_val_loss


def split_dataset(dataset, ratio):
    # Set a fixed random seed to insure same split
    seed = 314
    torch.manual_seed(seed)

    dataset_size = len(dataset)

    dataset1_size = int(ratio * dataset_size)
    dataset2_size = dataset_size - dataset1_size



    dataset1, dataset2 = random_split(dataset, [dataset1_size, dataset2_size])

    torch.seed()

    return  dataset1, dataset2