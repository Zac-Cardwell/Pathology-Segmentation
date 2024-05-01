import time
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau

def calculate_confusion_matrix(y_true, y_pred, num_classes):
    TP = torch.zeros(num_classes)
    FP = torch.zeros(num_classes)
    TN = torch.zeros(num_classes)
    FN = torch.zeros(num_classes)

    for i in range(num_classes):
        # Calculate TP, FP, TN, FN for class i
        TP[i] = ((y_true == i) & (y_pred == i)).sum().item()
        FP[i] = ((y_true != i) & (y_pred == i)).sum().item()
        TN[i] = ((y_true != i) & (y_pred != i)).sum().item()
        FN[i] = ((y_true == i) & (y_pred != i)).sum().item()

    #print(f"TP: {TP} FP: {FP} FN: {FN} TN: {TN}")
    return TP, FP, FN

def calculate_f1_score(y_true, y_pred, num_classes):

    TP, FP, FN = calculate_confusion_matrix(y_true, y_pred, num_classes)
    epsilon = 1e-7  # Small value to avoid division by zero
    precision = TP / (TP + FP + epsilon)
    #print(f"precision: {precision}")
    recall = TP / (TP + FN + epsilon)
    #print(f"recall: {recall}")
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    #print(f"f1_score: {f1_score}")
    return f1_score, precision, recall




def intersection_over_union(pred_mask, true_mask):
    intersection = torch.logical_and(pred_mask, true_mask).sum()
    union = torch.logical_or(pred_mask, true_mask).sum()
    iou = intersection.float() / union.float()
    return iou.item()  # Convert to Python float

def evaluate_model(model, criterion, val_loader, device='cuda'):
    model.eval()
    model.to(device)

    val_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    y_true = []
    y_pred = []
    f1_scores = []
    precisions = []
    recalls = []
    iou = 0
    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1).long())
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_pixels += (predicted == masks).sum().item()
            total_pixels += (masks >= 0).sum().item()

             # Calculate F1 score per batch
            y_true_flat = masks.view(-1).type(torch.int32)
            y_pred_flat = predicted.view(-1).type(torch.int32)
            #print(y_true_flat[:5], y_pred_flat[:5])
            # Calculate the F1 score
            f1, precision, recall = calculate_f1_score(y_true_flat, y_pred_flat, 5)
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            iou += intersection_over_union(predicted, masks)


    stacked_f1 = torch.stack(f1_scores, dim=0)
    average_f1 = torch.mean(stacked_f1, dim=0)

    stacked_p = torch.stack(precisions, dim=0)
    average_p = torch.mean(stacked_p, dim=0)

    stacked_r = torch.stack(recalls, dim=0)
    average_r = torch.mean(stacked_r, dim=0)

    val_accuracy = correct_pixels / total_pixels
    val_loss /= len(val_loader.dataset)
    iou /= len(val_loader.dataset)

    return val_loss, val_accuracy, average_f1, average_p, average_r, iou



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
    training_precision = []
    training_recall = []
    val_precision = []
    val_recall = []
    val_iou = []

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
        iou=0
        f1_scores = []
        precisions = []
        recalls = []

        start_time = time.time()
        for images, masks in tqdm(train_loader):

            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1).long())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)

            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_pixels += (predicted == masks).sum().item()
            total_pixels += (masks >= 0).sum().item()
            #print(predicted[0])

            iou += intersection_over_union(predicted, masks)
            # Calculate F1 score per batch
            y_true_flat = masks.view(-1).type(torch.int32)
            y_pred_flat = predicted.view(-1).type(torch.int32)
            #print(y_true_flat[:5], y_pred_flat[:5])
            # Calculate the F1 score
            f1, precision, recall = calculate_f1_score(y_true_flat, y_pred_flat, 5)
            #print(intersection_over_union(predicted, masks))
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)

        end_time = time.time()

        # Average F1 scores across all batches
        stacked_f1 = torch.stack(f1_scores, dim=0)
        average_f1 = torch.mean(stacked_f1, dim=0)

        stacked_p = torch.stack(precisions, dim=0)
        average_p = torch.mean(stacked_p, dim=0)

        stacked_r = torch.stack(recalls, dim=0)
        average_r = torch.mean(stacked_r, dim=0)

        # Calculate epoch accuracy, IoU, and training time
        epoch_accuracy = correct_pixels / total_pixels
        epoch_loss /= len(train_loader.dataset)
        iou /= len(train_loader.dataset)
        training_time = end_time - start_time

        # Evaluate on validation set
        with torch.no_grad():
            val_loss, val_accuracy, val_f1, val_p, val_r, viou = evaluate_model(model, criterion, val_loader, device=device)

        # Print and store metrics
        print(f"Epoch {epoch+1}/{num_epochs}, Learning Rate {optimizer.param_groups[0]['lr']}, Training Time: {training_time:.2f} sec\nTraining Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}, Training iou {iou} Training f1-score: {average_f1}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation IOU: {viou}, Validation f1-score: {val_f1}\nval_p: {val_p}, val_r: {val_r}")

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
        train_f1s.append(average_f1)
        training_precision.append(average_p)
        training_recall.append(average_r)
        val_precision.append(val_p)
        val_recall.append(val_r)
        val_iou.append(viou)

    if Save_model:
        torch.save(model.state_dict(), Save_model_path)

    print(f"Best model was at epoch {best_val_epoch} with a val loss of {best_val_loss}")
    return losses, accuracies, training_times, learning_rates, val_losses, val_accuracies, train_f1s, val_f1, training_precision, training_recall, val_precision, val_recall, val_iou


def save_best_model(model, val_loss, best_val_loss, save_path):
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), save_path)
        return val_loss
    else:
        return best_val_loss
