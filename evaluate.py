import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns

def evaluate_model(model, test_dataloader, num_class=10):
    """
    Evaluate a model on test data with various metrics including top-1 and top-5 error rates
    
    Args:
        model: The PyTorch model to evaluate
        test_dataloader: DataLoader for the test dataset
        batch_size: Batch size used in the dataloader
        num_class: Number of classes in the dataset
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.eval()
    
    correct = 0
    total = 0
    top5_correct = 0
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Evaluating", unit="batch"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Top-1 accuracy calculation
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            # Top-5 accuracy calculation (only if num_class > 5)
            if num_class > 5:
                _, top5_preds = outputs.topk(5, 1, largest=True, sorted=True)
                top5_preds = top5_preds.t()
                top5_correct_batch = top5_preds.eq(labels.view(1, -1).expand_as(top5_preds))
                top5_correct += top5_correct_batch.sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    eval_time = time.time() - start_time
    
    # Calculate metrics
    top1_accuracy = 100 * correct / total
    top1_error = 100 - top1_accuracy
    
    if num_class > 5:
        top5_accuracy = 100 * top5_correct / total
        top5_error = 100 - top5_accuracy
    else:
        top5_accuracy = None
        top5_error = None
    
    print(f'Evaluation Results:')
    print(f'Top-1 Accuracy: {top1_accuracy:.2f}%')
    print(f'Top-1 Error: {top1_error:.2f}%')
    
    if num_class > 5:
        print(f'Top-5 Accuracy: {top5_accuracy:.2f}%')
        print(f'Top-5 Error: {top5_error:.2f}%')
    
    print(f'Time taken for evaluation: {eval_time:.2f} seconds')
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    class_report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(num_class)])
    
    print(class_report)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[str(i) for i in range(num_class)], 
                yticklabels=[str(i) for i in range(num_class)])
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Compile results
    results = {
        'top1_accuracy': top1_accuracy,
        'top1_error': top1_error,
        'top5_accuracy': top5_accuracy if num_class > 5 else None,
        'top5_error': top5_error if num_class > 5 else None,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'eval_time': eval_time
    }
    
    return results

    