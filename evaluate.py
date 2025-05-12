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

def evaluate_model(model, test_dataloader, batch_size, num_class=10):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.eval()
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Evaluating", unit="batch"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    eval_time = time.time() - start_time
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
    print(f'Time taken for evaluation: {eval_time:.2f} seconds')
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    class_report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(num_class)])
    
    print(class_report)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[str(i) for i in range(num_class)], yticklabels=[str(i) for i in range(num_class)])
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'eval_time': eval_time
    }
    return results


    