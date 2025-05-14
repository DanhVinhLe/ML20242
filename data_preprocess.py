import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os 

def prepare_data(data_dir, input_size, batch_size):
    """
    Prepare data loaders for training, validation (if available), and testing
    
    Args:
        data_dir: Root directory for dataset
        input_size: Input image size (will be resized to this)
        batch_size: Batch size for dataloaders
        
    Returns:
        tuple: (dataloaders, dataset_sizes, class_names, num_classes)
    """
    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # Check for validation directory - it could be named 'val', 'valid' or 'validation'
    val_dir_name = None
    for val_name in ['val', 'valid', 'validation']:
        if os.path.exists(os.path.join(data_dir, val_name)):
            val_dir_name = val_name
            break
    
    if val_dir_name:
        data_transforms[val_dir_name] = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    # Define dataset splits to process
    dataset_splits = ['train', 'test']
    if val_dir_name:
        dataset_splits.append(val_dir_name)
    
    # Load datasets
    image_datasets = {}
    for split in dataset_splits:
        dataset_path = os.path.join(data_dir, split)
        if os.path.exists(dataset_path):
            image_datasets[split] = datasets.ImageFolder(
                root=dataset_path, 
                transform=data_transforms[split]
            )
    
    # Create dataloaders
    dataloaders = {
        split: DataLoader(
            image_datasets[split], 
            batch_size=batch_size, 
            shuffle=(split == 'train'), 
            num_workers=4
        ) for split in image_datasets.keys()
    }
    
    # Get dataset sizes
    dataset_sizes = {split: len(image_datasets[split]) for split in image_datasets.keys()}
    
    # Get class names and number of classes from training set
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    
    print(f"Data preparation complete:")
    print(f"- Found {len(dataset_splits)} splits: {', '.join(dataset_splits)}")
    if val_dir_name:
        print(f"- Validation directory found: '{val_dir_name}'")
    else:
        print(f"- No validation directory found")
    print(f"- Number of classes: {num_classes}")
    for split in dataloaders.keys():
        print(f"- {split} set: {dataset_sizes[split]} images")
    
    return dataloaders, dataset_sizes, class_names, num_classes