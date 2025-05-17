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

def prepare_data(train_dir,test_dir, input_size, batch_size):
    """
    Prepare data loaders for training, validation (if available), and testing
    
    Args:
        train_dir: Directory containing training images
        test_dir: Directory containing testing images
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
    
    # Load datasets
    image_datasets = {
        'train': datasets.ImageFolder(root=train_dir, transform=data_transforms['train']),
        'test': datasets.ImageFolder(root=test_dir, transform=data_transforms['test'])
    }
    
    #Create dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)
    }
    
    dataset_sizes = {
        'train': len(image_datasets['train']),
        'test': len(image_datasets['test'])
    }
    
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    
    print("Number of classes:", num_classes)
    print("Train set size:", dataset_sizes['train'])
    print("Test set size:", dataset_sizes['test'])
    
    return dataloaders, dataset_sizes, class_names, num_classes

def prepare_mnist_data(data_dir, batch_size):
    """
    Prepare data loaders for MNIST dataset for LinearSVM.
    Images will be flattened.
    
    Args:
        data_dir: Directory to download/load MNIST data
        batch_size: Batch size for dataloaders
        
    Returns:
        tuple: (dataloaders, dataset_sizes, class_names, num_classes)
    """
    mnist_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), 
    ])
    
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=mnist_transforms)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=mnist_transforms)
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }
    
    dataset_sizes = {
        'train': len(train_dataset),
        'test': len(test_dataset)
    }
    
    # MNIST class names are digits 0-9
    class_names = [str(i) for i in range(10)]
    num_classes = 10
    
    print("MNIST Data Prepared:")
    print("Number of classes:", num_classes)
    print("Train set size:", dataset_sizes['train'])
    print("Test set size:", dataset_sizes['test'])
    
    return dataloaders, dataset_sizes, class_names, num_classes