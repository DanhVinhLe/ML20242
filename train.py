from evaluate import evaluate_model
from trainer import Trainer, count_images_per_class, calculate_class_weights
from data_preprocess import prepare_data,prepare_mnist_data
from model import *
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
from typing import Type, Any, Callable, Union, List, Optional
import os
import argparse
from tqdm import tqdm
import seaborn as sns
import random
import json
import logging
import sys
import warnings
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LambdaLR
from evaluate import evaluate_model

def parse_args(input_args = None):
    parser = argparse.ArgumentParser(description="Example training script")
    parser.add_argument('--train_dir', type = str, help = 'Path to the training data directory')
    parser.add_argument('--test_dir', type = str, help = 'Path to the testing data directory')
    parser.add_argument('--mnist_data_dir', type=str, default=None, help='Directory to store MNIST data')    
    parser.add_argument('--input_size', type=int, default=224, help='Input size for the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--model_name', type = str, default = 'alexnet',
                        choices = ['linearsvm_mnist','alexnet', 'vgg16', 'lenet', 'vgg16', 'vgg16_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'inceptionv3', 'mobilenetv3', 'vit'],
                        help = 'Name of the model to use')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model weights')
    parser.add_argument('--save_path', type=str, default='best_model.pth', help='Path to save the best model')
    parser.add_argument('--criterion', type=str, default='cross_entropy', choices=['cross_entropy', 'mse','hinge'], help='Loss function to use')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='constant', choices=['constant', 'linear', 'cosine'], help='Learning rate scheduler to use')
    parser.add_argument('--num_warmup_steps', type=int, default=0, help='Number of warmup steps for the scheduler')
    parser.add_argument('--model_type', type = str, default = 'large', choices = ['large', 'small'], help = 'Model type of MobileNetV3')
    parser.add_argument('--dropout_rate', type = float, default = 0.4, help = 'Dropout rate for model')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for loss function')
    parser.add_argument('--weight_type', type=str, default='inverse', choices=['inverse', 'sqrt_inverse'], help='Type of class weights to use')
    args = parser.parse_args(input_args)
    return args

def main(args):
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    if args.mnist_data_dir is not None:
        dataloaders, dataset_sizes, class_names, num_classes = prepare_mnist_data(data_dir=args.mnist_data_dir, batch_size=args.batch_size)
    else:
        if not args.train_dir or not args.test_dir:
            raise ValueError("train_dir and test_dir must be specified for models other than linearsvm_mnist")
        dataloaders, dataset_sizes, class_names, num_classes = prepare_data(train_dir= args.train_dir, test_dir= args.test_dir, input_size= args.input_size, batch_size= args.batch_size)
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Class names: {class_names}")
    
    if args.model_name == 'linearsvm_mnist':
        model = LinearSVM(input_size=args.input_size, num_classes=num_classes)
    elif args.model_name == 'alexnet':
        model = AlexNet(num_classes=num_classes)
    elif args.model_name == 'lenet':
        model = LeNet(num_classes=num_classes, in_channels=1)
    elif args.model_name == 'vgg16':
        model = VGG16(num_classes = num_classes, in_channels = 3, dropout_rate= 0.4, input_size=args.input_size)
    elif args.model_name == 'vgg16_bn':
        model = VGG16BatchNorm(num_classes= num_classes, in_channels = 3, dropout_rate= 0.4, input_size=args.input_size)
    elif args.model_name == 'resnet18':
        model = resnet18(num_classes = num_classes, in_channels= 3)
    elif args.model_name == 'resnet34':
        model = resnet34(num_classes = num_classes, in_channels= 3)
    elif args.model_name == 'resnet50':
        model = resnet50(num_classes= num_classes, in_channels= 3)
    elif args.model_name == 'resnet101':
        model = resnet101(num_classes= num_classes, in_channels= 3)
    elif args.model_name == 'inceptionv3':
        model = InceptionV3(num_classes=num_classes, in_channels=3)
    elif args.model_name == 'mobilenetv3':
        model = MobileNetV3(mode = args.model_type, num_classes = num_classes, dropout=args.dropout_rate)
    elif args.model_name == 'vit':
        model = VisionTransformer(num_classes = num_classes, dropout_rate= args.dropout_rate)
    else:
        raise ValueError(f"Model {args.model_name} not recognized.")
    print(f"Model: {model}")
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not recognized.")
    
    if args.criterion == 'cross_entropy':
        if args.use_class_weights:
            class_counts = count_images_per_class(dataloaders['train'])
            class_weights = calculate_class_weights(class_counts, weight_type=args.weight_type)
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            criterion = nn.CrossEntropyLoss()
    elif args.criterion == "hinge":
        criterion = nn.MultiMarginLoss()      
    elif args.criterion == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Criterion {args.criterion} not recognized.")
        
    if args.scheduler == 'constant':
        scheduler = None
    elif args.scheduler == 'linear':
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.num_warmup_steps)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0)
        
    
    trainer = Trainer(model, dataloaders= dataloaders, dataset_sizes=dataset_sizes, criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device, num_epochs=args.num_epochs, save_path=args.save_path)
    model, history = trainer.train()
    trainer.plot_history()
    evaluate_model(model, dataloaders['test'], num_class = num_classes)
if __name__ == "__main__":
    args = parse_args()
    main(args)