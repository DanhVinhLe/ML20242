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


class Trainer:
    def __init__(self, model, dataloaders, dataset_sizes, criterion, optimizer, scheduler = None, 
                 device = None, num_epochs = 25, save_path = None):
        super().__init__()
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.save_path = save_path
        self.best_model = copy.deepcopy(model.state_dict())
        self.best_acc = 0.0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        self.model = model.to(self.device)
    
    def train(self):
        since = time.time()
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            print('-' * 10)
            epoch_start = time.time()
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                    
                running_loss = 0.0
                running_corrects = 0
                dataloader = self.dataloaders[phase]
                progress_bar = tqdm(dataloader, desc=f"{phase} epoch {epoch+1}/{self.num_epochs}", unit="batch")
                seen_samples = 0
                for inputs, labels in progress_bar:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        if isinstance(outputs, tuple) and len(outputs) == 2:
                            main_output, aux_output = outputs
                            main_loss = self.criterion(main_output, labels)
                            aux_loss = self.criterion(aux_output, labels)
                            loss = main_loss + 0.4 * aux_loss
                            _, preds = torch.max(main_output, 1)
                        else:
                            loss = self.criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)
                        
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                        
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        seen_samples += inputs.size(0)
                        current_loss = running_loss / seen_samples
                        progress_bar.set_postfix(loss = f"{current_loss:.4f}")
                    
                if phase == 'train' and self.scheduler is not None:
                    self.scheduler.step()
                    
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                
                if phase == 'train':
                    self.history['train_loss'].append(epoch_loss)
                    self.history['train_acc'].append(epoch_acc.item())
                else:
                    self.history['val_loss'].append(epoch_loss)
                    self.history['val_acc'].append(epoch_acc.item())
                    
                if phase == 'test':
                    if epoch_acc > self.best_acc:
                        self.best_acc = epoch_acc
                        self.best_model = copy.deepcopy(self.model.state_dict())
                        
                    if epoch_loss < self.best_val_loss:
                        self.best_val_loss = epoch_loss
                        if self.save_path:
                            torch.save(self.model.state_dict(), self.save_path)
                            print(f"Model saved to {self.save_path}")
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                        
            epoch_end = time.time()
            print(f'Epoch {epoch+1} completed in {epoch_end - epoch_start:.0f} seconds')
            
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best test Acc: {self.best_acc:.4f}')
        print(f'Best test Loss: {self.best_val_loss:.4f}')
        
        self.model.load_state_dict(self.best_model)
        return self.model, self.history
    def plot_history(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()
                      
                            
