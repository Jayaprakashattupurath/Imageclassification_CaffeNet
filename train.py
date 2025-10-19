"""
Training script for CaffeNet image classification
"""

import os
import argparse
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from models import caffenet
from utils.dataset import create_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch}], Step [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100 * correct / total:.2f}%')
    
    epoch_time = time.time() - start_time
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    print(f'Epoch [{epoch}] completed in {epoch_time:.2f}s - '
          f'Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Validate the model
    
    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    print(f'Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, epoch, best_acc, save_dir, class_to_idx):
    """
    Save model checkpoint
    
    Args:
        model: Neural network model
        optimizer: Optimizer
        epoch: Current epoch
        best_acc: Best validation accuracy
        save_dir: Directory to save checkpoint
        class_to_idx: Class to index mapping
    """
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'class_to_idx': class_to_idx
    }
    
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved to {checkpoint_path}')
    
    # Save best model separately
    best_path = os.path.join(save_dir, 'best_model.pth')
    torch.save(checkpoint, best_path)
    print(f'Best model saved to {best_path}')


def train(args):
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
    print('\nLoading dataset...')
    train_loader, val_loader, class_to_idx = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size
    )
    
    num_classes = len(class_to_idx)
    print(f'Number of classes: {num_classes}')
    print(f'Classes: {list(class_to_idx.keys())}')
    
    # Save class mapping
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'class_to_idx.json'), 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    
    # Create model
    print('\nInitializing model...')
    model = caffenet(num_classes=num_classes, dropout=args.dropout)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    if args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    else:
        scheduler = None
    
    # Training loop
    print('\nStarting training...')
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{args.epochs}')
        print(f'{"="*60}')
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        if scheduler:
            if args.scheduler == 'plateau':
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save checkpoint if best accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, best_acc, args.save_dir, class_to_idx)
        
        # Save training history
        with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    print(f'\n{"="*60}')
    print(f'Training completed!')
    print(f'Best validation accuracy: {best_acc:.2f}%')
    print(f'{"="*60}')


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train CaffeNet for image classification')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Path to dataset directory')
    parser.add_argument('--input-size', type=int, default=227,
                       help='Input image size (default: 227)')
    
    # Model parameters
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout probability (default: 0.5)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay (default: 5e-4)')
    
    # Scheduler parameters
    parser.add_argument('--scheduler', type=str, default='step',
                       choices=['step', 'plateau', 'none'],
                       help='Learning rate scheduler (default: step)')
    parser.add_argument('--step-size', type=int, default=10,
                       help='Step size for StepLR scheduler (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Gamma for StepLR scheduler (default: 0.1)')
    
    # Other parameters
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints (default: checkpoints)')
    
    args = parser.parse_args()
    
    # Print configuration
    print('Configuration:')
    print('-' * 60)
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print('-' * 60)
    
    # Start training
    train(args)


if __name__ == '__main__':
    main()

