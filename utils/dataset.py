"""
Dataset utilities for loading and preprocessing image data
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json


class ImageClassificationDataset(Dataset):
    """
    Custom dataset for image classification
    """
    
    def __init__(self, root_dir, split='train', transform=None, class_to_idx=None):
        """
        Initialize the dataset
        
        Args:
            root_dir (str): Root directory containing images
            split (str): Dataset split ('train', 'val', or 'test')
            transform (callable, optional): Optional transform to be applied
            class_to_idx (dict, optional): Dictionary mapping class names to indices
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = class_to_idx or {}
        
        # Load dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load all image paths and labels"""
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Dataset directory {self.root_dir} does not exist")
        
        # Get all class directories
        classes = sorted([d for d in os.listdir(self.root_dir) 
                         if os.path.isdir(os.path.join(self.root_dir, d))])
        
        if not classes:
            raise ValueError(f"No class directories found in {self.root_dir}")
        
        # Create class to index mapping if not provided
        if not self.class_to_idx:
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        
        self.classes = classes
        
        # Load all samples
        for class_name in classes:
            if class_name not in self.class_to_idx:
                continue
                
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(self.root_dir, class_name)
            
            # Get all image files
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        print(f"Loaded {len(self.samples)} samples from {len(classes)} classes")
    
    def __len__(self):
        """Return the total number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label) where image is transformed PIL image
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(input_size=227, augment=True):
    """
    Get data transforms for training and validation
    
    Args:
        input_size (int): Size to resize images to
        augment (bool): Whether to apply data augmentation
        
    Returns:
        dict: Dictionary containing 'train' and 'val' transforms
    """
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return {'train': train_transform, 'val': val_transform}


def create_dataloaders(data_dir, batch_size=32, num_workers=4, input_size=227):
    """
    Create data loaders for training and validation
    
    Args:
        data_dir (str): Root directory of the dataset
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        input_size (int): Input image size
        
    Returns:
        tuple: (train_loader, val_loader, class_to_idx)
    """
    transforms_dict = get_transforms(input_size=input_size, augment=True)
    
    # Create datasets
    train_dataset = ImageClassificationDataset(
        root_dir=data_dir,
        split='train',
        transform=transforms_dict['train']
    )
    
    val_dataset = ImageClassificationDataset(
        root_dir=data_dir,
        split='val',
        transform=transforms_dict['val'],
        class_to_idx=train_dataset.class_to_idx
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.class_to_idx

