"""
Evaluation script for CaffeNet image classification
Compute detailed metrics on test/validation set
"""

import os
import argparse
import json
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)

from models import caffenet
from utils.dataset import ImageClassificationDataset, get_transforms


def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        device: Device to load model on
        
    Returns:
        tuple: (model, class_to_idx)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_to_idx = checkpoint.get('class_to_idx', {})
    num_classes = len(class_to_idx)
    
    model = caffenet(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_to_idx


def evaluate_model(model, dataloader, device, class_to_idx):
    """
    Evaluate model on dataset
    
    Args:
        model: Neural network model
        dataloader: Data loader
        device: Device to evaluate on
        class_to_idx: Class to index mapping
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    
    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Processing batches"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Get predictions
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    
    avg_loss = running_loss / len(dataloader)
    
    # Create class name mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = \
        precision_recall_fscore_support(all_labels, all_predictions, average=None)
    
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i]),
            'f1_score': float(per_class_f1[i]),
            'support': int(per_class_support[i])
        }
    
    # Compile results
    results = {
        'overall': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'loss': float(avg_loss),
            'total_samples': len(all_labels)
        },
        'per_class': per_class_metrics,
        'confusion_matrix': conf_matrix.tolist(),
        'class_names': class_names
    }
    
    return results, all_predictions, all_labels, all_probabilities


def print_results(results):
    """
    Print evaluation results in a formatted way
    
    Args:
        results (dict): Evaluation results
    """
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Overall metrics
    print("\nOverall Metrics:")
    print("-" * 80)
    overall = results['overall']
    print(f"Accuracy:  {overall['accuracy']*100:.2f}%")
    print(f"Precision: {overall['precision']*100:.2f}%")
    print(f"Recall:    {overall['recall']*100:.2f}%")
    print(f"F1 Score:  {overall['f1_score']*100:.2f}%")
    print(f"Loss:      {overall['loss']:.4f}")
    print(f"Total Samples: {overall['total_samples']}")
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    
    for class_name, metrics in results['per_class'].items():
        print(f"{class_name:<20} "
              f"{metrics['precision']*100:<11.2f}% "
              f"{metrics['recall']*100:<11.2f}% "
              f"{metrics['f1_score']*100:<11.2f}% "
              f"{metrics['support']:<10}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print("-" * 80)
    conf_matrix = np.array(results['confusion_matrix'])
    class_names = results['class_names']
    
    # Print header
    print(f"{'True\\Pred':<15}", end="")
    for name in class_names:
        print(f"{name[:10]:<12}", end="")
    print()
    
    # Print matrix
    for i, name in enumerate(class_names):
        print(f"{name[:15]:<15}", end="")
        for j in range(len(class_names)):
            print(f"{conf_matrix[i, j]:<12}", end="")
        print()
    
    print("="*80 + "\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Evaluate CaffeNet model')
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='val',
                       help='Dataset split to evaluate (train/val/test)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--input-size', type=int, default=227,
                       help='Input image size')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'\nLoading model from {args.checkpoint}...')
    model, class_to_idx = load_model(args.checkpoint, device)
    print(f'Classes: {list(class_to_idx.keys())}')
    
    # Create dataset
    print(f'\nLoading {args.split} dataset from {args.data_dir}...')
    transforms_dict = get_transforms(input_size=args.input_size, augment=False)
    
    dataset = ImageClassificationDataset(
        root_dir=args.data_dir,
        split=args.split,
        transform=transforms_dict['val'],
        class_to_idx=class_to_idx
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Evaluate
    results, predictions, labels, probabilities = evaluate_model(
        model, dataloader, device, class_to_idx
    )
    
    # Print results
    print_results(results)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()

