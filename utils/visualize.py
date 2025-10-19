"""
Visualization utilities for training results and predictions
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_training_history(history_path, save_path=None):
    """
    Plot training and validation metrics
    
    Args:
        history_path (str): Path to training_history.json
        save_path (str, optional): Path to save the plot
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(results_path, save_path=None):
    """
    Plot confusion matrix from evaluation results
    
    Args:
        results_path (str): Path to evaluation_results.json
        save_path (str, optional): Path to save the plot
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    conf_matrix = np.array(results['confusion_matrix'])
    class_names = results['class_names']
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_class_metrics(results_path, save_path=None):
    """
    Plot per-class metrics (precision, recall, F1-score)
    
    Args:
        results_path (str): Path to evaluation_results.json
        save_path (str, optional): Path to save the plot
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    per_class = results['per_class']
    class_names = list(per_class.keys())
    
    # Extract metrics
    precision = [per_class[cls]['precision'] * 100 for cls in class_names]
    recall = [per_class[cls]['recall'] * 100 for cls in class_names]
    f1_score = [per_class[cls]['f1_score'] * 100 for cls in class_names]
    
    # Create figure
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class metrics plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_report(history_path, results_path, output_dir='reports'):
    """
    Create a comprehensive visualization report
    
    Args:
        history_path (str): Path to training_history.json
        results_path (str): Path to evaluation_results.json
        output_dir (str): Directory to save reports
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Generating visualization report...")
    
    # Plot training history
    plot_training_history(
        history_path,
        save_path=f"{output_dir}/training_history.png"
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        results_path,
        save_path=f"{output_dir}/confusion_matrix.png"
    )
    
    # Plot class metrics
    plot_class_metrics(
        results_path,
        save_path=f"{output_dir}/class_metrics.png"
    )
    
    print(f"\n✅ Report generated in '{output_dir}' directory")
    print(f"   - training_history.png")
    print(f"   - confusion_matrix.png")
    print(f"   - class_metrics.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize training and evaluation results')
    parser.add_argument('--history', type=str, help='Path to training_history.json')
    parser.add_argument('--results', type=str, help='Path to evaluation_results.json')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    if args.history and args.results:
        create_report(args.history, args.results, args.output_dir)
    elif args.history:
        plot_training_history(args.history)
    elif args.results:
        plot_confusion_matrix(args.results)
        plot_class_metrics(args.results)
    else:
        print("Please provide --history and/or --results arguments")

