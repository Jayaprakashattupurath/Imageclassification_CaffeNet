"""
Sample Use Case: Fruit Classification System

This example demonstrates how to use CaffeNet for classifying images of fruits.
The system can identify 5 different types of fruits: Apple, Banana, Orange, Strawberry, and Grape.

Use Case Scenario:
    A grocery store wants to automate their fruit sorting system. They need to classify
    incoming fruits into different categories for inventory management and pricing.
    
    The system can:
    1. Train on labeled fruit images
    2. Classify new fruit images with high accuracy
    3. Provide confidence scores for each prediction
    4. Process multiple images in batch
"""

import os
import sys
import shutil
from PIL import Image, ImageDraw, ImageFont
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_sample_dataset(data_dir='data_fruits', num_samples_per_class=50):
    """
    Create a sample fruit classification dataset with synthetic images
    
    Args:
        data_dir (str): Directory to create dataset in
        num_samples_per_class (int): Number of samples per class
    """
    print("Creating sample fruit classification dataset...")
    
    # Define fruit classes and their colors
    fruits = {
        'apple': [(220, 20, 60), (255, 0, 0), (139, 0, 0)],  # Red shades
        'banana': [(255, 255, 0), (255, 215, 0), (255, 239, 0)],  # Yellow shades
        'orange': [(255, 140, 0), (255, 165, 0), (255, 127, 80)],  # Orange shades
        'strawberry': [(255, 0, 127), (220, 20, 60), (255, 20, 147)],  # Pink-red shades
        'grape': [(128, 0, 128), (147, 112, 219), (138, 43, 226)]  # Purple shades
    }
    
    splits = ['train', 'val']
    split_ratios = {'train': 0.8, 'val': 0.2}
    
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        num_samples = int(num_samples_per_class * split_ratios[split])
        
        for fruit_name, colors in fruits.items():
            class_dir = os.path.join(split_dir, fruit_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for i in range(num_samples):
                # Create a simple colored circle to represent fruit
                img = Image.new('RGB', (227, 227), color=(240, 240, 240))
                draw = ImageDraw.Draw(img)
                
                # Random color from fruit's color palette
                color = random.choice(colors)
                
                # Draw circle with some randomness
                center_x = random.randint(100, 127)
                center_y = random.randint(100, 127)
                radius = random.randint(50, 70)
                
                draw.ellipse([center_x - radius, center_y - radius,
                            center_x + radius, center_y + radius],
                           fill=color, outline=(0, 0, 0))
                
                # Add some texture (small circles)
                for _ in range(random.randint(3, 8)):
                    x = random.randint(center_x - radius//2, center_x + radius//2)
                    y = random.randint(center_y - radius//2, center_y + radius//2)
                    r = random.randint(3, 8)
                    texture_color = tuple(max(0, c - random.randint(20, 40)) for c in color)
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=texture_color)
                
                # Save image
                img_path = os.path.join(class_dir, f'{fruit_name}_{i:04d}.png')
                img.save(img_path)
            
            print(f"  Created {num_samples} {fruit_name} images for {split} set")
    
    print(f"\nDataset created at: {data_dir}")
    print(f"Total images: {num_samples_per_class * len(fruits)}")
    print(f"Classes: {list(fruits.keys())}\n")


def run_training_example():
    """Run training example on fruit dataset"""
    import subprocess
    
    print("="*80)
    print("STEP 1: Training CaffeNet on Fruit Classification Dataset")
    print("="*80)
    
    cmd = [
        'python', 'train.py',
        '--data-dir', 'data_fruits',
        '--epochs', '20',
        '--batch-size', '16',
        '--lr', '0.001',
        '--save-dir', 'checkpoints_fruits'
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}\n")
    print("Note: This is a demonstration. For actual training, run the command above.")
    print("Training would take several minutes depending on your hardware.\n")


def run_inference_example():
    """Run inference example"""
    print("="*80)
    print("STEP 2: Running Inference on Test Images")
    print("="*80)
    
    cmd = [
        'python', 'predict.py',
        '--image-path', 'data_fruits/val',
        '--checkpoint', 'checkpoints_fruits/best_model.pth',
        '--top-k', '3',
        '--output', 'predictions.json'
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}\n")
    print("This will classify all validation images and save results to predictions.json")


def print_use_case_info():
    """Print detailed use case information"""
    print("\n" + "="*80)
    print("FRUIT CLASSIFICATION SYSTEM - USE CASE DEMONSTRATION")
    print("="*80)
    
    print("""
    📦 USE CASE: Automated Fruit Sorting System
    
    BUSINESS PROBLEM:
        A large grocery distribution center receives thousands of fruit items daily.
        Manual sorting is time-consuming and prone to errors. They need an automated
        system to classify fruits accurately for:
        - Inventory management
        - Quality control
        - Automated pricing
        - Storage optimization
    
    SOLUTION:
        Implement a CaffeNet-based image classification system that can:
        ✓ Identify 5 types of fruits: Apple, Banana, Orange, Strawberry, Grape
        ✓ Process images in real-time
        ✓ Provide confidence scores for quality assurance
        ✓ Scale to handle high volumes
    
    WORKFLOW:
        1. Data Collection: Gather labeled images of fruits
        2. Training: Train CaffeNet model on the dataset
        3. Validation: Validate model accuracy on test set
        4. Deployment: Deploy model for real-time inference
        5. Monitoring: Track performance and retrain as needed
    
    EXPECTED RESULTS:
        - Classification accuracy: >90% on validation set
        - Inference time: <100ms per image
        - Scalable to additional fruit categories
    
    """)


def main():
    """Main function to run the sample use case"""
    print_use_case_info()
    
    # Create sample dataset
    if not os.path.exists('data_fruits'):
        create_sample_dataset('data_fruits', num_samples_per_class=50)
    else:
        print("Sample dataset already exists at 'data_fruits'\n")
    
    # Show training example
    run_training_example()
    
    # Show inference example
    run_inference_example()
    
    print("\n" + "="*80)
    print("GETTING STARTED")
    print("="*80)
    print("""
    To run this example yourself:
    
    1. Install dependencies:
       pip install -r requirements.txt
    
    2. Create or use the sample dataset:
       python examples/sample_use_case.py
    
    3. Train the model:
       python train.py --data-dir data_fruits --epochs 20 --batch-size 16 \\
              --save-dir checkpoints_fruits
    
    4. Run predictions:
       python predict.py --image-path data_fruits/val/apple/apple_0001.png \\
              --checkpoint checkpoints_fruits/best_model.pth
    
    5. Evaluate on entire validation set:
       python predict.py --image-path data_fruits/val \\
              --checkpoint checkpoints_fruits/best_model.pth --output results.json
    
    """)
    
    print("="*80)
    print("CUSTOMIZATION")
    print("="*80)
    print("""
    To adapt this for your own use case:
    
    1. Prepare your dataset:
       - Organize images in folders: data/train/class1, data/train/class2, ...
       - Create validation set: data/val/class1, data/val/class2, ...
    
    2. Adjust hyperparameters:
       - Modify learning rate, batch size, epochs in train.py
       - Adjust data augmentation in utils/dataset.py
    
    3. Fine-tune the model:
       - Load pretrained weights if available
       - Adjust dropout rate for regularization
       - Experiment with different optimizers
    
    For more information, see README.md
    """)


if __name__ == '__main__':
    main()

