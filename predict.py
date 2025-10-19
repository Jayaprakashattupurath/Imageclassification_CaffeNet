"""
Inference script for CaffeNet image classification
"""

import os
import argparse
import json

import torch
from PIL import Image
import torchvision.transforms as transforms

from models import caffenet


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
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get class mapping
    class_to_idx = checkpoint.get('class_to_idx', {})
    num_classes = len(class_to_idx)
    
    # Create model
    model = caffenet(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Best validation accuracy: {checkpoint.get('best_acc', 0):.2f}%")
    
    return model, class_to_idx


def preprocess_image(image_path, input_size=227):
    """
    Preprocess an image for inference
    
    Args:
        image_path (str): Path to image file
        input_size (int): Size to resize image to
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def predict(model, image_tensor, class_to_idx, device, top_k=5):
    """
    Make prediction on an image
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        class_to_idx: Class to index mapping
        device: Device to run inference on
        top_k (int): Number of top predictions to return
        
    Returns:
        list: List of tuples (class_name, probability)
    """
    # Create reverse mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Move image to device
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Get top k predictions
    top_probs, top_indices = torch.topk(probabilities, min(top_k, len(class_to_idx)))
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    
    # Convert to class names
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        class_name = idx_to_class.get(idx, f"Class_{idx}")
        predictions.append((class_name, float(prob)))
    
    return predictions


def predict_image(args):
    """
    Main prediction function
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    
    # Load model
    print('Loading model...')
    model, class_to_idx = load_model(args.checkpoint, device)
    print(f'Classes: {list(class_to_idx.keys())}\n')
    
    # Process image(s)
    if os.path.isfile(args.image_path):
        # Single image
        image_paths = [args.image_path]
    elif os.path.isdir(args.image_path):
        # Directory of images
        image_paths = [
            os.path.join(args.image_path, f)
            for f in os.listdir(args.image_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]
    else:
        raise ValueError(f"Invalid image path: {args.image_path}")
    
    if not image_paths:
        print("No images found!")
        return
    
    print(f'Processing {len(image_paths)} image(s)...\n')
    
    # Make predictions
    results = []
    for image_path in image_paths:
        print(f'Image: {image_path}')
        
        try:
            # Preprocess image
            image_tensor = preprocess_image(image_path, input_size=args.input_size)
            
            # Make prediction
            predictions = predict(model, image_tensor, class_to_idx, device, top_k=args.top_k)
            
            # Print results
            print(f'Top {len(predictions)} predictions:')
            for i, (class_name, prob) in enumerate(predictions, 1):
                print(f'  {i}. {class_name}: {prob*100:.2f}%')
            
            results.append({
                'image': image_path,
                'predictions': [{'class': cls, 'probability': prob} for cls, prob in predictions]
            })
            
        except Exception as e:
            print(f'  Error: {str(e)}')
        
        print()
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Results saved to {args.output}')


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Predict image class using trained CaffeNet')
    
    parser.add_argument('--image-path', type=str, required=True,
                       help='Path to image file or directory of images')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint (default: checkpoints/best_model.pth)')
    parser.add_argument('--input-size', type=int, default=227,
                       help='Input image size (default: 227)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top predictions to show (default: 5)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save prediction results (JSON format)')
    
    args = parser.parse_args()
    
    predict_image(args)


if __name__ == '__main__':
    main()

