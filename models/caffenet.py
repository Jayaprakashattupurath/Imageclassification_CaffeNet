"""
CaffeNet Architecture Implementation in PyTorch
CaffeNet is a variant of AlexNet used in the Caffe deep learning framework
"""

import torch
import torch.nn as nn


class CaffeNet(nn.Module):
    """
    CaffeNet architecture for image classification
    
    Architecture:
    - 5 Convolutional layers
    - 3 Fully connected layers
    - ReLU activations
    - Dropout for regularization
    - Max pooling layers
    """
    
    def __init__(self, num_classes=1000, dropout=0.5):
        """
        Initialize CaffeNet model
        
        Args:
            num_classes (int): Number of output classes
            dropout (float): Dropout probability
        """
        super(CaffeNet, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv1: 227x227x3 -> 55x55x96
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2: 55x55x96 -> 27x27x256
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3: 27x27x256 -> 13x13x384
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: 13x13x384 -> 13x13x384
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: 13x13x384 -> 13x13x256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, H, W)
            
        Returns:
            torch.Tensor: Output logits of shape (N, num_classes)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def caffenet(num_classes=1000, pretrained=False, **kwargs):
    """
    CaffeNet model constructor
    
    Args:
        num_classes (int): Number of classes for classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        **kwargs: Additional arguments for the model
        
    Returns:
        CaffeNet: Initialized model
    """
    model = CaffeNet(num_classes=num_classes, **kwargs)
    
    if pretrained:
        # Note: You would need to download pretrained weights separately
        print("Warning: Pretrained weights not implemented. Training from scratch.")
    
    return model


if __name__ == "__main__":
    # Test the model
    model = caffenet(num_classes=10)
    print(model)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 227, 227)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

