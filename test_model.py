"""
Simple test script to verify the CaffeNet model implementation
"""

import torch
from models import caffenet


def test_model_creation():
    """Test model creation with different configurations"""
    print("Testing model creation...")
    
    # Test with default parameters
    model = caffenet(num_classes=10)
    assert model is not None, "Model creation failed"
    print("✓ Default model created successfully")
    
    # Test with custom dropout
    model = caffenet(num_classes=100, dropout=0.3)
    assert model is not None, "Model creation with custom dropout failed"
    print("✓ Model with custom dropout created successfully")
    
    # Test with different number of classes
    for num_classes in [2, 10, 100, 1000]:
        model = caffenet(num_classes=num_classes)
        assert model is not None, f"Model creation failed for {num_classes} classes"
    print(f"✓ Model creation tested for various class counts")


def test_forward_pass():
    """Test forward pass with different input sizes"""
    print("\nTesting forward pass...")
    
    model = caffenet(num_classes=10)
    model.eval()
    
    # Test with standard input size (227x227)
    x = torch.randn(1, 3, 227, 227)
    output = model(x)
    assert output.shape == (1, 10), f"Expected shape (1, 10), got {output.shape}"
    print(f"✓ Forward pass with 227x227 input: {x.shape} -> {output.shape}")
    
    # Test with batch
    x = torch.randn(4, 3, 227, 227)
    output = model(x)
    assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"
    print(f"✓ Forward pass with batch size 4: {x.shape} -> {output.shape}")
    
    # Test with different input size (should work due to adaptive pooling)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    assert output.shape == (1, 10), f"Expected shape (1, 10), got {output.shape}"
    print(f"✓ Forward pass with 256x256 input: {x.shape} -> {output.shape}")


def test_parameter_count():
    """Test that model has expected number of parameters"""
    print("\nTesting parameter count...")
    
    model = caffenet(num_classes=1000)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    assert total_params > 0, "Model has no parameters"
    assert trainable_params > 0, "Model has no trainable parameters"
    assert total_params == trainable_params, "Some parameters are frozen"


def test_gradients():
    """Test that gradients can be computed"""
    print("\nTesting gradient computation...")
    
    model = caffenet(num_classes=10)
    model.train()
    
    # Forward pass
    x = torch.randn(2, 3, 227, 227)
    output = model(x)
    
    # Compute loss
    target = torch.randint(0, 10, (2,))
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients are computed
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    assert has_gradients, "Gradients were not computed"
    print("✓ Gradients computed successfully")


def test_device_transfer():
    """Test model transfer to different devices"""
    print("\nTesting device transfer...")
    
    model = caffenet(num_classes=10)
    
    # Test CPU
    model = model.to('cpu')
    x = torch.randn(1, 3, 227, 227).to('cpu')
    output = model(x)
    assert output.device.type == 'cpu', "Model output not on CPU"
    print("✓ Model works on CPU")
    
    # Test GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')
        x = torch.randn(1, 3, 227, 227).to('cuda')
        output = model(x)
        assert output.device.type == 'cuda', "Model output not on CUDA"
        print("✓ Model works on CUDA")
    else:
        print("⚠ CUDA not available, skipping GPU test")


def test_eval_mode():
    """Test that model behaves differently in train vs eval mode"""
    print("\nTesting train/eval modes...")
    
    model = caffenet(num_classes=10)
    x = torch.randn(2, 3, 227, 227)
    
    # Train mode
    model.train()
    assert model.training, "Model not in training mode"
    output_train1 = model(x)
    output_train2 = model(x)
    print("✓ Model in training mode")
    
    # Eval mode
    model.eval()
    assert not model.training, "Model not in evaluation mode"
    output_eval1 = model(x)
    output_eval2 = model(x)
    
    # In eval mode with same input, outputs should be identical (no dropout randomness)
    assert torch.allclose(output_eval1, output_eval2), "Outputs differ in eval mode"
    print("✓ Model in evaluation mode")


def run_all_tests():
    """Run all tests"""
    print("="*80)
    print("Running CaffeNet Model Tests")
    print("="*80)
    
    try:
        test_model_creation()
        test_forward_pass()
        test_parameter_count()
        test_gradients()
        test_device_transfer()
        test_eval_mode()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        return True
        
    except AssertionError as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {str(e)}")
        print("="*80)
        return False
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ ERROR: {str(e)}")
        print("="*80)
        return False


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)

