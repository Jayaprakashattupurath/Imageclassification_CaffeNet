# Quick Start Guide

Get up and running with CaffeNet image classification in 5 minutes!

## 🚀 Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/Imageclassification_CaffeNet.git
cd Imageclassification_CaffeNet

# Install dependencies
pip install -r requirements.txt
```

## 🧪 Test the Model (1 minute)

Verify everything is working:

```bash
python test_model.py
```

You should see:
```
✅ ALL TESTS PASSED!
```

## 🎯 Run Sample Use Case (2 minutes)

### Step 1: Create Sample Dataset
```bash
python examples/sample_use_case.py
```

This creates a fruit classification dataset with 5 classes.

### Step 2: Train the Model (Quick Demo)
```bash
python train.py --data-dir data_fruits --epochs 5 --batch-size 16 --save-dir checkpoints_fruits
```

For better results, increase epochs to 20-50.

### Step 3: Make Predictions
```bash
python predict.py --image-path data_fruits/val --checkpoint checkpoints_fruits/best_model.pth --top-k 3
```

## 📊 Your Own Dataset

### 1. Organize Your Data
```
my_data/
├── train/
│   ├── cat/
│   │   ├── cat1.jpg
│   │   └── cat2.jpg
│   └── dog/
│       ├── dog1.jpg
│       └── dog2.jpg
└── val/
    ├── cat/
    │   └── cat_test.jpg
    └── dog/
        └── dog_test.jpg
```

### 2. Train
```bash
python train.py --data-dir my_data --epochs 50 --batch-size 32
```

### 3. Predict
```bash
python predict.py --image-path path/to/image.jpg --checkpoint checkpoints/best_model.pth
```

## 📈 Monitor Training

During training, you'll see:
```
Epoch [1/50], Step [10/25], Loss: 2.3456, Acc: 45.67%
...
Validation - Loss: 2.1234, Accuracy: 48.50%
Best model saved to checkpoints/best_model.pth
```

## 🔍 Evaluate Model

```bash
python evaluate.py --data-dir my_data --split val --checkpoint checkpoints/best_model.pth
```

## 🎓 Next Steps

1. **Tune Hyperparameters**: Adjust learning rate, batch size, epochs
2. **Add More Data**: Collect more training samples for better accuracy
3. **Customize**: Modify the model or training process
4. **Deploy**: Use the model in your application

## 💡 Tips

- **GPU**: Training is much faster with CUDA-enabled GPU
- **Batch Size**: Larger batch sizes train faster but need more memory
- **Epochs**: More epochs = better accuracy (but watch for overfitting)
- **Data**: More diverse training data = better generalization

## ❓ Need Help?

- See full documentation: [README.md](README.md)
- Check common issues: [README.md#troubleshooting](README.md#-troubleshooting)
- Run tests: `python test_model.py`

---

**That's it! You're ready to classify images with CaffeNet!** 🎉

