# Training Custom YOLOv8 Model

This guide explains how to train a YOLOv8 model with your own custom dataset.

## Dataset Preparation

1. Create a dataset directory structure:
```
custom_dataset/
├── images/
│   ├── train/  # Training images
│   └── val/    # Validation images
└── labels/
    ├── train/  # Training labels
    └── val/    # Validation labels
```

2. Image Requirements:
- Supported formats: JPG, JPEG, PNG
- Recommended size: 640x640 pixels
- Minimum size: 32x32 pixels
- Maximum size: No limit, but larger images will be resized

3. Label Format:
- YOLO format: One .txt file per image
- Each line in the .txt file represents one object
- Format: `class_id x_center y_center width height`
- All values should be normalized (0-1)
- Example: `0 0.5 0.5 0.2 0.2` (class 0, center at (0.5,0.5), width=0.2, height=0.2)

## Training Process

1. Install required packages:
```bash
pip install ultralytics pyyaml
```

2. Prepare your dataset:
- Place your images in `images/train/` and `images/val/`
- Place your labels in `labels/train/` and `labels/val/`
- Make sure image and label filenames match (except extension)

3. Modify the training script:
- Open `train.py`
- Update `class_names` with your class names
- Adjust training parameters as needed:
  - `model_size`: 'n' (fastest) to 'x' (most accurate)
  - `epochs`: Number of training iterations
  - `batch_size`: Number of images per batch
  - `img_size`: Input image size

4. Start training:
```bash
python train.py
```

## Training Parameters

- `model_size`: Choose from 'n', 's', 'm', 'l', 'x'
  - 'n': Fastest, least accurate
  - 'x': Slowest, most accurate
- `epochs`: Number of complete passes through the dataset
  - Start with 100 epochs
  - Increase if underfitting
  - Decrease if overfitting
- `batch_size`: Number of images processed at once
  - Depends on your GPU memory
  - Typical values: 16, 32, 64
- `img_size`: Input image size
  - Default: 640
  - Larger sizes: Better accuracy, slower training
  - Smaller sizes: Faster training, less accuracy

## Monitoring Training

During training, you'll see:
- Loss values (decreasing is good)
- mAP (mean Average Precision)
- Training progress
- Checkpoints saved in `custom_model/`

## Using the Trained Model

1. Find your trained model:
- Located in `custom_model/yolov8{size}/weights/best.pt`

2. Use in your application:
```python
model = YOLO('custom_model/yolov8n/weights/best.pt')
results = model(image)
```

## Tips for Better Results

1. Dataset Quality:
- Use diverse images
- Include various lighting conditions
- Add different angles and scales
- Include edge cases

2. Training Tips:
- Start with a small model (yolov8n)
- Use data augmentation
- Monitor validation loss
- Use early stopping
- Try different learning rates

3. Common Issues:
- Overfitting: Reduce epochs, increase dropout
- Underfitting: Increase epochs, use larger model
- Slow training: Reduce batch size, use smaller images
- Memory issues: Reduce batch size, use smaller model

## Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Dataset Preparation Guide](https://docs.ultralytics.com/datasets/detect/)
- [Training Parameters](https://docs.ultralytics.com/modes/train/) 