# Real-time Object Detection System

A professional computer vision application that provides real-time object detection, image detection, and custom model training capabilities using YOLOv8 and Streamlit.

![Application Screenshot](https://via.placeholder.com/800x400?text=Application+Screenshot)

## Features

### Real-time Object Detection
- 🎥 Live video stream processing
- 📷 Support for webcam and video file input
- 🎯 Multiple YOLOv8 model options
- 📊 Real-time object counting and statistics
- 🎨 Professional web interface

### Image Detection
- 🖼️ Single image object detection
- 📸 Support for JPG, JPEG, and PNG formats
- 🔍 Side-by-side original and annotated views
- 📊 Detailed object detection results
- 🎯 Custom model support

### Custom Model Training
- 🏋️‍♂️ Train YOLOv8 models with custom datasets
- 📈 Adjustable training parameters
- 📊 Real-time training metrics
- 💾 Automatic model saving
- 🔄 Easy model switching

## Screenshots

### Real-time Detection
![Real-time Detection](screenshots/realtime_detection.png)
*Real-time object detection using webcam feed*

### Image Detection
![Image Detection](screenshots/image_detection.png)
*Object detection in uploaded images*

### Training Interface
![Training Interface](screenshots/training_interface.png)
*Custom model training interface*

### Detection Results
![Detection Results](screenshots/detection_results.png)
*Detailed detection results and object counts*

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd computer-vision
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Application
```bash
streamlit run app.py
```

The application will open in your default web browser with three main tabs.

### Real-time Detection

1. Select a model from the sidebar:
   - YOLOv8n (Fastest)
   - YOLOv8s
   - YOLOv8m
   - YOLOv8l
   - YOLOv8x (Most Accurate)

2. Adjust the confidence threshold (0.0-1.0)

3. Choose input source:
   - Webcam
   - Video file

4. Click "Start Detection"

### Image Detection

1. Switch to the "Image Detection" tab

2. Upload an image (JPG, JPEG, or PNG)

3. Configure detection settings:
   - Select model size
   - Adjust confidence threshold
   - Option to use custom model

4. View results:
   - Original image
   - Annotated image with detections
   - Object counts and statistics

### Training Custom Models

1. Switch to the "Training" tab

2. Configure training settings:
   - Model size (n, s, m, l, x)
   - Number of epochs
   - Batch size
   - Image size
   - Class names

3. Upload your dataset:
   - Training images (JPG, JPEG, PNG)
   - Training labels (TXT)
   - Validation images
   - Validation labels

4. Click "Start Training"

## Dataset Preparation

### Image Requirements
- Format: JPG, JPEG, PNG
- Recommended size: 640x640 pixels
- Minimum size: 32x32 pixels
- Maximum size: No limit (will be resized)

### Label Format
YOLO format (one .txt file per image):
```
class_id x_center y_center width height
```
Example:
```
0 0.5 0.5 0.2 0.2
```

### Directory Structure
```
custom_dataset/
├── images/
│   ├── train/  # Training images
│   └── val/    # Validation images
└── labels/
    ├── train/  # Training labels
    └── val/    # Validation labels
```

## Training Parameters

### Model Size
- n: Fastest, least accurate
- s: Good balance of speed and accuracy
- m: Better accuracy, slower
- l: High accuracy, slower
- x: Highest accuracy, slowest

### Other Parameters
- Epochs: Number of training iterations
- Batch Size: Images processed per batch
- Image Size: Input resolution
- Confidence Threshold: Detection sensitivity

## Using Custom Models

1. After training, models are saved in:
```
custom_model/yolov8{size}/weights/best.pt
```

2. To use a custom model:
   - Copy the model to your working directory
   - Select it in the model dropdown
   - Or upload it in the Image Detection tab

## Best Practices

### For Better Detection
- Use appropriate model size for your needs
- Adjust confidence threshold based on requirements
- Ensure good lighting conditions
- Use high-quality images/video
- For images, use clear, well-lit photos
- For video, ensure stable camera position

### For Better Training
- Use diverse training images
- Include various lighting conditions
- Add different angles and scales
- Include edge cases
- Balance your dataset
- Start with a small model (yolov8n)
- Monitor validation metrics

## Troubleshooting

### Common Issues
1. Webcam not working
   - Check camera permissions
   - Verify camera is not in use by another application

2. Training errors
   - Verify dataset format
   - Check file permissions
   - Ensure sufficient disk space

3. Performance issues
   - Reduce batch size
   - Use smaller model
   - Decrease image size

4. Image detection issues
   - Check image format
   - Ensure image is not corrupted
   - Try different confidence threshold

## Requirements

- Python 3.8+
- Webcam (for live detection)
- CUDA-capable GPU (recommended)
- Sufficient RAM (8GB minimum)
- Disk space for models and datasets

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Streamlit](https://streamlit.io/) for the web interface
- [OpenCV](https://opencv.org/) for computer vision capabilities

## Support

For support, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue if needed

## Roadmap

- [ ] Add model export options
- [ ] Implement model quantization
- [ ] Add batch processing
- [ ] Support for more model architectures
- [ ] Enhanced visualization options
- [ ] Batch image processing
- [ ] Advanced image filtering options
- [ ] Export detection results to CSV/JSON 