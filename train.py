import os
from ultralytics import YOLO
import yaml

def prepare_dataset(data_dir):
    """
    Prepare the dataset structure for YOLOv8 training.
    Expected structure:
    data_dir/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    """
    # Create necessary directories if they don't exist
    os.makedirs(os.path.join(data_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'labels', 'val'), exist_ok=True)

def create_dataset_yaml(data_dir, class_names):
    """
    Create dataset.yaml file required for YOLOv8 training.
    """
    yaml_content = {
        'path': os.path.abspath(data_dir),  # dataset root dir
        'train': 'images/train',  # train images relative to 'path'
        'val': 'images/val',  # val images relative to 'path'
        'names': {i: name for i, name in enumerate(class_names)}  # class names
    }
    
    with open(os.path.join(data_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    
    return os.path.join(data_dir, 'dataset.yaml')

def train_model(data_yaml, model_size='n', epochs=100, batch_size=16, img_size=640):
    """
    Train a YOLOv8 model on custom data.
    
    Args:
        data_yaml (str): Path to dataset.yaml file
        model_size (str): Model size ('n', 's', 'm', 'l', 'x')
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        img_size (int): Input image size
    """
    # Initialize model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device='0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',  # Use GPU if available
        project='custom_model',
        name=f'yolov8{model_size}',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        patience=50,  # Early stopping patience
        save=True,  # Save checkpoints
        save_period=10,  # Save checkpoint every 10 epochs
    )
    
    return results

if __name__ == "__main__":
    # Example usage
    data_dir = 'custom_dataset'  # Your dataset directory
    class_names = ['class1', 'class2', 'class3']  # Replace with your class names
    
    # Prepare dataset structure
    prepare_dataset(data_dir)
    
    # Create dataset.yaml
    data_yaml = create_dataset_yaml(data_dir, class_names)
    
    # Train model
    train_model(
        data_yaml=data_yaml,
        model_size='n',  # Choose from 'n', 's', 'm', 'l', 'x'
        epochs=100,
        batch_size=16,
        img_size=640
    ) 