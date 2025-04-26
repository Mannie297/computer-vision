# Import necessary libraries
import streamlit as st  # For creating the web interface
import cv2  # OpenCV for image processing and video capture
from ultralytics import YOLO  # YOLOv8 model for object detection
import numpy as np  # For numerical operations
from PIL import Image  # For image processing
import tempfile  # For handling temporary files
import os  # For file operations
import yaml
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import time

# Configure the Streamlit page settings
st.set_page_config(
    page_title="Real-time Object Detection",  # Set the page title
    page_icon="🎥",  # Set the page icon
    layout="wide"  # Use wide layout for better space utilization
)

# Add custom CSS to improve the UI appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;  # Add padding to the main content area
    }
    .stButton>button {
        width: 100%;  # Make buttons full width
        border-radius: 5px;  # Add rounded corners to buttons
        height: 3em;  # Set button height
    }
    .stSelectbox {
        width: 100%;  # Make select boxes full width
    }
    .stTextInput {
        width: 100%;
    }
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1rem 0;
    }
    .detection-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Add title and description to the main page
st.title("🎥 Real-time Object Detection System")  # Main title
st.markdown("""
    This application performs real-time object detection using YOLOv8.
    Select your model and confidence threshold, then start the detection.
""")  # Description text

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Detection", "Training", "Image Detection"])

# Model mapping
MODEL_MAP = {
    "YOLOv8n (Fastest)": "yolov8n.pt",
    "YOLOv8s": "yolov8s.pt",
    "YOLOv8m": "yolov8m.pt",
    "YOLOv8l": "yolov8l.pt",
    "YOLOv8x (Most Accurate)": "yolov8x.pt",
    "Fruit Detection Model": "yolov8n-fruit.pt"
}

@st.cache_resource
def load_model(model_name: str) -> YOLO:
    """Load and cache the YOLO model.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        YOLO: Loaded YOLO model
        
    Raises:
        Exception: If model loading fails
    """
    try:
        return YOLO(model_name)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise

def process_image(image: np.ndarray, model: YOLO, conf_threshold: float) -> Tuple[np.ndarray, Dict[str, int]]:
    """Process an image for object detection.
    
    Args:
        image: Input image as numpy array
        model: YOLO model
        conf_threshold: Confidence threshold
        
    Returns:
        Tuple containing:
        - Annotated image
        - Dictionary of detected objects and their counts
    """
    try:
        results = model(image, conf=conf_threshold)
        detected_objects = results[0].names
        boxes = results[0].boxes
        
        object_counts = {}
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = detected_objects[class_id]
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        annotated_image = results[0].plot()
        return annotated_image, object_counts
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        raise

def validate_inputs(model_type: str, conf_threshold: float) -> bool:
    """Validate user inputs.
    
    Args:
        model_type: Selected model type
        conf_threshold: Confidence threshold
        
    Returns:
        bool: True if inputs are valid
    """
    if model_type not in MODEL_MAP:
        st.error("Invalid model type selected")
        return False
    if not 0 <= conf_threshold <= 1:
        st.error("Confidence threshold must be between 0 and 1")
        return False
    return True

def run_detection(model: YOLO, conf_threshold: float, input_source: str, uploaded_file: Optional[st.runtime.uploaded_file_manager.UploadedFile]) -> None:
    """Run object detection on video stream.
    
    Args:
        model: YOLO model
        conf_threshold: Confidence threshold
        input_source: Input source type
        uploaded_file: Uploaded video file
    """
    col1, col2 = st.columns([2, 1])
    
    # Initialize video capture
    if input_source == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            st.warning("Please upload a video file")
            return
    
    video_placeholder = col1.empty()
    info_placeholder = col2.empty()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("End of video stream")
                break
            
            annotated_frame, object_counts = process_image(frame, model, conf_threshold)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            video_placeholder.image(annotated_frame, channels="RGB")
            
            info_text = "### Detection Results\n"
            for obj, count in object_counts.items():
                info_text += f"- {obj}: {count}\n"
            info_placeholder.markdown(info_text)
            
            if not st.session_state.get('start_detection', False):
                break
    finally:
        cap.release()
        if input_source == "Upload Video" and uploaded_file is not None:
            os.unlink(tfile.name)

def main():
    st.title("🎥 Real-time Object Detection System")
    st.markdown("""
        This application performs real-time object detection using YOLOv8.
        Select your model and confidence threshold, then start the detection.
    """)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Detection", "Training", "Image Detection"])

    with tab1:
        # Sidebar controls
        with st.sidebar:
            st.header("Settings")
            
            model_type = st.selectbox(
                "Select Model",
                list(MODEL_MAP.keys()),
                index=0
            )
            
            conf_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05
            )
            
            input_source = st.radio(
                "Input Source",
                ["Webcam", "Upload Video"]
            )
            
            uploaded_file = None
            if input_source == "Upload Video":
                uploaded_file = st.file_uploader(
                    "Choose a video file",
                    type=["mp4", "avi", "mov"]
                )
            
            start_detection = st.button("Start Detection")
            if start_detection:
                st.session_state['start_detection'] = True

        # Main detection logic
        if st.session_state.get('start_detection', False):
            if not validate_inputs(model_type, conf_threshold):
                return
                
            try:
                model = load_model(MODEL_MAP[model_type])
                run_detection(model, conf_threshold, input_source, uploaded_file)
            except Exception as e:
                st.error(f"Detection failed: {str(e)}")
        else:
            st.info("Click 'Start Detection' to begin")

    with tab2:
        st.header("Train Custom Model")
        
        # Training settings
        with st.sidebar:
            st.header("Training Settings")
            
            # Model size selection
            model_size = st.selectbox(
                "Model Size",
                ["n (Fastest)", "s", "m", "l", "x (Most Accurate)"],
                index=0
            )
            
            # Training parameters
            epochs = st.number_input("Number of Epochs", min_value=1, value=100)
            batch_size = st.number_input("Batch Size", min_value=1, value=16)
            img_size = st.number_input("Image Size", min_value=32, value=640)
            
            # Class names input
            class_names = st.text_area(
                "Enter Class Names (one per line)",
                "class1\nclass2\nclass3"
            ).split('\n')
            
            # Dataset upload
            st.subheader("Upload Dataset")
            train_images = st.file_uploader("Training Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            train_labels = st.file_uploader("Training Labels", type=["txt"], accept_multiple_files=True)
            val_images = st.file_uploader("Validation Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            val_labels = st.file_uploader("Validation Labels", type=["txt"], accept_multiple_files=True)
            
            # Start training button
            start_training = st.button("Start Training")

        # Training function
        def prepare_dataset(data_dir, train_images, train_labels, val_images, val_labels):
            # Create directories
            os.makedirs(os.path.join(data_dir, 'images', 'train'), exist_ok=True)
            os.makedirs(os.path.join(data_dir, 'images', 'val'), exist_ok=True)
            os.makedirs(os.path.join(data_dir, 'labels', 'train'), exist_ok=True)
            os.makedirs(os.path.join(data_dir, 'labels', 'val'), exist_ok=True)
            
            # Save uploaded files
            for img in train_images:
                with open(os.path.join(data_dir, 'images', 'train', img.name), 'wb') as f:
                    f.write(img.getbuffer())
            
            for lbl in train_labels:
                with open(os.path.join(data_dir, 'labels', 'train', lbl.name), 'wb') as f:
                    f.write(lbl.getbuffer())
            
            for img in val_images:
                with open(os.path.join(data_dir, 'images', 'val', img.name), 'wb') as f:
                    f.write(img.getbuffer())
            
            for lbl in val_labels:
                with open(os.path.join(data_dir, 'labels', 'val', lbl.name), 'wb') as f:
                    f.write(lbl.getbuffer())

        def create_dataset_yaml(data_dir, class_names):
            yaml_content = {
                'path': os.path.abspath(data_dir),
                'train': 'images/train',
                'val': 'images/val',
                'names': {i: name for i, name in enumerate(class_names)}
            }
            
            with open(os.path.join(data_dir, 'dataset.yaml'), 'w') as f:
                yaml.dump(yaml_content, f, sort_keys=False)
            
            return os.path.join(data_dir, 'dataset.yaml')

        def train_model(data_yaml, model_size, epochs, batch_size, img_size):
            model = YOLO(f'yolov8{model_size[0]}.pt')
            
            with st.spinner('Training in progress...'):
                results = model.train(
                    data=data_yaml,
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=img_size,
                    device='0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
                    project='custom_model',
                    name=f'yolov8{model_size[0]}',
                    exist_ok=True,
                    pretrained=True,
                    optimizer='auto',
                    verbose=True,
                    seed=42,
                    patience=50,
                    save=True,
                    save_period=10,
                )
            
            return results

        # Main training section
        if start_training:
            if not (train_images and train_labels and val_images and val_labels):
                st.error("Please upload all required files")
            else:
                # Create temporary directory for dataset
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Prepare dataset
                    prepare_dataset(temp_dir, train_images, train_labels, val_images, val_labels)
                    
                    # Create dataset.yaml
                    data_yaml = create_dataset_yaml(temp_dir, class_names)
                    
                    # Train model
                    results = train_model(
                        data_yaml=data_yaml,
                        model_size=model_size,
                        epochs=epochs,
                        batch_size=batch_size,
                        img_size=img_size
                    )
                    
                    # Show results
                    st.success("Training completed!")
                    st.write("Model saved in: custom_model/")
                    
                    # Show training metrics
                    st.subheader("Training Metrics")
                    st.write(f"Best mAP: {results.results_dict['metrics/mAP50-95(B)']:.3f}")
                    st.write(f"Best Precision: {results.results_dict['metrics/precision(B)']:.3f}")
                    st.write(f"Best Recall: {results.results_dict['metrics/recall(B)']:.3f}")
        else:
            st.info("Configure training settings and upload your dataset to begin training")

    with tab3:
        st.header("Image Detection")
        
        # Image detection settings
        with st.sidebar:
            st.header("Settings")
            
            # Model selection
            model_type = st.selectbox(
                "Select Model",
                ["YOLOv8n (Fastest)", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x (Most Accurate)", "Fruit Detection Model"],
                index=0,
                key="image_model"
            )
            
            # Confidence threshold
            conf_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
                key="image_conf"
            )
            
            # Custom model option
            use_custom_model = st.checkbox("Use Custom Model")
            custom_model_path = None
            if use_custom_model:
                custom_model_path = st.file_uploader("Upload Custom Model", type=["pt"])
        
        # Image upload
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        
        if uploaded_image is not None:
            # Load image
            image = Image.open(uploaded_image)
            image_array = np.array(image)
            
            # Convert to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Load model
            if use_custom_model and custom_model_path is not None:
                # Save custom model temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                    tmp_file.write(custom_model_path.getvalue())
                    model_path = tmp_file.name
            else:
                model_path = MODEL_MAP[model_type]
            
            model = load_model(model_path)
            
            # Run detection
            with st.spinner('Detecting objects...'):
                results = model(image_bgr, conf=conf_threshold)
                
                # Get detection results
                detected_objects = results[0].names
                boxes = results[0].boxes
                
                # Count objects
                object_counts = {}
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = detected_objects[class_id]
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1
                
                # Draw results
                annotated_image = results[0].plot()
                
                # Convert to RGB for display
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Detected Objects")
                st.image(annotated_image_rgb, use_container_width=True)
            
            # Display detection info
            st.markdown("### Detection Results")
            for obj, count in object_counts.items():
                st.markdown(f"- **{obj}**: {count}")
            
            # Clean up temporary file if using custom model
            if use_custom_model and custom_model_path is not None:
                os.unlink(model_path)
        else:
            st.info("Upload an image to detect objects")

# Main application entry point
if __name__ == "__main__":
    main() 