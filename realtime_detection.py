import cv2
from ultralytics import YOLO
import numpy as np

def main():
    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')  # Load the YOLOv8n model
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide video file path
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Starting real-time object detection...")
    print("Press 'q' to quit")
    
    while True:
        # Read frame from video source
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 