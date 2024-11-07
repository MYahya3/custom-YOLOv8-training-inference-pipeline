import cv2
import torch
from ultralytics import YOLO
from utilis import YOLO_Detection, label_detection
import os

def setup_device():
    """Check if CUDA is available and set the device."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def load_yolo_model(device):
    """Load the YOLO model and configure it."""
    model = YOLO("runs/detect/1st_dataset_result/weights/best.pt")
    model.to(device)
    model.nms = 0.7
    print(f"Model classes: {model.names}")
    return model

def process_frame(model, frame):
    """Process a single frame to detect objects and apply labels."""
    boxes, classes, names, confidences = YOLO_Detection(model, frame, conf=0.4)

    for box, cls in zip(boxes, classes):
        if int(cls) == 0:
            label_detection(frame=frame, text=f"{names[int(cls)]}", tbox_color=(255, 144, 30), left=box[0],
                            top=box[1], bottom=box[2], right=box[3])
        else:
            label_detection(frame=frame, text=f"{names[int(cls)]}", tbox_color=(0, 0, 230), left=box[0],
                            top=box[1], bottom=box[2], right=box[3])

def main(source):
    device = setup_device()
    model = load_yolo_model(device)

    # Check if the source is a video file or an image
    if os.path.isfile(source):
        if source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Process a single image
            frame = cv2.imread(source)
            if frame is None:
                print(f"Error: Unable to read image {source}")
                return
            process_frame(model, frame)
            cv2.imshow('Processed Image', frame)
            cv2.waitKey(0)  # Wait for a key press to close the image window
        else:
            # Process a video file
            cap = cv2.VideoCapture(source)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                process_frame(model, frame)

                # Display the frame (press 'q' to quit early)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
    else:
        # Assume the source is a camera feed (e.g., '0' for the default camera)
        cap = cv2.VideoCapture(source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            process_frame(model, frame)

            # Display the frame (press 'q' to quit early)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example: Change this to the desired image/video source or camera index
    main(source= "3.mp4")  # Use an image path (e.g., "image.jpg") or video path
