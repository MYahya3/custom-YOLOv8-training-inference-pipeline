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
    model = YOLO("model_weights/best.pt")
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


def main(source, output_path="output_video.mp4"):
    device = setup_device()
    model = load_yolo_model(device)

    # Initialize VideoWriter if the source is a video file
    if os.path.isfile(source) and not source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        cap = cv2.VideoCapture(source)

        # Get video properties to set up the VideoWriter
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID' or others
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            process_frame(model, frame)
            out.write(frame)  # Write the processed frame to output video

            # Display the frame (press 'q' to quit early)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()  # Release VideoWriter object after the loop

    else:
        # Process a single image or a camera feed
        if os.path.isfile(source):
            # Process an image file
            frame = cv2.imread(source)
            if frame is None:
                print(f"Error: Unable to read image {source}")
                return
            process_frame(model, frame)
            cv2.imshow('Processed Image', frame)
            cv2.waitKey(0)  # Wait for a key press to close the image window
        else:
            # Assume the source is a camera feed
            cap = cv2.VideoCapture(source)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 30  # Set an estimated FPS for the camera feed

            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                process_frame(model, frame)
                out.write(frame)  # Write the frame to output video

                # Display the frame (press 'q' to quit early)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            out.release()  # Release VideoWriter object

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example: Change this to the desired image/video source or camera index
    main(source="1.mp4", output_path="output_video.mp4")  # Use an image path (e.g., "image.jpg") or video path
