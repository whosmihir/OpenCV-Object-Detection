import cv2
import numpy as np
from ultralytics import YOLO


KNOWN_WIDTH_OBJECT = 10  #cm
FOCAL_LENGTH_CAMERA = 800  # Focal length of the camera

def calculate_distance(known_width, focal_length, perceived_width):
   
    return (known_width * focal_length) / perceived_width

def main():
     # Load the YOLO model
    
    model = YOLO("yolov8n.pt")  # Loading a pre-trained YOLOv8 model


    # Open the video feed (use 0 for webcam or provide a video file path)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)

        # Process detection results
        for result in results:
            boxes = result.boxes.xywh.cpu().numpy()  # Get bounding boxes in (x, y, width, height) format
            classes = result.boxes.cls.cpu().numpy()  # Get class IDs
            confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores

            for box, cls, conf in zip(boxes, classes, confidences):
                x, y, w, h = box
                label = model.names[int(cls)]  # Get class label
                confidence = float(conf)

                # Draw bounding box and label
                cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (int(x - w / 2), int(y - h / 2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Calculate approximate distance
                distance = calculate_distance(KNOWN_WIDTH_OBJECT, FOCAL_LENGTH_CAMERA, w)
                cv2.putText(frame, f"Distance: {distance:.2f} cm", (int(x - w / 2), int(y + h / 2) + 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("YOLO Object Detection Window", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()