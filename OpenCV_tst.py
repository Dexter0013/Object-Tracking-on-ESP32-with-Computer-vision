
from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("D:/Hand tracking/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Optional: set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run inference
    results = model(frame, conf=0.5, verbose=False)

    # Draw results on frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("YOLOv8 Live Detection - Palm", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()