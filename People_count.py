import cv2
from ultralytics import YOLO
import numpy as np
import uuid

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Ensure the correct path to the model

# Define polygon points (adjust for your video)
polygon_points = np.array([[50, 50], [500, 50], [600, 300], [100, 300]], dtype=np.int32)

# Function to check if a point is inside a polygon
def is_point_inside_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# Initialize video capture
video_path = r'C:\Users\HINNOVIS\Desktop\People_couting\people.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Track unique IDs for people inside the polygon
tracked_ids = set()
current_ids = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for consistent processing
    frame = cv2.resize(frame, (640, 480))

    # Perform object detection
    results = model(frame)
    detections = results[0].boxes

    # Draw the polygon
    cv2.polylines(frame, [polygon_points], True, (0, 255, 0), 2)

    # Process detections and update counts
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        class_name = model.names[cls]

        if class_name == 'person':
            # Calculate bounding box center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Generate a unique ID if it's a new person
            person_id = str(uuid.uuid4())
            is_new = True

            # Check if this person was previously detected
            for existing_id, pos in current_ids.items():
                if abs(center_x - pos[0]) < 50 and abs(center_y - pos[1]) < 50:
                    person_id = existing_id
                    is_new = False
                    break

            current_ids[person_id] = (center_x, center_y)

            # Check if the center is inside the polygon
            if is_point_inside_polygon((center_x, center_y), polygon_points):
                if person_id not in tracked_ids:
                    tracked_ids.add(person_id)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f'ID: {person_id[:8]} {class_name} {conf:.2f}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the count of unique people inside or who have entered the polygon
    cv2.putText(frame, f'Unique People: {len(tracked_ids)}', (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Simulated infrared map
    infrared_map = cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
    infrared_map = cv2.resize(infrared_map, (640, 480))

    # Separate displays for original frame and infrared map
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Infrared Map', infrared_map)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()