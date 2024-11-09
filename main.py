import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
import firebase_admin
from firebase_admin import credentials, firestore
import time

# Initialize Firebase
cred = credentials.Certificate("smart-mess-16115-firebase-adminsdk-7sqjp-d07ce121ec.json")  # Replace with your service account key file path
firebase_admin.initialize_app(cred)
db = firestore.client()

# Firebase reference for entering and exiting counts
doc_ref = db.collection('people_count').document('current_status')

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Define two areas as vertical rectangles with a slight horizontal gap
area1 = [(3,201), (3,237), (639,228), (639,195)]  # Left rectangle
area2 = [(3,257), (3,293), (639, 283), (639, 250)]  # Right rectangle with a gap

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

# Set up window and mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture("testmess.mp4")

# Load class names from coco.txt
with open("coco.txt", "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")

count = 0
tracker = Tracker()

people_entering = {}
people_exiting = {}
entering = set()
exiting = set()

# Track the last time data was pushed to Firebase
last_push_time = time.time()

# Fetch initial values from Firestore
initial_data = doc_ref.get().to_dict() if doc_ref.get().exists else {}
out_count = initial_data.get('in', 0)
 
in_count = initial_data.get('out', 0)
capacity = initial_data.get('capacity', 0)

print(f"Initial values - IN: {in_count}, OUT: {out_count}, Capacity: {capacity}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 4 != 0:
        continue

    frame = cv2.resize(frame, (640, 360))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []

    for index, row in px.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        results = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
        if results >= 0:
            people_entering[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        if id in people_entering:
            results1 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
            if results1 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (x2, y2), 4, (255, 0, 255), -1)
                cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                entering.add(id)

        results2 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
        if results2 >= 0:
            people_exiting[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

        if id in people_exiting:
            results3 = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
            if results3 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.circle(frame, (x2, y2), 4, (255, 0, 255), -1)
                cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                exiting.add(id)

    # Draw areas and counts
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 3)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 3)

    out_count, in_count = len(entering), len(exiting)
    cv2.putText(frame, f"IN: {in_count}", (479,57), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"OUT: {out_count}", (498,127), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)

    # Push to Firebase every 10 seconds
    current_time = time.time()
    if current_time - last_push_time >= 10:
        doc_ref.set({'in': in_count, 'out': out_count, 'capacity': capacity})
        last_push_time = current_time

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
