import cv2
import numpy as np
import onnxruntime as ort
import time
from collections import Counter

# Load class labels
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ONNX session with AMD GPU
session = ort.InferenceSession("yolov5l.onnx", providers=["DmlExecutionProvider"])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not found.")
    exit()

conf_threshold = 0.99
nms_threshold = 0.45

print("🎥 Running YOLOv5 – press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_orig, w_orig = frame.shape[:2]
    img = cv2.resize(frame, (640, 640))
    img_input = img[:, :, ::-1].transpose(2, 0, 1)
    img_input = np.expand_dims(img_input, 0).astype(np.float32) / 255.0

    start = time.time()
    preds = session.run(None, {"images": img_input})[0][0]
    end = time.time()

    boxes = []
    confidences = []
    class_ids = []

    for pred in preds:
        scores = pred[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence >= conf_threshold:
            cx, cy, w, h = pred[:4]
            x1 = int((cx - w / 2) * w_orig / 640)
            y1 = int((cy - h / 2) * h_orig / 640)
            x2 = int((cx + w / 2) * w_orig / 640)
            y2 = int((cy + h / 2) * h_orig / 640)

            x1 = max(0, min(x1, w_orig - 1))
            y1 = max(0, min(y1, h_orig - 1))
            x2 = max(0, min(x2, w_orig - 1))
            y2 = max(0, min(y2, h_orig - 1))

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    visible_classes = []

    if len(indices) > 0:
        for idx in indices.flatten():
            x, y, w, h = boxes[idx]
            label = f"{class_names[class_ids[idx]]}: {confidences[idx]:.2f}"
            visible_classes.append(class_names[class_ids[idx]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Count and display object types
    counts = Counter(visible_classes)
    y_offset = 50
    for cls, count in counts.items():
        text = f"{cls}: {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        y_offset += 25

    # Display FPS
    fps = 1 / (end - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLOv5 AMD GPU - Object Count", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
