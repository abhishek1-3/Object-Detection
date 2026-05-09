import onnxruntime as ort
import cv2
import numpy as np

# Load class labels
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load and prepare image
image = cv2.imread("C:/Users/HP/PycharmProjects/PythonProject/Chapter-5 Running YOLO/images/13.jpg")
if image is None:
    print("❌ Error loading image.")
    exit()

original_height, original_width = image.shape[:2]
image_resized = cv2.resize(image, (640, 640))
image_input = image_resized[:, :, ::-1].transpose(2, 0, 1)
image_input = np.expand_dims(image_input, axis=0).astype(np.float32) / 255.0

# Run ONNX inference
session = ort.InferenceSession("yolov5l.onnx", providers=["DmlExecutionProvider"])
outputs = session.run(None, {"images": image_input})
predictions = outputs[0][0]  # (25200, 85)

# Filter predictions
boxes, confidences, class_ids = [], [], []
conf_threshold = 0.99
nms_threshold = 0.45

for pred in predictions:
    scores = pred[5:]
    class_id = np.argmax(scores)
    confidence = scores[class_id]

    if confidence > conf_threshold:
        cx, cy, w, h = pred[:4]

        # Convert to x1, y1, x2, y2 format, scale to original image
        x1 = int((cx - w / 2) * original_width / 640)
        y1 = int((cy - h / 2) * original_height / 640)
        x2 = int((cx + w / 2) * original_width / 640)
        y2 = int((cy + h / 2) * original_height / 640)

        # Clip coordinates to image bounds
        x1 = max(0, min(x1, original_width - 1))
        y1 = max(0, min(y1, original_height - 1))
        x2 = max(0, min(x2, original_width - 1))
        y2 = max(0, min(y2, original_height - 1))

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(float(confidence))
        class_ids.append(class_id)

# Apply Non-Maximum Suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Draw only high-confidence + NMS-filtered boxes
for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show result
cv2.imshow("Detections", image)
cv2.imwrite("filtered_detections.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
