import cv2
import numpy as np

# Load model
net = cv2.dnn.readNetFromONNX("dnn/yolov5s.onnx")

# Load image
image = cv2.imread("image.jpg")
h, w = image.shape[:2]

# Prepare blob
blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward()

# Parse outputs: shape = [1, 25200, 85]
preds = outputs[0]

conf_threshold = 0.5
nms_threshold = 0.4

boxes, confidences, class_ids = [], [], []

for detection in preds:
    scores = detection[5:]
    class_id = np.argmax(scores)
    confidence = scores[class_id]

    if confidence > conf_threshold:
        cx, cy, w_box, h_box = detection[0:4]
        x = int((cx - w_box / 2) * w)
        y = int((cy - h_box / 2) * h)
        width = int(w_box * w)
        height = int(h_box * h)

        boxes.append([x, y, width, height])
        confidences.append(float(confidence))
        class_ids.append(class_id)

# NMS
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Draw boxes
for i in indices:
    i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
    x, y, width, height = boxes[i]
    label = f"{class_ids[i]}: {confidences[i]:.2f}"
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
