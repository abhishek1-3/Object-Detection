import onnxruntime as ort
import cv2
import numpy as np

# Load image
image = cv2.imread("C:/Users/HP/PycharmProjects/PythonProject/Chapter-5 Running YOLO/images/13.jpg")
image_resized = cv2.resize(image, (640, 640))
image_input = image_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
image_input = np.expand_dims(image_input, axis=0).astype(np.float32) / 255.0

# Load YOLOv5l.onnx with DirectML
session = ort.InferenceSession("Yolo With GPU/yolov5s.onnx", providers=["DmlExecutionProvider"])

# Run inference
outputs = session.run(None, {"images": image_input})

print("Output shape:", outputs[0].shape)
