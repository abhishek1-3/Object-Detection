import os
import uuid

import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "yoloweights", "yolov8n.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "yoloweights", "yolov8l.pt")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("YOLO model not found under yoloweights/")

app = Flask(__name__, static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

model = YOLO(MODEL_PATH)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def draw_results(image, results):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{model.names.get(cls_id, cls_id)} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1 - text_size[1] - 8), (x1 + text_size[0] + 4, y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return image


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("index"))

    image_data = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image is None:
        return redirect(url_for("index"))

    results = model(image)
    image = draw_results(image, results)

    filename = f"result_{uuid.uuid4().hex[:12]}.jpg"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    cv2.imwrite(save_path, image)

    return render_template("index.html", filename=filename)


camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Cannot open webcam. Please check the camera and try again.")


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame)
        frame = draw_results(frame, results)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/webcam")
def webcam():
    return render_template("webcam.html")


@app.route("/video_feed")
def video_feed():
    return app.response_class(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
