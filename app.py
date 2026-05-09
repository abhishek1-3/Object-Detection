import os
import uuid

import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

LOCAL_MODEL_CANDIDATES = (
    os.path.join(BASE_DIR, "yoloweights", "yolov8m.pt"),
    os.path.join(BASE_DIR, "yoloweights", "yolov8l.pt"),
)
DEFAULT_MODEL = os.environ.get("MODEL_PATH") or next(
    (path for path in LOCAL_MODEL_CANDIDATES if os.path.exists(path)),
    "yolov8l.pt",
)
model = None


def get_model():
    global model
    if model is None:
        model = YOLO(DEFAULT_MODEL)
    return model


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def draw_results(image, results, detector):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{detector.names.get(cls_id, cls_id)} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1 - text_size[1] - 8), (x1 + text_size[0] + 4, y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return image


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}


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

    try:
        detector = get_model()
        results = detector(image)
        image = draw_results(image, results, detector)
    except Exception as exc:
        return render_template("index.html", error=f"Model failed to load or run: {exc}")

    filename = f"result_{uuid.uuid4().hex[:12]}.jpg"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    cv2.imwrite(save_path, image)

    return render_template("index.html", filename=filename)


camera = None


def initialize_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera.isOpened()


def generate_frames():
    global camera
    if camera is None or not camera.isOpened():
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n"
               b"\r\n")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        detector = get_model()
        results = detector(frame)
        frame = draw_results(frame, results, detector)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/webcam")
def webcam():
    camera_ok = initialize_camera()
    return render_template("webcam.html", camera_available=camera_ok)


@app.route("/video_feed")
def video_feed():
    return app.response_class(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
