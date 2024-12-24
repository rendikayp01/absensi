from flask import Flask, request, jsonify
import cv2
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)

TARGET_SIZE = (100, 100)

# Fungsi untuk memuat model wajah
def load_face_data():
    if not os.path.exists('face_data'):
        return None, None, None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels, label_map = [], [], {}
    for idx, file in enumerate(os.listdir('face_data')):
        with open(f"face_data/{file}", 'rb') as f:
            known_faces = pickle.load(f)
        faces.extend(known_faces)
        labels.extend([idx] * len(known_faces))
        label_map[idx] = os.path.splitext(file)[0]

    recognizer.train(faces, np.array(labels))
    return recognizer, label_map, faces

recognizer, label_map, faces = load_face_data()

attendance = []

@app.route('/process_absensi', methods=['POST'])
def process_absensi():
    if not recognizer:
        return jsonify({"success": False, "message": "Data wajah tidak tersedia."})

    data = request.json
    if "image" not in data:
        return jsonify({"success": False, "message": "Gambar tidak ditemukan."})

    # Decode Base64 image
    encoded_data = data["image"].split(",")[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_detected:
        roi = cv2.resize(gray[y:y+h, x:x+w], TARGET_SIZE)
        label, confidence = recognizer.predict(roi)
        if confidence < 100:
            name = label_map[label]
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if name not in [entry[0] for entry in attendance]:
                attendance.append((name, current_time))
                return jsonify({"success": True, "message": f"{name} absen pada {current_time}."})
        else:
            return jsonify({"success": False, "message": "Wajah tidak dikenali."})

    return jsonify({"success": False, "message": "Wajah tidak terdeteksi."})

if __name__ == "__main__":
    app.run(debug=True)
