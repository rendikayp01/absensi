import cv2
import os
import pickle
import numpy as np
from openpyxl import Workbook
from datetime import datetime

TARGET_SIZE = (100, 100)

# Fungsi untuk menyimpan daftar absensi ke dalam file Excel
def save_to_excel(attendance):
    # Sortir daftar absensi berdasarkan nama secara abjad
    sorted_attendance = sorted(attendance, key=lambda x: x[0])

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Daftar Hadir"

    sheet["A1"] = "No"
    sheet["B1"] = "Nama"
    sheet["C1"] = "Waktu Absensi"

    for idx, (name, time) in enumerate(sorted_attendance, start=1):
        sheet[f"A{idx + 1}"] = idx
        sheet[f"B{idx + 1}"] = name
        sheet[f"C{idx + 1}"] = time

    workbook.save("Daftar_Hadir.xlsx")
    print("Daftar hadir disimpan ke Daftar_Hadir.xlsx")

# Fungsi untuk memuat absensi yang sudah ada sebelumnya
def load_existing_attendance():
    if os.path.exists('attendance_data.pkl'):
        with open('attendance_data.pkl', 'rb') as f:
            return pickle.load(f)
    return []

# Fungsi untuk menyimpan absensi yang sudah ada ke file
def save_attendance(attendance):
    with open('attendance_data.pkl', 'wb') as f:
        pickle.dump(attendance, f)

# Fungsi utama untuk mengenali wajah dan mencatat absensi
def recognize_face():
    if not os.path.exists('face_data'):
        print("Tidak ada data wajah yang terdaftar.")
        return

    # Memuat absensi yang sudah ada
    existing_attendance = load_existing_attendance()

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Tidak dapat membuka kamera!")
        exit()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces, labels, label_map = [], [], {}
    for idx, file in enumerate(os.listdir('face_data')):
        with open(f"face_data/{file}", 'rb') as f:
            known_faces = pickle.load(f)
        faces.extend(known_faces)
        labels.extend([idx] * len(known_faces))
        label_map[idx] = os.path.splitext(file)[0]

    recognizer.train(faces, np.array(labels))

    attendance = set(existing_attendance)  # Menggabungkan absensi sebelumnya
    print("Mulai absensi. Tekan 'q' untuk keluar.")
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces_detected:
            roi = cv2.resize(gray[y:y+h, x:x+w], TARGET_SIZE)
            label, confidence = recognizer.predict(roi)
            if confidence < 100:
                name = label_map[label]
                if name not in [entry[0] for entry in attendance]:  # Memeriksa apakah sudah absen
                    # Menambahkan nama dan waktu ke absensi
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    attendance.add((name, current_time))
                    print(f"{name} absen pada {current_time}")
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    # Menyimpan data absensi yang terbaru
    save_attendance(list(attendance))

    # Menyimpan data absensi ke file Excel
    save_to_excel(attendance)

if __name__ == "__main__":
    recognize_face()
