import cv2
import os
import pickle
import sys

TARGET_SIZE = (100, 100)

def register_face(name):
    if not os.path.exists('face_data'):
        os.makedirs('face_data')

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Tidak dapat membuka kamera!")
        exit()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_data = []
    count = 0

    print(f"Mulai pendaftaran wajah untuk {name}. Tekan 'q' untuk berhenti.")
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            resized_face = cv2.resize(roi, TARGET_SIZE)
            face_data.append(resized_face)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Register Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
            break

    cam.release()
    cv2.destroyAllWindows()

    with open(f"face_data/{name}.pkl", 'wb') as f:
        pickle.dump(face_data, f)

    print(f"Wajah {name} berhasil didaftarkan!")

if __name__ == "__main__":
    name = sys.argv[1]  # Nama diambil dari argumen saat dipanggil
    register_face(name)
