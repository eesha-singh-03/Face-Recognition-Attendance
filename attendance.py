import cv2
import numpy as np
import os
from pathlib import Path
import csv
import pickle
from datetime import datetime

class FastAttendanceDNN:
    def __init__(self,
                 training_dir="training",
                 model_path="face_model.yml",
                 labels_path="labels.pkl",
                 attendance_dir="Attendance",
                 student_file="StudentDetails.csv"):

        self.training_dir = Path(training_dir)
        self.model_path = model_path
        self.labels_path = labels_path
        self.attendance_dir = Path(attendance_dir)
        self.student_file = student_file

        self.attendance_dir.mkdir(exist_ok=True)

        # Load DNN face detector
        self.face_net = cv2.dnn.readNetFromCaffe(
            "models/deploy.prototxt",
            "models/res10_300x300_ssd_iter_140000.caffemodel"
        )

        # LBPH recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_dict = {}
        self.marked_today = set()

    # ------------- FACE DETECTION (DNN) -------------- #
    def detect_faces(self, image):
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < 0.6:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = image[y1:y2, x1:x2]
            if face.size == 0:
                continue

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200))

            faces.append((gray, (x1, y1, x2, y2)))

        return faces

    # ---------------- TRAINING ---------------- #
    def train(self):
        faces = []
        labels = []
        label_map = {}
        current_label = 0

        person_dirs = [p for p in self.training_dir.iterdir() if p.is_dir()]
        if not person_dirs:
            print("[ERROR] No training folders found.")
            return

        for person_dir in person_dirs:
            name = person_dir.name
            label_map[current_label] = name
            print(f"[INFO] Training on {name}")

            image_files = list(person_dir.glob("*.jpg")) + \
                          list(person_dir.glob("*.png")) + \
                          list(person_dir.glob("*.jpeg"))

            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                detected = self.detect_faces(img)
                if not detected:
                    print(f"   [WARN] No face: {img_path.name}")
                    continue

                for face_roi, _ in detected:
                    faces.append(face_roi)
                    labels.append(current_label)
                    print(f"   [OK] Encoded: {img_path.name}")

            current_label += 1

        if len(faces) == 0:
            print("[ERROR] No faces collected.")
            return

        print(f"[INFO] Training LBPH on {len(faces)} samples...")
        self.recognizer.train(faces, np.array(labels))
        self.recognizer.save(self.model_path)

        with open(self.labels_path, "wb") as f:
            pickle.dump(label_map, f)

        self.label_dict = label_map
        print("[SUCCESS] Model trained successfully!")

    # ---------------- MODEL LOAD ---------------- #
    def load_model(self):
        if not os.path.exists(self.model_path):
            print("[ERROR] Train the model first!")
            return False
        self.recognizer.read(self.model_path)
        with open(self.labels_path, "rb") as f:
            self.label_dict = pickle.load(f)
        print("[INFO] Model loaded.")
        return True

    # ---------------- ATTENDANCE FILE ---------------- #
    def get_att_file(self):
        today = datetime.now().strftime("%d-%m-%Y")
        return self.attendance_dir / f"Attendance_{today}.csv"

    def init_attendance(self):
        file = self.get_att_file()
        if not os.path.exists(file):
            with open(file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Roll", "Time", "Date", "Status"])
        self.load_marked_today()

    def load_marked_today(self):
        self.marked_today.clear()
        file = self.get_att_file()
        if not file.exists(): return
        with open(file, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.marked_today.add(r["Name"])

    def mark_attendance(self, name):
        if name in self.marked_today:
            return
        file = self.get_att_file()
        now = datetime.now()
        with open(file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, "N/A",
                             now.strftime("%H:%M:%S"),
                             now.strftime("%d-%m-%Y"),
                             "Present"])
        self.marked_today.add(name)
        print(f"[MARKED] {name}")

    # ---------------- WEBCAM ---------------- #
    def start_webcam(self):
        if not self.load_model():
            return

        self.init_attendance()

        cap = cv2.VideoCapture(0)
        print("[INFO] Webcam started. Press Q to exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detect_faces(frame)

            for face_roi, (x1, y1, x2, y2) in detections:
                label, distance = self.recognizer.predict(face_roi)
                match = max(0, 100 - distance)

                name = self.label_dict.get(label, "Unknown") if distance < 80 else "Unknown"

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(frame, f"{name} ({match:.1f}%)",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)

                if name != "Unknown":
                    self.mark_attendance(name)

            cv2.imshow("Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Session ended.")

