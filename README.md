# **Face Recognition Attendance System (DNN + LBPH)**
This project is a **real-time face recognition attendance system** built using:

* **OpenCV DNN** (Caffe-based SSD) for face detection
* **OpenCV LBPH Recognizer** for face recognition
* **CSV attendance logging** per day
* **Training pipeline** that reads images from folders and builds a face recognition model

The system detects faces from the webcam, recognizes trained students, and automatically marks their attendance in date-wise CSV files.

## ⭐ **Features**
* Deep Learning–based face detection using OpenCV DNN
* LBPH face recognition (lightweight and fast)
* Automatic daily attendance CSV creation
* Prevents duplicate entries for the same day
* Easy to retrain with new student folders
* Fully modular class-based design

## 📂 **Project Structure**
```
/Attendance/                   # Auto-generated attendance CSV files
/models/
   deploy.prototxt             # DNN face detector config
   res10_300x300_ssd_iter_140000.caffemodel  # DNN weights
/training/
   /Student1/ image1.jpg...
   /Student2/ image1.jpg...
attendance.py                  # Main project file
StudentDetails.csv             # Optional CSV for roll numbers
face_model.yml                 # Saved LBPH model (after training)
labels.pkl                     # Saved label-name mapping
```

## 🛠️ **Dependencies**
Make sure you have the following installed:

| Library | Version (Recommended) |
| ------- | --------------------- |
| Python  | 3.8 – 3.12            |
| OpenCV  | 4.5+                  |
| Numpy   | latest                |
| Pickle  | builtin               |
| CSV     | builtin               |

Install required packages:
```bash
pip install opencv-python opencv-contrib-python numpy
```

> ⚠️ **Important**:
> You MUST install `opencv-contrib-python` because LBPHFaceRecognizer is included only in contrib package.

## 📥 **Setup Instructions**

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

## 2️⃣ Download Face Detection Model

Create a folder named **models/** and place the following inside it:

* **deploy.prototxt**
* **res10_300x300_ssd_iter_140000.caffemodel**

Download link (official OpenCV):
[https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)

Folder structure must be:

```
models/deploy.prototxt
models/res10_300x300_ssd_iter_140000.caffemodel
```

## 3️⃣ Prepare Training Images
Inside the `training/` folder, create one folder per student:

```
training/
   Test1/
      1.jpg
      2.jpg
      3.jpg
   Test2/
      1.jpg
      2.jpg
```

**Rules for training images:**
* Clear face, good lighting
* jpg / png / jpeg supported
* Multiple images recommended (10–20)

## 4️⃣ Train the Model
Run:

```bash
python attendance.py
```

Modify the last lines (if needed) to call:

```python
obj = FastAttendanceDNN()
obj.train()
```

This will:

* Detect faces from training images
* Train LBPH recognizer
* Save:

  * `face_model.yml`
  * `labels.pkl`

## 5️⃣ Start Webcam Attendance System

Use:

```python
obj = FastAttendanceDNN()
obj.start_webcam()
```

This will:

* Load model
* Initialize today's attendance CSV
* Start webcam feed
* Show bounding boxes + confidence
* Auto mark attendance on recognition

Press **Q** to exit.

## 📊 **Attendance Output**

Attendance files are stored as:

```
Attendance/Attendance_dd-mm-YYYY.csv
```

Columns:

| Name | Roll | Time | Date | Status |
| ---- | ---- | ---- | ---- | ------ |

Example:

```
Shashank, N/A, 14:22:01, 25-11-2025, Present
```

## 🧠 **How Recognition Works**

1. OpenCV DNN detects faces
2. Extracted face → converted to gray → resized to 200×200
3. LBPH model predicts label + confidence
4. If confidence < 80 → considers face as **Unknown**
5. If recognized → attendance recorded only once per day

## 📌 **Troubleshooting**

### ❗ Getting `cv2.dnn` error?

Ensure:

```
models/deploy.prototxt
models/res10_300x300_ssd_iter_140000.caffemodel
```

exist exactly at correct paths.

---

### ❗ Getting `cv2.face.LBPHFaceRecognizer_create()` error?
Install contrib version:
```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

### ❗ Recognizer accuracy is low?
* Add more training images
* Use clear front-face pictures
* Ensure good lighting during webcam attendance

## 📄 License
MIT License (Feel free to modify)

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first.

Just tell me!
