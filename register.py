import cv2
import os
import time
import database

FACES_DIR = "faces"

def ensure_faces_dir():
    """Ensures the root faces directory exists."""
    if not os.path.exists(FACES_DIR):
        os.makedirs(FACES_DIR)

def capture_student_faces(student_id: str, name: str, department: str = "General", sample_count: int = 5, frame_callback=None) -> tuple[bool, str]:
    """
    Captures face samples for a student using DirectShow camera backend.
    """
    student_id = student_id.strip()
    name = name.strip()

    if not student_id or not name:
        return False, "Student ID and Name are required!"

    ensure_faces_dir()
    student_dir = os.path.join(FACES_DIR, student_id)
    if not os.path.exists(student_dir):
        os.makedirs(student_dir)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Open camera with DirectShow backend for Windows stability
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        return False, "Could not open webcam. Please verify camera device."

    captured = 0
    start_time = time.time()

    print(f"Starting registration for {name} ({student_id}). Target samples: {sample_count}")

    while captured < sample_count:
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        all_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        # Select only the single main subject face (largest box)
        if len(all_faces) > 0:
            main_face = max(all_faces, key=lambda f: f[2] * f[3])
            faces = [main_face]
        else:
            faces = []

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing: {captured + 1}/{sample_count}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (200, 200))

            if time.time() - start_time > 0.4:
                captured += 1
                sample_path = os.path.join(student_dir, f"sample_{captured}.jpg")
                cv2.imwrite(sample_path, face_roi)
                start_time = time.time()
                print(f"Saved {sample_path}")
                break

        if frame_callback:
            frame_callback(frame)

    camera.release()

    if captured >= sample_count:
        database.add_student(student_id, name, department)
        return True, f"Successfully registered {name} ({student_id}) with {captured} face samples!"
    else:
        return False, f"Registration incomplete. Captured {captured}/{sample_count} samples."
