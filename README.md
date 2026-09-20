# 📷 AI Face Recognition Attendance Machine

A standalone desktop application for automated face recognition attendance using Python, OpenCV, SQLite, and a modern GUI.

## Features

- 👤 **Student Registration**: Capture 5 face samples via webcam and register student profiles (Roll No, Full Name, Department) into the SQLite database.
- 🎥 **Real-time Live Attendance**: Live webcam feed scanning for registered faces, displaying bounding boxes, matching confidence scores, and automatically marking attendance.
- 🚫 **Duplicate Prevention**: Prevents double-marking attendance on the same day for the same student.
- 📊 **Attendance Reports & Export**: Table view of all attendance logs with timestamps and one-click **CSV Export**.
- 🗄️ **Local SQLite Storage**: Secure offline storage (`attendance.db`) and image profile directory (`faces/`).

---

## File Structure

```text
attendance monitor/
├── main.py              # Main Graphical User Interface Application
├── attendance.py        # Real-time face detection & recognition engine
├── register.py          # Student face registration & sample capture module
├── database.py          # SQLite database schema & CRUD functions
├── requirements.txt     # Python dependencies
├── faces/               # Directory storing registered face image samples
└── attendance.db        # SQLite database storing student & attendance records
```

---

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Application**:
   ```bash
   python main.py
   ```

3. **Usage Workflow**:
   - Go to **👤 Register Student** tab, enter Student ID & Name, and click **Start Face Capture**.
   - Go to **🎥 Live Attendance** tab and click **Start Attendance Scan**. Look at the webcam to mark attendance.
   - Go to **📊 Attendance Reports** tab to view log history or click **Export CSV**.
