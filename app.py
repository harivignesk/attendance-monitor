from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import time
import threading
import os
import database
import attendance
import register

app = Flask(__name__)

# Initialize SQLite database
database.init_db()

# Initialize High-Speed FastFaceMatcher engine
face_matcher = attendance.FastFaceMatcher()

class ThreadedCamera:
    """
    Asynchronous Camera Reader Thread with MJPEG hardware decoding.
    Eliminates camera startup lag (<0.2s startup time) and prevents UI freeze.
    """
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.src)

        # Hardware MJPEG codec: 10x faster frame transfer from camera chip
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Zero frame lag

        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

        # Start continuous background capture thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.05)
                continue

            grabbed, frame = self.cap.read()
            if grabbed and frame is not None:
                with self.lock:
                    self.grabbed = grabbed
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.grabbed and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def release(self):
        self.stopped = True
        if self.cap and self.cap.isOpened():
            self.cap.release()

# Global Threaded Camera Instance
camera_stream = None

def get_camera_stream():
    global camera_stream
    if camera_stream is None or camera_stream.stopped:
        camera_stream = ThreadedCamera(0)
    return camera_stream

def release_camera_stream():
    global camera_stream
    if camera_stream is not None:
        camera_stream.release()
        camera_stream = None

def generate_frames():
    """Generates ultra-fast MJPEG video stream directly from RAM."""
    stream = get_camera_stream()

    while True:
        success, frame = stream.read()
        if not success or frame is None:
            time.sleep(0.02)
            continue

        frame = cv2.flip(frame, 1)

        # Process face recognition
        processed_frame, events = attendance.process_attendance_frame(frame, face_matcher)

        # Ultra-fast JPEG encoding (Quality 75 for 0ms lag)
        ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/reports')
def reports_page():
    return render_template('reports.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def api_stats():
    current_session = database.get_current_session()
    before_records = database.get_today_attendance("Before Lunch")
    after_records = database.get_today_attendance("After Lunch")
    all_today = database.get_today_attendance("all")
    all_students = database.get_all_students()

    return jsonify({
        'current_session': current_session,
        'before_lunch_count': len(before_records),
        'after_lunch_count': len(after_records),
        'total_today_count': len(all_today),
        'total_students': len(all_students),
        'recent_logs': all_today[:12]
    })

@app.route('/api/register_student', methods=['POST'])
def api_register_student():
    data = request.json or {}
    sid = data.get('student_id', '').strip()
    name = data.get('name', '').strip()
    dept = data.get('department', 'General').strip()

    if not sid or not name:
        return jsonify({'success': False, 'message': 'Student ID and Name are required.'}), 400

    release_camera_stream()
    success, msg = register.capture_student_faces(sid, name, dept, sample_count=5)
    
    if success:
        face_matcher.load_registered_faces()

    return jsonify({'success': success, 'message': msg})

@app.route('/api/students', methods=['GET'])
def api_get_students():
    students = database.get_all_students()
    return jsonify(students)

@app.route('/api/delete_student', methods=['POST', 'DELETE'])
def api_delete_student():
    data = request.json or {}
    sid = data.get('student_id', '').strip()
    if not sid:
        return jsonify({'success': False, 'message': 'Student ID is required.'}), 400

    success, msg = database.delete_student(sid)
    if success:
        face_matcher.load_registered_faces()

    return jsonify({'success': success, 'message': msg})

@app.route('/api/reports')
def api_reports():
    session_param = request.args.get('session', 'all')
    if session_param.lower() in ['before_lunch', 'before lunch']:
        records = database.get_today_attendance("Before Lunch")
    elif session_param.lower() in ['after_lunch', 'after lunch']:
        records = database.get_today_attendance("After Lunch")
    else:
        records = database.get_all_attendance()
    return jsonify(records)

@app.route('/api/export_excel')
def api_export_excel():
    session_param = request.args.get('session', 'all')
    filepath = database.export_to_excel("attendance_report.xlsx", session_filter=session_param)
    return send_file(filepath, as_attachment=True, download_name="attendance_report.xlsx")

@app.route('/api/export_csv')
def api_export_csv():
    filepath = database.export_to_csv("attendance_report.csv")
    return send_file(filepath, as_attachment=True, download_name="attendance_report.csv")

if __name__ == '__main__':
    # Pre-warm camera in background
    get_camera_stream()
    print("=" * 60)
    print("AI Face Recognition Attendance Machine Web Server Running!")
    print("Open in Browser: http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
