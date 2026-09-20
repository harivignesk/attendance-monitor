import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import time
import os

import database
import register
import attendance

class FaceAttendanceApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Face Recognition Attendance System")
        self.root.geometry("1100x700")
        self.root.minsize(950, 600)
        self.root.configure(bg="#1E1E2E")

        # Initialize Database
        database.init_db()

        # Face Recognition Matcher Engine
        self.matcher = attendance.FaceMatcher()

        # Video Streaming State
        self.cap = None
        self.is_camera_running = False
        self.is_registering = False

        # Apply Modern Styling
        self.setup_styles()

        # Build Main UI Components
        self.create_header()
        self.create_tabs()

    def setup_styles(self):
        """Sets up custom TTK widget styling."""
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Dark Palette Colors
        bg_dark = "#1E1E2E"
        panel_bg = "#2B2B3B"
        accent_blue = "#4E73DF"
        text_light = "#FFFFFF"

        # Notebook / Tabs
        self.style.configure("TNotebook", background=bg_dark, borderwidth=0)
        self.style.configure("TNotebook.Tab", background="#282A36", foreground="#A0A0B0",
                             font=("Helvetica", 11, "bold"), padding=[16, 8])
        self.style.map("TNotebook.Tab",
                       background=[("selected", accent_blue)],
                       foreground=[("selected", text_light)])

        # Treeview (Tables)
        self.style.configure("Treeview", background="#2B2B3B", foreground="#FFFFFF",
                             fieldbackground="#2B2B3B", font=("Helvetica", 10), rowheight=28)
        self.style.configure("Treeview.Heading", background="#181825", foreground="#4E73DF",
                             font=("Helvetica", 10, "bold"))
        self.style.map("Treeview", background=[("selected", "#3B3E5B")])

    def create_header(self):
        """Top Application Banner."""
        header_frame = tk.Frame(self.root, bg="#181825", height=65)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_label = tk.Label(
            header_frame,
            text="📷 AI FACE RECOGNITION ATTENDANCE MACHINE",
            font=("Helvetica", 16, "bold"),
            fg="#61AFEF",
            bg="#181825"
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=15)

        self.status_banner = tk.Label(
            header_frame,
            text="System Ready",
            font=("Helvetica", 10, "bold"),
            fg="#98C379",
            bg="#282C34",
            padx=12,
            pady=4
        )
        self.status_banner.pack(side=tk.RIGHT, padx=20, pady=15)

    def create_tabs(self):
        """Creates notebook tabs: Live Attendance, Register Student, View Reports."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Tab 1: Attendance Camera Feed
        self.tab_attendance = tk.Frame(self.notebook, bg="#1E1E2E")
        self.notebook.add(self.tab_attendance, text=" 🎥 Live Attendance ")
        self.setup_attendance_tab()

        # Tab 2: Student Registration
        self.tab_register = tk.Frame(self.notebook, bg="#1E1E2E")
        self.notebook.add(self.tab_register, text=" 👤 Register Student ")
        self.setup_register_tab()

        # Tab 3: Attendance Records
        self.tab_reports = tk.Frame(self.notebook, bg="#1E1E2E")
        self.notebook.add(self.tab_reports, text=" 📊 Attendance Reports ")
        self.setup_reports_tab()

    # ==========================================
    # TAB 1: LIVE ATTENDANCE SCANNER
    # ==========================================
    def setup_attendance_tab(self):
        # Left Panel: Video Display
        left_panel = tk.Frame(self.tab_attendance, bg="#2B2B3B", bd=2, relief=tk.RIDGE)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        video_header = tk.Label(left_panel, text="Webcam Live Scanner", font=("Helvetica", 12, "bold"),
                                fg="#E06C75", bg="#2B2B3B")
        video_header.pack(anchor="w", padx=15, pady=10)

        self.video_canvas = tk.Label(left_panel, bg="#1E1E2E", text="Camera Offline\nClick 'Start Camera' below",
                                     font=("Helvetica", 14), fg="#A0A0B0")
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Controls Bar
        btn_frame = tk.Frame(left_panel, bg="#2B2B3B")
        btn_frame.pack(fill=tk.X, padx=15, pady=10)

        self.btn_start_cam = tk.Button(
            btn_frame, text="▶ Start Attendance Scan", font=("Helvetica", 11, "bold"),
            bg="#98C379", fg="#1E1E2E", activebackground="#7EB35A", padx=15, pady=8,
            command=self.start_attendance_camera
        )
        self.btn_start_cam.pack(side=tk.LEFT, padx=5)

        self.btn_stop_cam = tk.Button(
            btn_frame, text="⏹ Stop Camera", font=("Helvetica", 11, "bold"),
            bg="#E06C75", fg="#FFFFFF", activebackground="#BE5046", padx=15, pady=8,
            state=tk.DISABLED, command=self.stop_attendance_camera
        )
        self.btn_stop_cam.pack(side=tk.LEFT, padx=5)

        # Right Panel: Stats & Activity Stream
        right_panel = tk.Frame(self.tab_attendance, bg="#2B2B3B", width=340, bd=2, relief=tk.RIDGE)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)
        right_panel.pack_propagate(False)

        stats_title = tk.Label(right_panel, text="Today's Overview", font=("Helvetica", 12, "bold"),
                               fg="#61AFEF", bg="#2B2B3B")
        stats_title.pack(anchor="w", padx=15, pady=10)

        # Cards Frame
        cards_frame = tk.Frame(right_panel, bg="#2B2B3B")
        cards_frame.pack(fill=tk.X, padx=10, pady=5)

        # Total Present Card
        card_present = tk.Frame(cards_frame, bg="#181825", padx=15, pady=10, bd=1, relief=tk.SOLID)
        card_present.pack(fill=tk.X, pady=5)
        tk.Label(card_present, text="MARKED PRESENT TODAY", font=("Helvetica", 9, "bold"), fg="#98C379", bg="#181825").pack(anchor="w")
        self.lbl_stat_present = tk.Label(card_present, text="0", font=("Helvetica", 22, "bold"), fg="#FFFFFF", bg="#181825")
        self.lbl_stat_present.pack(anchor="w")

        # Total Students Card
        card_students = tk.Frame(cards_frame, bg="#181825", padx=15, pady=10, bd=1, relief=tk.SOLID)
        card_students.pack(fill=tk.X, pady=5)
        tk.Label(card_students, text="REGISTERED STUDENTS", font=("Helvetica", 9, "bold"), fg="#61AFEF", bg="#181825").pack(anchor="w")
        self.lbl_stat_total = tk.Label(card_students, text="0", font=("Helvetica", 22, "bold"), fg="#FFFFFF", bg="#181825")
        self.lbl_stat_total.pack(anchor="w")

        # Recent Detections List
        tk.Label(right_panel, text="Live Activity Log", font=("Helvetica", 11, "bold"),
                 fg="#E5C07B", bg="#2B2B3B").pack(anchor="w", padx=15, pady=(15, 5))

        self.activity_box = tk.Text(right_panel, bg="#1E1E2E", fg="#ABB2BF", font=("Consolas", 9),
                                    bd=0, padx=10, pady=10)
        self.activity_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.update_stats_display()

    def update_stats_display(self):
        """Refreshes counter cards and logs."""
        today_records = database.get_today_attendance()
        all_students = database.get_all_students()

        self.lbl_stat_present.config(text=str(len(today_records)))
        self.lbl_stat_total.config(text=str(len(all_students)))

    def log_activity(self, message: str):
        """Appends message to live activity box."""
        timestamp = time.strftime("[%H:%M:%S] ")
        self.activity_box.insert(tk.END, timestamp + message + "\n")
        self.activity_box.see(tk.END)

    def start_attendance_camera(self):
        """Starts real-time attendance scanner loop."""
        if self.is_camera_running:
            return

        self.matcher.load_registered_faces()
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Unable to access webcam device 0.")
            return

        self.is_camera_running = True
        self.btn_start_cam.config(state=tk.DISABLED)
        self.btn_stop_cam.config(state=tk.NORMAL)
        self.status_banner.config(text="Scanning Active", fg="#1E1E2E", bg="#98C379")
        self.log_activity("Attendance scanning started.")

        threading.Thread(target=self.attendance_camera_loop, daemon=True).start()

    def stop_attendance_camera(self):
        """Stops the camera stream."""
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_canvas.config(image="", text="Camera Offline\nClick 'Start Camera' below")
        self.btn_start_cam.config(state=tk.NORMAL)
        self.btn_stop_cam.config(state=tk.DISABLED)
        self.status_banner.config(text="System Ready", fg="#98C379", bg="#282C34")
        self.log_activity("Attendance scanning stopped.")

    def attendance_camera_loop(self):
        """Background thread for continuous frame capture & matching."""
        last_logged = {} # { student_id: timestamp }

        while self.is_camera_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            processed_frame, events = attendance.process_attendance_frame(frame, self.matcher)

            for ev in events:
                sid = ev["student_id"]
                current_t = time.time()
                # Log event at most once every 3 seconds per person
                if sid != "UNKNOWN" and (sid not in last_logged or current_t - last_logged[sid] > 3):
                    last_logged[sid] = current_t
                    self.log_activity(ev["message"])
                    self.root.after(0, self.update_stats_display)

            # Convert BGR OpenCV image to PIL Tkinter format
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_image)
            pil_img = pil_img.resize((640, 440), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(image=pil_img)

            self.root.after(0, self.update_canvas_frame, tk_img)
            time.sleep(0.03) # ~30 FPS

    def update_canvas_frame(self, tk_img):
        """Updates video canvas on main thread."""
        if self.is_camera_running:
            self.video_canvas.config(image=tk_img)
            self.video_canvas.image = tk_img

    # ==========================================
    # TAB 2: STUDENT REGISTRATION
    # ==========================================
    def setup_register_tab(self):
        main_container = tk.Frame(self.tab_register, bg="#1E1E2E")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Form Section (Left)
        form_frame = tk.Frame(main_container, bg="#2B2B3B", width=400, bd=2, relief=tk.RIDGE)
        form_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        form_frame.pack_propagate(False)

        tk.Label(form_frame, text="Register New Student", font=("Helvetica", 14, "bold"),
                 fg="#61AFEF", bg="#2B2B3B").pack(anchor="w", padx=20, pady=20)

        # Student ID Entry
        tk.Label(form_frame, text="Student Roll No / ID *", font=("Helvetica", 10, "bold"),
                 fg="#ABB2BF", bg="#2B2B3B").pack(anchor="w", padx=20, pady=(10, 2))
        self.entry_sid = tk.Entry(form_frame, font=("Helvetica", 11), bg="#181825", fg="#FFFFFF",
                                  insertbackground="white", bd=1, relief=tk.SOLID)
        self.entry_sid.pack(fill=tk.X, padx=20, pady=5)

        # Full Name Entry
        tk.Label(form_frame, text="Full Name *", font=("Helvetica", 10, "bold"),
                 fg="#ABB2BF", bg="#2B2B3B").pack(anchor="w", padx=20, pady=(10, 2))
        self.entry_name = tk.Entry(form_frame, font=("Helvetica", 11), bg="#181825", fg="#FFFFFF",
                                   insertbackground="white", bd=1, relief=tk.SOLID)
        self.entry_name.pack(fill=tk.X, padx=20, pady=5)

        # Department Entry
        tk.Label(form_frame, text="Department", font=("Helvetica", 10, "bold"),
                 fg="#ABB2BF", bg="#2B2B3B").pack(anchor="w", padx=20, pady=(10, 2))
        self.entry_dept = tk.Entry(form_frame, font=("Helvetica", 11), bg="#181825", fg="#FFFFFF",
                                   insertbackground="white", bd=1, relief=tk.SOLID)
        self.entry_dept.insert(0, "Computer Science")
        self.entry_dept.pack(fill=tk.X, padx=20, pady=5)

        # Action Buttons
        self.btn_capture = tk.Button(
            form_frame, text="📷 Start Face Capture", font=("Helvetica", 11, "bold"),
            bg="#4E73DF", fg="#FFFFFF", activebackground="#2E59D9", padx=15, pady=8,
            command=self.start_registration_process
        )
        self.btn_capture.pack(fill=tk.X, padx=20, pady=(20, 5))

        self.btn_delete = tk.Button(
            form_frame, text="🗑️ Remove Student Profile", font=("Helvetica", 10, "bold"),
            bg="#E06C75", fg="#FFFFFF", activebackground="#BE5046", padx=15, pady=6,
            command=self.delete_student_action
        )
        self.btn_delete.pack(fill=tk.X, padx=20, pady=5)

        self.reg_status_lbl = tk.Label(form_frame, text="", font=("Helvetica", 10),
                                       fg="#98C379", bg="#2B2B3B", wraplength=340)
        self.reg_status_lbl.pack(padx=20, pady=10)

        # Preview Section (Right)
        preview_frame = tk.Frame(main_container, bg="#2B2B3B", bd=2, relief=tk.RIDGE)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(preview_frame, text="Camera Preview (Registration)", font=("Helvetica", 12, "bold"),
                 fg="#98C379", bg="#2B2B3B").pack(anchor="w", padx=15, pady=10)

        self.reg_canvas = tk.Label(preview_frame, bg="#1E1E2E", text="Click 'Start Face Capture'\nto record face samples.",
                                   font=("Helvetica", 13), fg="#A0A0B0")
        self.reg_canvas.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

    def start_registration_process(self):
        """Handles student face capture thread."""
        sid = self.entry_sid.get().strip()
        name = self.entry_name.get().strip()
        dept = self.entry_dept.get().strip()

        if not sid or not name:
            messagebox.showwarning("Missing Fields", "Please enter both Student ID and Full Name.")
            return

        if self.is_camera_running:
            self.stop_attendance_camera()

        self.btn_capture.config(state=tk.DISABLED)
        self.reg_status_lbl.config(text="Capturing 5 face samples... Look at the camera.", fg="#E5C07B")

        def frame_update(frame):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((480, 360), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(image=img)
            self.reg_canvas.config(image=tk_img)
            self.reg_canvas.image = tk_img

        def run_thread():
            success, msg = register.capture_student_faces(sid, name, dept, sample_count=5, frame_callback=frame_update)
            self.root.after(0, self.finish_registration, success, msg)

        threading.Thread(target=run_thread, daemon=True).start()

    def finish_registration(self, success: bool, msg: str):
        """Callback when face capture completes."""
        self.btn_capture.config(state=tk.NORMAL)
        if success:
            self.reg_status_lbl.config(text=msg, fg="#98C379")
            messagebox.showinfo("Success", msg)
            self.entry_sid.delete(0, tk.END)
            self.entry_name.delete(0, tk.END)
            self.update_stats_display()
            self.matcher.load_registered_faces()
        else:
            self.reg_status_lbl.config(text=msg, fg="#E06C75")
            messagebox.showerror("Registration Failed", msg)

    def delete_student_action(self):
        """Handles student deletion request."""
        sid = self.entry_sid.get().strip()
        if not sid:
            messagebox.showwarning("Student ID Required", "Please enter the Student ID/Roll No you wish to remove.")
            return

        student = database.get_student_by_id(sid)
        if not student:
            messagebox.showerror("Not Found", f"No student found with ID '{sid}'.")
            return

        confirm = messagebox.askyesno(
            "Confirm Student Removal",
            f"Are you sure you want to remove student:\n\nID: {sid}\nName: {student['name']}\n\nThis will permanently delete their profile, face samples, and attendance records."
        )

        if confirm:
            success, msg = database.delete_student(sid)
            if success:
                messagebox.showinfo("Student Removed", msg)
                self.entry_sid.delete(0, tk.END)
                self.entry_name.delete(0, tk.END)
                self.update_stats_display()
                self.matcher.load_registered_faces()
            else:
                messagebox.showerror("Error", msg)

    # ==========================================
    # TAB 3: ATTENDANCE REPORTS & LOGS
    # ==========================================
    def setup_reports_tab(self):
        top_bar = tk.Frame(self.tab_reports, bg="#2B2B3B", height=50)
        top_bar.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(top_bar, text="Attendance Log Records", font=("Helvetica", 13, "bold"),
                 fg="#61AFEF", bg="#2B2B3B").pack(side=tk.LEFT, padx=15, pady=10)

        btn_export = tk.Button(
            top_bar, text="📥 Export CSV", font=("Helvetica", 10, "bold"),
            bg="#98C379", fg="#1E1E2E", activebackground="#7EB35A", padx=12, pady=5,
            command=self.export_csv_action
        )
        btn_export.pack(side=tk.RIGHT, padx=15, pady=10)

        btn_refresh = tk.Button(
            top_bar, text="🔄 Refresh Table", font=("Helvetica", 10, "bold"),
            bg="#4E73DF", fg="#FFFFFF", activebackground="#2E59D9", padx=12, pady=5,
            command=self.refresh_reports_table
        )
        btn_refresh.pack(side=tk.RIGHT, padx=5, pady=10)

        # Treeview Table
        table_frame = tk.Frame(self.tab_reports, bg="#1E1E2E")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        columns = ("id", "student_id", "name", "department", "date", "time", "status")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")

        self.tree.heading("id", text="Log ID")
        self.tree.heading("student_id", text="Student ID")
        self.tree.heading("name", text="Full Name")
        self.tree.heading("department", text="Department")
        self.tree.heading("date", text="Date")
        self.tree.heading("time", text="Timestamp")
        self.tree.heading("status", text="Status")

        self.tree.column("id", width=60, anchor="center")
        self.tree.column("student_id", width=120, anchor="center")
        self.tree.column("name", width=200, anchor="w")
        self.tree.column("department", width=160, anchor="w")
        self.tree.column("date", width=120, anchor="center")
        self.tree.column("time", width=120, anchor="center")
        self.tree.column("status", width=120, anchor="center")

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.refresh_reports_table()

    def refresh_reports_table(self):
        """Loads records from database into Treeview table."""
        for row in self.tree.get_children():
            self.tree.delete(row)

        records = database.get_all_attendance()
        for r in records:
            self.tree.insert("", tk.END, values=(
                r["id"], r["student_id"], r["name"], r["department"],
                r["date"], r["time"], r["status"]
            ))

    def export_csv_action(self):
        """Exports attendance report to user selected file location."""
        default_path = os.path.abspath("attendance_report.csv")
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialfile="attendance_report.csv",
            title="Save Attendance Report CSV"
        )

        if file_path:
            saved_path = database.export_to_csv(file_path)
            messagebox.showinfo("Export Success", f"Attendance report exported to:\n{saved_path}")

def main():
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_attendance_camera(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()
