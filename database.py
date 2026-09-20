import sqlite3
import os
import shutil
import pandas as pd
from datetime import datetime

DB_PATH = "attendance.db"

def get_connection():
    """Returns SQLite database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_current_session() -> str:
    """
    Determines session based on current hour:
    Before 1:00 PM (13:00) -> 'Before Lunch'
    1:00 PM onwards -> 'After Lunch'
    """
    hour = datetime.now().hour
    return "Before Lunch" if hour < 13 else "After Lunch"

def init_db():
    """Initializes SQLite database with session support."""
    conn = get_connection()
    cursor = conn.cursor()

    # Students Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # Migration check: Add session column if missing
    cursor.execute("PRAGMA table_info(attendance)")
    columns = [row['name'] for row in cursor.fetchall()]

    if not columns:
        # Create Attendance Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                session TEXT NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students (student_id),
                UNIQUE(student_id, date, session)
            )
        """)
    elif "session" not in columns:
        cursor.execute("ALTER TABLE attendance ADD COLUMN session TEXT DEFAULT 'Before Lunch'")

    conn.commit()
    conn.close()

def add_student(student_id: str, name: str, department: str = "General"):
    """Adds a new student to the database."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT OR REPLACE INTO students (student_id, name, department, created_at)
        VALUES (?, ?, ?, ?)
    """, (student_id.strip(), name.strip(), department.strip(), created_at))

    conn.commit()
    conn.close()

def get_all_students():
    """Returns a list of all registered students."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students ORDER BY student_id ASC")
    students = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return students

def get_student_by_id(student_id: str):
    """Fetches student info by ID."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def delete_student(student_id: str) -> tuple[bool, str]:
    """
    Deletes a student profile from database, attendance records, and face images folder.
    """
    init_db()
    student_id = student_id.strip()
    student = get_student_by_id(student_id)
    if not student:
        return False, f"Student ID '{student_id}' not found."

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
        conn.commit()
        conn.close()

        # Delete face samples folder
        student_dir = os.path.join("faces", student_id)
        if os.path.exists(student_dir):
            shutil.rmtree(student_dir, ignore_errors=True)

        return True, f"Successfully removed student {student['name']} ({student_id}) and all associated records."
    except Exception as e:
        conn.close()
        return False, f"Failed to remove student: {str(e)}"

def is_already_marked_session(student_id: str, session: str = None) -> bool:
    """Checks if attendance is already marked for today in this session."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    today_date = datetime.now().strftime("%Y-%m-%d")
    target_session = session or get_current_session()

    cursor.execute("""
        SELECT 1 FROM attendance 
        WHERE student_id = ? AND date = ? AND session = ?
    """, (student_id, today_date, target_session))

    marked = cursor.fetchone() is not None
    conn.close()
    return marked

def mark_attendance(student_id: str, status: str = "Present") -> dict:
    """
    Marks attendance for student in the current session (Before Lunch / After Lunch).
    Returns dict with status details.
    """
    init_db()
    today_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    session = get_current_session()

    student = get_student_by_id(student_id)
    if not student:
        return {
            "success": False,
            "already_marked": False,
            "session": session,
            "message": f"Student ID '{student_id}' not found."
        }

    if is_already_marked_session(student_id, session):
        return {
            "success": False,
            "already_marked": True,
            "student_name": student["name"],
            "session": session,
            "message": f"{student['name']}, attendance already granted ({session})."
        }

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO attendance (student_id, date, time, session, status)
            VALUES (?, ?, ?, ?, ?)
        """, (student_id, today_date, current_time, session, status))
        conn.commit()
        conn.close()
        return {
            "success": True,
            "already_marked": False,
            "student_name": student["name"],
            "session": session,
            "message": f"Welcome {student['name']}! Attendance Granted ({session})."
        }
    except sqlite3.IntegrityError:
        conn.close()
        return {
            "success": False,
            "already_marked": True,
            "student_name": student["name"],
            "session": session,
            "message": f"{student['name']}, attendance already granted ({session})."
        }

def get_today_attendance(session_filter: str = None):
    """Returns today's attendance records, optionally filtered by session."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    today_date = datetime.now().strftime("%Y-%m-%d")

    if session_filter and session_filter.lower() != 'all':
        cursor.execute("""
            SELECT a.id, a.student_id, s.name, s.department, a.date, a.time, a.session, a.status
            FROM attendance a
            JOIN students s ON a.student_id = s.student_id
            WHERE a.date = ? AND LOWER(a.session) = LOWER(?)
            ORDER BY a.time DESC
        """, (today_date, session_filter))
    else:
        cursor.execute("""
            SELECT a.id, a.student_id, s.name, s.department, a.date, a.time, a.session, a.status
            FROM attendance a
            JOIN students s ON a.student_id = s.student_id
            WHERE a.date = ?
            ORDER BY a.time DESC
        """, (today_date,))

    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return records

def get_all_attendance():
    """Returns all attendance records."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT a.id, a.student_id, s.name, s.department, a.date, a.time, a.session, a.status
        FROM attendance a
        JOIN students s ON a.student_id = s.student_id
        ORDER BY a.date DESC, a.time DESC
    """)

    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return records

def export_to_excel(filename: str = "attendance_report.xlsx", session_filter: str = "all") -> str:
    """
    Exports attendance records to an Excel file (.xlsx) with multi-sheet support:
    - Sheet 1: Before Lunch Attendance
    - Sheet 2: After Lunch Attendance
    - Sheet 3: Full Combined Log
    """
    init_db()
    records = get_all_attendance()
    df_all = pd.DataFrame(records)

    if df_all.empty:
        df_all = pd.DataFrame(columns=["id", "student_id", "name", "department", "date", "time", "session", "status"])

    df_before = df_all[df_all["session"] == "Before Lunch"] if "session" in df_all.columns else pd.DataFrame()
    df_after = df_all[df_all["session"] == "After Lunch"] if "session" in df_all.columns else pd.DataFrame()

    filepath = os.path.abspath(filename)
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_before.to_excel(writer, sheet_name="Before Lunch", index=False)
        df_after.to_excel(writer, sheet_name="After Lunch", index=False)
        df_all.to_excel(writer, sheet_name="All Logs Summary", index=False)

    return filepath

if __name__ == "__main__":
    init_db()
    print("Database session schema initialized!")
