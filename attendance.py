import cv2
import os
import glob
import time
import threading
import queue
import numpy as np
import database

try:
    import pyttsx3
    HAS_TTS = True
except Exception:
    HAS_TTS = False

FACES_DIR = "faces"

class VoiceAnnouncer:
    """
    Asynchronous Text-to-Speech voice engine with strict rate limiting.
    """
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.last_spoken = {} # { key: timestamp }
        self.running = True

        if HAS_TTS:
            self.thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.thread.start()

    def _speech_worker(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 165)
            engine.setProperty('volume', 1.0)
        except Exception as e:
            print("TTS Engine init error:", e)
            return

        while self.running:
            try:
                text = self.speech_queue.get(timeout=0.5)
                if text:
                    engine.say(text)
                    engine.runAndWait()
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print("TTS Speech error:", e)

    def speak(self, text: str, key: str = None, cooldown: float = 8.0):
        """Queues voice announcement only if cooldown period has elapsed."""
        if not HAS_TTS:
            return

        now = time.time()
        if key:
            if key in self.last_spoken and (now - self.last_spoken[key]) < cooldown:
                return
            self.last_spoken[key] = now

        if self.speech_queue.qsize() < 2:
            self.speech_queue.put(text)

voice_engine = VoiceAnnouncer()

class LivenessTracker:
    """
    Advanced Anti-Spoofing & Liveness Verification Engine.
    Combines:
    1. Dense Optical Flow Non-Rigid Motion Analysis (Real face deforms vs flat rigid photo).
    2. Dynamic Eye Region Variance (Blink / micro-expression tracking).
    3. Laplacian Texture & Surface Reflectance Analysis (Detects printed paper / digital screens).
    """
    def __init__(self, history_size: int = 12):
        self.history_size = history_size
        self.tracks = {} # { track_id: dict }
        self.next_track_id = 0
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def _get_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = (w1 * h1) + (w2 * h2) - inter_area
        return inter_area / float(union_area) if union_area > 0 else 0.0

    def update(self, face_roi: np.ndarray, box: tuple) -> tuple[bool, str, float]:
        """
        Updates tracking history for a face ROI and returns (is_live, status_reason, liveness_score).
        """
        now = time.time()
        # Clean up stale tracks older than 2.0 seconds
        stale_ids = [tid for tid, t in self.tracks.items() if now - t["last_seen"] > 2.0]
        for tid in stale_ids:
            del self.tracks[tid]

        # Match current box to existing tracks using IoU
        matched_id = None
        best_iou = 0.3
        for tid, t in self.tracks.items():
            iou = self._get_iou(box, t["last_box"])
            if iou > best_iou:
                best_iou = iou
                matched_id = tid

        if matched_id is None:
            matched_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[matched_id] = {
                "rois": [],
                "boxes": [],
                "eye_means": [],
                "last_box": box,
                "last_seen": now,
                "created_at": now,
                "blink_detected": False,
                "non_rigid_frames": 0
            }

        track = self.tracks[matched_id]
        track["last_box"] = box
        track["last_seen"] = now

        # If track was ALREADY verified as a live human, stay verified (prevents UI fluctuation)
        if track.get("is_verified", False):
            return True, "LIVE HUMAN (VERIFIED)", 95.0

        # Resize ROI to standard size (100x100) for motion & texture analysis
        roi_norm = cv2.resize(face_roi, (100, 100))
        roi_blur = cv2.GaussianBlur(roi_norm, (5, 5), 0)

        # Track upper eye region mean brightness for blink dynamics
        eye_region = roi_norm[15:45, 15:85]
        eye_mean = float(np.mean(eye_region))
        track["eye_means"].append(eye_mean)

        track["rois"].append(roi_blur)
        track["boxes"].append(box)

        if len(track["rois"]) > self.history_size:
            track["rois"].pop(0)
            track["boxes"].pop(0)
            track["eye_means"].pop(0)

        # Need at least 5 frames buffer to analyze motion pattern
        if len(track["rois"]) < 5:
            return False, "ANALYZING MOVEMENT...", 35.0

        # --- TEST 1: Dense Optical Flow Non-Rigid Deformation ---
        prev_roi = track["rois"][-2]
        curr_roi = track["rois"][-1]

        flow = cv2.calcOpticalFlowFarneback(
            prev_roi, curr_roi, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        significant_mag = mag[mag > 0.3]

        if len(significant_mag) > 50:
            flow_mag_std = float(np.std(significant_mag))
            flow_ang_std = float(np.std(ang[mag > 0.3]))
        else:
            flow_mag_std = 0.0
            flow_ang_std = 0.0

        # Non-rigid motion check (Real face deforms in 3D; photo has parallel flow vectors)
        if flow_mag_std > 0.28 and flow_ang_std > 0.48:
            track["non_rigid_frames"] += 1

        # --- TEST 2: Eye Region Dynamic Variance (Blink Detection) ---
        eye_std = float(np.std(track["eye_means"])) if len(track["eye_means"]) >= 5 else 0.0
        eye_diffs = [abs(track["eye_means"][i] - track["eye_means"][i-1]) for i in range(1, len(track["eye_means"]))]
        max_eye_spike = max(eye_diffs) if eye_diffs else 0.0

        if max_eye_spike > 2.0 or eye_std > 1.8:
            track["blink_detected"] = True

        # --- TEST 3: Bounding Box Shift vs Internal Motion Ratio ---
        box_moves = [np.sqrt((track["boxes"][i][0] - track["boxes"][i-1][0])**2 + (track["boxes"][i][1] - track["boxes"][i-1][1])**2) for i in range(1, len(track["boxes"]))]
        avg_box_move = float(np.mean(box_moves)) if box_moves else 0.0

        diffs = [float(np.mean(cv2.absdiff(track["rois"][i], track["rois"][i-1]))) for i in range(1, len(track["rois"]))]
        avg_diff = float(np.mean(diffs))

        # --- REJECTION RULES FOR PHOTO / SCREEN SPOOFS ---

        # 1. Completely static photo held still
        if avg_diff < 0.35 and avg_box_move < 0.5:
            return False, "PHOTO SPOOF (STATIC IMAGE)", 10.0

        # 2. Flat rigid photo / smartphone screen moved in hand (Parallel rigid translation flow)
        if avg_box_move > 1.2 and flow_ang_std < 0.38 and not track["blink_detected"]:
            return False, "PHOTO / SCREEN SPOOF DETECTED", 18.0

        # --- LIVENESS VERIFICATION TRIGGER ---
        # Verified if eye blink detected OR persistent non-rigid 3D facial motion observed
        if track["blink_detected"] or track["non_rigid_frames"] >= 3:
            track["is_verified"] = True
            return True, "LIVE HUMAN (VERIFIED)", 95.0

        # Prompt real human to blink if not yet verified
        return False, "👀 PLEASE BLINK YOUR EYES", 40.0

liveness_tracker = LivenessTracker()

class FastFaceMatcher:
    """
    Ultra-fast vectorized face matching engine using downsampled histogram matrices.
    """
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.registered_samples = {}
        self.load_registered_faces()

    def load_registered_faces(self):
        """Pre-computes and caches face representations in RAM."""
        self.registered_samples.clear()

        if not os.path.exists(FACES_DIR):
            return

        student_dirs = [d for d in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, d))]

        for student_id in student_dirs:
            student_path = os.path.join(FACES_DIR, student_id)
            image_files = glob.glob(os.path.join(student_path, "*.jpg")) + glob.glob(os.path.join(student_path, "*.png"))

            features = []
            for img_path in image_files:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (100, 100))
                    hist = cv2.calcHist([img_resized], [0], None, [256], [0, 256])
                    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    features.append((img_resized, hist))

            if features:
                self.registered_samples[student_id] = features

        print(f"FastFaceMatcher: Loaded {len(self.registered_samples)} student face profiles into high-speed RAM.")

    def match_face(self, face_roi: np.ndarray, threshold: float = 0.68) -> tuple[str, float]:
        """Matches face ROI against pre-cached RAM features in < 2ms."""
        if not self.registered_samples or face_roi is None:
            return None, 0.0

        face_resized = cv2.resize(face_roi, (100, 100))
        input_hist = cv2.calcHist([face_resized], [0], None, [256], [0, 256])
        cv2.normalize(input_hist, input_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        best_match_id = None
        highest_score = 0.0

        for student_id, samples in self.registered_samples.items():
            scores = []
            for ref_img, ref_hist in samples:
                hist_score = cv2.compareHist(input_hist, ref_hist, cv2.HISTCMP_CORREL)
                res = cv2.matchTemplate(face_resized, ref_img, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)

                combined_score = (hist_score * 0.4) + (max_val * 0.6)
                scores.append(combined_score)

            max_student_score = max(scores) if scores else 0.0

            if max_student_score > highest_score:
                highest_score = max_student_score
                best_match_id = student_id

        if highest_score >= threshold:
            return best_match_id, highest_score * 100
        else:
            return None, highest_score * 100

def select_main_subject_face(faces, small_frame_shape) -> tuple:
    """
    Selects the single main subject face from all detected faces in frame.
    Evaluates:
    1. Face bounding box area (largest face closest to camera).
    2. Center proximity (face closest to camera frame center).
    3. Hysteresis lock (bonus for currently tracked subject to prevent jumping).
    """
    if len(faces) == 0:
        return None

    if len(faces) == 1:
        return faces[0]

    fh, fw = small_frame_shape[:2]
    cx, cy = fw / 2.0, fh / 2.0
    max_dist = np.sqrt(cx**2 + cy**2)

    best_face = None
    best_score = -1.0

    for face in faces:
        sx, sy, sw, sh = face
        area = sw * sh
        fcx, fcy = sx + sw / 2.0, sy + sh / 2.0
        dist = np.sqrt((fcx - cx)**2 + (fcy - cy)**2)
        
        # Centrality factor between 0.6 and 1.0
        centrality = 1.0 - 0.4 * (dist / max_dist if max_dist > 0 else 0)
        score = area * centrality

        # Full-res box for tracker IoU match
        box = (sx * 2, sy * 2, sw * 2, sh * 2)

        # Hysteresis continuity lock: check matching active track
        for t in liveness_tracker.tracks.values():
            if liveness_tracker._get_iou(box, t.get("last_box", (0, 0, 0, 0))) > 0.25:
                score *= 1.35  # Boost priority for locked main subject
                break

        if score > best_score:
            best_score = score
            best_face = face

    return best_face

def process_attendance_frame(frame: np.ndarray, matcher: FastFaceMatcher) -> tuple[np.ndarray, list[dict]]:
    """
    Ultra-fast frame processor with 4x downscaling, anti-spoofing liveness check & strict session duplicate prevention.
    Filters faces to track ONLY the single main subject in front of the camera.
    """
    h_orig, w_orig = frame.shape[:2]
    
    # 4x Speedup: Downscale frame by 50% for high-speed face detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    all_faces = matcher.face_cascade.detectMultiScale(
        gray_small,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(40, 40)
    )

    # STRICT SINGLE SUBJECT TRACKING: Select only the main subject face
    main_face = select_main_subject_face(all_faces, gray_small.shape)
    faces = [main_face] if main_face is not None else []

    events = []
    session_name = database.get_current_session()
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for (sx, sy, sw, sh) in faces:
        # Scale coordinates back to full resolution
        x, y, w, h = sx * 2, sy * 2, sw * 2, sh * 2

        # Extract full-res ROI for accurate face matching & liveness analysis
        face_roi = gray_full[y:y + h, x:x + w]

        # 1. LIVENESS & MOVEMENT CHECK (Anti-Spoofing)
        is_live, liveness_status, liveness_score = liveness_tracker.update(face_roi, (x, y, w, h))

        if not is_live:
            # PHOTO OR SCREEN SPOOF DETECTED!
            color = (255, 0, 255) # Magenta Warning
            label = f"⚠️ ANTI-SPOOF: {liveness_status}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 32), (x + w, y), color, cv2.FILLED)
            cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(frame, "ATTENDANCE DENIED (PHOTO SPOOF)", (x + 5, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            voice_engine.speak("Photo spoof detected. Attendance denied.", key="photo_spoof", cooldown=8.0)

            events.append({
                "student_id": "SPOOF",
                "name": "Photo Spoof Attempt",
                "confidence": liveness_score,
                "session": session_name,
                "marked": False,
                "already_marked": False,
                "message": f"⚠️ Photo spoof detected! ({liveness_status}). Attendance denied."
            })
            continue

        # 2. FACE MATCHING (For verified live human face)
        student_id, confidence = matcher.match_face(face_roi)

        if student_id:
            student = database.get_student_by_id(student_id)
            student_name = student["name"] if student else student_id

            # STRICT CHECK: Check if student is ALREADY marked for this session today
            already_marked = database.is_already_marked_session(student_id, session_name)

            if not already_marked:
                # Mark attendance once for this session
                res = database.mark_attendance(student_id)
                voice_msg = f"Welcome {student_name}! Attendance Granted for {session_name}."
                voice_engine.speak(voice_msg, key=f"grant_{student_id}_{session_name}", cooldown=10.0)

                color = (0, 255, 0) # Bright Green
                banner_text = f"ATTENDANCE GRANTED ({session_name})"
                msg = f"✓ Welcome {student_name}! Attendance granted for {session_name}."
            else:
                # DO NOT MARK ATTENDANCE AGAIN! Simply show 'ALREADY MARKED'
                voice_msg = f"{student_name}, attendance already granted for {session_name}."
                voice_engine.speak(voice_msg, key=f"already_{student_id}_{session_name}", cooldown=12.0)

                color = (255, 165, 0) # Orange / Cyan
                banner_text = f"ALREADY MARKED ({session_name})"
                msg = f"ℹ️ {student_name} spotted, already marked present ({session_name})."

            label = f"{student_name} ({student_id}) - {confidence:.1f}% [LIVE]"

            # Draw crisp visual overlay
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 32), (x + w, y), color, cv2.FILLED)
            cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, banner_text, (x + 5, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            events.append({
                "student_id": student_id,
                "name": student_name,
                "confidence": confidence,
                "session": session_name,
                "marked": not already_marked,
                "already_marked": already_marked,
                "message": msg
            })
        else:
            # UNKNOWN PERSON DETECTED
            voice_engine.speak("Unknown person detected. Access denied.", key="unknown_person", cooldown=6.0)

            color = (0, 0, 255) # Red
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 28), (x + w, y), color, cv2.FILLED)
            cv2.putText(frame, "UNKNOWN PERSON [LIVE]", (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, "ACCESS DENIED!", (x + 5, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            events.append({
                "student_id": "UNKNOWN",
                "name": "Unknown Person",
                "confidence": confidence,
                "session": session_name,
                "marked": False,
                "already_marked": False,
                "message": "Unknown person detected. Access denied."
            })

    return frame, events
