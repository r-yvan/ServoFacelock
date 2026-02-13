import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import onnxruntime as ort
import pickle
import os
from datetime import datetime
import traceback
import paho.mqtt.client as mqtt
import json
import time

# Get the project root directory (parent of 'src')
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===================== CONFIGURATION =====================
THRESHOLD = 0.62  # Face recognition similarity threshold
TARGET_NAME = input("Enter the identity to lock onto (e.g., your name): ").strip().lower()
MISS_TOLERANCE = 20  # Frames to tolerate no target before unlock
MOVEMENT_THRESHOLD = 40  # Adjusted for full-frame pixel scale
BLINK_EAR_THRESHOLD = 0.21
SMILE_CONFIDENCE_THRESHOLD = 0.65
CONSECUTIVE_SMILE_FRAMES = 3
MAX_FACES = 10  # Maximum faces to detect/process per frame

# ===================== MQTT CONFIGURATION =====================
TEAM_ID = "necromancers"  # Your unique team identifier
MQTT_BROKER = "157.173.101.159"  # Your VPS MQTT broker
MQTT_PORT = 1883
MQTT_TOPIC = f"vision/{TEAM_ID}/movement"
MQTT_HEARTBEAT_TOPIC = f"vision/{TEAM_ID}/heartbeat"

# ===================== SERVO CONFIGURATION =====================
SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180
FRAME_WIDTH = 1280  # Camera frame width
SERVO_SMOOTHING = 0.3  # Smoothing factor for servo movement

# ===================== PERFORMANCE OPTIMIZATION =====================
TRACKING_MODE = True  # Start in tracking mode after lock
PROCESS_EVERY_N_FRAMES = 3  # Process every Nth frame when tracking
MIN_FACE_SIZE = 50  # Minimum face size to process
MAX_FACE_SIZE = 500  # Maximum face size to process
STABILITY_THRESHOLD = 5  # Frames of stable position before reducing processing
BBOX_PADDING = 0.25  # Padding around landmarks for bounding box (extra head room)

# ===================== INITIALIZATION =====================
print("Initializing multi-face detection system...")

# Initialize FaceLandmarker detector using Tasks API
base_options = python.BaseOptions(model_asset_path=os.path.join(ROOT_DIR, "models", "face_landmarker.task"))
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=MAX_FACES
)
face_mesh = vision.FaceLandmarker.create_from_options(options)

# Initialize ONNX Runtime session for ArcFace
try:
    model_path = os.path.join(ROOT_DIR, "models", "embedder_arcface.onnx")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    session = ort.InferenceSession(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Alignment references
REF_POINTS = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014],
    [56.0252, 71.7366], [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)
INDICES_5PT = [33, 263, 1, 61, 291]

# Landmark groups
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# ===================== UTILITY FUNCTIONS =====================
def preprocess(aligned):
    img = aligned.astype(np.float32)
    img = (img - 127.5) / 127.5
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def get_embedding(aligned):
    blob = preprocess(aligned)
    emb = session.run(None, {'input.1': blob})[0][0]
    return emb / np.linalg.norm(emb)

def compute_ear(landmarks, eye_indices, h, w):
    points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_indices])
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C) if C > 0 else 0

def detect_smile(landmarks, h, w, baseline_mouth_width=None, baseline_lip_sep=None):
    left_mouth = np.array([landmarks[61].x * w, landmarks[61].y * h])
    right_mouth = np.array([landmarks[291].x * w, landmarks[291].y * h])
    upper_lip_top = np.array([landmarks[13].x * w, landmarks[13].y * h])
    lower_lip_bottom = np.array([landmarks[14].x * w, landmarks[14].y * h])

    mouth_width = np.linalg.norm(left_mouth - right_mouth)
    lip_separation = np.linalg.norm(upper_lip_top - lower_lip_bottom)

    left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    face_width = np.linalg.norm(left_eye - right_eye)

    normalized_width = mouth_width / face_width if face_width > 0 else 0
    normalized_sep = lip_separation / face_width if face_width > 0 else 0

    nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h])
    left_corner_height = left_mouth[1] - nose_tip[1]
    right_corner_height = right_mouth[1] - nose_tip[1]

    smile_score = 0
    if normalized_width > 0.35:
        width_score = min((normalized_width - 0.35) * 10, 1.0)
        smile_score += width_score * 0.4
    if normalized_sep > 0.08:
        sep_score = min((normalized_sep - 0.08) * 20, 1.0)
        smile_score += sep_score * 0.3
    corner_up_score = 0
    if left_corner_height < -5 and right_corner_height < -5:
        corner_up_score = 1.0
    elif left_corner_height < 0 or right_corner_height < 0:
        corner_up_score = 0.6
    smile_score += corner_up_score * 0.3

    if baseline_mouth_width and baseline_lip_sep:
        width_increase = mouth_width / baseline_mouth_width if baseline_mouth_width > 0 else 1
        sep_increase = lip_separation / baseline_lip_sep if baseline_lip_sep > 0 else 1
        if width_increase > 1.15:
            smile_score += min(width_increase - 1, 0.2)
        if sep_increase > 1.3:
            smile_score += min(sep_increase - 1, 0.2)

    smile_score = min(max(smile_score, 0), 1)
    return smile_score > SMILE_CONFIDENCE_THRESHOLD, smile_score, normalized_width, normalized_sep

class ActionDetector:
    def __init__(self):
        self.prev_nose_x = None
        self.baseline_mouth_width = None
        self.baseline_lip_sep = None
        self.smile_frames = 0
        self.blink_frames = 0

    def update_baseline(self, landmarks, h, w):
        left_mouth = np.array([landmarks[61].x * w, landmarks[61].y * h])
        right_mouth = np.array([landmarks[291].x * w, landmarks[291].y * h])
        upper_lip_top = np.array([landmarks[13].x * w, landmarks[13].y * h])
        lower_lip_bottom = np.array([landmarks[14].x * w, landmarks[14].y * h])
        self.baseline_mouth_width = np.linalg.norm(left_mouth - right_mouth)
        self.baseline_lip_sep = np.linalg.norm(upper_lip_top - lower_lip_bottom)

    def detect_actions(self, landmarks, h, w, locked):
        actions = []
        if not locked:
            return actions

        nose_x = int(landmarks[1].x * w)
        if self.prev_nose_x is not None:
            delta_x = nose_x - self.prev_nose_x
            if abs(delta_x) > MOVEMENT_THRESHOLD:
                direction = "right" if delta_x > 0 else "left"
                actions.append(f"moved {direction} ({abs(delta_x):.0f}px)")
        self.prev_nose_x = nose_x

        ear_left = compute_ear(landmarks, LEFT_EYE, h, w)
        ear_right = compute_ear(landmarks, RIGHT_EYE, h, w)
        ear = (ear_left + ear_right) / 2
        if ear < BLINK_EAR_THRESHOLD:
            self.blink_frames += 1
            if self.blink_frames == 2:
                actions.append(f"blink (EAR: {ear:.2f})")
        else:
            self.blink_frames = 0

        is_smiling, smile_score, mouth_width_norm, lip_sep_norm = detect_smile(
            landmarks, h, w, self.baseline_mouth_width, self.baseline_lip_sep
        )
        if is_smiling:
            self.smile_frames += 1
            if self.smile_frames >= CONSECUTIVE_SMILE_FRAMES:
                actions.append(f"smile (score: {smile_score:.2f})")
        else:
            self.smile_frames = 0

        if 0.25 < mouth_width_norm < 0.33 and 0.05 < lip_sep_norm < 0.08:
            self.update_baseline(landmarks, h, w)

        return actions

# ===================== MQTT FUNCTIONS =====================
mqtt_client = None
current_servo_angle = 90  # Start at center position
target_servo_angle = 90

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"✓ Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        # Send initial heartbeat
        send_heartbeat()
    else:
        print(f"✗ Failed to connect to MQTT broker: {rc}")

def on_mqtt_disconnect(client, userdata, rc):
    print(f"✗ Disconnected from MQTT broker: {rc}")

def init_mqtt():
    global mqtt_client
    try:
        mqtt_client = mqtt.Client()
        mqtt_client.on_connect = on_mqtt_connect
        mqtt_client.on_disconnect = on_mqtt_disconnect

        print(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        return True
    except Exception as e:
        print(f"✗ Failed to initialize MQTT: {e}")
        return False

def calculate_servo_angle(face_center_x):
    """Calculate servo angle based on face position in frame"""
    # Map face x-position (0 to FRAME_WIDTH) to servo angle (SERVO_MIN_ANGLE to SERVO_MAX_ANGLE)
    # Invert mapping so that left face movement turns servo left, right movement turns servo right
    normalized_x = face_center_x / FRAME_WIDTH
    angle = SERVO_MAX_ANGLE - (normalized_x * (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE))
    return max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle))

def send_heartbeat():
    """Send heartbeat message to MQTT broker"""
    if mqtt_client:
        heartbeat_msg = {
            "node": "pc",
            "status": "ONLINE",
            "timestamp": int(time.time())
        }
        try:
            mqtt_client.publish(MQTT_HEARTBEAT_TOPIC, json.dumps(heartbeat_msg))
        except Exception as e:
            print(f"Error sending heartbeat: {e}")

def publish_movement(status, confidence=0.0, face_center_x=None):
    """Publish movement status and servo angle to MQTT"""
    global current_servo_angle, target_servo_angle

    if not mqtt_client:
        return

    movement_msg = {
        "status": status,
        "confidence": float(confidence),  # Convert numpy float32 to Python float
        "timestamp": int(time.time())
    }

    # Calculate and add servo angle if we have a face position
    if face_center_x is not None:
        target_servo_angle = calculate_servo_angle(face_center_x)
        # Apply smoothing to reduce jitter
        current_servo_angle = (SERVO_SMOOTHING * target_servo_angle +
                              (1 - SERVO_SMOOTHING) * current_servo_angle)

        movement_msg["servo_angle"] = round(float(current_servo_angle), 1)  # Convert to Python float
        movement_msg["face_position"] = int(face_center_x)

        # Add degree movement information
        angle_change = abs(current_servo_angle - 90)  # Change from center
        movement_msg["degrees_from_center"] = round(float(angle_change), 1)  # Convert to Python float

        if status == "MOVE_LEFT":
            movement_msg["direction"] = "LEFT"
        elif status == "MOVE_RIGHT":
            movement_msg["direction"] = "RIGHT"
        else:
            movement_msg["direction"] = "CENTERED"

    try:
        mqtt_client.publish(MQTT_TOPIC, json.dumps(movement_msg))
        print(f"📡 Published: {movement_msg['status']} | Angle: {movement_msg.get('servo_angle', 'N/A')}° | Direction: {movement_msg.get('direction', 'N/A')}")
    except Exception as e:
        print(f"Error publishing movement: {e}")

# ===================== LOAD FACE DATABASE =====================
try:
    db_path = os.path.join(ROOT_DIR, "data", "db", "face_db.pkl")
    with open(db_path, 'rb') as f:
        db = pickle.load(f)
    reference = {}
    for name, embs in db.items():
        if len(embs) > 0:
            mean_emb = np.mean(np.array(embs), axis=0)
            mean_emb /= np.linalg.norm(mean_emb)
            reference[name.lower()] = mean_emb
    if TARGET_NAME not in reference:
        print(f"Error: {TARGET_NAME} not found in database!")
        print(f"Available identities: {list(reference.keys())}")
        exit(1)
    target_emb = reference[TARGET_NAME]
    print(f"Loaded database with {len(reference)} identities")
except Exception as e:
    print(f"Error loading database: {e}")
    traceback.print_exc()
    exit(1)

# ===================== MAIN LOOP =====================
action_detector = ActionDetector()
locked = False
locked_start = None
miss_count = 0
prev_bbox = None
history_file = None
fps_counter = 0
start_time = datetime.now()

# Performance optimization variables
frame_count = 0
stable_frames = 0
prev_position = None
processing_mode = "full"  # "full" or "tracking"

print("\n" + "=" * 50)
print(f"Target: {TARGET_NAME.capitalize()}")
print("Controls:")
print(" - Press 'q' to quit")
print(" - Press 'r' to manually release lock")
print(" - Colors: Thick Green = locked target | Thin Green = other target instances")
print("           Yellow = other enrolled people | Red = unknown")
print(" - Multi-face detection enabled (up to 10 faces per frame)")
print("=" * 50 + "\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Create resizable window
cv2.namedWindow('Face Locking System', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Locking System', 1280, 720)

print("Face locking window is resizable!")
print("Use mouse to resize or maximize the window")

# Initialize MQTT connection
if not init_mqtt():
    print("Warning: MQTT connection failed. Continuing without servo control.")

# Initialize heartbeat timer
last_heartbeat_time = time.time()
HEARTBEAT_INTERVAL = 30  # Send heartbeat every 30 seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h_frame, w_frame = frame.shape[:2]
    fps_counter += 1
    frame_count += 1

    # Performance optimization: Smart processing decision
    should_process = True

    if locked and processing_mode == "tracking":
        # In tracking mode, only process every Nth frame or if movement detected
        should_process = (frame_count % PROCESS_EVERY_N_FRAMES == 0)

        # Also process if significant movement detected
        if prev_position and should_process == False:
            current_position = (prev_bbox[0] + prev_bbox[2]//2, prev_bbox[1] + prev_bbox[3]//2)
            movement = abs(current_position[0] - prev_position[0]) + abs(current_position[1] - prev_position[1])
            should_process = movement > MOVEMENT_THRESHOLD

    elif locked:
        # Recently locked, check for stability before switching to tracking mode
        if prev_bbox and stable_frames < STABILITY_THRESHOLD:
            current_position = (prev_bbox[0] + prev_bbox[2]//2, prev_bbox[1] + prev_bbox[3]//2)
            if prev_position:
                movement = abs(current_position[0] - prev_position[0]) + abs(current_position[1] - prev_position[1])
                if movement < MOVEMENT_THRESHOLD:
                    stable_frames += 1
                    if stable_frames >= STABILITY_THRESHOLD:
                        processing_mode = "tracking"
                        print(f"✓ Switched to tracking mode (stable for {stable_frames} frames)")

    # Only run face detection if we should process
    recognized_faces = []
    if should_process or not locked:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = face_mesh.detect(mp_image)

        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                # Size filtering - skip small/large detections
                x_coords = np.array([l.x for l in face_landmarks])
                y_coords = np.array([l.y for l in face_landmarks])
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                face_width = (x_max - x_min) * w_frame
                face_height = (y_max - y_min) * h_frame

                if MIN_FACE_SIZE < face_width < MAX_FACE_SIZE and MIN_FACE_SIZE < face_height < MAX_FACE_SIZE:
                    # 5-point alignment on full frame
                    pts = np.array([[face_landmarks[i].x * w_frame,
                                     face_landmarks[i].y * h_frame] for i in INDICES_5PT],
                                   dtype=np.float32)
                    try:
                        M, _ = cv2.estimateAffinePartial2D(pts, REF_POINTS)
                        aligned = cv2.warpAffine(frame, M, (112, 112), flags=cv2.INTER_LINEAR)

                        query_emb = get_embedding(aligned)

                        sim_to_target = np.dot(query_emb, target_emb)
                        sims = {n: np.dot(query_emb, emb) for n, emb in reference.items()}
                        best_sim = max(sims.values()) if sims else -1
                        best_name = max(sims, key=sims.get) if best_sim >= THRESHOLD else "Unknown"

                        # Compute bounding box with padding
                        x_coords = np.array([l.x for l in face_landmarks])
                        y_coords = np.array([l.y for l in face_landmarks])
                        x_min, x_max = np.min(x_coords), np.max(x_coords)
                        y_min, y_max = np.min(y_coords), np.max(y_coords)

                        width = x_max - x_min
                        height = y_max - y_min

                        x_min -= width * BBOX_PADDING
                        x_max += width * BBOX_PADDING
                        y_min -= height * (BBOX_PADDING + 0.2)  # Extra room for forehead
                        y_max += height * BBOX_PADDING

                        x_min_pix = max(0, int(x_min * w_frame))
                        y_min_pix = max(0, int(y_min * h_frame))
                        x_max_pix = min(w_frame, int(x_max * w_frame))
                        y_max_pix = min(h_frame, int(y_max * h_frame))

                        w_bbox = x_max_pix - x_min_pix
                        h_bbox = y_max_pix - y_min_pix

                        if w_bbox <= 0 or h_bbox <= 0:
                            continue

                        recognized_faces.append({
                            'bbox': (x_min_pix, y_min_pix, w_bbox, h_bbox),
                            'name': best_name,
                            'sim': best_sim,
                            'sim_to_target': sim_to_target,
                            'lm': face_landmarks,
                            'aligned': aligned
                        })
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue

    # Update position tracking for optimization
    try:
        if recognized_faces:
            current_bbox = recognized_faces[0]['bbox']
            if prev_position:
                movement = abs((current_bbox[0] + current_bbox[2]//2) - prev_position[0]) + abs((current_bbox[1] + current_bbox[3]//2) - prev_position[1])
                if movement > MOVEMENT_THRESHOLD:
                    stable_frames = 0  # Reset stability if movement detected
            prev_position = (current_bbox[0] + current_bbox[2]//2, current_bbox[1] + current_bbox[3]//2)
        else:
            stable_frames = 0
            prev_position = None
    except Exception as e:
        print(f"Error updating position tracking: {e}")

    # Identify target instances
    target_faces = [fd for fd in recognized_faces if fd['sim_to_target'] >= THRESHOLD]

    locked_face = None
    actions = []

    if target_faces:
        if locked:
            # Continuity: closest to previous center
            prev_cx = prev_bbox[0] + prev_bbox[2] // 2
            prev_cy = prev_bbox[1] + prev_bbox[3] // 2
            def dist(fd):
                cx = fd['bbox'][0] + fd['bbox'][2] // 2
                cy = fd['bbox'][1] + fd['bbox'][3] // 2
                return (cx - prev_cx)**2 + (cy - prev_cy)**2
            locked_face = min(target_faces, key=dist)
        else:
            # New lock: highest similarity
            locked_face = max(target_faces, key=lambda fd: fd['sim_to_target'])
            locked = True
            locked_start = datetime.now()
            miss_count = 0
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_file = os.path.join(ROOT_DIR, "data", f"{TARGET_NAME}_history_{timestamp_str}.txt")
            action_detector.update_baseline(locked_face['lm'], h_frame, w_frame)
            with open(history_file, 'w') as f:
                f.write(f"Face locking started for {TARGET_NAME.capitalize()} at {datetime.now()}\n")
                f.write(f"Initial similarity: {locked_face['sim_to_target']:.4f}\n")
                f.write("-" * 50 + "\n")
            print(f"\n✓ LOCKED onto {TARGET_NAME.capitalize()} (similarity: {locked_face['sim_to_target']:.3f})")
            print(f" History saved to: {history_file}")

        # Detect actions on locked face
        actions = action_detector.detect_actions(
            locked_face['lm'], h_frame, w_frame, locked=True
        )
        if actions and history_file:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            with open(history_file, 'a') as f:
                for action in actions:
                    f.write(f"{timestamp} | {action}\n")

        prev_bbox = locked_face['bbox']
        miss_count = 0

        # Calculate face center and publish movement data
        face_center_x = locked_face['bbox'][0] + locked_face['bbox'][2] // 2
        face_center_y = locked_face['bbox'][1] + locked_face['bbox'][3] // 2

        # Determine movement status based on face position
        frame_center_x = w_frame // 2
        threshold = w_frame // 10  # 10% of frame width as dead zone

        if face_center_x < frame_center_x - threshold:
            status = "MOVE_LEFT"
        elif face_center_x > frame_center_x + threshold:
            status = "MOVE_RIGHT"
        else:
            status = "CENTERED"

        # Publish movement data with confidence
        publish_movement(status, locked_face['sim_to_target'], face_center_x)
    else:
        if locked:
            miss_count += 1
            if miss_count > MISS_TOLERANCE:
                locked = False
                print("\n⚠ Lock released - target disappeared")
                # Publish NO_FACE status
                publish_movement("NO_FACE")
                if history_file and locked_start:
                    duration = datetime.now() - locked_start
                    with open(history_file, 'a') as f:
                        f.write(f"\nLock released at {datetime.now()}\n")
                        f.write(f"Tracking duration: {str(duration).split('.')[0]}\n")
                history_file = None
                locked_start = None
                action_detector = ActionDetector()

    # Candidate for aligned view when searching
    candidate_face = None
    if not locked and recognized_faces:
        candidate_face = max(recognized_faces, key=lambda fd: fd['sim_to_target'])

    # Remove aligned face window - only show main window

    # Draw all faces
    for fd in recognized_faces:
        x, y, w, h = fd['bbox']
        sim_to_target = fd['sim_to_target']

        if locked_face and fd is locked_face:
            color = (0, 255, 0)
            thickness = 4
            text = f"LOCKED: {TARGET_NAME.capitalize()} ({sim_to_target:.3f})"
            for i, action in enumerate(actions[:3]):
                cv2.putText(frame, f"• {action}", (x, y + h + 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        elif sim_to_target >= THRESHOLD:
            color = (0, 255, 0)
            thickness = 2
            text = f"{TARGET_NAME.capitalize()} ({sim_to_target:.3f})"
        elif fd['name'] != "Unknown":
            color = (0, 255, 255)  # Yellow
            thickness = 2
            text = f"{fd['name'].capitalize()} ({fd['sim']:.3f})"
        else:
            color = (0, 0, 255)  # Red
            thickness = 2
            text = f"Unknown ({sim_to_target:.3f})"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(frame, text, (x, max(y - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Log other enrolled people
    other_names = {fd['name'] for fd in recognized_faces
                   if fd['name'] != "Unknown" and fd['sim_to_target'] < THRESHOLD}
    if locked and other_names:
        print(f"Other enrolled detected: {', '.join(sorted(other_names))}")
        if history_file:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            with open(history_file, 'a') as f:
                f.write(f"{timestamp} | Others detected: {', '.join(sorted(other_names))}\n")

    # FPS and status
    elapsed = (datetime.now() - start_time).total_seconds()
    fps = fps_counter / elapsed if elapsed > 0 else 0
    status_line = f"FPS: {fps:.1f} | Status: {'LOCKED' if locked else 'SEARCHING'} | Faces: {len(recognized_faces)}"
    cv2.putText(frame, status_line, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Get current window size and resize frame to match (maintain aspect ratio)
    window_width = cv2.getWindowImageRect('Face Locking System')[2]
    window_height = cv2.getWindowImageRect('Face Locking System')[3]

    if window_width > 0 and window_height > 0:
        h, w = frame.shape[:2]
        aspect_ratio = w / h

        new_width = window_width
        new_height = int(window_width / aspect_ratio)

        if new_height > window_height:
            new_height = window_height
            new_width = int(window_height * aspect_ratio)

        frame_resized = cv2.resize(frame, (new_width, new_height))
        cv2.imshow('Face Locking System', frame_resized)
    else:
        cv2.imshow('Face Locking System', frame)

    # Send periodic heartbeat
    current_time = time.time()
    if current_time - last_heartbeat_time > HEARTBEAT_INTERVAL:
        send_heartbeat()
        last_heartbeat_time = current_time

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nExiting...")
        break
    elif key == ord('r') and locked:
        print("\nManual lock release")
        locked = False
        if history_file and locked_start:
            duration = datetime.now() - locked_start
            with open(history_file, 'a') as f:
                f.write(f"\nManual lock release at {datetime.now()}\n")
                f.write(f"Tracking duration: {str(duration).split('.')[0]}\n")
        history_file = None
        locked_start = None
        action_detector = ActionDetector()

# Cleanup MQTT connection
if mqtt_client:
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("✓ MQTT connection closed")

cap.release()
cv2.destroyAllWindows()
