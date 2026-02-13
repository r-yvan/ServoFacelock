# Distributed Vision-Control System (Face-Locked Servo)

**Team: necromancers**
**Student: IRASUBIZA Saly Nelson**
**Live Dashboard: http://157.173.101.159:9005/src/dashboard.html**

## Overview

A complete distributed vision-control system that detects and tracks faces using computer vision, publishes movement commands via MQTT, and controls a servo motor to respond accordingly. The system allows multiple teams to operate simultaneously on the same MQTT broker without interference.

## System Architecture

```
┌──────────────────┐    MQTT     ┌─────────────────┐    WebSocket    ┌──────────────────┐
│   PC (Vision     │ ────────►   │  Backend API    │ ────────────►   │  Web Dashboard   │
│     Node)        │             │   Service       │                 │    (Browser)     │
│                  │             │     (VPS)       │                 │                  │
│ • Face Detection │             │ • MQTT Broker   │                 │ • Real-time      │
│ • Tracking       │             │ • WebSocket API │                 │   Visualization  │
│ • MQTT Publish   │             │ • Message Relay │                 │ • Status Display │
└──────────────────┘             └─────────────────┘                 └──────────────────┘
        │
        │ MQTT
        ▼
┌──────────────────┐
│  ESP8266 (Edge   │
│   Controller)    │
│                  │
│ • MQTT Subscribe │
│ • Servo Control  │
│ • Motor Actuation│
└──────────────────┘
```

## Development Phases

### Phase 1: Open-Loop Actuation (Simulation Stage)

- Camera remains fixed on PC
- Servo motor rotates based on face movement
- Camera frame does NOT change when servo rotates
- Validates MQTT communication, topic isolation, servo actuation logic

### Phase 2: Closed-Loop Tracking (Mechanical Integration)

- Camera mounted directly on servo
- Vision frame changes dynamically with servo rotation
- True feedback loop system

## Files Structure

```
face-detection/
├── src/
│   ├── face_locking.py          # PC Vision Node (main face tracking + MQTT)
│   ├── esp8266_servo_controller.py  # ESP8266 Edge Controller
│   ├── websocket_backend.py      # Backend API Service (VPS)
│   ├── dashboard.html           # Web Dashboard
│   ├── enroll.py               # Face enrollment utility
│   └── detect.py               # Face detection utility
├── models/
│   ├── face_landmarker.task     # MediaPipe face detection model
│   └── embedder_arcface.onnx    # Face recognition model
├── data/
│   ├── db/                      # Face database storage
│   └── enroll/                  # Enrolled face images
├── requirements.txt             # PC client dependencies
├── requirements_backend.txt      # Backend dependencies
└── README.md                    # This file
```

## Quick Start

### 1. Setup Face Recognition (Original System)

```bash
# Install dependencies
pip install -r requirements.txt

# Download required models
# Place face_landmarker.task in models/
# Place embedder_arcface.onnx in models/

# Enroll faces
python src/enroll.py

# Test face locking
python src/face_locking.py
```

### 2. Setup Distributed System (New)

#### PC Vision Node Configuration

Edit `src/face_locking.py`:

- `TEAM_ID = "necromancers"` (unique team identifier)
- `MQTT_BROKER = "157.173.101.159"` (VPS IP address)

#### ESP8266 Edge Controller

Edit `src/esp8266_servo_controller.ino`:

- `TEAM_ID = "necromancers"` (must match PC)
- `MQTT_BROKER = "157.173.101.159"` (VPS IP address)
- `WIFI_SSID = "RCA"` (WiFi network)
- `WIFI_PASSWORD = "@RcaNyabihu2023"` (WiFi password)

#### Backend API Service (VPS)

```bash
# Install MQTT broker
sudo apt install mosquitto mosquitto-clients
sudo systemctl start mosquitto
sudo ufw allow 1883

# Install Python dependencies
pip install websockets paho-mqtt

# Run backend service
python src/websocket_backend.py
```

#### Web Dashboard

Open `src/dashboard.html` in browser - WebSocket URL configured to:

```
ws://157.173.101.159:9005
```

## MQTT Topic Structure

### Movement Messages (PC → Broker)

**Topic:** `vision/necromancers/movement`

**Payload Example:**

```json
{
  "status": "MOVE_LEFT",
  "confidence": 0.87,
  "timestamp": 1730000000,
  "servo_angle": 45.5,
  "face_position": 320,
  "degrees_from_center": 44.5,
  "direction": "LEFT"
}
```

**Status Values:**

- `MOVE_LEFT`: Face detected on left side
- `MOVE_RIGHT`: Face detected on right side
- `CENTERED`: Face centered in frame
- `NO_FACE`: No target face detected

### Heartbeat Messages (Any Node → Broker)

**Topic:** `vision/necromancers/heartbeat`

**Payload Example:**

```json
{
  "node": "pc",
  "status": "ONLINE",
  "timestamp": 1730000000
}
```

## Hardware Setup

### ESP8266 Connections

```
ESP8266    Servo Motor
GPIO2      Signal (Orange/Yellow)
3V3        Power (Red)
GND        Ground (Brown/Black)
```

## System Operation

1. **Start Backend Service** (VPS)
2. **Power on ESP8266** (auto-connects to WiFi/MQTT)
3. **Run PC Vision Node** (`python src/face_locking.py`)
4. **Open Web Dashboard** (real-time visualization)

## Features

### Real-time Face Tracking

- Multi-face detection (up to 10 faces)
- Face recognition with similarity scoring
- Smooth servo movement with anti-jitter
- Performance optimization with tracking mode

### MQTT Communication

- Reliable message publishing
- Automatic reconnection
- Team topic isolation
- Heartbeat monitoring

### Web Dashboard

- Real-time WebSocket updates
- Visual servo position indicator
- Movement status display
- System heartbeat monitoring

## Original Face Recognition Details

### Validation Stages (Run in order)

- `python src/camera.py`
- `python src/detect.py`
- `python src/landmarks.py`
- `python src/haar_5pt.py`
- `python src/align.py`
- `python src/embed.py`

### Key Details

- 5-point alignment using similarity transform
- Embeddings L2-normalized, mean per identity
- Threshold: 0.62 (from evaluation)
- Unknown faces rejected properly

### Face Locking Features

- Normal recognition until high-confidence match → lock
- Once locked, focuses on same face region
- Tolerates brief misses up to 20 frames
- Logs actions to `data/<name>_history_*.txt`

### Actions Detected

- Left/right movement: Nose x-position change >30 pixels
- Blink: Average Eye Aspect Ratio (EAR) <0.25
- Smile/laugh: Mouth width / inter-eye distance >1.8

## Golden Rule

**Vision computes. Devices speak MQTT. Browsers speak WebSocket. The backend relays in real time.**
