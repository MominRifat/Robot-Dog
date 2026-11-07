<div align="center">

# 🐕 Robot-Dog — Autonomous Vision & Multi-Sensor Quadruped System

[![Python](https://img.shields.io/badge/Language-Python_3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/Computer_Vision-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Robotics](https://img.shields.io/badge/Hardware-Sensors_%26_Actuators-FF6F00?style=for-the-badge&logo=arduino&logoColor=white)](#)
[![Status](https://img.shields.io/badge/Domain-Robotics_%26_Embedded_AI-brightgreen?style=for-the-badge)](#)

<p align="center">
  <b>An intelligent, vision-guided autonomous robotics project combining Computer Vision with multi-sensor environmental safety interlocks.</b>
  <br />
  Equipped with real-time Image Recognition, Ultrasonic Collision Avoidance, and Active Flame/Fire Hazard Detection.
</p>

[📌 About the Project](#-about-the-project) • [🛠️ Tech Stack & Hardware](#️-comprehensive-tech-stack--hardware-bill-of-materials) • [🤖 Architecture](#-2d3d-perception--control-architecture) • [🚀 How to Run](#-complete-how-to-run--setup-guide)

---

</div>

## 📌 About the Project

**Robot-Dog** is an experimental robotic system designed for autonomous environmental navigation and surveillance. 

Rather than relying purely on pre-programmed pathing, Robot-Dog utilizes real-time optical streams via **OpenCV** to detect target markers and people, while continuous sensor polling (via **HC-SR04 Ultrasonic** and **Infrared Flame sensors**) provides an autonomous emergency stop and collision-avoidance mechanism.

---

## 🛠️ Comprehensive Tech Stack & Hardware Bill of Materials

| Layer / Component | Specification | Purpose & Function |
| :--- | :--- | :--- |
| **Vision & Image Processing** | **OpenCV (`cv2`)** | Real-time optical video capture, frame filtering, contour & object recognition |
| **Numerical Processing** | **NumPy** | High-performance matrix calculations for bounding coordinates and sensor timings |
| **Programming Language** | **Python 3.9+** | Controller logic, sensor polling loop, and motor coordination |
| **Hardware Controller** | **Raspberry Pi 4 / Arduino** | Microcomputer / Microcontroller running real-time I/O interrupts |
| **Vision Sensor** | **HD USB Webcam / Pi Cam** | Captures 30 FPS optical stream for target tracking |
| **Range Finder** | **HC-SR04 Ultrasonic** | Time-of-flight acoustic distance calculation (2 cm – 400 cm range) |
| **Thermal / Fire Hazard** | **KY-026 Flame Sensor** | Phototransistor detecting infrared light in the flame spectrum (760 nm – 1100 nm) |
| **Motor Driving Unit** | **L298N Dual H-Bridge** | High-power motor/servo driver delivering regulated voltage to actuators |

---

## 🤖 2D/3D Perception & Control Architecture

```mermaid
flowchart TD
    subgraph Sensation["👀 Perception Layer (Sensors)"]
        Cam["📷 Vision Camera (Continuous Frames)"]
        Sonar["📡 HC-SR04 Ultrasonic Sensor"]
        Flame["🔥 KY-026 Infrared Flame Sensor"]
    end

    subgraph Processing["🧠 Intelligence Layer (Python + OpenCV)"]
        VisionEngine["OpenCV Object Recognition & Bounding Box"]
        DistanceEngine["Time-of-Flight Distance Compute"]
        SafetyInterlock{"🚨 Flame / Emergency Detected?"}
        Kinematics["Motion & Gait Sequencer"]
    end

    subgraph Actuation["⚡ Output & Actuation Layer"]
        Motors["⚙️ L298N Motor Driver -> Wheels/Gait Servos"]
        Alarm["🚨 Warning Buzzer & Emergency Halt"]
        Log["📊 Real-time Terminal Telemetry Stream"]
    end

    Cam --> VisionEngine
    Sonar --> DistanceEngine
    Flame --> SafetyInterlock

    SafetyInterlock -- "YES: Hazard Detected" --> Alarm
    SafetyInterlock -- "YES: Hazard Detected" -->|Override Halt| Motors
    SafetyInterlock -- "NO: Safe to Proceed" --> Kinematics

    VisionEngine --> Kinematics
    DistanceEngine --> Kinematics
    Kinematics --> Motors
    VisionEngine --> Log
```

---

## 📂 Repository Layout

```plaintext
Robot-Dog/
├── 📁 Img Recognization/                 # OpenCV image detection scripts and test samples
├── 📄 Obstacle Avoid & Fire Sensor.txt   # Hardware pin mappings, sensor loop code & logic
└── 📄 README.md                          # Technical architecture & setup guide
```

---

## 🚀 Complete How-To-Run & Setup Guide

### 1. Software Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MominRifat/Robot-Dog.git
   cd Robot-Dog
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux / Raspberry Pi OS:
   source venv/bin/activate
   ```

3. **Install OpenCV and dependencies:**
   ```bash
   pip install opencv-python numpy
   # If running on Raspberry Pi, also install GPIO:
   # pip install RPi.GPIO
   ```

---

### 2. Hardware Wiring Guide

| Sensor / Module | Sensor Pin | Microcontroller / RPi Pin | Note |
| :--- | :--- | :--- | :--- |
| **HC-SR04 Ultrasonic** | `VCC` | `5V` | Regulated power |
| | `GND` | `GND` | Common ground |
| | `TRIG` | `GPIO 23` (or Arduino D9) | Trigger pulse input |
| | `ECHO` | `GPIO 24` (or Arduino D10)| Voltage divider recommended on Pi (5V to 3.3V) |
| **KY-026 Flame Sensor** | `VCC` | `3.3V / 5V` | Power supply |
| | `GND` | `GND` | Common ground |
| | `DO` (Digital Out) | `GPIO 17` (or Arduino D2) | High/Low trigger on flame detection |
| **L298N Driver** | `IN1, IN2, IN3, IN4`| `GPIO 5, 6, 13, 19` | Direction control logic |

---

### 3. Running the Systems

#### A. Running Real-Time Image Recognition:
Navigate to the image recognition directory and run the detection loop:
```bash
python "Img Recognization/image_recognition.py"
```
*(Press `q` in the camera preview window to safely terminate).*

#### B. Running the Autonomous Navigation & Flame Interlock:
Run the controller script executing distance polling and flame interrupts:
```bash
python "Obstacle Avoid & Fire Sensor.py" # (or load sketch onto Arduino)
```

---

## 👨‍💻 Author

**Momin Rifat**  
- GitHub: [@MominRifat](https://github.com/MominRifat)  
- Focus: Robotics, Computer Vision & Intelligent Embedded Systems
