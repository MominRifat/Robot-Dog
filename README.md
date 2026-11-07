<div align="center">

# 🐕 Robot-Dog — Autonomous Vision & Multi-Sensor Quadruped System

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/Computer_Vision-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Robotics](https://img.shields.io/badge/Hardware-Sensors_%26_Actuators-FF6F00?style=for-the-badge&logo=arduino&logoColor=white)](#)
[![Status](https://img.shields.io/badge/R%26D-Autonomous_Robotics-brightgreen?style=for-the-badge)](#)

<p align="center">
  <b>An intelligent, vision-guided autonomous robotic platform integrating Computer Vision and real-time sensory environmental hazard detection.</b>
  <br />
  Equipped with real-time Image Recognition, Ultrasonic Obstacle Avoidance, and Flame/Fire Detection safety protocols.
</p>

---

</div>

## 📌 Project Overview

**Robot-Dog** represents an applied exploration into embedded robotics and machine vision. By combining optical camera streams with sensory feedback (ultrasonic distance and flame detection), the robot can autonomously perceive its surroundings, navigate dynamic obstacles, and react to emergency fire hazards.

---

## 🤖 2D/3D Perception & Control Architecture

```mermaid
flowchart TD
    subgraph Sensor_Perception["👀 Sensor & Perception Array"]
        Cam["📷 Vision Camera (Image Recognition)"]
        Sonar["📡 Ultrasonic Sensor (Distance / Obstacles)"]
        Fire["🔥 Flame / Infrared Sensor (Fire Detection)"]
    end

    subgraph Decision_Core["🧠 Central Processing Unit (Python / OpenCV)"]
        VisionProc["Image Preprocessing & Feature Extraction"]
        ObstacleLogic["Dynamic Collision Avoidance Routine"]
        HazardSafety["Emergency Hazard Interruption"]
        MotionPlanner["Inverse Kinematics & Gait Sequencer"]
    end

    subgraph Actuation_Outputs["⚡ Actuation & Alerts"]
        Motors["⚙️ Servo & Motor Drivers (Forward / Turn / Reverse)"]
        AudioAlert["🚨 Warning Buzzer & Alarm State"]
        Telemetry["📊 Status Logs & Video Stream"]
    end

    Cam --> VisionProc
    Sonar --> ObstacleLogic
    Fire --> HazardSafety

    VisionProc --> MotionPlanner
    ObstacleLogic --> MotionPlanner
    HazardSafety -->|Override Priority| MotionPlanner
    HazardSafety --> AudioAlert

    MotionPlanner --> Motors
    VisionProc --> Telemetry
```

---

## 🔬 Core Capabilities

### 1. 📷 Real-time Image Recognition
- Frame capture and real-time object/target identification using image processing techniques.
- Boundary box computation and tracking of designated markers or target targets.

### 2. 📡 Ultrasonic Obstacle Avoidance
- Real-time time-of-flight distance calculation ($Distance = \frac{Time \times SpeedOfSound}{2}$).
- Dynamic path adjustment: Slows down on detection within warning threshold (< 30 cm) and executes 90° avoidance sweep if blocked (< 15 cm).

### 3. 🔥 Flame Detection & Safety Interlock
- Active monitoring via analog/digital flame sensors sensitive to infrared flame spectrum (760 nm – 1100 nm).
- Hardware-interrupt driven failsafe: halts navigation immediately and triggers alert signaling upon thermal spike detection.

---

## 📂 Repository Contents

```plaintext
Robot-Dog/
├── 📁 Img Recognization/                 # Computer vision scripts, weights, and detection pipelines
├── 📄 Obstacle Avoid & Fire Sensor.txt   # Hardware pinouts, sensor calibration code, and control logic
└── 📄 README.md                          # Comprehensive technical documentation
```

---

## 🛠️ Hardware Requirements & Software Stack

- **Microcontroller / SBC**: Raspberry Pi 4 / Arduino Mega or Uno
- **Camera Module**: USB Webcam / Raspberry Pi Camera Module V2
- **Sensors**: HC-SR04 Ultrasonic Sensor, Flame Sensor Module (KY-026 / analogue flame sensor)
- **Motor Control**: L298N Motor Driver / Multi-channel PWM Servo Driver
- **Libraries**: `opencv-python`, `numpy`, `RPi.GPIO` (or `pySerial`)

---

## 👨‍💻 Author

**Momin Rifat**  
- GitHub: [@MominRifat](https://github.com/MominRifat)  
- Location: Dhaka, Bangladesh
