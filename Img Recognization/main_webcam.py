import os
import sys
import time
import argparse
import cv2
import numpy as np
from datetime import datetime

from engine.arcface_engine import ArcFaceEngine
from engine.face_database import FaceDatabase
from engine.sound_alert import SoundAlertSystem


def draw_styled_bbox(frame, bbox, color, label=None, sub_label=None):
    """Draws a clean modern bounding box with corner accents and label badge."""
    x, y, w, h = bbox
    # Main rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Corner accents
    line_len = min(15, int(w * 0.15), int(h * 0.15))
    t = 2
    # Top-Left
    cv2.line(frame, (x, y), (x + line_len, y), color, t)
    cv2.line(frame, (x, y), (x, y + line_len), color, t)
    # Top-Right
    cv2.line(frame, (x + w, y), (x + w - line_len, y), color, t)
    cv2.line(frame, (x + w, y), (x + w, y + line_len), color, t)
    # Bottom-Left
    cv2.line(frame, (x, y + h), (x + line_len, y + h), color, t)
    cv2.line(frame, (x, y + h), (x, y + h - line_len), color, t)
    # Bottom-Right
    cv2.line(frame, (x + w, y + h), (x + w - line_len, y + h), color, t)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - line_len), color, t)

    # Label badge above the box
    if label:
        badge_text = label
        if sub_label:
            badge_text += f" | {sub_label}"

        (tw, th), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bg_y1 = max(0, y - th - 8)
        bg_y2 = y
        cv2.rectangle(frame, (x, bg_y1), (x + tw + 6, bg_y2), color, -1)
        cv2.putText(frame, badge_text, (x + 3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description="Real-Time Face Recognition with Barking Alert")
    parser.add_argument("--cam", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--url", type=str, default=None, help="ESP32-CAM stream URL or video path")
    parser.add_argument("--threshold", type=float, default=0.50, help="Matching similarity threshold (default: 0.50)")
    parser.add_argument("--cooldown", type=float, default=2.5, help="Bark alert cooldown in seconds (default: 2.5)")
    parser.add_argument("--no-sound", action="store_true", help="Disable sound alerts")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="Rotate video (0, 90, 180, 270 degrees)")

    args = parser.parse_args()
    current_threshold = args.threshold
    current_rotation = args.rotate
    is_mirrored = False
    is_vflip = False

    window_name = "Face Recognition & Intruder Alert (ArcFace)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n" + "=" * 60)
    print("  ArcFace Real-Time Face Recognition & Intruder Alert")
    print("=" * 60)

    # Initialize Engine, Database and Alert system
    engine = ArcFaceEngine(model_dir='models')
    db = FaceDatabase()
    sound_alert = SoundAlertSystem(cooldown_sec=args.cooldown)

    # Auto-scan 'known_faces' folder on launch
    if os.path.exists('known_faces') and os.listdir('known_faces'):
        print("[FaceDB] Syncing images from 'known_faces' folder...")
        db.scan_known_faces_directory(engine, known_faces_dir='known_faces')

    registered_people = db.get_people()
    print(f"[FaceDB] Registered identities ({len(registered_people)}): {', '.join(registered_people) if registered_people else 'None'}")

    # Open video source (Webcam, Phone IP Camera, or ESP32-CAM)
    source = args.url if args.url else args.cam

    # Auto-complete IP Webcam URL if user only passes ip:port (e.g. http://192.168.0.5:8080)
    if isinstance(source, str) and source.startswith("http://") and ":8080" in source and not source.endswith(("/video", "/stream")):
        source = source.rstrip("/") + "/video"
        print(f"[Camera] Auto-formatted IP camera URL to: {source}")

    print(f"[Camera] Connecting to source: {source} ...")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[Error] Could not open video source: {source}")
        print("Tip: If using laptop webcam, try `--cam 0` or `--cam 1`.")
        print("Tip: If using Phone IP Webcam, pass `--url http://<phone_ip>:8080/video`")
        print("Tip: If using ESP32-CAM, pass `--url http://<esp32_ip>:81/stream`")
        return

    # Minimize stream latency buffer
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if args.url is None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n[Controls]")
    print("  [Q] or [ESC] : Quit application")
    print("  [T]          : Rotate 90° clockwise (Press T to fix sideways camera)")
    print("  [V]          : Flip vertically (Press V if camera is upside-down!)")
    print("  [M]          : Mirror flip horizontally")
    print("  [+] / [-]    : Adjust similarity threshold (Current: {:.2f})".format(current_threshold))
    print("  [R]          : Reload and sync face database from disk")
    print("  [S]          : Save snapshot to screenshots folder")
    print("=" * 60 + "\n")

    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0
    unknown_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to grab frame. Reconnecting...")
            time.sleep(0.5)
            continue

        # 1. Apply Video Rotation if needed (Fix sideways phone camera)
        if current_rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif current_rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif current_rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 2. Apply Vertical Flip if needed (Fix upside-down camera)
        if is_vflip:
            frame = cv2.flip(frame, 0)

        # 3. Apply Mirror / Horizontal Flip if needed
        if is_mirrored:
            frame = cv2.flip(frame, 1)

        # 4. Resize frame so it comfortably fits on laptop screen (max width 640, max height 600)
        fh, fw = frame.shape[:2]
        scale = min(640.0 / fw, 600.0 / fh, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)

        fps_counter += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start_time = time.time()

        # Detect all faces in frame
        faces = engine.detect_all_faces(frame, score_threshold=0.55)

        unknown_detected_in_frame = False
        known_names_in_frame = []

        for face_info in faces:
            bbox = face_info['bbox']
            aligned_112 = face_info['aligned_112']

            # Extract 512-D embedding
            emb = engine.extract_embedding_from_aligned(aligned_112)

            # Match against database
            name, similarity, confidence, closest_person = db.identify(emb, engine, threshold=current_threshold)

            if name != "UNKNOWN":
                # Known registered person (Green)
                known_names_in_frame.append(name)
                box_color = (0, 200, 0)
                draw_styled_bbox(frame, bbox, box_color, label=name, sub_label=f"{confidence}% (Sim: {similarity:.2f})")
            else:
                # Intruder / Unknown person (Red)
                unknown_detected_in_frame = True
                box_color = (0, 0, 255)
                info_text = f"Sim: {similarity:.2f}"
                if closest_person != "None":
                    info_text += f" | Near {closest_person}"
                draw_styled_bbox(frame, bbox, box_color, label="UNKNOWN", sub_label=info_text)

        # Trigger bark alert if unknown face is present for at least 2 frames
        if unknown_detected_in_frame:
            unknown_frame_count += 1
            if unknown_frame_count >= 2 and not args.no_sound:
                played = sound_alert.trigger_bark()
                if played:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] [!] UNKNOWN INTRUDER DETECTED -> BARKING ALERT TRIGGERED!")
        else:
            unknown_frame_count = max(0, unknown_frame_count - 1)

        # Draw Clean HUD / Status Bar
        h, w = frame.shape[:2]
        hud_h = 44 if w < 540 else 28
        cv2.rectangle(frame, (0, 0), (w, hud_h), (20, 20, 20), -1)

        status_text = f"FPS: {current_fps:.1f} | [T] Rot {current_rotation}° | [V] Flip | Thresh: {current_threshold:.2f}"
        if len(faces) == 0:
            alert_badge = "[NO FACE / STANDBY]"
            alert_color = (150, 150, 150)
        elif unknown_detected_in_frame:
            alert_badge = "[ALERT: UNKNOWN INTRUDER]"
            alert_color = (0, 0, 255)
        else:
            names_summary = ", ".join(set(known_names_in_frame))
            alert_badge = f"[SECURE: {names_summary}]"
            alert_color = (0, 255, 0)

        if w >= 540:
            cv2.putText(frame, status_text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
            cv2.putText(frame, alert_badge, (w - 210, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.44, alert_color, 1)
        else:
            # 2 lines in narrow portrait mode
            cv2.putText(frame, status_text, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)
            cv2.putText(frame, alert_badge, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, alert_color, 1)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('t'):
            current_rotation = (current_rotation + 90) % 360
            print(f"[Rotate] Camera rotated to: {current_rotation}°")
        elif key == ord('v'):
            is_vflip = not is_vflip
            print(f"[Flip] Vertical flip (upside-down fix): {'ON' if is_vflip else 'OFF'}")
        elif key == ord('m'):
            is_mirrored = not is_mirrored
            print(f"[Mirror] Camera mirror: {'ON' if is_mirrored else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            current_threshold = min(0.80, round(current_threshold + 0.02, 2))
            print(f"[Threshold] Increased to: {current_threshold}")
        elif key == ord('-') or key == ord('_'):
            current_threshold = max(0.20, round(current_threshold - 0.02, 2))
            print(f"[Threshold] Decreased to: {current_threshold}")
        elif key == ord('r'):
            db.scan_known_faces_directory(engine, known_faces_dir='known_faces')
            print(f"[FaceDB] Reloaded & synced database: {db.get_people()}")
        elif key == ord('s'):
            os.makedirs("screenshots", exist_ok=True)
            shot_name = os.path.join("screenshots", f"shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(shot_name, frame)
            print(f"[Snapshot] Saved to: {shot_name}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[Application Closed]")


if __name__ == "__main__":
    main()
