import os
import sys
import argparse
import time
import cv2
from engine.arcface_engine import ArcFaceEngine
from engine.face_database import FaceDatabase


def register_from_webcam(name, camera_id=0, max_samples=5):
    """
    Interactively captures face photos from webcam and registers into database.
    """
    engine = ArcFaceEngine(model_dir='models')
    db = FaceDatabase()

    person_dir = os.path.join('known_faces', name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Error] Could not open webcam (ID: {camera_id}). Check your camera connection.")
        return

    print(f"\n=======================================================")
    print(f"  Registering: {name}")
    print(f"  1. Look at the camera.")
    print(f"  2. Press [SPACE] to capture a face sample (3 to 5 recommended).")
    print(f"  3. Slowly move your head (straight, slight left/right).")
    print(f"  4. Press [Q] or [ESC] when finished.")
    print(f"=======================================================\n")

    sample_count = len([f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png'))])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Failed to grab camera frame.")
            break

        # Detect faces in frame
        faces = engine.detect_all_faces(frame, score_threshold=0.6)
        display_frame = frame.copy()

        box_color = (0, 255, 0) if len(faces) == 1 else (0, 165, 255)
        status_text = f"Samples: {sample_count} | Press [SPACE] to capture"

        if len(faces) == 1:
            face = faces[0]
            x, y, w, h = face['bbox']
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(display_frame, f"Ready ({face['score']:.2f})", (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            # Draw landmarks
            for pt in face['landmarks']:
                cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)
        elif len(faces) > 1:
            status_text = "Multiple faces detected! Only 1 person should be in frame."
            for face in faces:
                x, y, w, h = face['bbox']
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            status_text = "No face detected. Please position face in camera."

        # Header bar
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 40), (40, 40, 40), -1)
        cv2.putText(display_frame, f"Registering: {name} | {status_text}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Face Registration", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACEBAR pressed
            if len(faces) == 1:
                aligned = faces[0]['aligned_112']
                emb = engine.extract_embedding_from_aligned(aligned)
                if emb is not None:
                    sample_count += 1
                    img_filename = os.path.join(person_dir, f"sample_{sample_count}.jpg")
                    cv2.imwrite(img_filename, frame)
                    db.add_face(name, emb)
                    db.save()
                    print(f"  [+] Sample #{sample_count} saved and registered for '{name}'.")
                    time.sleep(0.3)
                else:
                    print("  [-] Could not extract embedding. Try again.")
            else:
                print("  [!] Please ensure exactly ONE face is clearly visible before capturing.")

        elif key == ord('q') or key == 27:  # Q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[Done] Successfully registered {sample_count} samples for '{name}'.")


def register_from_image(name, image_path):
    """Registers a person from an existing image file."""
    engine = ArcFaceEngine(model_dir='models')
    db = FaceDatabase()

    if not os.path.exists(image_path):
        print(f"[Error] Image not found: {image_path}")
        return

    emb = engine.extract_embedding(image_path)
    if emb is None:
        print(f"[Error] No face detected in {image_path}")
        return

    db.add_face(name, emb)
    db.save()
    print(f"[Success] Successfully registered '{name}' from {image_path}")


def main():
    parser = argparse.ArgumentParser(description="Face Registration Tool (ArcFace)")
    parser.add_argument("--name", type=str, help="Person's name to register")
    parser.add_argument("--image", type=str, help="Path to image file to register")
    parser.add_argument("--cam", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--scan", action="store_true", help="Scan 'known_faces' folder and update database")
    parser.add_argument("--list", action="store_true", help="List all registered people")
    parser.add_argument("--remove", type=str, help="Remove a person from database")

    args = parser.parse_args()

    engine = ArcFaceEngine(model_dir='models')
    db = FaceDatabase()

    if args.list:
        people = db.get_people()
        print(f"\n[FaceDB] Total registered people: {len(people)}")
        for idx, p in enumerate(people, 1):
            count = len(db.database.get(p, []))
            print(f"  {idx}. {p} ({count} face samples)")
        print()
        return

    if args.remove:
        db.remove_person(args.remove)
        return

    if args.scan:
        total = db.scan_known_faces_directory(engine)
        print(f"\n[Done] Scanned and registered {total} photos.")
        return

    if args.name and args.image:
        register_from_image(args.name, args.image)
        return

    if args.name:
        register_from_webcam(args.name, camera_id=args.cam)
    else:
        name = input("Enter the person's name to register: ").strip()
        if not name:
            print("[Error] Name cannot be empty.")
            return
        register_from_webcam(name, camera_id=args.cam)


if __name__ == "__main__":
    main()
