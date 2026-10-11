import os
import argparse
import cv2
from engine.arcface_engine import ArcFaceEngine
from engine.face_database import FaceDatabase
from main_webcam import draw_styled_bbox


def test_single_image(image_path, threshold=0.50):
    if not os.path.exists(image_path):
        print(f"[Error] Image not found: {image_path}")
        return

    engine = ArcFaceEngine(model_dir='models')
    db = FaceDatabase()

    print(f"[FaceDB] Registered people: {db.get_people()}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] Could not read image: {image_path}")
        return

    faces = engine.detect_all_faces(img, score_threshold=0.5)
    print(f"\n[Detection] Found {len(faces)} face(s) in {image_path}")

    for idx, face_info in enumerate(faces, 1):
        bbox = face_info['bbox']
        aligned_112 = face_info['aligned_112']
        emb = engine.extract_embedding_from_aligned(aligned_112)

        name, sim, conf = db.identify(emb, engine, threshold=threshold)

        print(f"  Face #{idx}: Result = {name} | Similarity = {sim:.4f} | Confidence = {conf}%")

        if name != "UNKNOWN":
            draw_styled_bbox(img, bbox, (0, 200, 0), label=name, sub_label=f"{conf}%")
        else:
            draw_styled_bbox(img, bbox, (0, 0, 255), label="UNKNOWN", sub_label="ALERT!")

    out_path = "output_" + os.path.basename(image_path)
    cv2.imwrite(out_path, img)
    print(f"\n[Saved] Annotated result saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Test Face Recognition on a Static Image")
    parser.add_argument("--image", required=True, type=str, help="Path to input image")
    parser.add_argument("--threshold", type=float, default=0.50, help="Matching similarity threshold (default: 0.50)")
    args = parser.parse_args()

    test_single_image(args.image, threshold=args.threshold)


if __name__ == "__main__":
    main()
