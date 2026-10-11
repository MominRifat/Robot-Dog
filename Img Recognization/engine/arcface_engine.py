import os
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import base64


class ArcFaceEngine:
    """
    High-Performance Face Detection & Recognition Engine:
    - Detector: OpenCV YuNet (5 Facial Landmarks: Right Eye, Left Eye, Nose Tip, Mouth Corners)
    - Recognizer: MobileFaceNet ArcFace 512-D Deep Biometric Embedding (ONNX Runtime)
    """

    MATCH_THRESHOLD_SIMILARITY = 0.50  # >= 0.50 means match (Same person)
    MATCH_THRESHOLD_DISTANCE = 1.00    # <= 1.00 means match

    # ArcFace canonical 5-point landmark template (112x112 standard format)
    ARCFACE_REF = np.array([
        [38.2946, 51.6963],   # Right eye (subject's right, viewer's left)
        [73.5318, 51.5014],   # Left eye (subject's left, viewer's right)
        [56.0252, 71.7366],   # Nose tip
        [41.5493, 92.3655],   # Right mouth corner
        [70.7299, 92.2041]    # Left mouth corner
    ], dtype=np.float32)

    def __init__(self, model_dir='models'):
        """Initialize YuNet and ArcFace ONNX sessions."""
        self.rec_path = os.path.join(model_dir, 'w600k_mbf.onnx')
        self.det_path = os.path.join(model_dir, 'face_detection_yunet_2023mar.onnx')

        if not os.path.exists(self.rec_path):
            raise FileNotFoundError(f"ArcFace model not found at: {self.rec_path}")
        if not os.path.exists(self.det_path):
            raise FileNotFoundError(f"YuNet model not found at: {self.det_path}")

        # 1. ArcFace ONNX Runtime Inference Session
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self.rec_session = ort.InferenceSession(
            self.rec_path,
            sess_options=opts,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.rec_session.get_inputs()[0].name

        # 2. YuNet Face Landmark Detector
        self.detector = cv2.FaceDetectorYN.create(
            model=self.det_path,
            config='',
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )

    def load_image(self, img_input):
        """Loads image from path, numpy array, or base64 string to BGR format."""
        if isinstance(img_input, np.ndarray):
            return img_input
        if isinstance(img_input, str):
            if img_input.startswith('data:image') or (len(img_input) > 200 and not os.path.exists(img_input)):
                if ',' in img_input:
                    img_input = img_input.split(',', 1)[1]
                img_bytes = base64.b64decode(img_input)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            elif os.path.exists(img_input):
                return cv2.imread(img_input)
        return None

    def detect_all_faces(self, frame, score_threshold=0.6):
        """
        Detects all faces in the given frame using YuNet.
        Returns list of dicts: [
            {
                'bbox': (x, y, w, h),
                'landmarks': 5x2 array,
                'score': float,
                'aligned_112': 112x112 aligned face image
            }, ...
        ]
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        self.detector.setScoreThreshold(score_threshold)
        _, faces = self.detector.detect(frame)

        if faces is None or len(faces) == 0:
            # Low contrast fallback with CLAHE enhancement
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
            _, faces = self.detector.detect(enhanced)

        results = []
        if faces is not None and len(faces) > 0:
            for face in faces:
                x = max(0, int(face[0]))
                y = max(0, int(face[1]))
                width = int(face[2])
                height = int(face[3])
                score = float(face[14])
                landmarks = face[4:14].reshape((5, 2)).astype(np.float32)

                # 5-point similarity transformation alignment
                tform, _ = cv2.estimateAffinePartial2D(landmarks, self.ARCFACE_REF)
                if tform is not None:
                    aligned_112 = cv2.warpAffine(frame, tform, (112, 112), borderValue=0.0)
                else:
                    crop = frame[y:y+height, x:x+width]
                    if crop.size > 0:
                        aligned_112 = cv2.resize(crop, (112, 112))
                    else:
                        aligned_112 = cv2.resize(frame, (112, 112))

                results.append({
                    'bbox': (x, y, width, height),
                    'landmarks': landmarks,
                    'score': score,
                    'aligned_112': aligned_112
                })

        return results

    def extract_embedding_from_aligned(self, aligned_112):
        """
        Extracts 512-D L2-normalized embedding vector from a 112x112 aligned face.
        """
        if aligned_112 is None or aligned_112.size == 0:
            return None

        # 1. Bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(aligned_112, d=5, sigmaColor=35, sigmaSpace=35)

        # 2. BGR -> RGB, scale to [-1, 1], shape (1, 3, 112, 112)
        rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[None, :, :, :].astype(np.float32)

        # 3. ONNX inference
        emb = self.rec_session.run(None, {self.input_name: blob})[0][0]

        # 4. L2 Normalization
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)

    def extract_embedding(self, img_input):
        """
        Takes an image input, detects largest face, aligns it, and extracts embedding.
        """
        img = self.load_image(img_input)
        if img is None:
            return None

        faces = self.detect_all_faces(img, score_threshold=0.5)
        if not faces:
            return None

        # Pick largest face by area (width * height)
        faces.sort(key=lambda f: f['bbox'][2] * f['bbox'][3], reverse=True)
        aligned = faces[0]['aligned_112']
        return self.extract_embedding_from_aligned(aligned)

    @staticmethod
    def calculate_similarity(emb1, emb2):
        """Calculates cosine similarity between two 512-D vectors (-1.0 to 1.0)."""
        if emb1 is None or emb2 is None:
            return 0.0
        v1 = np.asarray(emb1, dtype=np.float32)
        v2 = np.asarray(emb2, dtype=np.float32)
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.clip(dot / (norm1 * norm2), -1.0, 1.0))

    @staticmethod
    def calculate_distance(emb1, emb2):
        """Calculates Euclidean distance between two vectors."""
        if emb1 is None or emb2 is None:
            return 2.0
        v1 = np.asarray(emb1, dtype=np.float32)
        v2 = np.asarray(emb2, dtype=np.float32)
        return float(np.linalg.norm(v1 - v2))

    def compute_confidence(self, similarity):
        """Maps cosine similarity to realistic percentage confidence (0% - 99.9%)."""
        sim = float(similarity)
        if sim >= self.MATCH_THRESHOLD_SIMILARITY:
            progress = (sim - self.MATCH_THRESHOLD_SIMILARITY) / max(0.01, (1.0 - self.MATCH_THRESHOLD_SIMILARITY))
            confidence = 88.0 + (progress * 11.8)  # 88.0% to 99.8%
        else:
            progress = max(0.0, sim) / max(0.01, self.MATCH_THRESHOLD_SIMILARITY)
            confidence = max(2.0, progress * 40.0)  # 2.0% to 40.0%
        return round(float(np.clip(confidence, 0.0, 99.9)), 1)
