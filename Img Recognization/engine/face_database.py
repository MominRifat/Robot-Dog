import os
import json
import glob
import numpy as np


class FaceDatabase:
    """
    Manages known faces database with multiple samples per person,
    JSON persistence, and fast similarity matching.
    """

    def __init__(self, db_path=os.path.join('database', 'face_db.json')):
        self.db_path = db_path
        # Schema: { "Person_Name": [ [512 floats], [512 floats], ... ] }
        self.database = {}
        self.load()

    def load(self):
        """Loads embeddings from JSON file if exists."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self.database = json.load(f)
                print(f"[FaceDB] Loaded {len(self.database)} people from {self.db_path}: {list(self.database.keys())}")
            except Exception as e:
                print(f"[FaceDB Error] Could not load database: {e}")
                self.database = {}
        else:
            self.database = {}

    def save(self):
        """Saves current embeddings to JSON file."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        try:
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.database, f, indent=2)
            print(f"[FaceDB] Successfully saved {len(self.database)} people to {self.db_path}")
            return True
        except Exception as e:
            print(f"[FaceDB Error] Failed to save database: {e}")
            return False

    def add_face(self, name, embedding):
        """
        Adds an embedding (512 floats) for a person.
        Supports multiple photos/embeddings for the same person.
        """
        if embedding is None:
            return False

        if isinstance(embedding, np.ndarray):
            emb_list = embedding.tolist()
        else:
            emb_list = list(embedding)

        if name not in self.database:
            self.database[name] = []

        self.database[name].append(emb_list)
        return True

    def remove_person(self, name):
        """Removes a person from the database."""
        if name in self.database:
            del self.database[name]
            self.save()
            print(f"[FaceDB] Removed {name}")
            return True
        return False

    def get_people(self):
        """Returns list of all registered person names."""
        return list(self.database.keys())

    def scan_known_faces_directory(self, engine, known_faces_dir='known_faces'):
        """
        Scans 'known_faces' directory:
        1. Direct files: 'known_faces/Sakib.jpg' -> Name: Sakib
        2. Subfolders: 'known_faces/Sakib/1.jpg' -> Name: Sakib (multiple samples)
        Extracts embeddings and updates the database.
        """
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir, exist_ok=True)
            return 0

        total_registered = 0
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

        # 1. Check direct image files in known_faces/ (e.g. known_faces/Sakib.jpg)
        for item in os.listdir(known_faces_dir):
            item_path = os.path.join(known_faces_dir, item)
            if os.path.isfile(item_path):
                base_name, ext = os.path.splitext(item)
                if ext.lower() in valid_extensions:
                    person_name = base_name.strip()
                    emb = engine.extract_embedding(item_path)
                    if emb is not None:
                        # Reset previous templates for this person
                        self.database[person_name] = [emb.tolist()]
                        print(f"  [+] Registered '{person_name}' from direct file: {item}")
                        total_registered += 1
                    else:
                        print(f"  [Warning] No face detected in: {item}")

        # 2. Check subfolders in known_faces/ (e.g. known_faces/Sakib/*.jpg)
        person_dirs = [d for d in os.listdir(known_faces_dir) if os.path.isdir(os.path.join(known_faces_dir, d))]

        for person_name in person_dirs:
            person_path = os.path.join(known_faces_dir, person_name)
            image_paths = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp'):
                image_paths.extend(glob.glob(os.path.join(person_path, ext)))

            if not image_paths:
                continue

            self.database[person_name] = []
            person_count = 0

            for img_path in image_paths:
                emb = engine.extract_embedding(img_path)
                if emb is not None:
                    self.add_face(person_name, emb)
                    person_count += 1
                else:
                    print(f"  [Warning] No face detected in: {img_path}")

            print(f"  [+] {person_name}: {person_count} photos processed from folder.")
            total_registered += person_count

        self.save()
        return total_registered

    def identify(self, test_embedding, engine, threshold=0.42):
        """
        Compares test_embedding with all known people in the database.
        Returns:
            match_name: str (Person name or 'UNKNOWN')
            best_similarity: float
            confidence: float (percentage)
            closest_person: str (closest match candidate regardless of threshold)
        """
        if test_embedding is None or not self.database:
            return "UNKNOWN", 0.0, 0.0, "None"

        best_sim = -1.0
        closest_person = "None"

        test_vec = np.asarray(test_embedding, dtype=np.float32)

        for name, emb_list in self.database.items():
            for stored_emb in emb_list:
                stored_vec = np.asarray(stored_emb, dtype=np.float32)
                sim = engine.calculate_similarity(test_vec, stored_vec)
                if sim > best_sim:
                    best_sim = sim
                    closest_person = name

        confidence = engine.compute_confidence(best_sim)

        if best_sim >= threshold:
            return closest_person, round(best_sim, 4), confidence, closest_person
        else:
            return "UNKNOWN", round(best_sim, 4), confidence, closest_person
