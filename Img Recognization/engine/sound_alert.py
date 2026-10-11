import os
import time
import threading

# Try importing pygame for sound playback
HAS_PYGAME = False
try:
    import pygame
    pygame.mixer.init()
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False

# Fallback for Windows built-in sound
HAS_WINSOUND = False
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class SoundAlertSystem:
    """
    Non-blocking sound alert system with cooldown to prevent audio stuttering.
    Plays a barking sound whenever an unknown face / intruder is detected.
    """

    def __init__(self, sound_path=os.path.join('sounds', 'bark.wav'), cooldown_sec=2.5):
        self.sound_path = sound_path
        self.cooldown_sec = cooldown_sec
        self.last_played_time = 0.0
        self._is_playing = False
        self._lock = threading.Lock()

        if not os.path.exists(self.sound_path):
            print(f"[SoundAlert Warning] Sound file not found at: {self.sound_path}")
        else:
            print(f"[SoundAlert] Ready with sound: {self.sound_path} (Cooldown: {cooldown_sec}s)")

    def _play_worker(self):
        try:
            if os.path.exists(self.sound_path):
                if HAS_PYGAME:
                    sound = pygame.mixer.Sound(self.sound_path)
                    sound.play()
                    # Sleep during sound length so it finishes cleanly
                    time.sleep(min(self.cooldown_sec, sound.get_length()))
                elif HAS_WINSOUND:
                    winsound.PlaySound(self.sound_path, winsound.SND_FILENAME)
                else:
                    print("\a", end="", flush=True)  # System beep fallback
            else:
                if HAS_WINSOUND:
                    winsound.Beep(1000, 500)
                else:
                    print("\a", end="", flush=True)
        except Exception as e:
            print(f"[SoundAlert Error] Failed to play sound: {e}")
        finally:
            with self._lock:
                self._is_playing = False

    def trigger_bark(self):
        """
        Triggers the barking sound if cooldown period has passed.
        Runs asynchronously in a background thread to avoid freezing video stream.
        """
        current_time = time.time()
        with self._lock:
            if current_time - self.last_played_time < self.cooldown_sec or self._is_playing:
                return False

            self.last_played_time = current_time
            self._is_playing = True

        thread = threading.Thread(target=self._play_worker, daemon=True)
        thread.start()
        return True
