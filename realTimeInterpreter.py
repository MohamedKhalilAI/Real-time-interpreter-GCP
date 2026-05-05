import argparse
import base64
import os
import queue
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import requests

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: pip install sounddevice numpy requests pygame webrtcvad")
    sys.exit(1)

try:
    import webrtcvad
except ImportError:
    print("ERROR: pip install webrtcvad")
    sys.exit(1)

try:
    import pygame
except ImportError:
    print("ERROR: pip install pygame")
    sys.exit(1)

SAMPLE_RATE     = 16000
FRAME_MS        = 30
FRAME_SAMPLES   = int(SAMPLE_RATE * FRAME_MS / 1000)
FRAME_BYTES     = FRAME_SAMPLES * 2

STT_URL = "https://speech.googleapis.com/v1/speech:recognize"
TRL_URL = "https://translation.googleapis.com/language/translate/v2"
TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

LANG_EN = "en-US"
LANG_AR = "ar"


def stt(audio_bytes: bytes, api_key: str) -> tuple[str, str]:
    b64 = base64.b64encode(audio_bytes).decode()
    body = {
        "config": {
            "encoding": "LINEAR16",
            "sampleRateHertz": SAMPLE_RATE,
            "languageCode": LANG_EN,
            "alternativeLanguageCodes": [LANG_AR],
            "enableAutomaticPunctuation": True,
            "model": "latest_long",
        },
        "audio": {"content": b64},
    }
    r = requests.post(STT_URL, params={"key": api_key}, json=body, timeout=20)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    if not results:
        return "", ""

    best       = results[0]["alternatives"][0]
    transcript = best["transcript"]
    confidence = best.get("confidence", 0)
    detected   = results[0].get("languageCode", LANG_EN)

    if confidence < 0.7:
        return "", ""

    return transcript, detected


def translate(text: str, target: str, api_key: str) -> str:
    r = requests.post(
        TRL_URL,
        params={"key": api_key},
        json={"q": text, "target": target.split("-")[0]},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["data"]["translations"][0]["translatedText"]


def tts(text: str, lang: str, api_key: str) -> bytes:
    body = {
        "input": {"text": text},
        "voice": {"languageCode": lang, "ssmlGender": "NEUTRAL"},
        "audioConfig": {"audioEncoding": "MP3"},
    }
    r = requests.post(TTS_URL, params={"key": api_key}, json=body, timeout=15)
    r.raise_for_status()
    return base64.b64decode(r.json()["audioContent"])


class AudioPlayer:
    def __init__(self):
        self._q = queue.Queue()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)

    def enqueue(self, mp3_bytes: bytes):
        self._q.put(mp3_bytes)

    def _loop(self):
        while True:
            mp3 = self._q.get()
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(mp3)
                    path = f.name
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
                pygame.mixer.music.unload()
                os.remove(path)
            except Exception as e:
                print(f"  [player] {e}")
            finally:
                self._q.task_done()


class VADCapture:
    def __init__(self, audio_queue: queue.Queue, sensitivity: int, silence_sec: float):
        self.audio_queue  = audio_queue
        self.vad          = webrtcvad.Vad(sensitivity)
        self.silence_sec  = silence_sec
        self._stop        = threading.Event()
        self._status      = "Listening..."
        self._lock        = threading.Lock()

    @property
    def status(self):
        with self._lock:
            return self._status

    def _set_status(self, s):
        with self._lock:
            self._status = s

    def start(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        pre_buffer     = deque(maxlen=int(200 / FRAME_MS))
        speech_buf     = []
        in_speech      = False
        silence_frames = 0
        silence_limit  = int(self.silence_sec * 1000 / FRAME_MS)

        raw_buf = bytes()

        def callback(indata, frames, time_info, status):
            nonlocal raw_buf
            raw_buf += bytes(indata)

        stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            dtype="int16",
            channels=1,
            callback=callback,
        )

        with stream:
            while not self._stop.is_set():
                if len(raw_buf) < FRAME_BYTES:
                    time.sleep(0.005)
                    continue

                frame   = raw_buf[:FRAME_BYTES]
                raw_buf = raw_buf[FRAME_BYTES:]

                try:
                    is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
                except Exception:
                    is_speech = False

                if not in_speech:
                    pre_buffer.append(frame)
                    if is_speech:
                        in_speech      = True
                        silence_frames = 0
                        speech_buf     = list(pre_buffer)
                        self._set_status("Speaking...")
                else:
                    speech_buf.append(frame)
                    if not is_speech:
                        silence_frames += 1
                        if silence_frames >= silence_limit:
                            audio = b"".join(speech_buf)
                            if len(audio) > FRAME_BYTES * 5:
                                self.audio_queue.put(audio)
                            speech_buf     = []
                            in_speech      = False
                            silence_frames = 0
                            self._set_status("Listening...")
                    else:
                        silence_frames = 0


class Processor:
    def __init__(self, audio_queue: queue.Queue, player: AudioPlayer,
                 api_key: str, n_workers: int, log_cb):
        self.audio_queue = audio_queue
        self.player      = player
        self.api_key     = api_key
        self.log         = log_cb
        self._workers    = [
            threading.Thread(target=self._work, daemon=True)
            for _ in range(n_workers)
        ]

    def start(self):
        for w in self._workers:
            w.start()

    def _work(self):
        while True:
            audio = self.audio_queue.get()
            try:
                self._process(audio)
            except Exception as e:
                self.log(f"[error] {e}")
            finally:
                self.audio_queue.task_done()

    def _process(self, audio: bytes):
        ts = datetime.now().strftime("%H:%M:%S")

        transcript, detected = stt(audio, self.api_key)

        if not transcript:
            self.log(f"[{ts}] Low confidence or silence — skipped")
            return

        is_english  = detected.startswith("en")
        target_lang = LANG_AR if is_english else LANG_EN
        src_label   = "EN" if is_english else "AR"
        tgt_label   = "AR" if is_english else "EN"

        self.log(f"[{ts}] {src_label}: {transcript}")
        translated = translate(transcript, target_lang, self.api_key)
        self.log(f"[{ts}] {tgt_label}: {translated}")
        mp3 = tts(translated, target_lang, self.api_key)
        self.player.enqueue(mp3)


class UI:
    def __init__(self, vad: VADCapture, audio_queue: queue.Queue):
        self.vad         = vad
        self.audio_queue = audio_queue
        self._log_lines  = deque(maxlen=10)
        self._lock       = threading.Lock()

    def log(self, msg: str):
        with self._lock:
            self._log_lines.append(msg)
        print(f"  {msg}")

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((640, 340))
        pygame.display.set_caption("Real-Time Interpreter  —  EN ↔ AR  |  Q to quit")
        font_title = pygame.font.SysFont("Arial", 18, bold=True)
        font_body  = pygame.font.SysFont("Arial", 14)
        font_small = pygame.font.SysFont("Arial", 12)
        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False

            screen.fill((22, 22, 26))

            status = self.vad.status
            q_size = self.audio_queue.qsize()
            color  = (100, 220, 100) if status == "Speaking..." else (160, 160, 180)
            screen.blit(font_title.render("EN ↔ AR  Real-Time Interpreter", True, (220, 220, 220)), (20, 16))
            screen.blit(font_body.render(f"Status: {status}", True, color), (20, 48))
            screen.blit(font_body.render(f"Queue: {q_size} chunk(s) pending", True, (140, 140, 160)), (20, 70))

            pygame.draw.line(screen, (50, 50, 60), (20, 96), (620, 96), 1)

            with self._lock:
                lines = list(self._log_lines)
            for i, line in enumerate(lines):
                col = (160, 160, 160)
                if "] EN:" in line:
                    col = (200, 220, 200)
                elif "] AR:" in line:
                    col = (100, 180, 255)
                screen.blit(font_small.render(line[:80], True, col), (20, 106 + i * 20))

            pygame.draw.line(screen, (50, 50, 60), (20, 310), (620, 310), 1)
            screen.blit(font_small.render("Just speak — VAD detects speech automatically  |  Q = quit", True, (90, 90, 110)), (20, 318))

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()
        self.vad.stop()


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    parser = argparse.ArgumentParser(description="Real-time bidirectional EN↔AR interpreter")
    parser.add_argument("--vad-sensitivity", type=int,   default=2,   help="0-3 (default 2)")
    parser.add_argument("--silence-sec",     type=float, default=0.8, help="Silence to end utterance (default 0.8s)")
    parser.add_argument("--workers",         type=int,   default=3,   help="Parallel processing threads (default 3)")
    args = parser.parse_args()

    audio_queue = queue.Queue()
    player      = AudioPlayer()
    vad_capture = VADCapture(audio_queue, args.vad_sensitivity, args.silence_sec)
    ui          = UI(vad_capture, audio_queue)
    processor   = Processor(audio_queue, player, api_key, args.workers, ui.log)

    vad_capture.start()
    processor.start()

    print("\n  Real-Time EN ↔ AR Interpreter started.")
    print("  Just speak — VAD will detect your voice automatically.")
    print("  Press Q in the window to quit.\n")

    ui.run()


if __name__ == "__main__":
    main()