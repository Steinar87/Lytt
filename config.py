"""Innstillinger for Lytt – lagres som JSON i data/config.json."""
from __future__ import annotations

import json
import os
import threading

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
MODELS_DIR = os.path.join(APP_DIR, "models")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")

DEFAULTS: dict = {
    # Tekst som settes foran transkripsjonen når du trykker "Kopier"
    "prefix": "Write meeting minutes from this transcript:\n\n",
    # Whisper-modell (navn kjent for faster-whisper, eller sti til mappe)
    "model": "large-v3-turbo",
    # Språk som skal gjenkjennes. Tom liste = alle språk.
    "languages": ["no", "sv", "en"],
    # Kilder
    "capture_system": True,
    "capture_mic": True,
    "mic_device": None,          # None = standard mikrofon, ellers PortAudio-indeks
    # Stemmeskilling
    "diarize": True,
    "my_name": "Me",
    "mic_muted": False,
    "speaker_threshold": 0.45,   # cosinus-likhet for "samme person" (ERes2Net: samme ≥0.65, ulike ≤0.35)
    # Beregning
    "device": "auto",            # auto | cuda | cpu
    "compute_type": "auto",      # auto | float16 | int8_float16 | int8
    "beam_size": 3,
    # Segmentering
    "min_speech_sec": 0.6,
    "silence_ms": 700,
    "max_chunk_sec": 25,
    # Kopiering
    "copy_speakers": True,
    "copy_time": False,
    # Vindu
    "start_hidden": False,
    "close_to_tray": True,       # False = the close button quits Lytt
}

APP_VERSION = "1.1.0"
APP_REPO = "https://github.com/Steinar87/Lytt"

_lock = threading.Lock()


def load() -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg


def save(cfg: dict) -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    clean = {k: cfg.get(k, v) for k, v in DEFAULTS.items()}
    with _lock:
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CONFIG_PATH)
    return clean
