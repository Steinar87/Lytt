"""Transkribering med faster-whisper (CTranslate2) i egen arbeidstråd."""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from typing import Callable, Optional

import numpy as np

from config import MODELS_DIR

# Kjente Whisper-hallusinasjoner på norsk/svensk/engelsk (fra undertekst-trening)
HALLUCINATIONS = (
    "teksting av", "tekstet av", "undertekster av", "undertekst av", "takk for at du så",
    "takk for at dere så", "amara.org", "nicolai winther", "textning", "undertexter av",
    "översättning", "tack för att du tittade", "subtitles by", "thanks for watching",
    "thank you for watching", "www.", "©", "ai-media", "captions by",
)


def _add_cuda_dlls() -> None:
    """Gjør pip-installerte cuBLAS/cuDNN-DLLer synlige for CTranslate2 (Windows)."""
    if not sys.platform.startswith("win"):
        return
    for sp in sys.path:
        base = os.path.join(sp, "nvidia")
        if not os.path.isdir(base):
            continue
        for sub in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
            d = os.path.join(base, sub, "bin")
            if os.path.isdir(d):
                try:
                    os.add_dll_directory(d)
                    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                except Exception:
                    pass


class Transcriber:
    """Kø av lydbiter -> tekst. Modellen lastes lat, første gang noe skal transkriberes."""

    def __init__(self, on_result: Callable[[dict], None], on_status: Callable[[str, str], None]):
        self.on_result = on_result
        self.on_status = on_status          # (state, message)   state: idle|loading|ready|error|busy
        self._q: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._model = None
        self._model_key = None
        self._thread = threading.Thread(target=self._run, name="transcriber", daemon=True)
        self._thread.start()
        self.backlog = 0
        self.device_info = ""
        self._last_lang: Optional[str] = None

    # ---------- offentlig ----------
    def submit(self, job: dict) -> None:
        self.backlog += 1
        self._q.put(job)

    def preload(self, cfg: dict) -> None:
        self._q.put({"_preload": True, "cfg": cfg})

    def close(self) -> None:
        self._q.put(None)

    # ---------- intern ----------
    def _ensure_model(self, cfg: dict):
        key = (cfg["model"], cfg["device"], cfg["compute_type"])
        if self._model is not None and self._model_key == key:
            return self._model
        self._model = None
        self.on_status("loading", f"Loading model {cfg['model']} …")
        _add_cuda_dlls()
        from faster_whisper import WhisperModel

        model_name = cfg["model"]
        local = os.path.join(MODELS_DIR, model_name)
        if os.path.isdir(local):
            model_name = local

        attempts = []
        dev = cfg["device"]
        ct = cfg["compute_type"]
        if dev in ("auto", "cuda"):
            attempts.append(("cuda", ct if ct != "auto" else "float16"))
        if dev in ("auto", "cpu"):
            attempts.append(("cpu", ct if ct not in ("auto", "float16", "int8_float16") else "int8"))

        last_err = None
        for device, compute_type in attempts:
            try:
                t0 = time.time()
                model = WhisperModel(model_name, device=device, compute_type=compute_type,
                                     download_root=MODELS_DIR, cpu_threads=4)
                # Varm opp med et kort stille klipp så CUDA-kjerner er JIT-kompilert
                list(model.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)[0])
                self._model = model
                self._model_key = key
                self.device_info = f"{device.upper()} · {compute_type}"
                self.on_status("ready", f"Model ready on {self.device_info} ({time.time() - t0:.0f}s)")
                return model
            except Exception as e:  # prøv neste
                last_err = e
                print(f"[transcriber] {device}/{compute_type} feilet: {e}")
        self.on_status("error", f"Could not load model: {last_err}")
        raise RuntimeError(str(last_err))

    def _pick_language(self, model, audio: np.ndarray, allowed: list[str]) -> Optional[str]:
        if not allowed:
            return None
        if len(allowed) == 1:
            return allowed[0]
        try:
            lang, prob, all_probs = model.detect_language(audio)
        except Exception:
            return None
        if lang in allowed and prob >= 0.6:
            self._last_lang = lang
            return lang
        probs = dict(all_probs or [])
        # Whisper forveksler gjerne norsk med dansk/nynorsk; regn dem som norsk
        for alias, target in (("da", "no"), ("nn", "no")):
            if target in allowed and alias in probs:
                probs[target] = probs.get(target, 0.0) + probs[alias]
        best = max(allowed, key=lambda l: probs.get(l, 0.0))
        # Språket i et møte skifter sjelden: bytt bare når gjenkjenningen er tydelig
        last = self._last_lang
        if last in allowed and best != last and probs.get(best, 0.0) < 0.6:
            return last
        self._last_lang = best
        return best

    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                break
            try:
                if job.get("_preload"):
                    self._ensure_model(job["cfg"])
                    continue
                self._process(job)
            except Exception as e:
                print("[transcriber] feil:", e)
            finally:
                if not job.get("_preload"):
                    self.backlog = max(0, self.backlog - 1)

    def _process(self, job: dict) -> None:
        cfg = job["cfg"]
        audio: np.ndarray = job["audio"]
        model = self._ensure_model(cfg)
        self.on_status("busy", "")
        lang = self._pick_language(model, audio, cfg.get("languages") or [])
        segments, info = model.transcribe(
            audio,
            language=lang,
            beam_size=int(cfg.get("beam_size", 3)),
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 200},
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            word_timestamps=False,
        )
        texts = []
        for s in segments:
            t = s.text.strip()
            if not t:
                continue
            if s.no_speech_prob > 0.7 and s.avg_logprob < -0.8:
                continue
            low = t.lower()
            if any(h in low for h in HALLUCINATIONS):
                continue
            texts.append(t)
        text = " ".join(texts).strip()
        self.on_status("ready", "")
        if not text:
            return
        self.on_result({
            **{k: v for k, v in job.items() if k not in ("audio", "cfg")},
            "text": text,
            "language": info.language if info else lang,
            "audio": audio,     # brukes av stemmeskilling
        })
