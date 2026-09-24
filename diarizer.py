"""Enkel stemmeskilling: talerembedding (ERes2Net/VoxCeleb via sherpa-onnx) + online klustring.

Hver ytring får én embedding. Sammenlignes med kjente taleres sentroider med
cosinus-likhet; over terskel = samme person, ellers ny "Person N".
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

from config import MODELS_DIR

EMBED_MODEL = os.path.join(MODELS_DIR, "eres2net_voxceleb.onnx")
MIN_SEC_FOR_EMBED = 1.0


class Diarizer:
    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold
        self._extractor = None
        self._lock = threading.Lock()
        self.available = os.path.exists(EMBED_MODEL)
        self.reset()

    def reset(self) -> None:
        self._centroids: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}
        self._last_speaker: Optional[str] = None

    def _load(self):
        if self._extractor is None and self.available:
            import sherpa_onnx
            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=EMBED_MODEL, num_threads=2, debug=False, provider="cpu"
            )
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        return self._extractor

    def embed(self, audio16k: np.ndarray) -> Optional[np.ndarray]:
        ext = self._load()
        if ext is None:
            return None
        stream = ext.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=audio16k.astype(np.float32))
        stream.input_finished()
        if not ext.is_ready(stream):
            return None
        emb = np.asarray(ext.compute(stream), dtype=np.float32)
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else None

    def assign(self, audio16k: np.ndarray) -> str:
        """Returnerer talernavn ("Person 1", "Person 2", ...)."""
        with self._lock:
            if len(audio16k) < MIN_SEC_FOR_EMBED * 16000:
                return self._last_speaker or self._new_speaker(None)
            emb = self.embed(audio16k)
            if emb is None:
                return self._last_speaker or self._new_speaker(None)

            best, best_sim = None, -1.0
            for name, c in self._centroids.items():
                sim = float(np.dot(emb, c) / (np.linalg.norm(c) + 1e-9))
                if sim > best_sim:
                    best, best_sim = name, sim

            if best is not None and best_sim >= self.threshold:
                # Oppdater sentroid (glidende snitt, tak på vekt så den forblir adaptiv)
                n = min(self._counts[best], 20)
                self._centroids[best] = (self._centroids[best] * n + emb) / (n + 1)
                self._counts[best] += 1
                self._last_speaker = best
                return best
            return self._new_speaker(emb)

    def _new_speaker(self, emb: Optional[np.ndarray]) -> str:
        name = f"Speaker {len(self._centroids) + 1}"
        if emb is not None:
            self._centroids[name] = emb
            self._counts[name] = 1
        elif not self._centroids:
            # ingen embedding og ingen kjente talere: bruk navnet uten sentroid
            pass
        self._last_speaker = name
        return name

    def rename(self, old: str, new: str) -> None:
        with self._lock:
            if old in self._centroids and new not in self._centroids:
                self._centroids[new] = self._centroids.pop(old)
                self._counts[new] = self._counts.pop(old)
            if self._last_speaker == old:
                self._last_speaker = new
