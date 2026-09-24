"""Lydopptak via WASAPI (pyaudiowpatch): systemlyd (loopback) og mikrofon.

Hver kilde deler lyden opp i ytringer basert på enkel, adaptiv RMS-terskel
(nesten null CPU-bruk), og leverer 16 kHz mono float32 til en callback.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Callable, Optional

import numpy as np

try:
    import pyaudiowpatch as pyaudio
except ImportError:  # pragma: no cover
    import pyaudio  # type: ignore

TARGET_SR = 16000
BLOCK_MS = 100

ChunkCallback = Callable[[str, np.ndarray, float, float], None]  # (source, audio16k, t_start, t_end)


def _resample(x: np.ndarray, sr_in: int, sr_out: int = TARGET_SR) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    from scipy.signal import resample_poly
    g = math.gcd(sr_in, sr_out)
    y = resample_poly(x.astype(np.float32), sr_out // g, sr_in // g)
    return y.astype(np.float32, copy=False)


class UtteranceChunker:
    """Samler lydblokker til ytringer. Alt skjer på kildens egen samplerate."""

    def __init__(self, source: str, sr: int, on_chunk: ChunkCallback, *,
                 silence_ms: int = 700, min_speech_sec: float = 0.6,
                 max_chunk_sec: float = 25.0, preroll_ms: int = 300):
        self.source = source
        self.sr = sr
        self.on_chunk = on_chunk
        self.silence_blocks = max(1, silence_ms // BLOCK_MS)
        self.min_speech_sec = min_speech_sec
        self.max_chunk_sec = max_chunk_sec
        self.preroll_blocks = max(1, preroll_ms // BLOCK_MS)

        self.noise_floor = 0.003
        self.level = 0.0            # siste RMS, for nivåmåler i UI
        self.speaking = False
        self._preroll: list[np.ndarray] = []
        self._buf: list[np.ndarray] = []
        self._buf_start: float = 0.0
        self._speech_blocks = 0
        self._silence_run = 0
        self._buf_samples = 0

    def feed(self, block: np.ndarray, t_block: float) -> None:
        rms = float(np.sqrt(np.mean(block * block) + 1e-12))
        self.level = rms
        # Adaptivt støygulv: faller raskt, stiger sakte
        if rms < self.noise_floor:
            self.noise_floor = 0.7 * self.noise_floor + 0.3 * rms
        else:
            self.noise_floor = min(self.noise_floor * 1.01 + 1e-5, 0.02)
        thr = max(self.noise_floor * 3.5, 0.0035)
        is_speech = rms > thr

        if not self._buf:
            self._preroll.append(block)
            if len(self._preroll) > self.preroll_blocks:
                self._preroll.pop(0)
            if is_speech:
                self._buf = list(self._preroll)
                self._buf_samples = sum(len(b) for b in self._buf)
                self._buf_start = t_block - (len(self._buf) - 1) * BLOCK_MS / 1000.0
                self._preroll = []
                self._speech_blocks = 1
                self._silence_run = 0
                self.speaking = True
            return

        self._buf.append(block)
        self._buf_samples += len(block)
        if is_speech:
            self._speech_blocks += 1
            self._silence_run = 0
        else:
            self._silence_run += 1

        dur = self._buf_samples / self.sr
        if self._silence_run >= self.silence_blocks:
            self._flush(t_block, trim_tail=True)
        elif dur >= self.max_chunk_sec:
            self._flush(t_block, trim_tail=False)

    def flush(self) -> None:
        if self._buf:
            self._flush(time.time(), trim_tail=False)

    def _flush(self, t_now: float, trim_tail: bool) -> None:
        buf = self._buf
        self._buf = []
        self.speaking = False
        speech_sec = self._speech_blocks * BLOCK_MS / 1000.0
        if trim_tail and self._silence_run > 2:
            # Behold ~200 ms stillhet etter talen, kast resten
            buf = buf[: len(buf) - (self._silence_run - 2)]
        self._speech_blocks = 0
        self._silence_run = 0
        self._buf_samples = 0
        if speech_sec < self.min_speech_sec or not buf:
            return
        audio = np.concatenate(buf)
        t_end = self._buf_start + len(audio) / self.sr
        audio16 = _resample(audio, self.sr)
        peak = float(np.max(np.abs(audio16))) if len(audio16) else 0.0
        if peak > 0:
            audio16 = audio16 * min(1.0 / peak, 8.0) * 0.9   # normaliser forsiktig
        self.on_chunk(self.source, audio16, self._buf_start, t_end)


class AudioSource:
    """Én PortAudio-strøm (mic eller loopback) med tilhørende chunker."""

    def __init__(self, pa: "pyaudio.PyAudio", device: dict, source: str,
                 on_chunk: ChunkCallback, chunk_kwargs: dict):
        self.pa = pa
        self.device = device
        self.source = source
        self.sr = int(device["defaultSampleRate"])
        self.channels = max(1, min(int(device["maxInputChannels"]), 2))
        self.chunker = UtteranceChunker(source, self.sr, on_chunk, **chunk_kwargs)
        self._stream = None
        self._frames = int(self.sr * BLOCK_MS / 1000)

    def start(self) -> None:
        self._stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sr,
            input=True,
            input_device_index=int(self.device["index"]),
            frames_per_buffer=self._frames,
            stream_callback=self._cb,
        )
        self._stream.start_stream()

    def _cb(self, in_data, frame_count, time_info, status):
        try:
            x = np.frombuffer(in_data, dtype=np.float32)
            if self.channels > 1:
                x = x.reshape(-1, self.channels).mean(axis=1)
            self.chunker.feed(x, time.time())
        except Exception as e:  # pragma: no cover
            print("audio callback error:", e)
        return (None, pyaudio.paContinue)

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.chunker.flush()

    @property
    def level(self) -> float:
        return self.chunker.level


class Recorder:
    """Starter/stopper kildene og eier PyAudio-instansen."""

    def __init__(self, on_chunk: ChunkCallback):
        self.on_chunk = on_chunk
        self._pa: Optional["pyaudio.PyAudio"] = None
        self.sources: dict[str, AudioSource] = {}
        self._lock = threading.Lock()

    # ---------- enheter ----------
    def _ensure_pa(self):
        if self._pa is None:
            self._pa = pyaudio.PyAudio()
        return self._pa

    def list_mics(self) -> list[dict]:
        pa = self._ensure_pa()
        out = []
        try:
            wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except Exception:
            return out
        default_idx = wasapi.get("defaultInputDevice", -1)
        for i in range(wasapi["deviceCount"]):
            try:
                d = pa.get_device_info_by_host_api_device_index(wasapi["index"], i)
            except Exception:
                continue
            if d.get("maxInputChannels", 0) > 0 and not d.get("isLoopbackDevice", False):
                out.append({"index": int(d["index"]), "name": d["name"],
                            "default": int(d["index"]) == default_idx})
        return out

    def _default_mic(self, preferred: Optional[int]) -> Optional[dict]:
        pa = self._ensure_pa()
        if preferred is not None:
            try:
                d = pa.get_device_info_by_index(int(preferred))
                if d.get("maxInputChannels", 0) > 0:
                    return d
            except Exception:
                pass
        try:
            wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            return pa.get_device_info_by_index(wasapi["defaultInputDevice"])
        except Exception:
            return None

    def _default_loopback(self) -> Optional[dict]:
        pa = self._ensure_pa()
        try:
            wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            speakers = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        except Exception:
            return None
        if speakers.get("isLoopbackDevice"):
            return speakers
        for lb in pa.get_loopback_device_info_generator():
            if speakers["name"] in lb["name"]:
                return lb
        return None

    # ---------- start/stopp ----------
    def start(self, *, capture_mic: bool, capture_system: bool, mic_device: Optional[int],
              chunk_kwargs: dict) -> list[str]:
        """Returnerer liste over advarsler (f.eks. kilde som ikke kunne åpnes)."""
        warnings: list[str] = []
        with self._lock:
            self.stop_locked()
            pa = self._ensure_pa()
            if capture_system:
                dev = self._default_loopback()
                if dev is None:
                    warnings.append("No loopback device found for system audio.")
                else:
                    try:
                        src = AudioSource(pa, dev, "system", self.on_chunk, chunk_kwargs)
                        src.start()
                        self.sources["system"] = src
                    except Exception as e:
                        warnings.append(f"Could not open system audio: {e}")
            if capture_mic:
                dev = self._default_mic(mic_device)
                if dev is None:
                    warnings.append("No microphone found.")
                else:
                    try:
                        src = AudioSource(pa, dev, "mic", self.on_chunk, chunk_kwargs)
                        src.start()
                        self.sources["mic"] = src
                    except Exception as e:
                        warnings.append(f"Could not open microphone: {e}")
        return warnings

    def set_mic(self, enabled: bool, mic_device: Optional[int], chunk_kwargs: dict) -> Optional[str]:
        """Turns the microphone source on/off while recording. Returns a warning string or None."""
        with self._lock:
            src = self.sources.pop("mic", None)
            if src is not None:
                src.stop()
            if not enabled:
                return None
            dev = self._default_mic(mic_device)
            if dev is None:
                return "No microphone found."
            try:
                src = AudioSource(self._ensure_pa(), dev, "mic", self.on_chunk, chunk_kwargs)
                src.start()
                self.sources["mic"] = src
            except Exception as e:
                return f"Could not open microphone: {e}"
        return None

    def stop_locked(self) -> None:
        for src in list(self.sources.values()):
            src.stop()
        self.sources.clear()

    def stop(self) -> None:
        with self._lock:
            self.stop_locked()

    def levels(self) -> dict[str, float]:
        return {k: v.level for k, v in self.sources.items()}

    def speaking(self) -> dict[str, bool]:
        return {k: v.chunker.speaking for k, v in self.sources.items()}

    def device_names(self) -> dict[str, str]:
        return {k: v.device["name"] for k, v in self.sources.items()}

    def terminate(self) -> None:
        self.stop()
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
