"""Ende-til-ende-test med syntetisk tale (Windows TTS): Whisper på GPU + stemmeskilling."""
import os, sys, time, tempfile
import numpy as np
from scipy.io import wavfile
from audio import _resample
import config
from transcriber import Transcriber
from diarizer import Diarizer

d = os.path.join(tempfile.gettempdir(), "lytt_tts")
clips = []
for name in ("david", "zira", "david"):
    sr, x = wavfile.read(os.path.join(d, f"{name}.wav"))
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    if x.ndim > 1:
        x = x.mean(axis=1)
    clips.append((name, _resample(x, sr)))

cfg = config.load()
results = []
t = Transcriber(results.append, lambda st, m: print("  status:", st, m) if m else None)
t0 = time.time()
t.preload(cfg)
for i, (name, a) in enumerate(clips):
    t.submit({"cfg": cfg, "audio": a, "source": "system", "session_id": 0, "t_start": i * 5, "t_end": i * 5 + 4, "name": name})
while t.backlog and time.time() - t0 < 900:
    time.sleep(0.1)
print(f"\nModell: {t.device_info}, total {time.time()-t0:.1f}s inkl. lasting")

dz = Diarizer(cfg["speaker_threshold"])
t1 = time.time()
for r in results:
    who = dz.assign(r["audio"])
    print(f"  [{r['name']:>5}] -> {who:9} ({r['language']}) {r['text']}")
print(f"Stemmeskilling {time.time()-t1:.2f}s for {len(results)} klipp")

# Ren transkriberingstid uten lasting
t2 = time.time()
for name, a in clips:
    t.submit({"cfg": cfg, "audio": a, "source": "system", "session_id": 0, "t_start": 0, "t_end": 4, "name": name})
while t.backlog:
    time.sleep(0.05)
tot = sum(len(a) for _, a in clips) / 16000
print(f"Transkribering: {tot:.1f}s lyd på {time.time()-t2:.2f}s  (x{tot/(time.time()-t2):.0f} sanntid)")
t.close()
