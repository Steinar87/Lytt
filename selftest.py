"""Selvtest: CUDA, lydenheter, stemmemodell, chunker, lagring. Kjør: .venv\\Scripts\\python.exe selftest.py"""
import sys, time
import numpy as np

ok = True

def step(name, fn):
    global ok
    t = time.time()
    try:
        r = fn()
        print(f"[OK]   {name}: {r}  ({time.time()-t:.1f}s)")
    except Exception as e:
        ok = False
        print(f"[FEIL] {name}: {e!r}")

def cuda():
    import transcriber; transcriber._add_cuda_dlls()
    import ctranslate2
    n = ctranslate2.get_cuda_device_count()
    return f"cuda devices={n}, compute types={sorted(ctranslate2.get_supported_compute_types('cuda')) if n else '-'}"

def devices():
    from audio import Recorder
    r = Recorder(lambda *a: None)
    mics = r.list_mics()
    lb = r._default_loopback()
    mic = r._default_mic(None)
    r.terminate()
    return f"mics={[m['name'] for m in mics]} | loopback={lb['name'] if lb else None} @ {lb['defaultSampleRate'] if lb else '-'} Hz | default mic={mic['name'] if mic else None}"

def diar():
    from diarizer import Diarizer
    d = Diarizer(0.55)
    rng = np.random.default_rng(0)
    a = rng.standard_normal(16000 * 2).astype(np.float32) * 0.1
    b = np.sin(np.linspace(0, 2000, 32000)).astype(np.float32)
    s1 = d.assign(a); s2 = d.assign(a * 0.9); s3 = d.assign(b)
    return f"available={d.available}, dim={len(d.embed(a))}, assign=({s1},{s2},{s3})"

def chunker():
    from audio import UtteranceChunker
    out = []
    c = UtteranceChunker("t", 48000, lambda src, au, t0, t1: out.append((len(au), round(t1 - t0, 2))), silence_ms=700, min_speech_sec=0.6)
    blk = 4800
    t = 0.0
    for i in range(60):   # 6 s: 1 s stille, 2 s "tale", 3 s stille
        loud = 10 <= i < 30
        x = (np.random.default_rng(i).standard_normal(blk) * (0.2 if loud else 0.001)).astype(np.float32)
        c.feed(x, t); t += 0.1
    return f"chunks={out}"

def store():
    import os, tempfile
    from store import Store
    p = os.path.join(tempfile.gettempdir(), "lytt_test.db")
    if os.path.exists(p): os.remove(p)
    s = Store(p)
    sess = s.create_session("Test")
    for i, (sp, tx) in enumerate([("Meg", "Hei alle."), ("Person 1", "Hei."), ("Person 1", "Skal vi begynne?"), ("Meg", "Ja.")]):
        s.add_segment(dict(session_id=sess["id"], t_start=i * 3.0, t_end=i * 3.0 + 2, source="mic", speaker=sp, language="no", text=tx))
    s.rename_speaker(sess["id"], "Person 1", "Kari")
    txt = s.transcript_text(sess["id"])
    assert "Kari: Hei. Skal vi begynne?" in txt, txt
    return txt.replace("\n\n", " | ")

def rec_open():
    """Åpner faktiske strømmer i 1.5 s og rapporterer nivå."""
    from audio import Recorder
    r = Recorder(lambda src, au, t0, t1: print(f"      chunk fra {src}: {len(au)/16000:.1f}s"))
    w = r.start(capture_mic=True, capture_system=True, mic_device=None,
                chunk_kwargs=dict(silence_ms=700, min_speech_sec=0.6, max_chunk_sec=25))
    time.sleep(1.5)
    lv = r.levels(); names = r.device_names()
    r.terminate()
    return f"warnings={w} levels={ {k: round(v, 4) for k, v in lv.items()} } devices={names}"

step("CUDA/CTranslate2", cuda)
step("Lydenheter", devices)
step("Stemmeskilling", diar)
step("Chunker", chunker)
step("Lagring", store)
step("Åpne lydstrømmer", rec_open)
if "--model" in sys.argv:
    def model():
        from transcriber import Transcriber
        import config
        res = []
        t = Transcriber(res.append, lambda st, m: print("      ", st, m))
        cfg = config.load()
        t.preload(cfg)
        # 3 s syntetisk "lyd" – forventer tom/ingen tekst, men bekrefter at kjeden kjører
        t.submit({"cfg": cfg, "audio": np.zeros(16000 * 3, dtype=np.float32), "source": "mic", "session_id": 0, "t_start": 0, "t_end": 3})
        t0 = time.time()
        while t.backlog and time.time() - t0 < 600:
            time.sleep(0.2)
        return f"device={t.device_info} results={res}"
    step("Whisper-modell", model)
print("\nALT OK" if ok else "\nNOE FEILET")
