"""Lytt – lokal tale-til-tekst for møter. Startpunkt.

Kjør:  .venv\\Scripts\\pythonw.exe app.py
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import threading
import time
import traceback

import numpy as np
import webview

import config as cfgmod
from audio import Recorder
from diarizer import Diarizer
from store import Store
from transcriber import Transcriber
from tray import Tray

APP_DIR = cfgmod.APP_DIR
UI_PATH = os.path.join(APP_DIR, "ui", "index.html")
LOG_PATH = os.path.join(cfgmod.DATA_DIR, "lytt.log")


# ---------------------------------------------------------------- hjelpere
def set_clipboard(text: str) -> bool:
    """Legger tekst på Windows-utklippstavlen (CF_UNICODETEXT) via ctypes."""
    try:
        CF_UNICODETEXT = 13
        GMEM_MOVEABLE = 0x0002
        u32, k32 = ctypes.windll.user32, ctypes.windll.kernel32
        # 64-bit handles/pointers: without these restypes the values are truncated to 32 bit
        k32.GlobalAlloc.restype = ctypes.c_void_p
        k32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
        k32.GlobalLock.restype = ctypes.c_void_p
        k32.GlobalLock.argtypes = [ctypes.c_void_p]
        k32.GlobalUnlock.argtypes = [ctypes.c_void_p]
        u32.SetClipboardData.restype = ctypes.c_void_p
        u32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
        u32.OpenClipboard.argtypes = [ctypes.c_void_p]
        data = text.encode("utf-16-le") + b"\x00\x00"
        for _ in range(5):
            if u32.OpenClipboard(None):
                break
            time.sleep(0.05)
        else:
            return False
        try:
            u32.EmptyClipboard()
            h = k32.GlobalAlloc(GMEM_MOVEABLE, len(data))
            p = k32.GlobalLock(h)
            ctypes.memmove(p, data, len(data))
            k32.GlobalUnlock(h)
            if not u32.SetClipboardData(CF_UNICODETEXT, h):
                raise OSError("SetClipboardData failed")
        finally:
            u32.CloseClipboard()
        return True
    except Exception as e:
        log("clipboard via ctypes failed:", e)
        # Fallback: PowerShell Set-Clipboard (reads UTF-8 from stdin)
        try:
            import subprocess
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "[Console]::InputEncoding=[Text.Encoding]::UTF8; $t=[Console]::In.ReadToEnd(); Set-Clipboard -Value $t"],
                input=text.encode("utf-8"), check=True, timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return True
        except Exception as e2:
            log("clipboard via powershell failed:", e2)
            return False


def log(*a):
    line = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(line)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------- kjerne
class Engine:
    """Binder sammen opptak, transkribering, stemmeskilling og lagring."""

    def __init__(self):
        self.cfg = cfgmod.load()
        self.store = Store()
        self.diarizer = Diarizer(self.cfg["speaker_threshold"])
        self.recorder = Recorder(self._on_chunk)
        self.transcriber = Transcriber(self._on_result, self._on_model_status)
        self.state = "stopped"          # recording | paused | stopped
        self.session: dict | None = None
        self.session_t0 = 0.0
        self.model_state = "idle"
        self.model_msg = ""
        self.window: webview.Window | None = None
        self.tray: Tray | None = None
        self.warnings: list[str] = []
        self._lock = threading.RLock()
        threading.Thread(target=self._level_loop, daemon=True).start()

    # ---- hendelser til UI
    def emit(self, evt: dict) -> None:
        w = self.window
        if w is None:
            return
        try:
            w.evaluate_js(f"window.Lytt && Lytt.onEvent({json.dumps(evt, ensure_ascii=False)})")
        except Exception:
            pass

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "session": self.session,
            "model_state": self.model_state,
            "model_msg": self.model_msg,
            "device_info": self.transcriber.device_info,
            "backlog": self.transcriber.backlog,
            "warnings": self.warnings,
            "sources": self.recorder.device_names(),
            "diarize_available": self.diarizer.available,
            "mic_muted": bool(self.cfg.get("mic_muted")),
            "capture_mic": bool(self.cfg.get("capture_mic")),
        }

    def push_state(self) -> None:
        self.emit({"type": "state", **self.snapshot()})
        if self.tray:
            self.tray.update()

    # ---- opptaksstyring
    def start(self, title: str | None = None, new_session: bool = False) -> dict:
        with self._lock:
            if self.state == "recording" and not new_session:
                return self.snapshot()
            if new_session and self.session and self.session.get("status") != "stopped":
                self.recorder.stop()
                self.store.set_status(self.session["id"], "stopped")
            if new_session or self.session is None or self.session.get("status") == "stopped":
                self.session = self.store.create_session((title or "").strip() or None)
                self.session_t0 = self.session["started_at"]
                self.diarizer.reset()
                self.diarizer.threshold = float(self.cfg["speaker_threshold"])
            chunk_kwargs = self._chunk_kwargs()
            self.warnings = self.recorder.start(
                capture_mic=bool(self.cfg["capture_mic"]) and not bool(self.cfg.get("mic_muted")),
                capture_system=bool(self.cfg["capture_system"]),
                mic_device=self.cfg.get("mic_device"),
                chunk_kwargs=chunk_kwargs,
            )
            if not self.recorder.sources:
                self.warnings.append("No audio sources active – check the settings.")
                self.state = "paused"
            else:
                self.state = "recording"
            self.store.set_status(self.session["id"], self.state)
            self.session = self.store.get_session(self.session["id"])
            self.transcriber.preload(self.cfg)
            for w in self.warnings:
                log("warning:", w)
        self.push_state()
        self.emit({"type": "sessions"})
        return self.snapshot()

    def pause(self) -> dict:
        with self._lock:
            if self.state == "recording":
                self.recorder.stop()
                self.state = "paused"
                if self.session:
                    self.store.set_status(self.session["id"], "paused")
                    self.session = self.store.get_session(self.session["id"])
        self.push_state()
        return self.snapshot()

    def toggle(self) -> dict:
        return self.pause() if self.state == "recording" else self.start()

    def _chunk_kwargs(self) -> dict:
        return dict(
            silence_ms=int(self.cfg["silence_ms"]),
            min_speech_sec=float(self.cfg["min_speech_sec"]),
            max_chunk_sec=float(self.cfg["max_chunk_sec"]),
        )

    def set_mic_muted(self, muted: bool) -> dict:
        """Mutes/unmutes the microphone. Takes effect immediately while recording."""
        with self._lock:
            self.cfg = cfgmod.save({**self.cfg, "mic_muted": bool(muted)})
            if self.state == "recording" and self.cfg["capture_mic"]:
                warn = self.recorder.set_mic(not muted, self.cfg.get("mic_device"), self._chunk_kwargs())
                self.warnings = [w for w in self.warnings if "microphone" not in w.lower()]
                if warn:
                    self.warnings.append(warn)
        self.push_state()
        return self.snapshot()

    def stop(self) -> dict:
        with self._lock:
            self.recorder.stop()
            if self.session:
                self.store.set_status(self.session["id"], "stopped")
                self.session = self.store.get_session(self.session["id"])
            self.state = "stopped"
        self.push_state()
        self.emit({"type": "sessions"})
        return self.snapshot()

    # ---- lyd -> transkribering -> lagring
    def _on_chunk(self, source: str, audio: np.ndarray, t_start: float, t_end: float) -> None:
        if self.session is None:
            return
        self.transcriber.submit({
            "cfg": self.cfg,
            "audio": audio,
            "source": source,
            "session_id": self.session["id"],
            "t_start": max(0.0, t_start - self.session_t0),
            "t_end": max(0.0, t_end - self.session_t0),
        })
        self.emit({"type": "backlog", "backlog": self.transcriber.backlog})

    def _on_result(self, res: dict) -> None:
        try:
            audio = res.pop("audio")
            if res["source"] == "mic":
                speaker = self.cfg["my_name"] or "Me"
            elif self.cfg["diarize"] and self.diarizer.available:
                speaker = self.diarizer.assign(audio)
            else:
                speaker = "Speaker"
            seg = self.store.add_segment({**res, "speaker": speaker})
            self.emit({"type": "segment", "segment": seg, "backlog": self.transcriber.backlog})
        except Exception:
            log(traceback.format_exc())

    def _on_model_status(self, state: str, msg: str) -> None:
        self.model_state = state
        if msg:
            self.model_msg = msg
            log("modell:", msg)
        self.emit({"type": "model", "model_state": state, "model_msg": self.model_msg,
                   "device_info": self.transcriber.device_info, "backlog": self.transcriber.backlog})

    def _level_loop(self) -> None:
        while True:
            time.sleep(0.15)
            if self.state == "recording" and self.window is not None:
                self.emit({"type": "levels", "levels": self.recorder.levels(),
                           "speaking": self.recorder.speaking()})

    # ---- innstillinger
    def save_config(self, cfg: dict) -> dict:
        self.cfg = cfgmod.save({**self.cfg, **cfg})
        self.diarizer.threshold = float(self.cfg["speaker_threshold"])
        return self.cfg

    def shutdown(self) -> None:
        try:
            self.stop()
        finally:
            self.transcriber.close()
            self.recorder.terminate()


# ---------------------------------------------------------------- JS-API
class Api:
    """Metoder som kalles fra index.html via window.pywebview.api.*"""

    def __init__(self, engine: Engine):
        self._e = engine

    def get_state(self):
        return self._e.snapshot()

    def start(self):
        return self._e.start()

    def new_session(self, title=None):
        return self._e.start(title=title, new_session=True)

    def search_sessions(self, query):
        return self._e.store.search_sessions(query or "")

    def get_about(self):
        return {"version": cfgmod.APP_VERSION, "repo": cfgmod.APP_REPO,
                "device_info": self._e.transcriber.device_info, "model": self._e.cfg.get("model"),
                "python": sys.version.split()[0]}

    def open_url(self, url):
        if str(url).startswith("https://"):
            os.startfile(url)
        return True

    def pause(self):
        return self._e.pause()

    def stop(self):
        return self._e.stop()

    def set_mic_muted(self, muted):
        return self._e.set_mic_muted(bool(muted))

    def list_sessions(self):
        return self._e.store.list_sessions()

    def get_session(self, session_id):
        s = self._e.store.get_session(int(session_id))
        if not s:
            return None
        s["segments"] = self._e.store.get_segments(int(session_id))
        return s

    def delete_session(self, session_id):
        sid = int(session_id)
        if self._e.session and self._e.session["id"] == sid:
            self._e.stop()
            self._e.session = None
        self._e.store.delete_session(sid)
        return True

    def rename_session(self, session_id, title):
        self._e.store.rename_session(int(session_id), title)
        if self._e.session and self._e.session["id"] == int(session_id):
            self._e.session = self._e.store.get_session(int(session_id))
        return True

    def rename_speaker(self, session_id, old, new):
        self._e.store.rename_speaker(int(session_id), old, new)
        if self._e.session and self._e.session["id"] == int(session_id):
            self._e.diarizer.rename(old, new)
        return True

    def delete_segment(self, segment_id):
        self._e.store.delete_segment(int(segment_id))
        return True

    def update_segment(self, segment_id, text):
        self._e.store.update_segment_text(int(segment_id), text)
        return True

    def copy_transcript(self, session_id, with_prefix=True, with_speakers=True, with_time=False):
        text = self._e.store.transcript_text(int(session_id), with_speakers, with_time)
        if with_prefix:
            text = (self._e.cfg.get("prefix") or "") + text
        ok = set_clipboard(text)
        return {"ok": ok, "chars": len(text)}

    def export_transcript(self, session_id, with_speakers=True, with_time=True):
        s = self._e.store.get_session(int(session_id))
        if not s:
            return None
        text = self._e.store.transcript_text(int(session_id), with_speakers, with_time)
        safe = "".join(c if c.isalnum() or c in " -_." else "_" for c in s["title"]).strip() or "okt"
        out_dir = os.path.join(cfgmod.DATA_DIR, "export")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{safe}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{s['title']}\n{time.strftime('%d.%m.%Y %H:%M', time.localtime(s['started_at']))}\n\n{text}\n")
        os.startfile(out_dir)
        return path

    def get_config(self):
        return self._e.cfg

    def save_config(self, cfg):
        return self._e.save_config(cfg)

    def list_mics(self):
        return self._e.recorder.list_mics()

    def hide(self):
        if self._e.window:
            self._e.window.hide()
        return True

    def minimize(self):
        if self._e.window:
            self._e.window.minimize()
        return True

    def quit(self):
        threading.Thread(target=_quit, daemon=True).start()
        return True

    def open_data_folder(self):
        os.startfile(cfgmod.DATA_DIR)
        return True


# ---------------------------------------------------------------- oppstart
engine: Engine | None = None
_quitting = False


def _quit():
    global _quitting
    if _quitting:
        return
    _quitting = True
    try:
        if engine:
            engine.shutdown()
            if engine.tray:
                engine.tray.stop()
            if engine.window:
                engine.window.destroy()
    except Exception:
        pass
    os._exit(0)


def _show_window():
    if engine and engine.window:
        try:
            engine.window.show()
            engine.window.restore()
        except Exception:
            pass


ICON_PATH = os.path.join(APP_DIR, "ui", "lytt.ico")


PORT_FILE = os.path.join(cfgmod.DATA_DIR, "lytt.port")


def _single_instance() -> bool:
    """Returns True if this is the only instance. Otherwise asks the running one to show itself."""
    import socket
    try:
        with open(PORT_FILE) as f:
            port = int(f.read().strip())
        with socket.create_connection(("127.0.0.1", port), timeout=1) as s:
            s.sendall(b"show\n")
            return False          # another instance answered
    except Exception:
        pass                      # no instance, or a stale port file

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(2)
    with open(PORT_FILE, "w") as f:
        f.write(str(srv.getsockname()[1]))

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
                with conn:
                    if b"show" in conn.recv(64):
                        _show_window()
            except Exception:
                pass

    threading.Thread(target=serve, name="single-instance", daemon=True).start()
    return True


def main():
    global engine
    os.makedirs(cfgmod.DATA_DIR, exist_ok=True)
    if not _single_instance():
        return
    # Own taskbar identity, so Windows shows Lytt's icon instead of grouping it under Python
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Expectit.Lytt")
    except Exception:
        pass
    engine = Engine()
    api = Api(engine)

    # Cache-buster: WebView2 has been seen serving a stale disk-cached copy of the UI
    ui_url = f"{UI_PATH}?v={int(os.path.getmtime(UI_PATH))}"
    window = webview.create_window(
        "Lytt", ui_url, js_api=api,
        width=1180, height=760, min_size=(860, 560),
        background_color="#0e1016", text_select=True,
    )
    engine.window = window

    def on_closing():
        # Close button: hide to tray (default) or quit, depending on settings. Never a notification.
        if engine.cfg.get("close_to_tray", True):
            window.hide()
            return False
        threading.Thread(target=_quit, daemon=True).start()
        return False

    window.events.closing += on_closing

    tray = Tray(
        get_state=lambda: engine.state,
        on_toggle=lambda: engine.toggle(),
        on_stop=lambda: engine.stop(),
        on_show=_show_window,
        on_quit=_quit,
    )
    engine.tray = tray
    tray.run_detached()

    def after_start():
        if engine.cfg.get("start_hidden") or "--hidden" in sys.argv:
            window.hide()
        if "--start" in sys.argv:
            engine.start()

    webview.start(after_start, gui="edgechromium", debug="--debug" in sys.argv, private_mode=False,
                  icon=ICON_PATH, storage_path=os.path.join(cfgmod.DATA_DIR, "webview"))
    _quit()


if __name__ == "__main__":
    main()
