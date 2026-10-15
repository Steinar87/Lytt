"""Start-with-Windows via the per-user Run key (HKCU). No admin rights required."""
from __future__ import annotations

import os
import sys
import winreg

from config import APP_DIR

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "Lytt"


def _command() -> str:
    vbs = os.path.join(APP_DIR, "Start Lytt.vbs")
    wscript = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "wscript.exe")
    return f'"{wscript}" "{vbs}"'


def _startup_shortcut() -> str:
    return os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows",
                        "Start Menu", "Programs", "Startup", "Lytt.lnk")


def is_enabled() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            winreg.QueryValueEx(k, VALUE_NAME)
            return True
    except OSError:
        return False


def set_enabled(enabled: bool) -> None:
    """Writes or removes the Run entry. Also removes any old Startup-folder shortcut."""
    if not sys.platform.startswith("win"):
        return
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
        if enabled:
            winreg.SetValueEx(k, VALUE_NAME, 0, winreg.REG_SZ, _command())
        else:
            try:
                winreg.DeleteValue(k, VALUE_NAME)
            except OSError:
                pass
    lnk = _startup_shortcut()
    if os.path.exists(lnk):
        try:
            os.remove(lnk)      # the Run entry replaces the old shortcut
        except OSError:
            pass


def sync(enabled: bool) -> None:
    """Called at startup: make the system match the setting."""
    try:
        if is_enabled() != enabled or (not enabled and os.path.exists(_startup_shortcut())):
            set_enabled(enabled)
    except Exception:
        pass
