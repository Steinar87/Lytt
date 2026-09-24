"""Ikon i systemstatusfeltet (ved klokka). Farger når opptak er aktivt, grått når stoppet."""
from __future__ import annotations

from typing import Callable

from PIL import Image, ImageDraw

import pystray

SIZE = 128
BARS = [0.30, 0.55, 0.85, 1.0, 0.7, 0.45, 0.25]   # relative høyder på "lydbølgen"


def _gradient(size: int, c1, c2) -> Image.Image:
    """Diagonal fargegradient."""
    img = Image.new("RGBA", (size, size), c1)
    px = img.load()
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * size - 2)
            px[x, y] = tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3)) + (255,)
    return img


def make_icon(state: str, size: int = SIZE) -> Image.Image:
    """state: recording | paused | stopped"""
    s = size
    r = int(s * 0.24)
    if state == "recording":
        bg = _gradient(s, (124, 92, 255), (56, 222, 255))    # lilla -> cyan
        bar = (255, 255, 255, 255)
    elif state == "paused":
        bg = _gradient(s, (255, 170, 60), (255, 110, 110))   # oransje -> rosa
        bar = (255, 255, 255, 235)
    else:
        bg = Image.new("RGBA", (s, s), (92, 96, 104, 255))
        bar = (200, 203, 208, 255)

    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, s - 1, s - 1), radius=r, fill=255)
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    img.paste(bg, (0, 0), mask)

    d = ImageDraw.Draw(img)
    n = len(BARS)
    pad = s * 0.18
    w_avail = s - 2 * pad
    bw = w_avail / (n * 1.9 - 0.9)
    gap = bw * 0.9
    max_h = s * 0.56
    cy = s / 2
    for i, h in enumerate(BARS):
        x0 = pad + i * (bw + gap)
        bh = max(bw, max_h * h)
        d.rounded_rectangle((x0, cy - bh / 2, x0 + bw, cy + bh / 2), radius=bw / 2, fill=bar)

    if state == "paused":
        # liten pause-markør nederst til høyre
        dot = s * 0.26
        d.ellipse((s - dot - s * 0.06, s - dot - s * 0.06, s - s * 0.06, s - s * 0.06),
                  fill=(20, 22, 28, 255))
        pw = dot * 0.16
        px0 = s - dot / 2 - s * 0.06
        py0 = s - dot / 2 - s * 0.06
        d.rectangle((px0 - pw * 1.6, py0 - dot * 0.22, px0 - pw * 0.6, py0 + dot * 0.22), fill=(255, 255, 255, 255))
        d.rectangle((px0 + pw * 0.6, py0 - dot * 0.22, px0 + pw * 1.6, py0 + dot * 0.22), fill=(255, 255, 255, 255))
    return img


class Tray:
    def __init__(self, *, get_state: Callable[[], str], on_toggle: Callable[[], None],
                 on_stop: Callable[[], None], on_show: Callable[[], None], on_quit: Callable[[], None]):
        self._get_state = get_state
        self._icons = {st: make_icon(st) for st in ("recording", "paused", "stopped")}

        def toggle_text(_item):
            st = get_state()
            return "Pause recording" if st == "recording" else ("Resume recording" if st == "paused" else "Start recording")

        menu = pystray.Menu(
            pystray.MenuItem("Show Lytt", lambda i, it: on_show(), default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(toggle_text, lambda i, it: on_toggle()),
            pystray.MenuItem("Stop session", lambda i, it: on_stop(), enabled=lambda it: get_state() != "stopped"),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda i, it: on_quit()),
        )
        self.icon = pystray.Icon("Lytt", self._icons["stopped"], "Lytt – speech to text", menu)

    def run_detached(self) -> None:
        self.icon.run_detached()

    def update(self) -> None:
        st = self._get_state()
        self.icon.icon = self._icons.get(st, self._icons["stopped"])
        self.icon.title = {"recording": "Lytt – recording", "paused": "Lytt – paused",
                           "stopped": "Lytt – ready"}.get(st, "Lytt")
        try:
            self.icon.update_menu()
        except Exception:
            pass

    def notify(self, msg: str, title: str = "Lytt") -> None:
        try:
            self.icon.notify(msg, title)
        except Exception:
            pass

    def stop(self) -> None:
        try:
            self.icon.stop()
        except Exception:
            pass


if __name__ == "__main__":
    # Lager ikonfiler til snarveier
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    make_icon("recording", 256).save(os.path.join(here, "ui", "lytt.ico"),
                                     sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    make_icon("recording", 256).save(os.path.join(here, "ui", "lytt.png"))
    print("ikoner skrevet")
