# Lytt – local speech-to-text for meetings

Lytt listens to your meetings (system audio + microphone), transcribes them locally with Whisper on your GPU,
separates speakers, and stores every session so you can copy the transcript into an LLM with one click.
Nothing leaves your machine.

![Lytt](ui/lytt.png)

## Features

- **System audio + microphone** captured separately via WASAPI loopback. Your own voice is labelled "Me";
  other voices are separated automatically into "Speaker 1", "Speaker 2", … (rename them by clicking the label).
- **Multilingual**: Norwegian, Swedish, English (plus Danish/German optional). Language is detected per utterance.
- **Tray icon** next to the clock: colourful while recording, orange when paused, grey when stopped.
  Right-click for start / pause / stop.
- **Sessions** are stored in SQLite. Right-click a session to rename, copy or delete it.
- **Copy** button (Ctrl+Shift+C) puts your own prefix text (e.g. *"Write meeting minutes from this transcript:"*)
  in front of the transcript and copies everything to the clipboard.
- **Mute yourself** by clicking the microphone meter while recording.
- Low resource use: the model only runs while someone is speaking, ~1.6 GB VRAM, ~8× realtime on an RTX PRO 500.

## Requirements

- Windows 10/11, Python 3.12 (`py -3.12`), Microsoft Edge WebView2 runtime (included in Windows 11).
- NVIDIA GPU with a recent driver (CUDA 12 compatible) is recommended. Falls back to CPU (int8) automatically.

## Install

```bash
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the models (~1.7 GB total) into `models/`:

```bash
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('mobiuslabsgmbh/faster-whisper-large-v3-turbo', local_dir='models/large-v3-turbo', allow_patterns=['*.bin','*.json','*.txt'])"
curl -L -o models/eres2net_voxceleb.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx
```

Generate the icon and run the self-test:

```bash
.venv\Scripts\python.exe tray.py
.venv\Scripts\python.exe selftest.py
```

## Run

Double-click **Start Lytt.vbs** (no console window), or create shortcuts:

```bash
powershell -ExecutionPolicy Bypass -File "Create shortcuts.ps1" -Startup
```

`-Startup` also adds Lytt to your Startup folder so it launches at login. If the shortcut does not work,
right-click `Start Lytt.vbs` → *Send to* → *Desktop (create shortcut)*.

The first time, Windows may hide the tray icon behind the `^` overflow arrow; drag it out to keep it visible.

## Usage

| Action | Where |
|---|---|
| Show window | Left-click the tray icon |
| Start / pause / stop | Tray right-click menu, or the buttons top-right in the window |
| Mute / unmute microphone | Click the microphone meter while recording |
| Copy with prefix | **Copy** button or `Ctrl+Shift+C` |
| Rename a speaker | Click the speaker label |
| Edit / delete an utterance | Double-click the text / hover and press ✕ |
| Rename / delete a session | Right-click the session in the list |
| Close the window | Lytt keeps running in the tray. *Quit* is in the tray menu and in Settings |

**Start** creates a new session. **Pause** stops capturing; the next Start continues the same session.
**Stop** ends the session. Everything is saved continuously to `data/lytt.db`.

If you change the audio output device (e.g. switch to a headset) mid-session, press pause and start again.

## How it works

- Audio is split into utterances with a cheap adaptive RMS gate (no CPU cost), resampled to 16 kHz.
- Whisper `large-v3-turbo` via faster-whisper / CTranslate2 (float16 on CUDA, int8 on CPU), with Silero VAD
  and a hallucination filter. Language is chosen among the enabled ones and is "sticky" so it does not flip
  between Norwegian/Swedish on short utterances.
- Speaker separation: ERes2Net speaker embeddings (sherpa-onnx, CPU, ~0.2 s/utterance) with online
  cosine clustering against per-session centroids.
- UI: pywebview (WebView2) + pystray. Storage: SQLite.

## Files

- `data/config.json` – settings (also editable in the window)
- `data/lytt.db` – sessions and transcript (SQLite)
- `data/export/` – exported text files
- `data/lytt.log` – log
- `models/` – downloaded models (not in git)

## Troubleshooting

Run `app.py` with `python.exe` instead of `pythonw.exe` to see errors in a console, and check `data/lytt.log`.
`selftest.py` tests GPU, audio devices, the speaker model and storage; add `--model` to also load Whisper.

## Licenses and commercial use

Lytt itself is MIT licensed (see `LICENSE`). It depends on third-party software and models with their own
licenses. To the best of my knowledge (this is not legal advice) they are:

| Component | License | Commercial use |
|---|---|---|
| OpenAI Whisper (architecture) and `large-v3-turbo` weights (mobiuslabsgmbh CT2 conversion) | MIT | Yes |
| faster-whisper, CTranslate2 | MIT | Yes |
| Silero VAD (bundled in faster-whisper) | MIT | Yes |
| sherpa-onnx, onnxruntime | Apache-2.0 / MIT | Yes |
| ERes2Net speaker model (3D-Speaker, `..._eres2net_sv_en_voxceleb_16k.onnx`) | Apache-2.0 (code/model). **Trained on VoxCeleb**, whose dataset license is CC BY 4.0 with terms that some interpret as research-only. | Model file: yes per Apache-2.0. If you need certainty for a commercial product, check the VoxCeleb terms or swap in a model trained on a clearly licensed dataset. |
| pywebview | BSD-3-Clause | Yes |
| pystray | LGPL-3.0 | Yes when used as a library (dynamic import); modifications to pystray itself must be shared. |
| PyAudioWPatch (PortAudio) | MIT | Yes |
| Pillow, NumPy, SciPy | MIT-CMU / BSD | Yes |
| NVIDIA cuBLAS / cuDNN (pip wheels) | NVIDIA EULA | Use is allowed; redistribution has conditions. |
| Microsoft Edge WebView2 runtime | Microsoft license | Yes, distributed by Microsoft with Windows |
| Hugging Face Hub client | Apache-2.0 | Yes |

Language support (Norwegian, Swedish, English, …) comes from the Whisper weights and carries the same MIT license.
Nothing is sent to any online service; the Hugging Face download is only used to fetch the model once.
