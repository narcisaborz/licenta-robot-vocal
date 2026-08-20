import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI
from app_config import FS

def pick_working_input_device(fs=FS):
    devices = sd.query_devices()
    candidates = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            candidates.append((i, d["name"], d.get("default_samplerate", None)))

    print()
    for i, name, sr in candidates:
        print(f"{i} : {name} (sr={sr})")
    print()

    for i, name, sr in candidates:
        try:
            sd.check_input_settings(device=i, samplerate=fs, channels=1)
            print(f"Aleg microfonul: {i} : {name}")
            return i
        except Exception:
            pass

    raise RuntimeError(f"Nu am găsit niciun microfon compatibil pentru FS={fs}, mono.")

def record_wav(input_device: int, path="ask.wav", seconds=3, fs=FS) -> float:
    print(f"Vorbește acum ({seconds}s)... [device={input_device}]")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="int16", device=input_device)
    sd.wait()
    write(path, fs, audio)
    mean_amp = float(np.abs(audio).mean())
    print("Mean amp:", mean_amp)
    return mean_amp

def stt_whisper(client: OpenAI, path="ask.wav") -> str:
    with open(path, "rb") as f:
        tr = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f
        )
    return tr.text.strip()

