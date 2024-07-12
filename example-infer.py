import os
from zerorvc import RVC
import soundfile as sf

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL = "JacobLinCool/my-rvc-model3"

rvc = RVC.from_pretrained(MODEL, token=HF_TOKEN)
samples = rvc.convert("test.mp3")
sf.write("output.wav", samples, rvc.sr)

pitch_modifications = [-12, -8, -4, 4, 8, 12]
for pitch_modification in pitch_modifications:
    samples = rvc.convert("test.mp3", pitch_modification=pitch_modification)
    sf.write(f"output-{pitch_modification}.wav", samples, rvc.sr)
