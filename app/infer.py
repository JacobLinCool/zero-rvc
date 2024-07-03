import os
import shutil
import hashlib
from pathlib import Path
from typing import Tuple
from demucs.separate import main as demucs
import gradio as gr
import numpy as np
import soundfile as sf
from zerorvc import RVC
from .zero import zero
from .model import device


@zero(duration=120)
def infer(
    exp_dir: str, original_audio: str, pitch_mod: int, protect: float
) -> Tuple[int, np.ndarray]:
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        raise gr.Error("Model not found")

    # rename the original audio to the hash
    with open(original_audio, "rb") as f:
        original_audio_hash = hashlib.md5(f.read()).hexdigest()
    ext = Path(original_audio).suffix
    original_audio_hashed = os.path.join(exp_dir, f"{original_audio_hash}{ext}")
    shutil.copy(original_audio, original_audio_hashed)

    out = os.path.join("separated", "htdemucs", original_audio_hash, "vocals.wav")
    if not os.path.exists(out):
        demucs(
            [
                "--two-stems",
                "vocals",
                "-d",
                str(device),
                "-n",
                "htdemucs",
                original_audio_hashed,
            ]
        )

    rvc = RVC.from_pretrained(checkpoint_dir)
    samples = rvc.convert(out, pitch_modification=pitch_mod, protect=protect)
    file = os.path.join(exp_dir, "infer.wav")
    sf.write(file, samples, rvc.sr)

    return file


def merge(exp_dir: str, original_audio: str, vocal: Tuple[int, np.ndarray]) -> str:
    with open(original_audio, "rb") as f:
        original_audio_hash = hashlib.md5(f.read()).hexdigest()
    music = os.path.join("separated", "htdemucs", original_audio_hash, "no_vocals.wav")

    tmp = os.path.join(exp_dir, "tmp.wav")
    sf.write(tmp, vocal[1], vocal[0])

    os.system(
        f"ffmpeg -i {music} -i {tmp} -filter_complex '[1]volume=2[a];[0][a]amix=inputs=2:duration=first:dropout_transition=2' -ac 2 -y {tmp}.merged.mp3"
    )

    return f"{tmp}.merged.mp3"


class InferenceTab:
    def __init__(self):
        pass

    def ui(self):
        gr.Markdown("# Inference")
        gr.Markdown(
            "After trained model is pruned, you can use it to infer on new music. \n"
            "Upload the original audio and adjust the F0 add value to generate the inferred audio."
        )

        with gr.Row():
            self.original_audio = gr.Audio(
                label="Upload original audio",
                type="filepath",
                show_download_button=True,
            )

            with gr.Column():
                self.pitch_mod = gr.Slider(
                    label="Pitch Modification +/-",
                    minimum=-16,
                    maximum=16,
                    step=1,
                    value=0,
                )
                self.protect = gr.Slider(
                    label="Protect",
                    minimum=0,
                    maximum=0.5,
                    step=0.01,
                    value=0.33,
                )

            self.infer_btn = gr.Button(value="Infer", variant="primary")
        with gr.Row():
            self.infer_output = gr.Audio(
                label="Inferred audio", show_download_button=True, format="mp3"
            )
        with gr.Row():
            self.merge_output = gr.Audio(
                label="Merged audio", show_download_button=True, format="mp3"
            )

    def build(self, exp_dir: gr.Textbox):
        self.infer_btn.click(
            fn=infer,
            inputs=[
                exp_dir,
                self.original_audio,
                self.pitch_mod,
                self.protect,
            ],
            outputs=[self.infer_output],
        ).success(
            fn=merge,
            inputs=[exp_dir, self.original_audio, self.infer_output],
            outputs=[self.merge_output],
        )
