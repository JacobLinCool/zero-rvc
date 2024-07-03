import gradio as gr
from .constants import HF_TOKEN


class SettingsTab:
    def __init__(self):
        pass

    def ui(self):
        self.exp_dir = gr.Textbox(
            label="Temporary Experiment Directory (auto-managed)",
            placeholder="It will be auto-generated after setup",
            interactive=True,
        )
        gr.Markdown(
            "### Sync with Hugging Face 🤗\n\nThe access token will be use to upload/download the dataset and model."
        )
        self.hf_token = gr.Textbox(
            label="Hugging Face Access Token",
            placeholder="Paste your Hugging Face access token here (hf_...)",
            value=HF_TOKEN,
            interactive=True,
        )

    def build(self):
        pass
