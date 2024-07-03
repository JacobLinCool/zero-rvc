import gradio as gr
from app.settings import SettingsTab
from app.tutorial import TutotialTab
from app.dataset import DatasetTab
from app.train import TrainTab
from app.infer import InferenceTab


with gr.Blocks() as app:
    gr.Markdown("# ZeroRVC")
    gr.Markdown(
        "Run Retrieval-based Voice Conversion training and inference on Hugging Face ZeroGPU or locally."
    )

    settings = SettingsTab()
    tutorial = TutotialTab()
    dataset = DatasetTab()
    training = TrainTab()
    inference = InferenceTab()

    with gr.Accordion(label="Environment Settings"):
        settings.ui()

    with gr.Tabs():
        with gr.Tab(label="Tutorial", id=0):
            tutorial.ui()

        with gr.Tab(label="Dataset", id=1):
            dataset.ui()

        with gr.Tab(label="Training", id=2):
            training.ui()

        with gr.Tab(label="Inference", id=3):
            inference.ui()

    settings.build()
    tutorial.build()
    dataset.build(settings.exp_dir, settings.hf_token)
    training.build(settings.exp_dir, settings.hf_token)
    inference.build(settings.exp_dir)

    app.launch()
