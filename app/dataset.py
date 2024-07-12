import os
import gradio as gr
import zipfile
import tempfile
from zerorvc import prepare
from datasets import load_dataset, load_from_disk
from .constants import ROOT_EXP_DIR, BATCH_SIZE
from .zero import zero
from .model import accelerator


def extract_audio_files(zip_file: str, target_dir: str) -> list[str]:
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    audio_files = [
        os.path.join(target_dir, f)
        for f in os.listdir(target_dir)
        if f.endswith((".wav", ".mp3", ".ogg"))
    ]
    if not audio_files:
        raise gr.Error("No audio files found at the top level of the zip file")

    return audio_files


def make_dataset_from_zip(exp_dir: str, zip_file: str):
    if not exp_dir:
        exp_dir = tempfile.mkdtemp(dir=ROOT_EXP_DIR)
        print(f"Using exp dir: {exp_dir}")

    data_dir = os.path.join(exp_dir, "raw_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    extract_audio_files(zip_file, data_dir)

    ds = prepare(
        data_dir,
        accelerator=accelerator,
        batch_size=BATCH_SIZE,
        stage=1,
    )

    return exp_dir, str(ds)


@zero(duration=120)
def make_dataset_from_zip_stage_2(exp_dir: str):
    data_dir = os.path.join(exp_dir, "raw_data")
    ds = prepare(
        data_dir,
        accelerator=accelerator,
        batch_size=BATCH_SIZE,
        stage=2,
    )
    return exp_dir, str(ds)


def make_dataset_from_zip_stage_3(exp_dir: str):
    data_dir = os.path.join(exp_dir, "raw_data")
    ds = prepare(
        data_dir,
        accelerator=accelerator,
        batch_size=BATCH_SIZE,
        stage=3,
    )

    dataset = os.path.join(exp_dir, "dataset")
    ds.save_to_disk(dataset)
    return exp_dir, str(ds)


def make_dataset_from_repo(repo: str, hf_token: str):
    ds = load_dataset(repo, token=hf_token)
    ds = prepare(
        ds,
        accelerator=accelerator,
        batch_size=BATCH_SIZE,
        stage=1,
    )
    return str(ds)


@zero(duration=120)
def make_dataset_from_repo_stage_2(repo: str, hf_token: str):
    ds = load_dataset(repo, token=hf_token)
    ds = prepare(
        ds,
        accelerator=accelerator,
        batch_size=BATCH_SIZE,
        stage=2,
    )
    return str(ds)


def make_dataset_from_repo_stage_3(exp_dir: str, repo: str, hf_token: str):
    ds = load_dataset(repo, token=hf_token)
    ds = prepare(
        ds,
        accelerator=accelerator,
        batch_size=BATCH_SIZE,
        stage=3,
    )

    if not exp_dir:
        exp_dir = tempfile.mkdtemp(dir=ROOT_EXP_DIR)
        print(f"Using exp dir: {exp_dir}")

    dataset = os.path.join(exp_dir, "dataset")
    ds.save_to_disk(dataset)
    return exp_dir, str(ds)


def use_dataset(exp_dir: str, repo: str, hf_token: str):
    gr.Info("Fetching dataset")
    ds = load_dataset(repo, token=hf_token)

    if not exp_dir:
        exp_dir = tempfile.mkdtemp(dir=ROOT_EXP_DIR)
        print(f"Using exp dir: {exp_dir}")

    dataset = os.path.join(exp_dir, "dataset")
    ds.save_to_disk(dataset)
    return exp_dir, str(ds)


def upload_dataset(exp_dir: str, repo: str, hf_token: str):
    dataset = os.path.join(exp_dir, "dataset")
    if not os.path.exists(dataset):
        raise gr.Error("Dataset not found")

    gr.Info("Uploading dataset")
    ds = load_from_disk(dataset)
    ds.push_to_hub(repo, token=hf_token, private=True)
    gr.Info("Dataset uploaded successfully")


class DatasetTab:
    def __init__(self):
        pass

    def ui(self):
        gr.Markdown("# Dataset")
        gr.Markdown("The suggested dataset size is > 5 minutes of audio.")

        gr.Markdown("## Create Dataset from ZIP")
        gr.Markdown(
            "Create a dataset by simply upload a zip file containing audio files. The audio files should be at the top level of the zip file."
        )
        with gr.Row():
            self.zip_file = gr.File(
                label="Upload a zip file containing audio files",
                file_types=["zip"],
            )
            self.make_ds_from_dir = gr.Button(
                value="Create Dataset from ZIP", variant="primary"
            )

        gr.Markdown("## Create Dataset from Dataset Repository")
        gr.Markdown(
            "You can also create a dataset from any Hugging Face dataset repository that has 'audio' column."
        )
        with gr.Row():
            self.repo = gr.Textbox(
                label="Hugging Face Dataset Repository",
                placeholder="username/dataset-name",
            )
            self.make_ds_from_repo = gr.Button(
                value="Create Dataset from Repo", variant="primary"
            )

        gr.Markdown("## Sync Preprocessed Dataset")
        gr.Markdown(
            "After you have preprocessed the dataset, you can upload the dataset to Hugging Face. And fetch it back later directly."
        )
        with gr.Row():
            self.preprocessed_repo = gr.Textbox(
                label="Hugging Face Dataset Repository",
                placeholder="username/dataset-name",
            )
            self.fetch_ds = gr.Button(value="Fetch Dataset", variant="primary")
            self.upload_ds = gr.Button(value="Upload Dataset", variant="primary")

        self.ds_state = gr.Textbox(label="Dataset Info", lines=5)

    def build(self, exp_dir: gr.Textbox, hf_token: gr.Textbox):
        self.make_ds_from_dir.click(
            fn=make_dataset_from_zip,
            inputs=[exp_dir, self.zip_file],
            outputs=[exp_dir, self.ds_state],
        ).success(
            fn=make_dataset_from_zip_stage_2,
            inputs=[exp_dir],
            outputs=[exp_dir, self.ds_state],
        ).success(
            fn=make_dataset_from_zip_stage_3,
            inputs=[exp_dir],
            outputs=[exp_dir, self.ds_state],
        )

        self.make_ds_from_repo.click(
            fn=make_dataset_from_repo,
            inputs=[self.repo, hf_token],
            outputs=[self.ds_state],
        ).success(
            fn=make_dataset_from_repo_stage_2,
            inputs=[self.repo, hf_token],
            outputs=[self.ds_state],
        ).success(
            fn=make_dataset_from_repo_stage_3,
            inputs=[exp_dir, self.repo, hf_token],
            outputs=[exp_dir, self.ds_state],
        )

        self.fetch_ds.click(
            fn=use_dataset,
            inputs=[exp_dir, self.preprocessed_repo, hf_token],
            outputs=[exp_dir, self.ds_state],
        )

        self.upload_ds.click(
            fn=upload_dataset,
            inputs=[exp_dir, self.preprocessed_repo, hf_token],
            outputs=[],
        )
