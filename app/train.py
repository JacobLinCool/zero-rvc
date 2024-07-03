import os
import tempfile
import gradio as gr
import torch
from zerorvc import RVCTrainer, pretrained_checkpoints, SynthesizerTrnMs768NSFsid
from zerorvc.trainer import TrainingCheckpoint
from datasets import load_from_disk
from huggingface_hub import snapshot_download
from .zero import zero
from .model import accelerator, device
from .constants import BATCH_SIZE, ROOT_EXP_DIR, TRAINING_EPOCHS


@zero(duration=240)
def train_model(exp_dir: str, progress=gr.Progress()):
    dataset = os.path.join(exp_dir, "dataset")
    if not os.path.exists(dataset):
        raise gr.Error("Dataset not found. Please prepare the dataset first.")

    ds = load_from_disk(dataset)
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    trainer = RVCTrainer(checkpoint_dir)

    resume_from = trainer.latest_checkpoint()
    if resume_from is None:
        resume_from = pretrained_checkpoints()
        gr.Info(f"Starting training from pretrained checkpoints.")
    else:
        gr.Info(f"Resuming training from {resume_from}")

    tqdm = progress.tqdm(
        trainer.train(
            dataset=ds["train"],
            resume_from=resume_from,
            batch_size=BATCH_SIZE,
            epochs=TRAINING_EPOCHS,
            accelerator=accelerator,
        ),
        total=TRAINING_EPOCHS,
        unit="epochs",
        desc="Training",
    )

    for ckpt in tqdm:
        info = f"Epoch: {ckpt.epoch} loss: (gen: {ckpt.loss_gen:.4f}, fm: {ckpt.loss_fm:.4f}, mel: {ckpt.loss_mel:.4f}, kl: {ckpt.loss_kl:.4f}, disc: {ckpt.loss_disc:.4f})"
        print(info)
        latest: TrainingCheckpoint = ckpt

    latest.save(trainer.checkpoint_dir)
    latest.G.save_pretrained(trainer.checkpoint_dir)

    result = f"{TRAINING_EPOCHS} epochs trained. Latest loss: (gen: {latest.loss_gen:.4f}, fm: {latest.loss_fm:.4f}, mel: {latest.loss_mel:.4f}, kl: {latest.loss_kl:.4f}, disc: {latest.loss_disc:.4f})"

    del trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def upload_model(exp_dir: str, repo: str, hf_token: str):
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        raise gr.Error("Model not found")

    gr.Info("Uploading model")
    model = SynthesizerTrnMs768NSFsid.from_pretrained(checkpoint_dir)
    model.push_to_hub(repo, token=hf_token, private=True)
    gr.Info("Model uploaded successfully")


def upload_checkpoints(exp_dir: str, repo: str, hf_token: str):
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        raise gr.Error("Checkpoints not found")

    gr.Info("Uploading checkpoints")
    trainer = RVCTrainer(checkpoint_dir)
    trainer.push_to_hub(repo, token=hf_token, private=True)
    gr.Info("Checkpoints uploaded successfully")


def fetch_model(exp_dir: str, repo: str, hf_token: str):
    if not exp_dir:
        exp_dir = tempfile.mkdtemp(dir=ROOT_EXP_DIR)
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")

    gr.Info("Fetching model")
    files = ["README.md", "config.json", "model.safetensors"]
    snapshot_download(
        repo, token=hf_token, local_dir=checkpoint_dir, allow_patterns=files
    )
    gr.Info("Model fetched successfully")

    return exp_dir


def fetch_checkpoints(exp_dir: str, repo: str, hf_token: str):
    if not exp_dir:
        exp_dir = tempfile.mkdtemp(dir=ROOT_EXP_DIR)
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")

    gr.Info("Fetching checkpoints")
    snapshot_download(repo, token=hf_token, local_dir=checkpoint_dir)
    gr.Info("Checkpoints fetched successfully")

    return exp_dir


class TrainTab:
    def __init__(self):
        pass

    def ui(self):
        gr.Markdown("# Training")
        gr.Markdown(
            "You can start training the model by clicking the button below. "
            f"Each time you click the button, the model will train for {TRAINING_EPOCHS} epochs, which takes about 3 minutes on ZeroGPU (A100). "
        )

        with gr.Row():
            self.train_btn = gr.Button(value="Train", variant="primary")
            self.result = gr.Textbox(label="Training Result", lines=3)

        gr.Markdown("## Sync Model and Checkpoints with Hugging Face")
        gr.Markdown(
            "You can upload the trained model and checkpoints to Hugging Face for sharing or further training."
        )

        self.repo = gr.Textbox(label="Repository ID", placeholder="username/repo")
        with gr.Row():
            self.upload_model_btn = gr.Button(value="Upload Model", variant="primary")
            self.upload_checkpoints_btn = gr.Button(
                value="Upload Checkpoints", variant="primary"
            )
        with gr.Row():
            self.fetch_mode_btn = gr.Button(value="Fetch Model", variant="primary")
            self.fetch_checkpoints_btn = gr.Button(
                value="Fetch Checkpoints", variant="primary"
            )

    def build(self, exp_dir: gr.Textbox, hf_token: gr.Textbox):
        self.train_btn.click(
            fn=train_model,
            inputs=[exp_dir],
            outputs=[self.result],
        )

        self.upload_model_btn.click(
            fn=upload_model,
            inputs=[exp_dir, self.repo, hf_token],
        )

        self.upload_checkpoints_btn.click(
            fn=upload_checkpoints,
            inputs=[exp_dir, self.repo, hf_token],
        )

        self.fetch_mode_btn.click(
            fn=fetch_model,
            inputs=[exp_dir, self.repo, hf_token],
            outputs=[exp_dir],
        )

        self.fetch_checkpoints_btn.click(
            fn=fetch_checkpoints,
            inputs=[exp_dir, self.repo, hf_token],
            outputs=[exp_dir],
        )
