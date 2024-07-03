import os
from datasets import load_dataset
from tqdm import tqdm
from zerorvc import RVCTrainer, pretrained_checkpoints

HF_TOKEN = os.environ.get("HF_TOKEN")
EPOCHS = 100
BATCH_SIZE = 8
DATASET = "JacobLinCool/my-rvc-dataset"
MODEL = "JacobLinCool/my-rvc-model"

dataset = load_dataset(DATASET, token=HF_TOKEN)
print(dataset)

trainer = RVCTrainer(checkpoint_dir="./checkpoints")
training = tqdm(
    trainer.train(
        dataset=dataset["train"],
        resume_from=pretrained_checkpoints(),  # resume training from the pretrained VCTK checkpoint
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    ),
    total=EPOCHS,
)

# Training loop: iterate over epochs
for checkpoint in training:
    training.set_description(
        f"Epoch {checkpoint.epoch}/{EPOCHS} loss: (gen: {checkpoint.loss_gen:.4f}, fm: {checkpoint.loss_fm:.4f}, mel: {checkpoint.loss_mel:.4f}, kl: {checkpoint.loss_kl:.4f}, disc: {checkpoint.loss_disc:.4f})"
    )

    # Save checkpoint every 10 epochs
    if checkpoint.epoch % 10 == 0:
        checkpoint.save(checkpoint_dir=trainer.checkpoint_dir)
        # Directly push the synthesizer to the Hugging Face Hub
        checkpoint.G.push_to_hub(MODEL, token=HF_TOKEN, private=True)

print("Training completed.")
