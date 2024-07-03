# ZeroRVC

Run Retrieval-based Voice Conversion training and inference with ease.

## Features

- [x] Dataset Preparation
- [x] Hugging Face Datasets Integration
- [x] Hugging Face Accelerate Integration
- [x] Trainer API
- [x] Inference API
  - [ ] Index Support
- [ ] Tensorboard Support
- [ ] FP16 Support

## Dataset Preparation

ZeroRVC provides a simple API to prepare your dataset for training. You only need to provide the path to your audio files. The feature extraction models will be downloaded automatically, or you can provide your own with the `hubert` and `rmvpe` arguments.

```py
from zerorvc import prepare

dataset = prepare("./my-voices")
```

Since `dataset` is a Hugging Face Dataset object, you can easily push it to the Hugging Face Hub.

```py
dataset.push_to_hub("my-rvc-dataset", token=HF_TOKEN)
```

And bring the preprocessed dataset back with the following code.

```py
from datasets import load_dataset

dataset = load_dataset("my-rvc-dataset")
```

## Training

Once you've prepared your dataset, you can start training your model with the `RVCTrainer`.

```py
from tqdm import tqdm
from zerorvc import RVCTrainer

epochs = 100
trainer = RVCTrainer(checkpoint_dir="./checkpoints")
training = tqdm(
    trainer.train(
        dataset=dataset["train"], # preprocessed dataset
        resume_from=trainer.latest_checkpoint(), # resume training from the latest checkpoint if any
        epochs=epochs, batch_size=8
    )
)

# Training loop: iterate over epochs
for checkpoint in training:
    training.set_description(
        f"Epoch {checkpoint.epoch}/{epochs} loss: (gen: {checkpoint.loss_gen:.4f}, fm: {checkpoint.loss_fm:.4f}, mel: {checkpoint.loss_mel:.4f}, kl: {checkpoint.loss_kl:.4f}, disc: {checkpoint.loss_disc:.4f})"
    )

    # Save checkpoint every 10 epochs
    if checkpoint.epoch % 10 == 0:
        checkpoint.save(checkpoint_dir=trainer.checkpoint_dir)
        # Directly push the synthesizer to the Hugging Face Hub
        checkpoint.G.push_to_hub("my-rvc-model", token=HF_TOKEN)

print("Training completed.")
```

You can also push the whole GAN weights to the Hugging Face Hub.

```py
checkpoint.push_to_hub("my-rvc-model", token=HF_TOKEN)
```

## Inference

ZeroRVC provides an easy API to convert your voice with the trained model.

```py
from zerorvc import RVC
import soundfile as sf

rvc = RVC.from_pretrained("my-rvc-model")
samples = rvc.convert("test.mp3")
sf.write("output.wav", samples, rvc.sr)
```
