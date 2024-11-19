# ZeroRVC

Run Retrieval-based Voice Conversion training and inference with ease.

## Features

- [x] Dataset Preparation
- [x] Hugging Face Datasets Integration
- [x] Hugging Face Accelerate Integration
- [x] Trainer API
- [x] Inference API
  - [ ] Index Support
- [x] Tensorboard Support
- [ ] FP16 Support

## Dataset Preparation

ZeroRVC provides a simple API to prepare your dataset for training. You only need to provide the path to your audio files. The feature extraction models will be downloaded automatically, or you can provide your own with the `hubert` and `rmvpe` arguments.

```py
from datasets import load_dataset
from zerorvc import prepare, RVCTrainer

dataset = load_dataset("my-audio-dataset")
dataset = prepare(dataset)

trainer = RVCTrainer(
    "my-rvc-model",
    dataset_train=dataset["train"],
    dataset_test=dataset["test"],
)
trainer.train(epochs=100, batch_size=8, upload="someone/rvc-test-1")
```

## Inference

ZeroRVC provides an easy API to convert your voice with the trained model.

```py
from zerorvc import RVC
import soundfile as sf

rvc = RVC.from_pretrained("someone/rvc-test-1")
samples = rvc.convert("test.mp3")
sf.write("output.wav", samples, rvc.sr)
```
