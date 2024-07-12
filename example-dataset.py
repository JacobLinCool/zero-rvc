import os
from zerorvc import prepare

HF_TOKEN = os.environ.get("HF_TOKEN")

dataset = prepare("./my-voices")
print(dataset)

dataset.push_to_hub("my-rvc-dataset", token=HF_TOKEN, private=True)
