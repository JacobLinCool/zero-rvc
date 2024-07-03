import gradio as gr


class TutotialTab:
    def __init__(self):
        pass

    def ui(self):
        gr.Markdown(
            """
            # Welcome to ZeroRVC!

            > If you are more satisfied with Python code, you can also [use the Python API to run ZeroRVC](https://pypi.org/project/zerorvc/).

            ZeroRVC is a toolkit for training and inference of retrieval-based voice conversion models.

            By leveraging the power of Hugging Face ZeroGPU, you can train your model in minutes without setting up the environment.

            ## How to Use

            There are 3 main steps to use ZeroRVC:

            - **Make Dataset**: Prepare your dataset for training. You can upload a zip file containing audio files.
            - **Model Training**: Train your model using the prepared dataset.
            - **Model Inference**: Try your model.
            """
        )

    def build(self):
        pass
