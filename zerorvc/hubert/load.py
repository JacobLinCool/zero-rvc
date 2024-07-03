import torch
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.models.hubert import HubertModel
from huggingface_hub import hf_hub_download
from ..auto_loader import auto_loaded_model


def load_hubert(
    hubert: str | HubertModel | None = None,
    device: torch.device = torch.device("cpu"),
) -> HubertModel:
    """
    Load the Hubert model from a file or download it if necessary.
    If a loaded model is provided, it will be returned as is.

    Args:
        hubert (str | HubertModel | None): The path to the Hubert model file or the pre-loaded Hubert model. If None, the default model will be downloaded.
        device (torch.device): The device to load the model on.

    Returns:
        HubertModel: The loaded Hubert model.

    Raises:
        If the model file does not exist.
    """
    if isinstance(hubert, HubertModel):
        return hubert.to(device)
    if isinstance(hubert, str):
        models, _, _ = load_model_ensemble_and_task([hubert])
        model = models[0].to(device)
        return model
    if "hubert" not in auto_loaded_model:
        hubert = hf_hub_download("lj1995/VoiceConversionWebUI", "hubert_base.pt")
        models, _, _ = load_model_ensemble_and_task([hubert])
        model = models[0].to(device)
        auto_loaded_model["hubert"] = model
    return auto_loaded_model["hubert"]
