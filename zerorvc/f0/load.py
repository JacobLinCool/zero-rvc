import torch
from huggingface_hub import hf_hub_download
from .rmvpe import RMVPE
from ..auto_loader import auto_loaded_model


def load_rmvpe(
    rmvpe: str | RMVPE | None = None, device: torch.device = torch.device("cpu")
) -> RMVPE:
    """
    Load the RMVPE model from a file or download it if necessary.
    If a loaded model is provided, it will be returned as is.

    Args:
        rmvpe (str | RMVPE | None): The path to the RMVPE model file or the pre-loaded RMVPE model. If None, the default model will be downloaded.
        device (torch.device): The device to load the model on.

    Returns:
        RMVPE: The loaded RMVPE model.

    Raises:
        If the model file does not exist.
    """
    if isinstance(rmvpe, RMVPE):
        return rmvpe.to(device)
    if isinstance(rmvpe, str):
        model = RMVPE(4, 1, (2, 2))
        model.load_state_dict(torch.load(rmvpe, map_location=device))
        model.to(device)
        return model
    if "rmvpe" not in auto_loaded_model:
        rmvpe = hf_hub_download("lj1995/VoiceConversionWebUI", "rmvpe.pt")
        model = RMVPE(4, 1, (2, 2))
        model.load_state_dict(torch.load(rmvpe, map_location="cpu"))
        model.to(device)
        auto_loaded_model["rmvpe"] = model
    return auto_loaded_model["rmvpe"]
