import torch
from .rmvpe import RMVPE


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
        model = RMVPE.from_pretrained(rmvpe).to(device)
        return model
    return RMVPE.from_pretrained("safe-models/RMVPE").to(device)
