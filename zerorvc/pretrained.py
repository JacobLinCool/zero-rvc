from typing import Tuple
from huggingface_hub import hf_hub_download


def pretrained_checkpoints() -> Tuple[str, str]:
    """
    The pretrained checkpoints from the Hugging Face Hub.

    Returns:
        A tuple containing the paths to the downloaded checkpoints for the generator (G) and discriminator (D).
    """
    G = hf_hub_download("lj1995/VoiceConversionWebUI", "pretrained_v2/f0G48k.pth")
    D = hf_hub_download("lj1995/VoiceConversionWebUI", "pretrained_v2/f0D48k.pth")
    return G, D
