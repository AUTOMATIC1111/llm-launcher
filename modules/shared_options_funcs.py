import os

from modules import shared


def list_llamacpp_models():
    if not os.path.exists(shared.opts.llamacpp_model_dir):
        return []

    return [f for f in os.listdir(shared.opts.llamacpp_model_dir) if f.endswith(".gguf")]
