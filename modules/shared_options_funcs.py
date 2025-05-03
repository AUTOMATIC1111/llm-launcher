import os

from modules import shared


def list_llamacpp_models():
    if not os.path.exists(shared.opts.llamacpp_model_dir):
        return []

    models = [f for f in os.listdir(shared.opts.llamacpp_model_dir) if f.endswith(".gguf")]
    models = [x for x in models if not ]
    models = sorted(models)

    return models
