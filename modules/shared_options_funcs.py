import os
import re

from modules import shared


def is_multipart_extra(x):
    m = re.search(r"(\d+)-of-(\d+)", x)
    if not m:
        return False

    return int(m.group(1)) > 1


def list_llamacpp_models():
    if not os.path.exists(shared.opts.llamacpp_model_dir):
        return []

    models = [f for f in os.listdir(shared.opts.llamacpp_model_dir) if f.endswith(".gguf")]
    models = [x for x in models if not is_multipart_extra(x)]
    models = sorted(models)

    return models
