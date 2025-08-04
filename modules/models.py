import dataclasses
import os
import re
import shutil

from modules import shared, backend_llamacpp, backend_tabbyapi


@dataclasses.dataclass
class ModelInfo:
    path: str
    model_dir: str
    backend_type: type
    label: str = None
    fullpath: str = None

    def __post_init__(self):
        self.label = f"{self.path} [{self.backend_type.backend_type}]"
        self.fullpath = os.path.join(self.model_dir, self.path)


models: dict[str, ModelInfo] = {}


def is_multipart_extra(x):
    m = re.search(r"(\d+)-of-(\d+)", x)
    if not m:
        return False

    return int(m.group(1)) > 1


def list_models():
    if not os.path.exists(shared.opts.model_dir):
        return models

    models_list = []

    have_tabbyapi = shared.opts.tabbyapi_path and os.path.exists(shared.opts.tabbyapi_path)
    have_llamacpp = shared.opts.llamacpp_exe and shutil.which(shared.opts.llamacpp_exe)

    for root, dirs, files in os.walk(shared.opts.model_dir):
        for filename in files:
            if filename.endswith(".gguf") and have_llamacpp:
                if is_multipart_extra(filename):
                    continue

                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, shared.opts.model_dir)
                models_list.append(ModelInfo(relative_path, shared.opts.model_dir, backend_llamacpp.BackendLlamacpp))

        for dirname in dirs:
            if not have_tabbyapi:
                break

            config_fn = os.path.join(root, dirname, "config.json")
            tokenizer_config_fn = os.path.join(root, dirname, "tokenizer_config.json")

            if not os.path.exists(config_fn) or not os.path.exists(tokenizer_config_fn):
                continue

            full_path = os.path.join(root, dirname)
            relative_path = os.path.relpath(full_path, shared.opts.model_dir)
            models_list.append(ModelInfo(relative_path, shared.opts.model_dir, backend_tabbyapi.BackendTabbyapi))

    models_list.sort(key=lambda x: x.path)

    models.clear()
    models.update({x.label: x for x in models_list})

