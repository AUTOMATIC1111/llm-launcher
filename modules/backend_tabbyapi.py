import json
import os
import re
import shlex

import safetensors

from modules import backend, shared, output_reader_tabbyapi, utils


class BackendTabbyapi(backend.BackendBase):
    backend_type = 'tabbyapi'

    def cmd(self):
        self.chdir = shared.opts.tabbyapi_path
        return self.prepare_commandline_options()

    def create_server_reader(self):
        return output_reader_tabbyapi.ReaderTabbyapi(self.server_process.stdout)

    def detect_started_line(self, line):
        if "Uvicorn running on" not in line:
            return False

        self.access_url = utils.extract_url(line)
        return True

    def process_startup_log(self):
        h = utils.get_hash(self.chdir)
        m = re.search(r'(\w+) version: (\S+)', self.startup_log)
        self.build_info = "<br />".join([
            "tabbyapi",
            *([f'<b>{h}</b>'] if h else []),
            *([f'<em>{m.group(1)}: {m.group(2)}</em>'] if m else []),
        ])

    def read_model_info(self):
        model_dir = self.model.fullpath

        def load_json_config(fn):
            path = os.path.join(model_dir, fn)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return {}

        config = load_json_config('config.json')
        tokenizer_config = load_json_config('tokenizer_config.json')

        self.model_arch = config.get('model_type', '*unknown*')

        chat_template = tokenizer_config.get('chat_template')
        if not chat_template:
            chat_template = config.get('chat_template')
        self.model_chat_template = chat_template or ''  # Ensure it's a string

        tensors_info = []
        total_size = 0

        for root, _, files in os.walk(model_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                total_size += os.path.getsize(filepath)

                if filename.endswith('.safetensors'):
                    try:
                        with safetensors.safe_open(filepath, framework="numpy", device="cpu") as f:
                            for key in f.keys():
                                tensor_metadata = f.get_tensor(key)
                                dims = list(tensor_metadata.shape)
                                dtype_str = str(tensor_metadata.dtype).replace('torch.', '')

                                tensors_info.append({
                                    'name': key,
                                    'type': dtype_str,
                                    'dimensions': dims
                                })
                    except Exception as e:
                        print(f"Warning: Could not process {filename}: {e}")

        # doesn't seem to be an easy way to calculate the total number of parameters
        self.model_size = total_size

        def tensor_info_formatter(x):
            cells = [
                x.get('name', ''),
                x.get('type', 'UNKNOWN'),
                x.get('dimensions', ''),
            ]
            return '|' + '|'.join(str(cell) for cell in cells) + '|'

        self.model_tensor_info = "| name | type | size |\n|---|---|---|\n" + "\n".join(tensor_info_formatter(x) for x in tensors_info)

        def get_special_token(cfg, key):
            token = cfg.get(key)
            if isinstance(token, str):
                return token

            if isinstance(token, dict) and 'content' in token:
                return token['content']

            return ""

        self.model_chat_template_vars = {
            'messages': self.sample_messages(),
            'bos_token': get_special_token(tokenizer_config, 'bos_token'),
            'eos_token': get_special_token(tokenizer_config, 'eos_token'),
            'pad_token': get_special_token(tokenizer_config, 'pad_token'),
            'unk_token': get_special_token(tokenizer_config, 'unk_token'),
        }

    def prepare_commandline_options(self):
        model_name = self.model.path
        model_path = os.path.join(self.model.model_dir, self.model.path)
        model_alias = os.path.splitext(os.path.basename(model_path))[0]

        permodel_opts_all = [x.partition(':') for x in shared.opts.tabbyapi_cmdline_permodel.split('\n')]
        permodel_opts = next((opts for model, _, opts in permodel_opts_all if model and model.lower() in model_name.lower()), '')

        python_paths = [
            os.path.join(shared.opts.tabbyapi_path, "venv", "scripts", "python.exe"),
            os.path.join(shared.opts.tabbyapi_path, "venv", "bin", "python"),
        ]

        python_path = next((x for x in python_paths if os.path.exists(x)), None)
        if python_path is None:
            raise Exception(f"Couldn't find python venv in {shared.opts.tabbyapi_path}")

        cmd = [
            python_path,
            os.path.join(shared.opts.tabbyapi_path, "start.py"),
            "--model-name", model_path,
            "--dummy-model-names", model_alias
        ] + shlex.split(permodel_opts.strip()) + shlex.split(shared.opts.tabbyapi_cmdline.strip())

        if shared.opts.tabbyapi_port:
            cmd += ["--port", shared.opts.tabbyapi_port]

        if shared.opts.tabbyapi_host:
            cmd += ["--host", shared.opts.tabbyapi_host]

        return cmd
