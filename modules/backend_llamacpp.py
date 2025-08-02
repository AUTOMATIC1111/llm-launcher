import html
import math
import os
import re
import shlex

from modules import backend, shared, output_reader_llamacpp, utils

import gguf_parser


class BackendLlamacpp(backend.BackendBase):
    backend_type = 'llama.cpp'

    def cmd(self):
        return self.prepare_commandline_options()

    def create_server_reader(self):
        return output_reader_llamacpp.ReaderLlamacpp(self.server_process.stdout)

    def detect_started_line(self, line):
        if "starting the main loop" not in line:
            return False

        self.access_url = utils.extract_url(line)

        return True

    def process_startup_log(self):
        m = re.search(r'build: ([^ ]+) (\([^)]+\))', self.startup_log)
        self.build_info = f'llama.cpp<br /><b>{html.escape(m.group(1))}</b><br /><em>{m.group(2)}</em>' if m else '<em>unknown<em>'

    def read_model_info(self):
        parser = gguf_parser.GGUFParser(self.model.fullpath)
        parser.parse()

        def tensor_info(x):
            cells = [
                x.get('name', ''),
                parser.TENSOR_TYPES.get(x.get('type', -1), 'UNKNOWN').replace("GGML_TYPE_", ''),
                x.get('dimensions', ''),
            ]

            return '|' + '|'.join(str(x) for x in cells) + '|'

        self.model_arch = parser.metadata.get('general.architecture', '*unknown*')
        self.model_chat_template = parser.metadata.get('tokenizer.chat_template', '')
        self.model_tensor_info = "| name | type | size |\n|---|---|---|\n" + "\n".join(tensor_info(x) for x in parser.tensors_info)
        self.model_size = os.path.getsize(self.model.fullpath)
        self.model_param_count = sum(math.prod(x.get('dimensions', [0])) for x in parser.tensors_info)

        tokens = parser.metadata.get('tokenizer.ggml.tokens', [])

        def find_token(token_id):
            return "" if token_id < 0 or token_id >= len(tokens) else tokens[token_id]

        self.model_chat_template_vars = {
            'messages': self.sample_messages(),
            "bos_token": find_token(parser.metadata.get('tokenizer.ggml.bos_token_id', -1)),
            "eos_token": find_token(parser.metadata.get('tokenizer.ggml.eos_token_id', -1)),
            "pad_token": find_token(parser.metadata.get('tokenizer.ggml.unknown_token_id', -1)),
            "unk_token": find_token(parser.metadata.get('tokenizer.ggml.padding_token_id', -1)),
        }

    def prepare_commandline_options(self):
        model_name = self.model.path
        model_path = os.path.join(self.model.model_dir, self.model.path)
        model_alias = os.path.splitext(os.path.basename(model_path))[0]

        permodel_opts_all = [x.partition(':') for x in shared.opts.llamacpp_cmdline_permodel.split('\n')]
        permodel_opts = next((opts for model, _, opts in permodel_opts_all if model and model.lower() in model_name.lower()), '')

        cmd = [shared.opts.llamacpp_exe, "-m", model_path, "--alias", model_alias] + shlex.split(permodel_opts.strip()) + shlex.split(shared.opts.llamacpp_cmdline.strip())

        if shared.opts.llamacpp_port:
            cmd += ["--port", shared.opts.llamacpp_port]

        if shared.opts.llamacpp_host:
            cmd += ["--host", shared.opts.llamacpp_host]

        return cmd
