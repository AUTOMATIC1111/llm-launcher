import math
import queue
import re
import shlex
import time

import gradio as gr
import subprocess
import os
import signal

import gguf_parser

from modules import shared, errors, llamacpp_output_reader, templating


def nvidia_smi():
    if os.name == 'nt':
        command = 'nvidia-smi'
    else:
        command = 'nvidia-smi; echo; free -h; echo; df -h -T -x tmpfs'

    return '```\n' + subprocess.check_output(command, shell=True).decode('utf8', errors='ignore') + '\n```'


class LlamaServerLauncher:
    def __init__(self):
        self.server_process = None
        self.server_loading = False
        self.server_reader: llamacpp_output_reader.Reader = None

        self.server_status = ''
        self.model_name = None
        self.model_path = None
        self.model_arch = None
        self.model_chat_template = None
        self.model_chat_template_example = None
        self.model_chat_template_markdown = None
        self.model_tensor_info = None
        self.model_size = None
        self.model_param_count = None
        self.build_info = None
        self.startup_log = ''
        self.commandline = ''

        if shared.opts.llamacpp_model:
            self.start_server()

    def model_info(self):
        return f"""
Model: **{self.model_name}** ({self.model_arch}, {(self.model_param_count or 0) // 1000000000}B, {(self.model_size or 0)/1024/1024/1024:.1f}GB)

Server: {self.build_info}
        """.strip()

    def read_model_info(self):
        parser = gguf_parser.GGUFParser(self.model_path)
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
        self.model_size = os.path.getsize(self.model_path)
        self.model_param_count = sum(math.prod(x.get('dimensions', [0])) for x in parser.tensors_info)

        self.write_down_template(parser)

    def prepare_commandline_options(self):
        permodel_opts_all = [x.partition(':') for x in shared.opts.llamacpp_cmdline_permodel.split('\n')]
        permodel_opts = next((opts for model, _, opts in permodel_opts_all if model and model.lower() in self.model_name.lower()), '')

        cmd = [shared.opts.llamacpp_exe, "-m", self.model_path] + shlex.split(permodel_opts.strip()) + shlex.split(shared.opts.llamacpp_cmdline.strip())

        if shared.opts.llamacpp_port:
            cmd += ["--port", shared.opts.llamacpp_port]

        if shared.opts.llamacpp_host:
            cmd += ["--host", shared.opts.llamacpp_host]

        return cmd

    def write_down_template(self, parser):
        self.model_chat_template_example = ''

        if not self.model_chat_template:
            self.model_chat_template_markdown = "*Chat ttemplate missing!*"
            return

        tokens = parser.metadata.get('tokenizer.ggml.tokens', [])

        def find_token(token_id):
            return "" if token_id < 0 or token_id >= len(tokens) else tokens[token_id]

        template_vars = {
            'messages': [
                {"role": "user", "content": "What is 1+1?"},
                {"role": "assistant", "content": "It's 2."},
                {"role": "user", "content": "Thank you."},
                {"role": "assistant", "content": "No problem."},
            ],
            "bos_token": find_token(parser.metadata.get('tokenizer.ggml.bos_token_id', -1)),
            "eos_token": find_token(parser.metadata.get('tokenizer.ggml.eos_token_id', -1)),
            "pad_token": find_token(parser.metadata.get('tokenizer.ggml.unknown_token_id', -1)),
            "unk_token": find_token(parser.metadata.get('tokenizer.ggml.padding_token_id', -1)),
        }

        try:
            rendered = templating.render(self.model_chat_template, template_vars)
            self.model_chat_template_example = rendered
            self.model_chat_template_markdown = "Example:\n```\n" + str(rendered) + "\n```\n\nFull chat template:\n```\n" + str(self.model_chat_template) + "\n```"
        except Exception as e:
            self.model_chat_template_markdown = f"Error rendering example: {e}\n\nFull chat template:\n```\n" + str(self.model_chat_template) + "\n```"

    def stop_server(self):
        if self.server_process and self.server_process.poll() is None:
            os.kill(self.server_process.pid, signal.SIGTERM if os.name == 'nt' else signal.SIGKILL)
            self.server_process.wait()
            self.server_process = None

    def start_server(self):
        if not shared.opts.llamacpp_model:
            self.server_status = 'Model not selected.'
            yield self.server_status
            return

        self.server_loading = True
        self.startup_log = ''
        self.model_name = shared.opts.llamacpp_model
        self.model_path = os.path.join(shared.opts.llamacpp_model_dir, self.model_name)

        try:
            self.read_model_info()

            self.server_status = 'Stopping server...'
            yield self.server_status
            self.stop_server()

            self.server_status = 'Starting server...'
            yield self.server_status

            cmd = self.prepare_commandline_options()
            self.commandline = shlex.join(cmd)

            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
            )

            self.server_reader = llamacpp_output_reader.Reader(self.server_process.stdout)

            ready = False
            start = time.time()

            self.server_status = 'Waiting for server to start...'
            yield self.server_status

            while True:
                try:
                    line = self.server_reader.queue.get(timeout=0.2)
                    if line:
                        self.startup_log += line
                        start = time.time()

                        if "starting the main loop" in line:
                            ready = True
                            break

                except queue.Empty:
                    if self.server_process.poll() is not None:
                        self.server_status = f"❌ Server exited before it was ready. See log for more."
                        yield self.server_status
                        self.server_loading = False
                        return

                if time.time() - start > 20:
                    self.startup_log += "\nTimed out."
                    break

            if not ready:
                self.server_status = f"❌ Failed to start. See log for more."
                yield self.server_status
                self.server_loading = False
                return

            m = re.search('build: ([^ ]+) ([^\n]+)', self.startup_log)
            self.build_info = f'**{m.group(1)}** *{m.group(2)}*' if m else '*unknown*'

            self.server_status = f"✅ Ready!"
            self.server_loading = False
            yield self.server_status
        except Exception as e:
            errors.display(e, full_traceback=True)
            self.server_status = f'❌ {e}. See log for more.'
            self.server_loading = False
            yield self.server_status

    def load_status(self):
        status = None

        while self.server_loading:
            if status != self.server_status:
                status = self.server_status
                yield self.server_status

            time.sleep(0.2)

        yield self.server_status

    def stats(self, current_value=None):
        tokens_generated = sum(x.tokens_generate for x in self.server_reader.requests)
        tokens_processed = sum(x.tokens_process for x in self.server_reader.requests)
        time_generating = sum(x.time_generate for x in self.server_reader.requests) / 1000
        time_processing = sum(x.time_process for x in self.server_reader.requests) / 1000

        v = f"""
        *In last {self.server_reader.keep_requests_duration // 60} minutes:*
        > Completed requests: **{len(self.server_reader.requests)}**
        >
        > Tokens generated: **{tokens_generated}**
        >
        > Tokens processed: **{tokens_processed}**
        >
        > Avg generation rate: **{round(tokens_generated / time_generating, 1) if time_generating else 'none'}** tokens/sec
        >
        > Avg processing rate: **{round(tokens_processed / time_processing, 1) if time_processing else 'none'}** tokens/sec
        """.strip()
        return v if v != current_value else gr.update()

    def create_ui(self, settings_ui):
        with gr.Blocks(css_paths=['style.css'], title="Llama.cpp launcher") as demo:

            with gr.Tabs() as tabs:
                with gr.Tab("Llama.cpp"):
                    with gr.Row():
                        with gr.Column(scale=6):
                            model_info = gr.Markdown(value='', elem_classes=['compact'])
                        with gr.Column(scale=1, min_width=40):
                            restart = gr.Button("Restart")

                    stats = gr.Markdown(value='', elem_classes=['no-flicker', 'compact'])

                    status = gr.Markdown(value='')

                with gr.Tab("System"):
                    refresh_system = gr.Button("Refresh")
                    nvidia_smi_view = gr.Markdown()

                with gr.Tab("Info"):
                    with gr.Accordion("Full command line", open=False):
                        commandline = gr.Markdown(value='')

                    with gr.Accordion("Startup log", open=False):
                        startup_log = gr.Markdown(value='')

                    with gr.Accordion("Chat template", open=False):
                        chat_template = gr.Markdown(value='')

                    with gr.Accordion("Tensors", open=False):
                        tensor_info = gr.Markdown(value='')

                with gr.Tab("Settings"):
                    settings_ui.create_ui(demo)

            demo.load(fn=self.load_status, outputs=[status])

            init_fields = dict(
                fn=lambda: [self.model_info(), '```\n' + self.startup_log + '\n```', self.model_chat_template_markdown, self.model_tensor_info, '```\n' + self.commandline + '\n```'],
                outputs=[model_info, startup_log, chat_template, tensor_info, commandline]
            )

            demo.load(**init_fields)
            restart.click(fn=self.start_server, outputs=[status]).then(**init_fields)

            refresh_system.click(fn=nvidia_smi, outputs=[nvidia_smi_view])
            demo.load(fn=nvidia_smi, outputs=[nvidia_smi_view])

            gr.Timer(1).tick(fn=self.stats, inputs=[stats], outputs=[stats],  show_progress="hidden")
            demo.load(fn=self.stats, outputs=[stats],  show_progress="hidden")

        return demo
