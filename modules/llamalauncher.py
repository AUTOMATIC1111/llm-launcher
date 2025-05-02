import math
import re
import shlex
import sys
import time

import gradio as gr
import subprocess
import os
import signal

import select
import gguf_parser

from modules import shared, errors


def nvidia_smi():
    return '```\n' + subprocess.check_output('nvidia-smi; echo; free -h; echo; df -h -T -x tmpfs', shell=True).decode('utf8', errors='ignore') + '\n```'


class LlamaServerLauncher:
    def __init__(self):
        self.server_process = None
        self.server_status = ''
        self.model_name = None
        self.model_path = None
        self.model_arch = None
        self.model_chat_template = None
        self.model_tensor_info = None
        self.model_size = None
        self.model_param_count = None
        self.build_info = None
        self.startup_log = ''

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
        self.model_chat_template = "```\n" + parser.metadata.get('tokenizer.chat_template', '-') + "```\n"
        self.model_tensor_info = "| name | type | size |\n|---|---|---|\n" + "\n".join(tensor_info(x) for x in parser.tensors_info)
        self.model_size = os.path.getsize(self.model_path)
        self.model_param_count = sum(math.prod(x.get('dimensions', [0])) for x in parser.tensors_info)

    def stop_server(self):
        if self.server_process and self.server_process.poll() is None:
            os.kill(self.server_process.pid, signal.SIGKILL)
            self.server_process.wait()
            self.server_process = None

    def start_server(self):
        if not shared.opts.llamacpp_model:
            self.server_status = 'Model not selected.'
            yield self.server_status
            return

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

            cmd = [shared.opts.llamacpp_exe, "-m", self.model_path] + shlex.split(shared.opts.llamacpp_cmdline.strip())
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
            )

            fd = self.server_process.stdout.fileno()
            ready = False
            start = time.time()

            self.server_status = 'Waiting for model to load...'
            yield self.server_status

            while True:
                rlist, _, _ = select.select([fd], [], [], 0.1)

                if rlist:
                    start = time.time()
                    data = os.read(fd, 1024).decode(errors="ignore")
                    self.startup_log += data

                    print(data, end='')
                    sys.stdout.flush()

                    if "starting the main loop" in self.startup_log:
                        ready = True
                        break

                if self.server_process.poll() is not None:
                    break

                if time.time() - start > 20:
                    self.startup_log += "\nTimed out."
                    break

            if not ready:
                self.server_status = f"❌ Failed to start. See log for more."
                yield self.server_status
                return

            m = re.search('build: ([^ ]+) ([^\n]+)', self.startup_log)
            self.build_info = f'**{m.group(1)}** *{m.group(2)}*' if m else '*unknown*'

            self.server_status = f"✅ Ready!"
            yield self.server_status
        except Exception as e:
            errors.display(e, full_traceback=True)
            self.server_status = f'❌ {e}. See log for more.'
            yield self.server_status

    def create_ui(self, settings_ui):
        with gr.Blocks(css_paths=['style.css'], title="Llama.cpp launcher") as demo:

            with gr.Tabs() as tabs:
                with gr.Tab("Llama.cpp"):
                    with gr.Row():
                        with gr.Column(scale=6):
                            model_info = gr.Markdown(value='', show_label=False, elem_classes=['model-info'])
                        with gr.Column(scale=1, min_width=40):
                            restart = gr.Button("Restart")

                    with gr.Accordion("Startup log", open=False):
                        startup_log = gr.Markdown(value='', show_label=False)

                    with gr.Accordion("Chat template", open=False):
                        chat_template = gr.Markdown(value='', show_label=False)

                    with gr.Accordion("Tensors", open=False):
                        tensor_info = gr.Markdown(value='', show_label=False)

                    status = gr.Markdown(value=lambda: self.server_status, show_label=False)

                with gr.Tab("Settings"):
                    settings_ui.create_ui(demo)

                with gr.Tab("System"):
                    refresh_system = gr.Button("Refresh")
                    nvidia_smi_view = gr.Markdown()

            init_fields = dict(
                fn=lambda: [self.model_info(), '```\n' + self.startup_log + '\n```', self.model_chat_template, self.model_tensor_info],
                outputs=[model_info, startup_log, chat_template, tensor_info]
            )

            demo.load(**init_fields)
            restart.click(fn=self.start_server, outputs=[status]).then(**init_fields)

            refresh_system.click(fn=nvidia_smi, outputs=[nvidia_smi_view])
            demo.load(fn=nvidia_smi, outputs=[nvidia_smi_view])

        return demo
