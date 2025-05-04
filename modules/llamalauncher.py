import html
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
        self.server_reader: llamacpp_output_reader.Reader = None
        self.busy = 0

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
                {"role": "system", "content": "You are a helpful assistant"},
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
            try:
                rendered = templating.render(self.model_chat_template, template_vars)
            except Exception:
                template_vars['messages'].pop(0)
                rendered = templating.render(self.model_chat_template, template_vars)

            self.model_chat_template_example = rendered
            self.model_chat_template_markdown = "Example:\n```\n" + str(rendered) + "\n```\n\nFull chat template:\n```\n" + str(self.model_chat_template) + "\n```"
        except Exception as e:
            self.model_chat_template_markdown = f"Error rendering example: {e}\n\nFull chat template:\n```\n" + str(self.model_chat_template) + "\n```"

    def stop_server(self, skip_check=False):
        if self.busy and not skip_check:
            gr.Warning('Already working!')
            return

        self.busy += 1

        if self.server_process and self.server_process.poll() is None:
            self.server_status = 'Stopping server...'
            yield self.server_status

            os.kill(self.server_process.pid, signal.SIGTERM if os.name == 'nt' else signal.SIGKILL)
            self.server_process.wait()
            self.server_process = None

        self.server_status = '✋🏻 Stopped by user.'
        yield self.server_status

        self.busy -= 1

    def start_server(self):
        if not shared.opts.llamacpp_model:
            self.server_status = 'Model not selected.'
            yield self.server_status
            return

        if self.busy:
            gr.Warning('Already working!')
            return

        self.busy += 1
        self.startup_log = ''
        self.model_name = shared.opts.llamacpp_model
        self.model_path = os.path.join(shared.opts.llamacpp_model_dir, self.model_name)

        try:
            self.read_model_info()

            for _ in self.stop_server(skip_check=True):
                pass

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
                errors='ignore',
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
                        self.server_status = "❌ Server exited before it was ready."
                        yield self.server_status
                        self.busy -= 1
                        return

                if time.time() - start > shared.opts.llamacpp_startup_timeout:
                    self.server_status = "❌ Timed out waiting for output from server."
                    yield self.server_status
                    self.startup_log += "\nTimed out."
                    self.busy -= 1
                    break

            if not ready:
                self.server_status = "❌ Failed to start."
                yield self.server_status
                self.busy -= 1
                return

            m = re.search(r'build: ([^ ]+) (\([^)]+\))', self.startup_log)
            self.build_info = f'<b>{html.escape(m.group(1))}</b><br /><em>{m.group(2)}</em>' if m else '<em>unknown<em>'
        except Exception as e:
            errors.display(e, full_traceback=True)
            self.server_status = f'❌ {e}. See log for more.'
            self.busy -= 1
            yield self.server_status

        self.server_status = "✅ Ready!"
        self.busy -= 1
        yield self.server_status

    def load_status(self):
        status = None

        while self.busy:
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

        if self.busy:
            loaded_model = "<em>Loading...</em>"
            build_info = '<em>Loading...</em>'
        else:
            loaded_model = f"<b>{html.escape(self.model_name)}</b> <br />({html.escape(self.model_arch)}, {round((self.model_param_count or 0) / 1000000000)}B, {(self.model_size or 0) / 1024 / 1024 / 1024:.1f} GB)"
            build_info = self.build_info

        v = f"""
<table class='stats'>
    <thead>
    <tr>
        <th><span>Model</span></th>
        <th><span>Version</span></th>
        <th><span>Completed</span></th>
        <th><span>Generation</span></th>
        <th><span>Processing</span></th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td class='stat-model'>
            <span class='textstat'>{loaded_model}</span>
        </td>
        <td class='stat-build'>
            <span class='textstat'>{build_info}</span>
        </td>
        <td class='stat-requests'>
            <span class='bigstat'>{len(self.server_reader.requests)}</span>
            <span class='bigstat-subtitle'>Requests</span>
        </td>
        <td class='stat-generated'>
            <span class='bigstat'>{round(tokens_generated / time_generating, 1) if time_generating else '0'}</span>
            <span class='bigstat-subtitle'>Tokens/sec</span>
            <span class='ministat'>Total: {tokens_generated}</span>
        </td>
        <td class='stat-processed'>
            <span class='bigstat'>{round(tokens_processed / time_processing, 1) if time_processing else '0'}</span>
            <span class='bigstat-subtitle'>Tokens/sec</span>
            <span class='ministat'>Total: {tokens_processed}</span>
        </td>
    </tr>
    </tbody>
</table>
""".strip()

        return v if v != current_value else gr.update()

    def create_ui(self, settings_ui):
        with gr.Blocks(css_paths=['assets/style.css'], title="Llama.cpp launcher") as demo:

            with gr.Tabs():
                with gr.Tab("Llama.cpp"):

                    with gr.Row():
                        with gr.Column(scale=6):
                            settings_ui.render('llamacpp_model')
                        with gr.Column(scale=1, min_width=40):
                            stop = gr.Button("Stop", elem_classes=['aligned-to-label'])
                        with gr.Column(scale=1, min_width=50):
                            restart = gr.Button("Start", elem_classes=['aligned-to-label'], variant="primary")

                    stats = gr.HTML(value='', elem_classes=['no-flicker', 'compact'])
                    status = gr.Markdown(value='*Loading...*', elem_classes=['status'])

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

            init_fields = dict(
                fn=lambda: ['```\n' + self.startup_log + '\n```', self.model_chat_template_markdown, self.model_tensor_info, '```\n' + self.commandline + '\n```'],
                outputs=[startup_log, chat_template, tensor_info, commandline]
            )

            demo.load(fn=self.load_status, outputs=[status]).then(**init_fields)
            demo.load(**init_fields)
            restart.click(fn=self.start_server, outputs=[status]).then(**init_fields)

            stop.click(fn=self.stop_server, outputs=[status]).then(**init_fields)

            refresh_system.click(fn=nvidia_smi, outputs=[nvidia_smi_view])
            demo.load(fn=nvidia_smi, outputs=[nvidia_smi_view])

            gr.Timer(1).tick(fn=self.stats, inputs=[stats], outputs=[stats],  show_progress="hidden")
            demo.load(fn=self.stats, outputs=[stats],  show_progress="hidden")

        return demo
