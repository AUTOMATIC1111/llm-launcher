import html
import time

import gradio as gr
import subprocess
import os

from modules import shared, errors, ui_download, backend, models


def nvidia_smi():
    if os.name == 'nt':
        command = 'nvidia-smi'
    else:
        command = 'nvidia-smi; echo; free -h; echo; df -h -T -x tmpfs'

    return '```\n' + subprocess.check_output(command, shell=True).decode('utf8', errors='ignore') + '\n```'


class LlmLauncher:
    def __init__(self):
        self.server_status = "Not started"
        self.backend: backend.BackendBase = None

        self.downloader = ui_download.HuggingfaceDownloader()
        self.busy = 0

    def launch_at_startup(self):
        if shared.opts.model and shared.opts.run_at_startup:
            for _ in self.start_server():
                pass

    def status(self):
        bknd = self.backend

        if bknd is not None and bknd.status_message is not None:
            return bknd.status_message

        return self.server_status

    def runbusy(self, func, skip_check=False):
        if self.busy and not skip_check:
            gr.Warning('Already working!')
            return

        self.busy += 1

        try:
            yield from func()
        except Exception as e:
            errors.display(e, full_traceback=True)
            yield f'❌ {e}'

        self.busy -= 1

    def stop_server(self):
        bknd = self.backend

        if bknd is not None:
            self.server_status = 'Stopping...'
            yield self.server_status

            bknd.stop_server()
            self.backend = None

        self.server_status = '✋🏻 Stopped by user.'
        yield self.server_status

    def stop_server_gradio(self):
        yield from self.runbusy(self.stop_server)

    def start_server(self):
        model_label = shared.opts.model
        model_info = models.models.get(model_label)
        if model_info is None:
            self.server_status = f'❌ Model not found: {model_label}'
            yield self.server_status
            return

        yield from self.stop_server()

        bknd = model_info.backend_type()
        self.backend = bknd
        bknd.model = model_info

        self.server_status = 'Reading model info...'
        yield self.server_status
        try:
            bknd.read_model_info()
        except Exception as e:
            errors.display(e, full_traceback=True)

        self.server_status = 'Writing down template...'
        yield self.server_status
        try:
            bknd.write_down_template()
        except Exception as e:
            errors.display(e, full_traceback=True)

        bknd.run()
        yield bknd.status_message

    def start_server_gradio(self):
        if not shared.opts.model:
            self.server_status = 'Model not selected.'
            yield self.server_status
            return

        yield from self.runbusy(self.start_server)

    def load_status(self):
        status = None

        while self.busy:
            if status != self.status():
                status = self.status()
                yield status

            time.sleep(0.2)

        yield self.status()

    def stats(self, current_value):
        bknd = self.backend

        if not bknd or not bknd.server_reader:
            return "", self.status(), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

        reqs_generating = [x for x in bknd.server_reader.requests if x.time_generate is not None]
        reqs_processing = [x for x in bknd.server_reader.requests if x.time_process is not None]

        tokens_generated = sum(x.tokens_generate for x in reqs_generating)
        tokens_processed = sum(x.tokens_process for x in reqs_processing)
        time_generating = sum(x.time_generate for x in reqs_generating) / 1000
        time_processing = sum(x.time_process for x in reqs_processing) / 1000

        if self.busy:
            loaded_model = "<em>Loading...</em>"
            build_info = '<em>Loading...</em>'
        else:
            total_params = f"{round(bknd.model_param_count / 1000000000)}B, " if bknd.model_param_count is not None else ""
            loaded_model = f"<b>{html.escape(bknd.model.path or '')}</b> <br />({html.escape(bknd.model_arch or '')}, {total_params}{(bknd.model_size or 0) / 1024 / 1024 / 1024:.1f} GB)"
            build_info = bknd.build_info

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
            <span class='bigstat'>{len(bknd.server_reader.requests)}</span>
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

        is_running = not bknd.over
        return (v if v != current_value else gr.update()), self.status(), gr.update(visible=not is_running), gr.update(visible=is_running), gr.update(visible=is_running)

    def create_ui(self, settings_ui):
        js = """
        function(){
            const script = document.createElement('script');
            script.src = "/gradio_api/file=assets/script.js";
            (document.head || document.documentElement).appendChild(script);
        }
        """

        with gr.Blocks(css_paths=['assets/style.css'], title="Llama.cpp launcher", js=js) as demo:

            with gr.Tabs():
                with gr.Tab("Backend"):

                    with gr.Row():
                        with gr.Column(scale=6):
                            settings_ui.render('model')
                        with gr.Column(scale=2, min_width=100):
                            with gr.Row():
                                start = gr.Button("Start", elem_classes=['aligned-to-label'], variant="primary", min_width=40, visible=False)
                                restart = gr.Button("Restart", elem_classes=['aligned-to-label'], variant="primary", min_width=40, visible=False)
                                stop = gr.Button("Stop", elem_classes=['aligned-to-label'], min_width=40, visible=False)

                    status = gr.Markdown(value='*Loading...*', elem_classes=['status'])
                    stats = gr.HTML(value='', elem_classes=['no-flicker', 'compact'])

                with gr.Tab("Download"):
                    self.downloader.create_ui(demo)

                with gr.Tab("Info"):
                    with gr.Accordion("System", open=False):
                        refresh_system = gr.Button("Refresh")
                        nvidia_smi_view = gr.Markdown()

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

            def init_fields_func():
                bknd = self.backend
                if bknd is None:
                    return ["", "", "", ""]

                return [
                    f'```\n{bknd.startup_log}\n```',
                    bknd.model_chat_template_markdown,
                    bknd.model_tensor_info,
                    f'```\n{bknd.commandline}\n```'
                ]

            get_info = dict(fn=init_fields_func, outputs=[startup_log, chat_template, tensor_info, commandline], show_progress="hidden")
            get_stats = dict(fn=self.stats, inputs=[stats], outputs=[stats, status, start, stop, restart], show_progress="hidden")
            disable_buttons = dict(fn=lambda: [gr.update(interactive=False) for _ in range(3)], outputs=[start, stop, restart])
            enable_buttons = dict(fn=lambda: [gr.update(interactive=True) for _ in range(3)], outputs=[start, stop, restart])

            demo.load(api_name="get_info", **get_info)
            start.click(**disable_buttons).then(fn=self.start_server_gradio, outputs=[status], show_progress="hidden").then(**enable_buttons).then(**get_info)
            restart.click(**disable_buttons).then(fn=self.start_server_gradio, outputs=[status], show_progress="hidden").then(**enable_buttons).then(**get_info)
            stop.click(**disable_buttons).then(fn=self.stop_server_gradio, outputs=[status], show_progress="hidden").then(**enable_buttons).then(**get_info)

            refresh_system.click(fn=nvidia_smi, outputs=[nvidia_smi_view])
            demo.load(fn=nvidia_smi, outputs=[nvidia_smi_view])

            gr.Timer(1).tick(**get_stats)
            demo.load(api_name="get_stats", **get_stats)

        return demo
