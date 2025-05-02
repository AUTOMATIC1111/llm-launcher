import pty
import shlex
import sys
import time

import gradio as gr
import subprocess
import os
import signal

import select

from modules import settings, shared, shared_options, errors


def nvidia_smi():
    return '```\n' + subprocess.check_output(['nvidia-smi']).decode('utf8', errors='ignore') + '\n```'

class LlamaServerLauncher:
    def __init__(self):
        self.server_process = None
        if shared.opts.llamacpp_model:
            self.start_server()

    def stop_server(self):
        if self.server_process and self.server_process.poll() is None:
            os.kill(self.server_process.pid, signal.SIGKILL)
            self.server_process.wait()
            self.server_process = None

    def start_server(self):
        try:
            yield 'Stopping server...'
            self.stop_server()

            yield 'Starting server...'
            master_fd, slave_fd = pty.openpty()

            model_path = os.path.join(shared.opts.llamacpp_model_dir, shared.opts.llamacpp_model or '')
            cmd = [shared.opts.llamacpp_exe, "-m", model_path] + shlex.split(shared.opts.llamacpp_cmdline.strip())
            self.server_process = subprocess.Popen(
                cmd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=subprocess.STDOUT,
                close_fds=True
            )

            ready = False
            start = time.time()
            lines = ""

            yield 'Waiting for model to load...'
            while True:
                rlist, _, _ = select.select([master_fd], [], [], 0.1)

                if rlist:
                    start = time.time()
                    data = os.read(master_fd, 1024).decode(errors="ignore")
                    lines += data

                    print(data, end='')
                    sys.stdout.flush()

                    if "starting the main loop" in lines:
                        ready = True
                        break

                if self.server_process.poll() is not None:
                    break

                if time.time() - start > 20:
                    lines += "\nTimed out."
                    break

            if not ready:
                yield f"❌ Failed to start:\n```\n{''.join(lines)}\n```"
                return

            yield f"✅ Ready!"
        except Exception as e:
            errors.display(e, full_traceback=True)
            yield f'❌ {e}'

    def create_ui(self, settings_ui):
        with gr.Blocks(css_paths=['style.css']) as demo:
            with gr.Tabs() as tabs:
                with gr.Tab("Llama.cpp"):
                    restart = gr.Button("Restart")
                    status = gr.Markdown(show_label=False)

                with gr.Tab("Settings"):
                    settings_ui.create_ui(demo)

                with gr.Tab("System"):
                    refresh_system = gr.Button("Refresh")
                    nvidia_smi_view = gr.Markdown()

            restart.click(fn=self.start_server, outputs=[status])

            refresh_system.click(fn=nvidia_smi, outputs=[nvidia_smi_view])
            demo.load(fn=nvidia_smi, outputs=[nvidia_smi_view])

        return demo


def main():
    shared.opts = settings.Settings(shared_options.temlates)
    settings_ui = settings.SettingsUi(shared.opts, shared.config_filename)

    launcher = LlamaServerLauncher()
    ui = launcher.create_ui(settings_ui)

    for _ in launcher.start_server():
        pass

    ui.queue().launch()


if __name__ == "__main__":
    main()

