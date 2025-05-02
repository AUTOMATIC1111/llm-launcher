import shlex

import gradio as gr
import subprocess
import os
import signal

from modules import settings, shared, shared_options, errors


class LlamaServerLauncher:

    def __init__(self):
        self.server_process = None
        if shared.opts.llamacpp_model:
            self.start_server()

    def stop_server(self):
        if self.server_process and self.server_process.poll() is None:
            os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
            self.server_process.wait()
            self.server_process = None

    def start_server(self):
        try:
            yield 'Stopping server...'
            self.stop_server()

            yield 'Starting server...'
            model_path = os.path.join(shared.opts.llamacpp_model_dir, shared.opts.llamacpp_model or '')
            cmd = [shared.opts.llamacpp_exe, "-m", model_path] + shlex.split(shared.opts.llamacpp_cmdline.strip())
            self.server_process = subprocess.Popen(cmd)
            yield f"✅ Started!"
        except Exception as e:
            errors.display(e, full_traceback=True)
            yield str(e)

    def create_ui(self, settings_ui):
        with gr.Blocks(css_paths=['style.css']) as demo:
            with gr.Tabs() as tabs:
                with gr.Tab("Llama.cpp"):
                    gr.Markdown("# LLaMA.cpp Server Launcher")

                    restart = gr.Button("Restart server")
                    status = gr.Markdown(show_label=False)

                with gr.Tab("Settings"):
                    settings_ui.create_ui(demo)

            restart.click(fn=self.start_server, outputs=[status])

        return demo


def main():
    shared.opts = settings.Settings(shared_options.temlates)
    settings_ui = settings.SettingsUi(shared.opts, shared.config_filename)

    launcher = LlamaServerLauncher()
    ui = launcher.create_ui(settings_ui)
    ui.queue().launch()


if __name__ == "__main__":
    main()

