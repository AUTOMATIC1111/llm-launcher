import os
import queue
import shlex
import signal
import subprocess
import threading
import time

from modules import templating, models, shared


class BackendBase:
    backend_type = 'none'

    def __init__(self):
        self.server_process = None
        self.server_reader = None
        self.chdir = None

        self.model: models.ModelInfo = None
        self.model_arch = None
        self.model_chat_template = None
        self.model_chat_template_vars = None

        self.model_chat_template_example = None
        self.model_chat_template_markdown = None

        self.model_tensor_info = None
        self.model_size = None
        self.model_param_count = None
        self.build_info = None
        self.access_url = None

        self.over = False
        self.ready = False
        self.server_thread = None
        self.status_message: str = None
        self.startup_log = ''
        self.commandline = ''
        self.extra_paths = None

    def cmd(self) -> list[str]:
        raise NotImplementedError()

    def create_server_reader(self):
        raise NotImplementedError()

    def detect_started_line(self, line):
        raise NotImplementedError()

    def process_startup_log(self):
        raise NotImplementedError()

    def read_model_info(self):
        raise NotImplementedError()

    def status(self, message):
        self.status_message = message

    def start_server(self):
        self.ready = False

        cmd = self.cmd()

        env = {**os.environ, **dict(COLUMNS="9999")}
        if self.extra_paths:
            env["PATH"] = os.pathsep.join(self.extra_paths) + os.pathsep + os.environ.get("PATH", "")

        self.commandline = shlex.join(cmd)
        self.server_process = subprocess.Popen(
            cmd,
            cwd=self.chdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            errors='ignore',
            env=env,
        )

        self.server_reader = self.create_server_reader()

        ready = False
        start = time.time()

        self.status('Waiting for server to start...')

        while True:
            try:
                line = self.server_reader.queue.get(timeout=0.2)
                if line:
                    self.startup_log += line
                    start = time.time()

                    if self.detect_started_line(line):
                        ready = True
                        break

            except queue.Empty:
                if self.server_process.poll() is not None:
                    self.status("❌ Server exited before it was ready.")
                    self.over = True
                    break

            if time.time() - start > shared.opts.backend_startup_timeout:
                self.status("❌ Timed out waiting for output from server.")
                self.startup_log += "\nTimed out."
                self.over = True
                break

        if not ready:
            return

        self.process_startup_log()

        if self.access_url is not None:
            self.status(f"✅ Listening on {self.access_url}")
        else:
            self.status("✅ Ready!")

        self.ready = True

    def run(self):
        if self.server_thread is not None:
            return

        self.status("Launching server process.")
        self.server_thread = threading.Thread(target=self.server_thread_main, daemon=True)
        self.server_thread.start()

    def server_thread_main(self):
        while not self.over:
            self.start_server()

            while True:
                code = self.server_process.poll()

                if code is not None:
                    break

                time.sleep(1)

            self.ready = False
            self.status(f"Server process exited with code {code}; {'quitting' if self.over else 'restarting'}")

        self.server_process = None

    def stop_server(self):
        self.over = True

        if self.server_process and self.server_process.poll() is None:
            self.status('Stopping server...')

            os.kill(self.server_process.pid, signal.SIGTERM if os.name == 'nt' else signal.SIGKILL)
            self.server_process.wait()

    def sample_messages(self):
        return [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "What is 1+1?"},
                {"role": "assistant", "content": "It's 2."},
                {"role": "user", "content": "Thank you."},
                {"role": "assistant", "content": "No problem."},
            ]

    def write_down_template(self):
        self.model_chat_template_example = ''

        if not self.model_chat_template:
            self.model_chat_template_markdown = "*Chat template missing!*"
            return

        try:
            try:
                rendered = templating.render(self.model_chat_template, self.model_chat_template_vars)
            except Exception:
                vars2 = self.model_chat_template_vars.copy()
                vars2['messages'] = vars2['messages'][1:]
                rendered = templating.render(self.model_chat_template, vars2)

            self.model_chat_template_example = rendered
            self.model_chat_template_markdown = "Example:\n```\n" + str(rendered) + "\n```\n\nFull chat template:\n```\n" + str(self.model_chat_template) + "\n```"
        except Exception as e:
            self.model_chat_template_markdown = f"Error rendering example: {e}\n\nFull chat template:\n```\n" + str(self.model_chat_template) + "\n```"
