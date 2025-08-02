import dataclasses
import html
import os
import time
import threading
from pathlib import Path
import requests
import urllib.parse

from modules import shared
import gradio as gr


base_url = 'https://huggingface.co'


def format_file_size(size_bytes):
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB")
    i = 0
    while size_bytes >= 1024 and i < len(size_name) - 1:
        size_bytes /= 1024.0
        i += 1

    if i < 2:
        return f"{size_bytes:.0f} {size_name[i]}"

    return f"{size_bytes:.1f} {size_name[i]}"


@dataclasses.dataclass
class Progress:
    total: int
    done: int = 0
    history: list[tuple] = dataclasses.field(default_factory=list)
    history_length: int = 3

    def advance(self, add):
        self.done += add

        if add != 0:
            self.history.append((time.time(), self.done))

    def finish(self):
        self.advance(self.total - self.done)

    def speed(self):
        now = time.time()
        cutoff = now - self.history_length

        while self.history and self.history[0][0] < cutoff:
            self.history.pop(0)

        self.history.append((now, self.done))

        start, start_done = self.history[0]
        end, end_done = self.history[-1]
        elapsed = end - start

        if elapsed == 0:
            return 0

        return (end_done - start_done) / elapsed

    def percentage(self):
        if not self.total:
            return 0.0

        return self.done / self.total * 100


@dataclasses.dataclass
class DownloadTask:
    model_id: str
    revision: str
    file_url: str
    path: str
    local_path: Path
    status: str = "queued"
    progress: Progress = None
    total_size: int = 0
    start_time: float = dataclasses.field(default_factory=time.time)
    error: str = None
    thread: threading.Thread = None
    stop: bool = False
    in_progress: bool = False
    is_junk: bool = False

    def __post_init__(self):
        self.progress = Progress(total=self.total_size)


class HuggingfaceDownloader:
    VERIFY_CHUNK_SIZE = 0

    def __init__(self):
        self.downloads: list[DownloadTask] = []
        self.lock = threading.Lock()

    def get_downloads_html(self):
        htmls = []

        cleanable = 0

        with self.lock:
            for task in self.downloads:
                status = f"""
                    <span class='status'>{html.escape(task.status)}</span>
                    {f"— <span class='error'>{html.escape(task.error)}</span>" if task.error else ""}
                    
                """

                if task.in_progress:
                    status += f"""
                        —
                        {format_file_size(task.progress.done)} of {format_file_size(task.progress.total)}
                        —
                        {format_file_size(task.progress.speed())}/sec
                    """
                elif task.status == "completed":
                    status += f"— {format_file_size(task.progress.total)}"
                    cleanable += 1
                else:
                    cleanable += 1

                if task.in_progress:
                    btn = "⏹"
                    action = "Stop download"
                elif task.is_junk:
                    btn = "🗑"
                    action = "Delete"
                else:
                    btn = "╳"
                    action = "Remove entry from the list"

                htmls += [f"""
                <div class='download{" active" if task.in_progress else ""}'>
                    <div class='cross' onclick='stopDownload("{html.escape(task.file_url, quote=True)}")' title='{action}'>{btn}</div>
                    <div class='upper'>
                        {html.escape(task.local_path.name)}
                    </div>
                    <div class='progressbar'>
                        <div class='progress' style='width: {task.progress.percentage()}%;'></div>
                    </div>
                    <div class='lower'>
                        {status}
                    </div>
                </div>
                """]

        return "".join(htmls), gr.update(visible=cleanable>0)

    def list_files(self, model_id, revision):
        revision = revision or 'main'
        api_url = f"{base_url}/api/models/{model_id}/tree/{revision}"
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            items = [item for item in response.json() if item["type"] == "file"]
        except Exception as e:
            gr.Warning(f'{e}')
            return gr.update(choices=[], value=None), [], gr.update(interactive=False)

        for item in items:
            item['model_id'] = model_id
            item['revision'] = revision

        choices = [('All files', -1), *[(item.get("path"), i) for i, item in enumerate(items)]]
        return gr.update(choices=choices, value=-1), items, gr.update(interactive=True)

    def download_worker(self, task: DownloadTask):
        task.status = "preparing"
        task.in_progress = True

        errors = []

        while True:
            if len(errors) >= 5 and time.time() - errors[0][0] < 10.0:
                task.status = "failed"
                task.error = errors[-1][1]
                break

            try:
                task.local_path.parent.mkdir(parents=True, exist_ok=True)

                initial_size = task.local_path.stat().st_size if task.local_path.exists() else 0
                if initial_size > self.VERIFY_CHUNK_SIZE > 0:
                    task.status = "verifying"

                    verify_range_start = initial_size - self.VERIFY_CHUNK_SIZE
                    verify_headers = {"Range": f"bytes={verify_range_start}-"}

                    with open(task.local_path, "rb") as f:
                        f.seek(verify_range_start)
                        local_chunk = f.read(self.VERIFY_CHUNK_SIZE)

                    with requests.get(task.file_url, headers=verify_headers, stream=True, timeout=10) as response:
                        response.raise_for_status()

                        remote_chunk = b''
                        for chunk in response.iter_content(chunk_size=self.VERIFY_CHUNK_SIZE):
                            remote_chunk = remote_chunk + chunk
                            if len(remote_chunk) >= self.VERIFY_CHUNK_SIZE:
                                break

                    remote_chunk = remote_chunk[0:self.VERIFY_CHUNK_SIZE]

                    if remote_chunk != local_chunk:
                        task.status = "failed"
                        task.error = "data mismatch; can't resume"
                        task.is_junk = True
                        break

                task.progress.advance(initial_size)

                headers = {"Range": f"bytes={initial_size}-"} if initial_size else {}
                if 0 < task.total_size <= initial_size:
                    task.status = "completed"
                    task.progress.finish()
                    break

                with requests.get(task.file_url, headers=headers, stream=True, timeout=10) as response:
                    response.raise_for_status()

                    if task.total_size is None:
                        task.total_size = int(response.headers.get('content-length', 0)) + initial_size

                    task.start_time = time.time()

                    task.status = "downloading"

                    with open(task.local_path, "ab") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if task.stop:
                                break
                            if chunk:
                                f.write(chunk)
                                with self.lock:
                                    task.progress.advance(len(chunk))

                if task.stop:
                    task.status = "canceled"
                    task.stop = False
                else:
                    task.status = "completed"
                    task.progress.finish()

                break
            except Exception as e:
                errors.append((time.time(), str(e)))
                if len(errors) > 5:
                    errors.pop(0)

        task.in_progress = False
        self.clean_threads()

    def clean_threads(self):
        with self.lock:
            for task in self.downloads:
                if task.thread is not None and not task.thread.is_alive():
                    task.thread.join()
                    task.thread = None

    def start_download(self, file_data, selection):
        if selection == -1:
            to_download = file_data
        else:
            to_download = [file_data[selection]]

        for item in to_download:
            model_id, revision, path, size = (item[x] for x in ['model_id', 'revision', 'path', 'size'])

            with self.lock:
                file_url = f"{base_url}/{model_id}/resolve/{revision}/{path}"

                task = next((x for x in self.downloads if x.file_url == file_url), None)
                if task is not None:
                    if task.status == "completed" or task.in_progress:
                        continue
                else:
                    parsed = urllib.parse.urlparse(file_url)
                    file_path = parsed.path.split(f'{model_id}/resolve/{revision}/')[-1]
                    if selection == -1:
                        local_path = Path(shared.opts.model_dir) / model_id.replace("/", '-') / file_path
                    else:
                        local_path = Path(shared.opts.model_dir) / os.path.basename(file_path)

                    task = DownloadTask(
                        model_id=model_id,
                        revision=revision,
                        path=path,
                        file_url=file_url,
                        local_path=local_path,
                        total_size=size,
                    )
                    self.downloads.append(task)

                task.thread = threading.Thread(target=self.download_worker, args=(task,))
                task.thread.start()

    def stop_download(self, url):
        with self.lock:
            task = next((x for x in self.downloads if x.file_url == url), None)
            if not task:
                return

            if task.in_progress:
                task.stop = True
            elif task.is_junk:
                os.unlink(task.local_path)
                self.downloads.remove(task)
            else:
                self.downloads.remove(task)

    def do_cleanup(self):
        with self.lock:
            for i in reversed(range(len(self.downloads))):
                if not self.downloads[i].in_progress:
                    self.downloads.pop(i)

    def create_ui(self, demo):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    list_files = gr.Button("List files", variant="secondary")
                    download_btn = gr.Button("Download", variant="primary", interactive=False)

                with gr.Row():
                    with gr.Column(scale=6):
                        model_id = gr.Textbox(label="Model ID", placeholder="username/model-name")
                    with gr.Column(scale=1, min_width=60):
                        revision = gr.Textbox(label="Revision", placeholder="main", value="", min_width=60)

                with gr.Row():
                    file_selection = gr.Dropdown(label="File", choices=[])
                    file_data = gr.JSON(visible=False)

            with gr.Column():
                cleanup = gr.Button("Clean up", variant="secondary", visible=False)
                downloads_panel = gr.HTML('', elem_classes=['downloads'])

        stop_btn = gr.Button("Stop", visible=False, elem_id='stop_download')
        refresh_btn = gr.Button("Refresh", visible=False, elem_id='refresh_downloads')

        update_download_list = dict(fn=self.get_downloads_html, inputs=[], outputs=[downloads_panel, cleanup], show_progress='hidden')

        list_files.click(self.list_files, inputs=[model_id, revision], outputs=[file_selection, file_data, download_btn])
        download_btn.click(self.start_download, inputs=[file_data, file_selection]).then(**update_download_list)
        stop_btn.click(fn=self.stop_download, inputs=[stop_btn], js="getTargetForStopDownload").then(**update_download_list)
        cleanup.click(fn=self.do_cleanup).then(**update_download_list)

        refresh_btn.click(**update_download_list)
        demo.load(**update_download_list)
