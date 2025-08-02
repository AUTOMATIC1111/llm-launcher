import dataclasses
import queue
import sys
import threading
import time


@dataclasses.dataclass
class RequestStat:
    time: float = 0
    time_process: float = 0
    time_generate: float = 0
    tokens_process: int = 0
    tokens_generate: int = 0


class BaseReader:
    """
    Abstract base class for reading and parsing model server stats.

    Handles the common functionality of reading from a pipe in a background
    thread, maintaining a list of recent requests, and printing output.
    Subclasses must implement the `process_line` method to handle
    the specific log format.
    """
    def __init__(self, pipe, keep_requests_duration_sec: int = 30 * 60):
        self.pipe = pipe
        self.queue = queue.Queue()
        self.requests: list[RequestStat] = []
        self.keep_requests_duration = keep_requests_duration_sec

        thread = threading.Thread(target=self.main, args=(), daemon=True)
        thread.start()

    def main(self):
        """Main loop to read lines from the pipe and process them."""

        try:
            for line in iter(self.pipe.readline, ''):
                self.queue.put(line)
                print(line, end='')
                sys.stdout.flush()

                self.process_line(line.strip())

                cutoff_time = time.time() - self.keep_requests_duration
                while self.requests and self.requests[0].time < cutoff_time:
                    self.requests.pop(0)
        finally:
            self.pipe.close()

    def process_line(self, line: str):
        raise NotImplementedError()
