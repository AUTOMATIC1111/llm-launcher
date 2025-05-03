import dataclasses
import queue
import re
import sys
import threading
import time


@dataclasses.dataclass
class RequestStat:
    time: int = 0
    time_process: float = 0
    time_generate: float = 0
    tokens_process: int = 0
    tokens_generate: int = 0


class Reader:
    def __init__(self, pipe):
        self.queue = queue.Queue()
        self.pipe = pipe
        self.requests: list[RequestStat] = []
        self.current_request = RequestStat()
        self.keep_requests_duration = 10 * 60

        thread = threading.Thread(target=self.main, args=(), daemon=True)
        thread.start()

    def main(self):
        try:
            for line in iter(self.pipe.readline, ''):
                self.queue.put(line)

                print(line, end='')
                sys.stdout.flush()

                m = re.search(r'prompt eval time =\s*([\d.]+) ms\s*/\s*(\d+) tokens', line)
                if m:
                    self.current_request.time = time.time()
                    self.current_request.time_process = float(m.group(1))
                    self.current_request.tokens_process = int(m.group(2))
                else:
                    m = re.search(r'eval time =\s*([\d.]+) ms\s*/\s*(\d+) tokens', line)
                    if m:
                        self.current_request.time_generate = float(m.group(1))
                        self.current_request.tokens_generate = int(m.group(2))
                        self.requests.append(self.current_request)
                        self.current_request = RequestStat()

                while self.requests and self.requests[0].time < time.time() - self.keep_requests_duration:
                    self.requests.pop()

        finally:
            self.pipe.close()
