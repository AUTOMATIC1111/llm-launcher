import re
import time

from modules import output_reader


class ReaderLlamacpp(output_reader.BaseReader):
    def __init__(self, pipe):
        super().__init__(pipe)
        self.current_request = output_reader.RequestStat()

        self.re_prompt = re.compile(r'prompt eval time =\s*([\d.]+) ms\s*/\s*(\d+) tokens')
        self.re_eval = re.compile(r'eval time =\s*([\d.]+) ms\s*/\s*(\d+) tokens')

    def process_line(self, line):
        m = self.re_prompt.search(line)
        if m:
            self.current_request.time = time.time()
            self.current_request.time_process = float(m.group(1))
            self.current_request.tokens_process = int(m.group(2))
            return

        m = self.re_eval.search(line)
        if m:
            self.current_request.time_generate = float(m.group(1))
            self.current_request.tokens_generate = int(m.group(2))
            self.requests.append(self.current_request)
            self.current_request = output_reader.RequestStat()
