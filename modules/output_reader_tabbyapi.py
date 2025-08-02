import re
import time

from modules import output_reader


class ReaderTabbyapi(output_reader.BaseReader):
    def __init__(self, pipe):
        super().__init__(pipe)

        self.re_total = re.compile(r"(\d+) tokens generated .*? Generate: ([\d.]+) T/s")
        self.re_process = re.compile(r"Process: .*? (\d+) new tokens at ([\d.]+) T/s")

    def process_line(self, line: str):
        m = self.re_total.search(line)
        if not m:
            return

        total_generated = int(m.group(1))
        gen_rate = float(m.group(2))

        m = self.re_process.search(line)
        total_processed = int(m.group(1)) if m else None
        process_rate = float(m.group(2)) if m else None

        stat = output_reader.RequestStat(
            time=time.time(),
            time_process=total_processed/process_rate * 1000 if process_rate else None,
            time_generate=total_generated/gen_rate * 1000 if gen_rate else None,
            tokens_process=total_processed,
            tokens_generate=total_generated,
        )

        self.requests.append(stat)
