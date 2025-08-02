import re
import socket
import subprocess


def get_local_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # (doesn't send actual packets)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


def extract_url(text):
    m = re.search(r'https?://[^ ]+', text)
    if m:
        return re.sub(r'\b0.0.0.0\b', lambda x: get_local_address(), m.group(0))

    return None


def get_hash(repo_path):
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
