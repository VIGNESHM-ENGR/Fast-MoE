"""Locate llama.cpp binaries and supervise a `llama-server` process."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

BIN_DIR_ENV = "LLAMA_CPP_BIN_DIR"
REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_BUILD_DIR = REPO_ROOT / "third_party" / "llama.cpp" / "build" / "bin"


class ServerError(RuntimeError):
    pass


def find_binary(name: str) -> Path:
    candidates = []
    if os.environ.get(BIN_DIR_ENV):
        candidates.append(Path(os.environ[BIN_DIR_ENV]) / name)
    candidates.append(NATIVE_BUILD_DIR / name)
    on_path = shutil.which(name)
    if on_path:
        candidates.append(Path(on_path))
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise ServerError(f"{name} not found. Build llama.cpp with scripts/build_llama_cpp.sh "
                      f"or set {BIN_DIR_ENV} to the directory containing it.")


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


class LlamaServer:
    def __init__(self, binary: Path, args: list[str], host: str, port: int, log_path: Path):
        self.command = [str(binary), *args, "--host", host, "--port", str(port)]
        self.url = f"http://{host}:{port}"
        self.log_path = log_path
        self.tail: deque[str] = deque(maxlen=40)
        self.proc: subprocess.Popen | None = None

    def start(self, echo: bool = False) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1)
        threading.Thread(target=self._pump_logs, args=(echo,), daemon=True).start()

    def _pump_logs(self, echo: bool) -> None:
        with self.log_path.open("w") as log:
            for line in self.proc.stdout:
                log.write(line)
                log.flush()
                self.tail.append(line.rstrip())
                if echo:
                    sys.stdout.write(line)

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=2) as resp:
                return json.loads(resp.read()).get("status") == "ok"
        except (OSError, ValueError):
            return False

    def wait_healthy(self, timeout_s: float = 900) -> float:
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            if self.proc.poll() is not None:
                raise ServerError(f"llama-server exited with code {self.proc.returncode}:\n"
                                  + "\n".join(self.tail))
            if self.healthy():
                return time.monotonic() - start
            time.sleep(1)
        raise ServerError(f"llama-server not healthy after {timeout_s:.0f}s; see {self.log_path}")

    def stop(self, timeout_s: float = 30) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
