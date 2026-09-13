"""Plan and run llama-server; restartable with new settings (used by the CLI and the UI)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from engine.llama.fit import FitResult, run_fit_params
from engine.llama.layout import LayerPlacement, ModelLayout, expert_placement, read_layout
from engine.llama.server import LlamaServer, ServerError, find_binary, port_in_use
from engine.models.model_downloader import DEFAULT_QUANT, DEFAULT_REPO, download, local_model

LOG_DIR = Path.home() / ".cache" / "fast-moe" / "logs"


@dataclass(frozen=True)
class RunSettings:
    model: str = DEFAULT_REPO
    quant: str = DEFAULT_QUANT
    host: str = "127.0.0.1"
    port: int = 8080
    ctx: int | None = None
    vram_margin_mib: int | None = None
    no_mmap: bool = False
    extra_args: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RunPlan:
    settings: RunSettings
    model_path: Path
    layout: ModelLayout
    fit: FitResult
    placement: tuple[LayerPlacement, ...]
    server_args: tuple[str, ...]


def resolve_model(model: str, quant: str) -> Path:
    path = Path(model).expanduser()
    if path.suffix == ".gguf":
        if not path.is_file():
            raise ServerError(f"Model file not found: {path}")
        return path
    return local_model(model, quant) or download(model, quant)[0]


@lru_cache(maxsize=8)
def _cached_layout(path: str, size: int, mtime: float) -> ModelLayout:
    return read_layout(Path(path))


def layout_for(path: Path) -> ModelLayout:
    """Reading a GGUF header takes seconds, so reuse it until the file changes."""
    st = path.stat()
    return _cached_layout(str(path), st.st_size, st.st_mtime)


def fit_params_args(settings: RunSettings) -> list[str]:
    args = []
    if settings.ctx:
        args += ["--fit-ctx", str(settings.ctx)]
    if settings.vram_margin_mib is not None:
        args += ["--fit-target", str(settings.vram_margin_mib)]
    return args


def server_args(settings: RunSettings, model_path: Path, fit: FitResult) -> tuple[str, ...]:
    # --fit off: llama-server must run exactly the placement we computed and displayed.
    args = ["-m", str(model_path), "--jinja", "--metrics", "--fit", "off", *fit.args]
    if settings.no_mmap:
        args += ["--load-mode", "none"]
    return (*args, *(a for a in settings.extra_args if a != "--"))


def plan_run(settings: RunSettings) -> RunPlan:
    fit_bin = find_binary("llama-fit-params")
    model_path = resolve_model(settings.model, settings.quant)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fit_future = pool.submit(run_fit_params, fit_bin, model_path, fit_params_args(settings))
        layout_future = pool.submit(layout_for, model_path)
        fit, layout = fit_future.result(), layout_future.result()
    placement = expert_placement(layout, fit.gpu_layers or 0, fit.cpu_patterns)
    return RunPlan(settings, model_path, layout, fit, placement, server_args(settings, model_path, fit))


class Runner:
    """Owns at most one llama-server; `start` replaces a running one."""

    def __init__(self) -> None:
        self.plan: RunPlan | None = None
        self.server: LlamaServer | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.server is not None and self.server.proc is not None and self.server.proc.poll() is None

    def start(self, settings: RunSettings, on_plan: Callable[[RunPlan], None] | None = None) -> float:
        # Stop first: llama-fit-params measures free VRAM, which a running server would occupy.
        self.stop()
        plan = plan_run(settings)
        if on_plan:
            on_plan(plan)
        return self.launch(plan)

    def launch(self, plan: RunPlan) -> float:
        """Start llama-server for a plan made while no server was running; returns seconds to healthy."""
        with self._lock:
            self._stop_locked()
            settings = plan.settings
            server_bin = find_binary("llama-server")
            if port_in_use(settings.host, settings.port):
                raise ServerError(f"Port {settings.port} is already in use; pick another port.")
            self.plan = plan
            log_path = LOG_DIR / f"llama-server-{time.strftime('%Y%m%d-%H%M%S')}.log"
            self.server = LlamaServer(server_bin, list(plan.server_args), settings.host, settings.port, log_path)
            self.server.start()
            server = self.server
        return server.wait_healthy()

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None
