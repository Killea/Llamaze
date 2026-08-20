#!/usr/bin/env -S /home/hank/llama.cpp/.venv-llama/bin/python3
"""GUI frontend for running GGUF models with llama-server (PyQt6)."""

from __future__ import annotations

import os

# Disable atk-bridge to avoid GTK warning:
# "Not loading module 'atk-bridge': The functionality is provided by GTK natively."
os.environ["NO_AT_BRIDGE"] = "1"
import codecs
import datetime
import functools
import hashlib
import http.server
import json
import platform
import re
import shlex
import socket
import subprocess
import sys
import threading
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast

from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import QSettings, QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QCursor, QFont, QFontDatabase, QIcon
from PyQt6.QtNetwork import QLocalServer, QLocalSocket

from gguf_utils import (
    detect_mtp_cached,
    inspect_gguf_metadata,
    model_context_length,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR.parent / "llama_model"
DEFAULT_MTP_CACHE = SCRIPT_DIR / "llamaze_mtp.ini"
LOG_DIR = SCRIPT_DIR / "gui_log"
DEFAULT_CTX = 256000
ROSETTA_DIR = SCRIPT_DIR / "llm-rosetta"
ROSETTA_SRC_DIR = ROSETTA_DIR / "src"
ROSETTA_CONFIG_PATH = SCRIPT_DIR / "rosetta_config.jsonc"
ROSETTA_GATEWAY_PORT = 8920

_log_file_lock = threading.Lock()
_log_file_path: Path | None = None
_log_file_handle = None


def _append_log_file(timestamped_msg: str) -> None:
    global _log_file_path, _log_file_handle
    try:
        with _log_file_lock:
            if _log_file_path is None:
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                _log_file_path = LOG_DIR / f"gui_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                _log_file_handle = open(_log_file_path, "a", encoding="utf-8", buffering=1)
            if _log_file_handle:
                _log_file_handle.write(timestamped_msg + "\n")
    except Exception:
        pass


def _close_log_file() -> None:
    global _log_file_handle
    try:
        with _log_file_lock:
            if _log_file_handle:
                _log_file_handle.close()
                _log_file_handle = None
    except Exception:
        pass
CONTEXT_CHOICES = [""] + [f"{value}K" for value in range(32, 257, 16)] + ["184K"]
CACHE_TYPE_CHOICES = ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
FLASH_ATTN_CHOICES = ["auto", "on", "off"]
DEFAULT_ON_OFF_CHOICES = ["default", "on", "off"]
DEFAULT_BOOL_CHOICES = ["default", "on"]
IMAGE_MODEL_DEVICE_CHOICES = ["CPU", "GPU"]
QWEN35MOE_STABLE_AUTO_CTX = 131072
LATEST_VERIFIED_SERVER_CANDIDATE = SCRIPT_DIR / "build-qwen38-fa-allquants/bin/llama-server"
NO_GRAPHS_SERVER_CANDIDATE = SCRIPT_DIR / "build-qwen38-fa-allquants/bin/llama-server"
DEFAULT_SERVER_CANDIDATES = (
    LATEST_VERIFIED_SERVER_CANDIDATE,
    SCRIPT_DIR / "build-b9813-cuda13-nographs/bin/llama-server",
    SCRIPT_DIR / "build-latest-cuda13-graphs/bin/llama-server",
    SCRIPT_DIR / "build-cuda/bin/llama-server",
    SCRIPT_DIR / "build/bin/llama-server",
)
MODEL_SETTINGS_VERSION = 1
PRESETS_FILE = SCRIPT_DIR / "llamaze_presets.json"
PRESETS_DOC_VERSION = 1
GUI_API_DEFAULT_PORT = 8910
GUI_API_PORT_RANGE = 10


def _generate_rosetta_config(model_name: str, llama_port: str | int,
                             gateway_port: int = ROSETTA_GATEWAY_PORT) -> dict[str, Any]:
    """Build a Rosetta gateway config dict pointing at the local llama-server."""
    return {
        "providers": {
            "openai_chat": {
                "api_key": "sk-no-key-needed",
                "base_url": f"http://127.0.0.1:{llama_port}/v1",
            },
        },
        "models": {
            model_name: "openai_chat",
        },
        "server": {
            "host": "127.0.0.1",
            "port": gateway_port,
            "open_on_no_keys": True,
        },
    }


def _write_rosetta_config(model_name: str, llama_port: str | int) -> Path:
    """Write the Rosetta gateway config file and return its path."""
    config = _generate_rosetta_config(model_name, llama_port)
    with ROSETTA_CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return ROSETTA_CONFIG_PATH


@functools.cache
def server_help_flags(server_raw: str) -> frozenset[str]:
    """Return option tokens accepted by a llama-server binary."""
    server = _script_relative_path(Path(server_raw))
    try:
        proc = subprocess.run(
            [str(server), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    return frozenset(re.findall(r"(?<!\w)-{1,2}[A-Za-z0-9][A-Za-z0-9_-]*", proc.stdout))


def server_supports_flag(server_raw: str, flag: str) -> bool:
    """Return whether the selected llama-server advertises a command-line flag."""
    flags = server_help_flags(server_raw)
    return not flags or flag in flags


# ── Log reader thread ─────────────────────────────────────────────────


class LogChunkParser:
    """Decode subprocess output and split on terminal-style line updates."""

    def __init__(self):
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._buffer = ""
        self._last_sep_was_cr = False

    def feed(self, chunk: bytes) -> list[str]:
        return self._feed_text(self._decoder.decode(chunk, final=False))

    def flush(self) -> list[str]:
        lines = self._feed_text(self._decoder.decode(b"", final=True))
        if self._buffer:
            lines.append(self._buffer)
            self._buffer = ""
        return lines

    def _feed_text(self, text: str) -> list[str]:
        lines: list[str] = []
        for char in text:
            if char in "\r\n":
                if char == "\n" and self._last_sep_was_cr:
                    self._last_sep_was_cr = False
                    continue
                lines.append(self._buffer)
                self._buffer = ""
                self._last_sep_was_cr = char == "\r"
            else:
                self._buffer += char
                self._last_sep_was_cr = False
        return lines


class LlamaLogAnalyzer:
    """Extract high-signal diagnostics from llama-server logs."""

    _OFFLOADED_RE = re.compile(
        r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU",
        re.IGNORECASE,
    )
    _BUFFER_RE = re.compile(
        r":\s*(?P<backend>[A-Za-z0-9_ -]+?)\s+model buffer size\s*=\s*"
        r"(?P<mib>[0-9]+(?:\.[0-9]+)?)\s+MiB",
        re.IGNORECASE,
    )
    _GPU_BACKEND_TOKENS = (
        "CUDA",
        "HIP",
        "ROCM",
        "VULKAN",
        "METAL",
        "SYCL",
        "CANN",
        "OPENCL",
    )

    def __init__(self):
        self._emitted: set[str] = set()

    def analyze_line(self, line: str) -> list[str]:
        diagnostics: list[str] = []
        diagnostics.extend(self._analyze_offload(line))
        diagnostics.extend(self._analyze_model_buffer(line))
        diagnostics.extend(self._analyze_errors(line))
        return diagnostics

    def _emit_once(self, key: str, message: str) -> list[str]:
        if key in self._emitted:
            return []
        self._emitted.add(key)
        return [message]

    def _analyze_offload(self, line: str) -> list[str]:
        match = self._OFFLOADED_RE.search(line)
        if not match:
            return []

        offloaded = int(match.group(1))
        total = int(match.group(2))
        if total <= 0 or offloaded >= total:
            return []

        if offloaded == 0:
            return self._emit_once(
                "offload:none",
                "WARNING: No model layers were offloaded to GPU. Check the CUDA build, "
                "-ngl/GPU Layers setting, VRAM availability, and selected server binary.",
            )

        return self._emit_once(
            f"offload:partial:{offloaded}:{total}",
            f"WARNING: Only {offloaded}/{total} layers were offloaded to GPU; "
            "remaining layers are on CPU and generation may be much slower.",
        )

    def _analyze_model_buffer(self, line: str) -> list[str]:
        match = self._BUFFER_RE.search(line)
        if not match:
            return []

        backend = " ".join(match.group("backend").split())
        mib = float(match.group("mib"))
        backend_upper = backend.upper()

        if "CPU" in backend_upper:
            return self._emit_once(
                f"buffer:cpu:{backend}",
                f"WARNING: CPU model buffer detected: {backend} {mib:.2f} MiB. "
                "Some model weights are resident on CPU; generation may be slower.",
            )

        if any(token in backend_upper for token in self._GPU_BACKEND_TOKENS):
            return self._emit_once(
                f"buffer:gpu:{backend}",
                f"DIAG: GPU model buffer detected: {backend} {mib:.2f} MiB.",
            )

        return []

    def _analyze_errors(self, line: str) -> list[str]:
        lowered = line.lower()
        if (
            "out of memory" in lowered
            or "failed to allocate" in lowered
            or "cuda error" in lowered
        ):
            return self._emit_once(
                "error:allocation",
                "ERROR: CUDA/host allocation failure detected. Try lower context, "
                "lower batch, lower GPU layers, or close other GPU workloads.",
            )

        if (
            "ggml_assert" in lowered
            or "segmentation fault" in lowered
            or "fatal signal" in lowered
            or "core dumped" in lowered
        ):
            return self._emit_once(
                "error:crash",
                "ERROR: llama-server crash signature detected. Check the preceding "
                "log lines for the failing backend or kernel.",
            )

        return []


class ServerLogThread(QThread):
    """Read server stdout in a background thread and emit log lines."""

    line_received = pyqtSignal(str)
    finished_with_code = pyqtSignal(int)
    server_started = pyqtSignal(int, str)  # pid, port_url

    def __init__(self, process: subprocess.Popen):
        super().__init__()
        self._process = process
        self._stop_event = threading.Event()

    def run(self) -> None:
        import sys as _sys

        stream = getattr(_sys.stdout, "buffer", _sys.stdout)
        parser = LogChunkParser()
        stdout = self._process.stdout
        if stdout is None:
            return
        while not self._stop_event.is_set():
            chunk = stdout.read(4096)
            if not chunk:
                break
            # Mirror to console
            try:
                if isinstance(chunk, str):
                    stream.write(chunk)
                else:
                    try:
                        stream.write(chunk)
                    except TypeError:
                        stream.write(chunk.decode("utf-8", "replace"))
                stream.flush()
            except Exception:
                pass
            chunk_bytes = chunk.encode("utf-8", "replace") if isinstance(chunk, str) else chunk
            for line in parser.feed(chunk_bytes):
                self.line_received.emit(line)
        for line in parser.flush():
            if self._stop_event.is_set():
                break
            self.line_received.emit(line)
        retcode = self._process.poll()
        if retcode is not None:
            self.finished_with_code.emit(retcode)

    def stop(self) -> None:
        self._stop_event.set()


class RosettaGatewayThread(QThread):
    """Run the Rosetta gateway subprocess and emit log lines."""

    line_received = pyqtSignal(str)
    started_ok = pyqtSignal()
    finished_with_code = pyqtSignal(int)

    def __init__(self, config_path: Path):
        super().__init__()
        self._config_path = config_path
        self._process: subprocess.Popen | None = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        env = os.environ.copy()
        # Ensure llm_rosetta is importable from the submodule
        src_dir = str(ROSETTA_SRC_DIR)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = src_dir + (os.pathsep + existing if existing else "")
        # Use the same Python interpreter that runs the GUI
        python = sys.executable
        cmd = [python, "-m", "llm_rosetta.gateway", "--config", str(self._config_path), "--no-banner"]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env=env,
            )
        except Exception as exc:
            self.line_received.emit(f"[Rosetta gateway failed to start: {exc}]")
            self.finished_with_code.emit(-1)
            return

        self.line_received.emit(f"Rosetta gateway PID: {self._process.pid}")
        self.started_ok.emit()

        parser = LogChunkParser()
        stdout = self._process.stdout
        if stdout is None:
            return
        while not self._stop_event.is_set():
            chunk = stdout.read(4096)
            if not chunk:
                break
            chunk_bytes = chunk.encode("utf-8", "replace") if isinstance(chunk, str) else chunk
            for line in parser.feed(chunk_bytes):
                self.line_received.emit(line)
        for line in parser.flush():
            if self._stop_event.is_set():
                break
            self.line_received.emit(line)
        retcode = self._process.poll()
        if retcode is not None:
            self.finished_with_code.emit(retcode)

    def stop(self) -> None:
        self._stop_event.set()
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()


# ── Helpers ───────────────────────────────────────────────────────────


def resolve_server_path() -> Path:
    """Auto-detect llama-server binary."""
    for candidate in DEFAULT_SERVER_CANDIDATES:
        if candidate.is_file():
            return candidate
    return DEFAULT_SERVER_CANDIDATES[0]


def _kill_process_on_port(port: int) -> None:
    """Kill any process listening on the given TCP port (best-effort)."""
    import signal
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            result = s.connect_ex(("127.0.0.1", port))
            if result != 0:
                return  # port is free
    except OSError:
        return
    # Find and kill the process using fuser
    try:
        subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        pass


def format_server_build_info(server_raw: str) -> str:
    """Return a compact one-line summary of the selected llama-server binary."""
    server = Path(server_raw).expanduser()
    if not server_raw.strip():
        return "llama: no server selected"
    if not server.is_file():
        return f"llama: server not found: {server}"

    try:
        version = subprocess.run(
            [str(server), "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=3,
            check=False,
        ).stdout.strip()
    except Exception as exc:
        version = f"version unavailable: {exc}"

    version = " | ".join(line.strip() for line in version.splitlines() if line.strip())
    if not version:
        version = "version unavailable"

    built_at = datetime.datetime.fromtimestamp(server.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return f"llama: {version} | compiled: {built_at} | path: {server}"


def refresh_models(model_dir: Path) -> list[Path]:
    """Return sorted list of .gguf models in directory."""
    if not model_dir.is_dir():
        return []
    return sorted(model_dir.glob("*.gguf"))


def resolve_initial_model_dir(default_model_dir: Path, last_model: str | None) -> Path:
    """Return the model directory to show at startup."""
    if last_model:
        last_model_path = Path(last_model)
        if last_model_path.is_file():
            return last_model_path.parent
    return default_model_dir


def resolve_preferred_model(
    models: list[Path],
    current_model: str | None,
    last_model: str | None,
) -> str | None:
    """Return the model path that should be selected after refreshing the list."""
    available = {str(model) for model in models}
    if current_model and current_model in available and Path(current_model).is_file():
        return current_model
    if last_model and last_model in available and Path(last_model).is_file():
        return last_model
    return str(models[0]) if models else None


def resolve_mtp_badge(is_mtp: bool) -> tuple[str, str]:
    """Return the text and stylesheet for the model MTP badge."""
    if is_mtp:
        return (
            "MTP",
            "QLabel { background-color: #2e7d32; color: white; "
            "padding: 3px 8px; border-radius: 4px; }",
        )
    return (
        "non MTP",
        "QLabel { color: #555555; padding: 3px 8px; border: 1px solid transparent; }",
    )


def _metadata_name(metadata: dict[str, object]) -> str:
    value = metadata.get("general.name")
    return value if isinstance(value, str) else ""


def _normalized_model_identity(model: Path, metadata: dict[str, object]) -> str:
    return f"{model.name} {_metadata_name(metadata)}".lower()


def _model_family(metadata: dict[str, object], fallback_name: str = "") -> str:
    """Return a coarse model family for matching companion GGUF files."""
    arch = _model_architecture(metadata).lower()
    name = f"{_metadata_name(metadata)} {fallback_name}".lower()
    if arch.startswith("gemma") or "gemma" in name:
        return "gemma"
    if arch.startswith("qwen") or "qwen" in name or "huihui" in name:
        return "qwen"
    return arch


def _file_rank(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    if "q8_0" in name:
        return (0, name)
    if "f16" in name:
        return (1, name)
    if "bf16" in name:
        return (2, name)
    if "q5" in name:
        return (3, name)
    return (4, name)


def _metadata_int(metadata: dict[str, object], suffix: str) -> int | None:
    for key, value in metadata.items():
        if key.endswith(suffix) and isinstance(value, int):
            return value
    return None


def _mmproj_matches_text_model(
    model_metadata: dict[str, object],
    mmproj_metadata: dict[str, object],
) -> bool:
    text_embedding = _metadata_int(model_metadata, ".embedding_length")
    projection_dim = _metadata_int(mmproj_metadata, ".projection_dim")
    if text_embedding is None or projection_dim is None:
        return True
    return text_embedding == projection_dim


def _is_auto_mtp_draft_supported(metadata: dict[str, object]) -> bool:
    """Return whether this draft architecture is supported by the current server build."""
    arch = _model_architecture(metadata)
    return bool(arch)


def resolve_default_mmproj(model: Path) -> Path | None:
    """Return a matching local multimodal projector for the selected model."""
    model_dir = model.parent

    # Qwen3.8 and variants: prefer the dedicated qwen3.8 mmproj if present
    try:
        model_metadata = inspect_gguf_metadata(model)
    except Exception:
        model_metadata = {}
    if _is_qwen38_27b(model, model_metadata):
        qwen38_mmproj = model_dir / "mmproj-F16-qwen3.8.gguf"
        if qwen38_mmproj.is_file():
            return qwen38_mmproj

    candidates = sorted(model_dir.glob("mmproj*.gguf"))
    if not candidates:
        return None

    if not model_metadata:
        preferred = model_dir / "mmproj-model-f16.gguf"
        if preferred.is_file():
            return preferred
        return candidates[0]

    family = _model_family(model_metadata, model.name)
    scored_candidates: list[tuple[int, tuple[int, str], Path]] = []
    for candidate in candidates:
        try:
            candidate_metadata = inspect_gguf_metadata(candidate)
        except Exception:
            candidate_metadata = {}
        if not _mmproj_matches_text_model(model_metadata, candidate_metadata):
            continue
        candidate_family = _model_family(candidate_metadata, candidate.name)
        score = 0 if candidate_family == family else 1
        scored_candidates.append((score, _file_rank(candidate), candidate))

    if not scored_candidates:
        return None
    scored_candidates.sort()
    return scored_candidates[0][2]


def resolve_draft_mtp_model(model: Path, cache_path: Path) -> Path | None:
    """Return a companion MTP draft model for a non-MTP base model, if available."""
    try:
        is_mtp, _reason = detect_mtp_cached(model, cache_path)
    except Exception:
        is_mtp = False
    if is_mtp:
        return None

    search_dirs = [model.parent / "MTP", model.parent]
    candidates: list[Path] = []
    for search_dir in search_dirs:
        if search_dir.is_dir():
            candidates.extend(sorted(search_dir.glob("*MTP*.gguf")))

    try:
        model_metadata = inspect_gguf_metadata(model)
    except Exception:
        model_metadata = {}
    family = _model_family(model_metadata, model.name)

    matches: list[tuple[int, tuple[int, str], Path]] = []
    for candidate in candidates:
        if candidate.resolve() == model.resolve():
            continue
        try:
            candidate_is_mtp, _reason = detect_mtp_cached(candidate, cache_path)
        except Exception:
            continue
        if not candidate_is_mtp:
            continue

        try:
            candidate_metadata = inspect_gguf_metadata(candidate)
        except Exception:
            candidate_metadata = {}
        if not _is_auto_mtp_draft_supported(candidate_metadata):
            continue
        candidate_family = _model_family(candidate_metadata, candidate.name)
        if candidate_family != family:
            continue
        matches.append((0, _file_rank(candidate), candidate))

    if not matches:
        return None
    matches.sort()
    return matches[0][2]


def _is_qwythos_9b_claude_mythos_1m(model: Path, metadata: dict[str, object]) -> bool:
    identity = _normalized_model_identity(model, metadata)
    return (
        "qwythos" in identity
        and "9b" in identity
        and "claude" in identity
        and "mythos" in identity
        and ("1m" in identity or model_context_length(metadata) == 1048576)
    )


def _is_huihui_qwen36_27b_abliterated(model: Path, metadata: dict[str, object]) -> bool:
    identity = _normalized_model_identity(model, metadata)
    return (
        "huihui" in identity
        and "qwen3.6" in identity
        and "27b" in identity
        and "abliterated" in identity
    )


def _is_qwen3_coder_next(model: Path, metadata: dict[str, object]) -> bool:
    """Detect Qwen3-Coder-Next 512x2.5B MoE models by architecture."""
    arch = _model_architecture(metadata)
    if arch != "qwen3next":
        return False
    # Confirm it's the Coder-Next variant with MoE signature
    n_experts = metadata.get("qwen3next.expert_count")
    return isinstance(n_experts, int) and n_experts > 0


def _is_gemma4_31b_it(model: Path, metadata: dict[str, object]) -> bool:
    identity = _normalized_model_identity(model, metadata)
    return _model_architecture(metadata) == "gemma4" and "31b" in identity and "it" in identity


def _is_qwen38_27b(model: Path, metadata: dict[str, object]) -> bool:
    identity = _normalized_model_identity(model, metadata)
    return "qwen3.8" in identity and "27b" in identity


def model_default_parameter_settings(
    model: Path,
    metadata: dict[str, object],
) -> dict[str, str | bool]:
    """Return first-run GUI parameter defaults for known local models."""
    if _is_huihui_qwen36_27b_abliterated(model, metadata):
        return {
            "specType": "auto",
            "specDraftModel": "",
        }

    if _is_qwen3_coder_next(model, metadata):
        return {
            "server": str(NO_GRAPHS_SERVER_CANDIDATE),
            "context": "",
            "device": "CUDA0",
            "splitMode": "none",
            "mainGpu": "0",
            "gpuLayers": "28",
            "cacheTypeK": "q8_0",
            "cacheTypeV": "q8_0",
            "flashAttention": "on",
            "batch": "2048",
            "ubatch": "512",
            "parallel": "1",
            "fit": "on",
            "fitTarget": "1024",
            "fitCtx": "131072",
            "ctxCheckpoints": "",
            "checkpointEveryNTokens": "",
            "noHost": "on",
            "kvOffload": "on",
            "opOffload": "on",
            "repack": "on",
            "mmap": "off",
            "mmproj": "",
            "mmprojAuto": "off",
            "backendSampling": "default",
            "uiMcpProxy": "default",
            "extraArgs": "",
            "specType": "none",
            "specDraftModel": "",
            "reasoning": "off",
            "reasoningFormat": "auto",
            "reasoningBudget": "1024",
            "timeout": "1800",
            "temperature": "0.7",
            "topP": "0.95",
            "minP": "0.05",
            "repeatPenalty": "1.05",
            "samplers": "top_k;top_p;min_p;temperature",
            "cachePrompt": "on",
            "threads": "",
            "threadsBatch": "",
            "keep": "",
            "seed": "",
            "tensorSplit": "",
            "numa": "default",
            "mlock": "default",
            "swaFull": "default",
            "cpuMoe": "default",
            "nCpuMoe": "",
            "defragThold": "",
            "specDraftMax": "3",
            "specDraftMin": "0",
            "specDraftPSplit": "",
            "specDraftPMin": "",
            "specDraftCacheK": "default",
            "specDraftCacheV": "default",
            "specDraftGpuLayers": "",
            "specDraftDevice": "",
            "slots": "default",
            "metrics": "default",
            "props": "default",
            "jinja": "default",
            "skipChatParsing": "default",
            "prefillAssistant": "default",
            "embedding": "default",
            "rerank": "default",
            "pooling": "default",
            "ignoreEos": "default",
            "cacheReuse": "",
            "threadsHttp": "",
            "alias": "",
            "tags": "",
            "apiKey": "",
            "apiKeyFile": "",
            "tools": "",
            "staticPath": "",
            "apiPrefix": "",
            "mediaPath": "",
            "slotSavePath": "",
            "chatTemplate": "",
            "chatTemplateFile": "",
            "chatTemplateKwargs": "",
            "imageMinTokens": "",
            "imageMaxTokens": "",
            "mmprojOffload": "CPU",
            "mtpCache": str(DEFAULT_MTP_CACHE),
            "noUi": False,
            "predict": "",
            "topK": "",
            "typicalP": "",
            "repeatLastN": "",
            "presencePenalty": "",
            "frequencyPenalty": "",
            "dynatempRange": "",
            "dynatempExp": "",
            "dryMultiplier": "",
            "dryBase": "",
            "dryAllowedLength": "",
            "dryPenaltyLastN": "",
            "reasoningBudgetMessage": "",
        }

    if _is_gemma4_31b_it(model, metadata):
        return {
            "context": "64K",
            "device": "CUDA0",
            "splitMode": "none",
            "mainGpu": "0",
            "gpuLayers": "all",
            "cacheTypeK": "q8_0",
            "cacheTypeV": "q8_0",
            "flashAttention": "on",
            "batch": "2048",
            "ubatch": "512",
            "parallel": "1",
            "fit": "on",
            "fitTarget": "1024",
            "fitCtx": "65536",
            "ctxCheckpoints": "",
            "checkpointEveryNTokens": "",
            "noHost": "on",
            "kvOffload": "on",
            "opOffload": "on",
            "repack": "on",
            "mmap": "off",
            "mmproj": "",
            "mmprojAuto": "off",
            "backendSampling": "default",
            "uiMcpProxy": "default",
            "extraArgs": "",
            "specType": "auto",
            "specDraftModel": "",
            "specDraftGpuLayers": "all",
            "specDraftDevice": "CUDA0",
            "specDraftCacheK": "q8_0",
            "specDraftCacheV": "q8_0",
            "reasoning": "off",
            "reasoningFormat": "auto",
            "reasoningBudget": "1024",
            "timeout": "1800",
            "temperature": "0.7",
            "topP": "0.95",
            "minP": "0.05",
            "repeatPenalty": "1.05",
            "samplers": "top_k;top_p;min_p;temperature",
            "cachePrompt": "on",
        }

    if _is_qwen38_27b(model, metadata):
        return {
            "server": str(SCRIPT_DIR / "build-qwen38-fa-allquants/bin/llama-server"),
            "context": "192K",
            "device": "CUDA0",
            "splitMode": "none",
            "mainGpu": "0",
            "gpuLayers": "all",
            "cacheTypeK": "q8_0",
            "cacheTypeV": "q8_0",
            "flashAttention": "on",
            "batch": "1024",
            "ubatch": "512",
            "parallel": "1",
            "fit": "default",
            "fitTarget": "1024",
            "fitCtx": "131072",
            "ctxCheckpoints": "48",
            "checkpointEveryNTokens": "16384",
            "noHost": "default",
            "kvOffload": "default",
            "opOffload": "default",
            "repack": "off",
            "mmap": "default",
            "mmproj": "",
            "mmprojAuto": "on",
            "backendSampling": "off",
            "uiMcpProxy": "default",
            "extraArgs": "",
            "specType": "auto",
            "specDraftModel": "",
            "specDraftGpuLayers": "all",
            "specDraftDevice": "CUDA0",
            "specDraftCacheK": "q8_0",
            "specDraftCacheV": "q8_0",
            "reasoning": "auto",
            "reasoningFormat": "auto",
            "reasoningBudget": "-1",
            "reasoningPreserve": "off",
            "timeout": "1800",
            "temperature": "0.7",
            "topP": "0.80",
            "minP": "0.0",
            "topK": "20",
            "repeatPenalty": "1.0",
            "samplers": "top_k;top_p;min_p;temperature",
            "cachePrompt": "on",
        }

    if not _is_qwythos_9b_claude_mythos_1m(model, metadata):
        return {}

    return {
        "context": "512K",
        "device": "CUDA0",
        "splitMode": "none",
        "mainGpu": "0",
        "gpuLayers": "all",
        "cacheTypeK": "q8_0",
        "cacheTypeV": "q5_1",
        "flashAttention": "on",
        "batch": "2048",
        "ubatch": "512",
        "parallel": "1",
        "fit": "on",
        "fitTarget": "1024",
        "fitCtx": "131072",
        "ctxCheckpoints": "",
        "checkpointEveryNTokens": "",
        "noHost": "on",
        "kvOffload": "on",
        "opOffload": "on",
        "repack": "on",
        "mmap": "off",
        "mmproj": "",
        "mmprojAuto": "off",
        "backendSampling": "default",
        "uiMcpProxy": "default",
        "extraArgs": "",
        "specType": "none",
        "specDraftModel": "",
        "reasoning": "off",
        "reasoningFormat": "auto",
        "reasoningBudget": "1024",
        "timeout": "1800",
        "temperature": "0.7",
        "topP": "0.95",
        "minP": "0.05",
        "repeatPenalty": "1.05",
        "samplers": "top_k;top_p;min_p;temperature",
        "cachePrompt": "on",
    }


def parse_context_choice(ctx_raw: str) -> str:
    """Convert UI context presets like 128K to llama-server integer values."""
    value = ctx_raw.strip()
    if not value:
        return ""

    upper_value = value.upper()
    if upper_value.endswith("K") and upper_value[:-1].isdigit():
        return str(int(upper_value[:-1]) * 1024)

    return value


def normalize_image_model_device(value: str) -> str:
    """Return the image model device value, accepting legacy offload settings."""
    normalized = value.strip().lower()
    if normalized in {"gpu", "on", "true", "1"}:
        return "GPU"
    return "CPU"


def _script_relative_path(path: Path) -> Path:
    return path if path.is_absolute() else SCRIPT_DIR / path


@dataclass(frozen=True)
class LaunchDefaults:
    """Process-level defaults derived from model metadata."""

    env: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _model_architecture(metadata: dict[str, object]) -> str:
    value = metadata.get("general.architecture")
    return value if isinstance(value, str) else ""


def _is_qwen35moe_long_context(metadata: dict[str, object]) -> bool:
    arch = _model_architecture(metadata)
    train_ctx = model_context_length(metadata) or 0
    return arch == "qwen35moe" and train_ctx >= QWEN35MOE_STABLE_AUTO_CTX


def _has_mtp_metadata(metadata: dict[str, object]) -> bool:
    for key, value in metadata.items():
        key_lower = key.lower()
        if key_lower.endswith(".nextn_predict_layers") and isinstance(value, int) and value > 0:
            return True
        if "mtp" in key_lower:
            return True
    return False


def _needs_no_graphs_cuda_server(metadata: dict[str, object]) -> bool:
    arch = _model_architecture(metadata)
    if _is_qwen35moe_long_context(metadata):
        return True
    if arch == "qwen3next":
        return True
    return arch == "qwen35" and _has_mtp_metadata(metadata)


def resolve_auto_context(ctx_raw: str, metadata: dict[str, object]) -> tuple[int, list[str]]:
    """Resolve the context size and notes from user input plus GGUF metadata."""
    if ctx_raw:
        return int(ctx_raw), []

    train_ctx = model_context_length(metadata)
    ctx = min(DEFAULT_CTX, train_ctx) if train_ctx is not None else DEFAULT_CTX
    notes: list[str] = []

    if _is_qwen35moe_long_context(metadata) and ctx > QWEN35MOE_STABLE_AUTO_CTX:
        ctx = QWEN35MOE_STABLE_AUTO_CTX
        notes.append(
            f"Auto context capped to {ctx} for qwen35moe long-context stability; "
            "set Context manually to override."
        )

    return ctx, notes


def build_model_launch_defaults(metadata: dict[str, object], ctx: int) -> LaunchDefaults:
    """Return environment overrides and notes for safer model launch defaults."""
    env: dict[str, str] = {}
    notes: list[str] = []

    return LaunchDefaults(env=env, notes=notes)


def _server_build_dir(server: Path) -> Path | None:
    if server.name != "llama-server" or server.parent.name != "bin":
        return None
    return server.parent.parent


def _cmake_cache_bool(build_dir: Path, key: str) -> bool | None:
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.is_file():
        return None
    prefix = f"{key}:BOOL="
    try:
        for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(prefix):
                return line[len(prefix):].strip().upper() in {"1", "ON", "TRUE", "YES"}
    except OSError:
        return None
    return None


def server_cuda_graphs_enabled(server: Path) -> bool | None:
    """Return whether the server build has CUDA graphs enabled, when discoverable."""
    build_dir = _server_build_dir(server)
    if build_dir is None:
        return None
    return _cmake_cache_bool(build_dir, "GGML_CUDA_GRAPHS")


def resolve_server_for_model(server_raw: str, metadata: dict[str, object]) -> tuple[str, list[str]]:
    """Switch to a safer server binary for model/backend combinations known to crash."""
    server = _script_relative_path(Path(server_raw))
    no_graphs_server = _script_relative_path(NO_GRAPHS_SERVER_CANDIDATE)
    notes: list[str] = []

    # qwen3next: use no-graphs server for stability with MoE architecture
    arch = _model_architecture(metadata)
    if arch == "qwen3next":
        if server.resolve() == no_graphs_server.resolve():
            return server_raw, notes
        if not no_graphs_server.is_file():
            notes.append(
                f"qwen3next recommends {no_graphs_server}; "
                "keeping selected server as no-graphs binary not found."
            )
            return server_raw, notes
        notes.append(
            f"Switched server to {no_graphs_server} for qwen3next MoE architecture."
        )
        return str(no_graphs_server), notes

    # Only qwen35moe long-context genuinely needs a no-graphs binary. Plain qwen35
    # MTP works fine on a graphs build once --backend-sampling is stripped (handled
    # in _build_command), so don't force-swap the server for it -- that would throw
    # away CUDA graphs acceleration unnecessarily.
    if not _is_qwen35moe_long_context(metadata):
        return server_raw, notes

    if server.resolve() == no_graphs_server.resolve():
        return server_raw, notes

    if not no_graphs_server.is_file():
        notes.append(
            "qwen35moe long-context CUDA graphs workaround needs "
            f"{no_graphs_server}; keeping selected server."
        )
        return server_raw, notes

    cuda_graphs_enabled = server_cuda_graphs_enabled(server)
    if cuda_graphs_enabled is True:
        notes.append(
            f"Switched server to {no_graphs_server} for qwen35moe long-context; "
            "the selected CUDA build has GGML_CUDA_GRAPHS=ON and crashes during warmup decode."
        )
        return str(no_graphs_server), notes

    return server_raw, notes


def resolve_display_server_for_model(server_raw: str, metadata: dict[str, object]) -> tuple[str, list[str]]:
    """Resolve the server path that should be shown in the GUI for the selected model."""
    server, notes = resolve_server_for_model(server_raw, metadata)
    if notes or _needs_no_graphs_cuda_server(metadata):
        return server, notes

    no_graphs_server = _script_relative_path(NO_GRAPHS_SERVER_CANDIDATE)
    current_server = _script_relative_path(Path(server_raw))
    if current_server.resolve() != no_graphs_server.resolve():
        return server, notes

    default_server = resolve_server_path()
    if default_server.resolve() == no_graphs_server.resolve():
        return server, notes
    if not default_server.is_file():
        return server, notes

    notes.append(f"Restored server to {default_server} for non-qwen35moe model.")
    return str(default_server), notes


# ── Font helpers ──────────────────────────────────────────────────────


def _pick_font(family_candidates: tuple[str, ...], fallback: str) -> str:
    """Return the first available font family."""
    available = set(QFontDatabase.families())
    for name in family_candidates:
        if name in available:
            return name
    return fallback


_UI_FONT_CANDIDATES = (
    "Noto Sans",
    "Lato",
    "DejaVu Sans",
    "Liberation Sans",
    "Segoe UI",
    "Helvetica Neue",
    "Arial",
)
_LOG_FONT_CANDIDATES = (
    "Noto Mono",
    "Liberation Mono",
    "DejaVu Sans Mono",
    "Cascadia Code",
    "Consolas",
    "Menlo",
)

_ui_family: str = ""
_log_family: str = ""


def _init_fonts() -> None:
    global _ui_family, _log_family
    _ui_family = _pick_font(_UI_FONT_CANDIDATES, "Sans Serif")
    _log_family = _pick_font(_LOG_FONT_CANDIDATES, "Monospace")


def ui_font(bold: bool = False, size_override: int | None = None) -> QFont:
    f = QFont(_ui_family, size_override or 11)
    f.setBold(bold)
    return f


def log_font() -> QFont:
    return QFont(_log_family, 10)


# ── UI Builder ────────────────────────────────────────────────────────


class NoWheelComboBox(QComboBox):
    """Project-wide combo box: dropdown values must not change on mouse wheel."""

    def wheelEvent(self, e) -> None:
        # Avoid accidental parameter changes while scrolling the settings panel.
        # New dropdowns in this GUI should use this class instead of QComboBox.
        if e is not None:
            e.ignore()


class ClickableButton(QPushButton):
    """QPushButton that shows a hand cursor when enabled and a forbidden cursor when disabled.

    Qt does not render custom cursors on disabled widgets, so we keep the widget
    Qt-enabled and simulate the disabled state: a forbidden cursor plus swallowed
    mouse events so clicked never fires.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._logical_enabled = True
        self._refresh_cursor()

    def setEnabled(self, enabled: bool) -> None:
        self._logical_enabled = enabled
        self._refresh_cursor()

    def setDisabled(self, disabled: bool) -> None:
        self.setEnabled(not disabled)

    def isEnabled(self) -> bool:
        return self._logical_enabled

    def mousePressEvent(self, e) -> None:
        if not self._logical_enabled:
            if e is not None:
                e.ignore()
            return
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if not self._logical_enabled:
            if e is not None:
                e.ignore()
            return
        super().mouseReleaseEvent(e)

    def _refresh_cursor(self) -> None:
        if self._logical_enabled:
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ForbiddenCursor))


def _labeled_input(
    label_text: str,
    default: str = "",
    placeholder: str = "",
    hint: str = "",
    width: int = 20,
) -> tuple[QWidget, QLineEdit]:
    """Return (container_widget, line_edit)."""
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    lbl = QLabel(label_text)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    lbl.setMinimumWidth(120)
    lbl.setFont(ui_font())

    edit = QLineEdit()
    edit.setText(default)
    if placeholder:
        edit.setPlaceholderText(placeholder)
    edit.setFont(ui_font())

    layout.addWidget(lbl)
    layout.addWidget(edit, 1)

    if hint:
        hint_lbl = QLabel(hint)
        hint_lbl.setStyleSheet("color: grey;")
        hint_lbl.setFont(ui_font())
        layout.addWidget(hint_lbl)

    return container, edit


def _labeled_combo(
    label_text: str,
    choices: list[str],
    default: str = "",
    editable: bool = False,
) -> tuple[QWidget, QComboBox]:
    """Return (container_widget, combo_box)."""
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    lbl = QLabel(label_text)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    lbl.setMinimumWidth(120)
    lbl.setFont(ui_font())

    combo = NoWheelComboBox()
    combo.addItems(choices)
    combo.setEditable(editable)
    if default and default in choices:
        combo.setCurrentText(default)
    combo.setFont(ui_font())

    layout.addWidget(lbl)
    layout.addWidget(combo, 1)

    return container, combo


def _browse_row(
    label_text: str,
    default: str,
    is_directory: bool = False,
    title: str = "Select",
    extra_buttons: list[tuple[str, Callable[[], None]]] | None = None,
) -> tuple[QWidget, QLineEdit]:
    """Row with label, text input, Browse button and optional extra buttons."""
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    lbl = QLabel(label_text)
    lbl.setFont(ui_font())
    layout.addWidget(lbl)

    edit = QLineEdit(default)
    edit.setFont(ui_font())
    layout.addWidget(edit, 1)

    def do_browse():
        if is_directory:
            d = QFileDialog.getExistingDirectory(
                None, title, edit.text()
            )
        else:
            d = QFileDialog.getOpenFileName(
                None, title, edit.text()
            )[0]
        if d:
            edit.setText(d)

    browse_btn = ClickableButton("Browse")
    browse_btn.setFont(ui_font(True))
    browse_btn.clicked.connect(do_browse)
    layout.addWidget(browse_btn)

    if extra_buttons:
        for btn_text, callback in extra_buttons:
            btn = ClickableButton(btn_text)
            btn.setFont(ui_font(True))
            btn.clicked.connect(callback)
            layout.addWidget(btn)

    return container, edit


# ── HTTP control API ─────────────────────────────────────────────────


class GuiApiHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for the GUI control API (localhost only)."""

    protocol_version = "HTTP/1.1"

    def _send_json(self, code: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _bridge(self) -> Any:
        return self.server.api_bridge  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path.rstrip("/") or "/"
        bridge = self._bridge()

        if path == "/":
            self._send_json(200, {
                "service": "llama-server-gui",
                "endpoints": [
                    "GET  /status", "GET  /models", "GET  /params",
                    "GET  /log", "POST /start", "POST /stop",
                    "POST /model", "POST /params", "POST /refresh",
                ],
            })
        elif path == "/status":
            self._send_json(200, bridge.api_execute("api_get_status"))
        elif path == "/models":
            self._send_json(200, bridge.api_execute("api_get_models"))
        elif path == "/params":
            self._send_json(200, bridge.api_execute("api_get_params"))
        elif path == "/log":
            self._send_json(200, bridge.api_execute("api_get_log"))
        else:
            self._send_json(404, {"error": f"unknown path: {path}"})

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path.rstrip("/") or "/"
        bridge = self._bridge()

        try:
            body = self._read_body()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_json(400, {"error": f"invalid JSON body: {exc}"})
            return

        if path == "/start":
            self._send_json(200, bridge.api_execute("api_start"))
        elif path == "/stop":
            self._send_json(200, bridge.api_execute("api_stop"))
        elif path == "/refresh":
            self._send_json(200, bridge.api_execute("api_refresh_models"))
        elif path == "/model":
            model = body.get("model")
            if not model:
                self._send_json(400, {"error": "missing 'model' field"})
                return
            self._send_json(200, bridge.api_execute("api_set_model", (model,)))
        elif path == "/params":
            if not isinstance(body, dict):
                self._send_json(400, {"error": "body must be a JSON object of {key: value}"})
                return
            self._send_json(200, bridge.api_execute("api_set_params", (body,)))
        else:
            self._send_json(404, {"error": f"unknown path: {path}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        pass


# ── Main Window ───────────────────────────────────────────────────────


class PresetManagerDialog(QDialog):
    """Manage named parameter presets bound to a single model."""

    def __init__(self, parent: QWidget, model: str, main_window: "MainWindow") -> None:
        super().__init__(parent)
        self._model = model
        self._main = main_window
        self.setWindowTitle("Model Presets")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        stem = main_window._normalize_model_path(model).stem
        info = QLabel(f"Presets for: <b>{stem}</b>")
        info.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(info)

        self.list_widget = QListWidget()
        self.list_widget.setMinimumHeight(180)
        self.list_widget.itemDoubleClicked.connect(self._on_load)
        layout.addWidget(self.list_widget, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        save_btn = QPushButton("Save As\u2026")
        save_btn.clicked.connect(self._on_save_as)
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._on_load)
        rename_btn = QPushButton("Rename\u2026")
        rename_btn.clicked.connect(self._on_rename)
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._on_delete)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        for btn in (save_btn, load_btn, rename_btn, delete_btn):
            btn_row.addWidget(btn)
        btn_row.addStretch(1)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._refresh()

    def _refresh(self) -> None:
        self.list_widget.clear()
        doc = self._main._load_presets_doc()
        presets = self._main._model_presets(doc, self._model)
        for preset in presets:
            name = str(preset.get("name", ""))
            saved_at = str(preset.get("savedAt", ""))
            item = QListWidgetItem(f"{name}  -  {saved_at}" if saved_at else name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.list_widget.addItem(item)

    def _current_name(self) -> str | None:
        item = self.list_widget.currentItem()
        if not item:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _on_save_as(self) -> None:
        default = self._main._recommended_preset_name(self._model)
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:", text=default)
        if not ok:
            return
        if self._main._save_preset_as(name):
            self._refresh()

    def _on_load(self) -> None:
        name = self._current_name()
        if not name:
            return
        if self._main._load_preset(name):
            self.accept()

    def _on_rename(self) -> None:
        old = self._current_name()
        if not old:
            return
        new, ok = QInputDialog.getText(self, "Rename Preset", "New name:", text=old)
        if not ok:
            return
        if self._main._rename_preset(old, new):
            self._refresh()
        else:
            QMessageBox.information(self, "Rename", "Could not rename (name missing or already used).")

    def _on_delete(self) -> None:
        name = self._current_name()
        if not name:
            return
        reply = QMessageBox.question(
            self,
            "Delete preset",
            f"Delete preset \"{name}\"?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes and self._main._delete_preset(name):
            self._refresh()


class MainWindow(QMainWindow):
    """Main GUI application for llama-server."""

    _api_execute_signal = pyqtSignal(str, object)

    def __init__(self):
        super().__init__()
        self._server_process: subprocess.Popen | None = None
        self._log_thread: ServerLogThread | None = None
        self._stop_event = threading.Event()
        self._launch_env_overrides: dict[str, str] = {}
        self._launch_notes: list[str] = []
        self._launch_control_states: list[tuple[QWidget, bool]] | None = None
        self._log_analyzer = LlamaLogAnalyzer()
        self._running_port: str | None = None
        self._startup_last_model = self._load_last_model()
        self._active_model_path: str | None = None
        self._loading_model_settings = False
        self._default_parameter_values: dict[str, str | bool] = {}

        self.setWindowTitle("llama-server Web UI Launcher")
        self.setWindowIcon(QIcon(str(SCRIPT_DIR / ".." / "media" / "llama1-icon-transparent.png")))
        self.resize(960, 860)
        self.setMinimumSize(880, 640)

        self._vram_label: QLabel | None = None
        self._ram_label: QLabel | None = None
        self._status_timer: QTimer | None = None
        self._local_server: QLocalServer | None = None
        self._advanced_widgets: list[QWidget] = []
        self._show_advanced = False
        self._rosetta_thread: RosettaGatewayThread | None = None
        self._api_server: http.server.HTTPServer | None = None
        self._api_result: Any = None
        self._api_result_event = threading.Event()

        self._api_execute_signal.connect(
            self._on_api_execute, Qt.ConnectionType.QueuedConnection
        )

        self._build_ui()
        self._default_parameter_values = self._capture_parameter_settings()
        self._load_parameter_settings()
        self._connect_parameter_autosave()
        self._setup_local_server()
        self._refresh_model_list()
        self._start_status_monitor()
        self._start_api_server()

        settings = QSettings("llamacpp", "llama-server-gui")
        self._show_advanced = settings.value("showAdvanced", False, type=bool)
        self._action_show_advanced.setChecked(self._show_advanced)
        self._apply_advanced_visibility()

    # ───────────────────────────── UI construction ────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # ---- Tab widget ----
        self._tab_widget = QTabWidget()
        outer_layout.addWidget(self._tab_widget)

        # Tab 1: Server
        server_tab = QWidget()
        main_layout = QVBoxLayout(server_tab)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(4)
        self._tab_widget.addTab(server_tab, "Server")

        # Tab 2: Rosetta Gateway
        rosetta_tab = QWidget()
        self._rosetta_layout = QVBoxLayout(rosetta_tab)
        self._rosetta_layout.setContentsMargins(8, 8, 8, 8)
        self._rosetta_layout.setSpacing(6)
        self._tab_widget.addTab(rosetta_tab, "API")

        # ---- Menu bar ----
        menubar = cast(QMainWindow, self).menuBar()
        view_menu = menubar.addMenu("&View")
        self._action_show_advanced = QAction("Show &Advanced Settings", self)
        self._action_show_advanced.setCheckable(True)
        self._action_show_advanced.setChecked(self._show_advanced)
        self._action_show_advanced.triggered.connect(self._toggle_advanced_settings)
        view_menu.addAction(self._action_show_advanced)

        # ---- Row 1: Server binary ----
        self._srv_row, self.server_edit = _browse_row(
            "Server:",
            str(resolve_server_path()),
            is_directory=False,
            title="Select llama-server binary",
        )
        main_layout.addWidget(self._srv_row)

        # ---- Row 2: Model directory ----
        self._dir_row, self.model_dir_edit = _browse_row(
            "Model Dir:",
            str(resolve_initial_model_dir(DEFAULT_MODEL_DIR, self._startup_last_model)),
            is_directory=True,
            title="Select model directory",
            extra_buttons=[("Refresh", self._refresh_model_list)],
        )
        main_layout.addWidget(self._dir_row)

        # ---- Row 3: Model selector + Predict ----
        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(4)
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = NoWheelComboBox()
        self.model_combo.setFont(ui_font())
        self.model_combo.setMinimumHeight(28)
        model_layout.addWidget(self.model_combo, 1)
        self.mtp_badge_label = QLabel("")
        self.mtp_badge_label.setFont(ui_font(True))
        self.mtp_badge_label.setMinimumHeight(24)
        self.mtp_badge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        model_layout.addWidget(self.mtp_badge_label)

        self.presets_btn = QPushButton("Presets\u2026")
        self.presets_btn.setFont(ui_font())
        self.presets_btn.setMinimumHeight(28)
        self.presets_btn.setToolTip(
            "Save the current parameter values as a named preset, or load a saved preset."
        )
        self.presets_btn.clicked.connect(self._open_preset_dialog)
        model_layout.addWidget(self.presets_btn)

        # Predict (max output tokens) - placed next to model selector for visibility
        predict_lbl = QLabel("Predict:")
        predict_lbl.setFont(ui_font(True))
        predict_lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        model_layout.addWidget(predict_lbl)
        self.predict_edit = QLineEdit("")
        self.predict_edit.setFont(ui_font())
        self.predict_edit.setPlaceholderText("default: unlimited")
        self.predict_edit.setMinimumWidth(100)
        self.predict_edit.setToolTip(
            "Maximum tokens to generate. "
            "Default: -1 (unlimited, until context is filled). "
            "Set a value to limit output length."
        )
        model_layout.addWidget(self.predict_edit)

        main_layout.addWidget(model_row)

        # ---- Row 4: selected llama-server build info ----
        self.server_info_label = QLabel("")
        self.server_info_label.setFont(ui_font())
        self.server_info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.server_info_label.setWordWrap(True)
        main_layout.addWidget(self.server_info_label)
        self.server_edit.textChanged.connect(self._update_server_info)
        self._update_server_info()

        # ---- Parameters group (scrollable) ----
        params_group = QGroupBox("Parameters")
        params_group.setFont(ui_font(True))
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(2)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._params_widget = QWidget()
        self._params_layout = QVBoxLayout(self._params_widget)
        self._params_layout.setContentsMargins(4, 2, 4, 4)
        self._params_layout.setSpacing(2)
        scroll.setWidget(self._params_widget)
        params_layout.addWidget(scroll)
        main_layout.addWidget(params_group, 1)

        # ---- Build parameter rows ----
        default_threads = str(max((os.cpu_count() or 8) // 2, 1))

        def _register(widget: QWidget, advanced: bool) -> None:
            if advanced:
                self._advanced_widgets.append(widget)

        def _add_input(label, default="", hint="", advanced=False):
            widget, edit = _labeled_input(label, default, hint=hint)
            self._params_layout.addWidget(widget)
            _register(widget, advanced)
            return edit

        def _add_combo(label, choices, default="", advanced=False, hint=""):
            widget, combo = _labeled_combo(label, choices, default)
            if hint:
                combo.setToolTip(hint)
            self._params_layout.addWidget(widget)
            _register(widget, advanced)
            return combo

        # ── Two-in-one-row helpers ──────────────────────────────────

        def _add_input_row(left_label: str, left_default: str,
                           right_label: str, right_default: str,
                           left_hint: str = "", right_hint: str = "",
                           advanced: bool = False):
            """Return (left_edit, right_edit) placed side by side."""
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(8)

            left_cont, left_e = _labeled_input(left_label, left_default, hint=left_hint)
            right_cont, right_e = _labeled_input(right_label, right_default, hint=right_hint)
            rl.addWidget(left_cont)
            rl.addWidget(right_cont)
            self._params_layout.addWidget(row)
            _register(row, advanced)
            return left_e, right_e

        def _add_combo_row(left_label: str, left_choices: list[str], left_default: str,
                           right_label: str, right_choices: list[str], right_default: str,
                           advanced: bool = False):
            """Return (left_combo, right_combo) placed side by side."""
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(8)

            left_cont, left_c = _labeled_combo(left_label, left_choices, left_default)
            right_cont, right_c = _labeled_combo(right_label, right_choices, right_default)
            rl.addWidget(left_cont)
            rl.addWidget(right_cont)
            self._params_layout.addWidget(row)
            _register(row, advanced)
            return left_c, right_c

        def _add_input_combo_row(left_label: str, left_default: str,
                                 right_label: str, right_choices: list[str], right_default: str,
                                 advanced: bool = False):
            """Return (left_edit, right_combo) placed side by side."""
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(8)

            left_cont, left_e = _labeled_input(left_label, left_default)
            right_cont, right_c = _labeled_combo(right_label, right_choices, right_default)
            rl.addWidget(left_cont)
            rl.addWidget(right_cont)
            self._params_layout.addWidget(row)
            _register(row, advanced)
            return left_e, right_c

        # ── Packed rows ─────────────────────────────────────────────

        # Row: Host + Port
        self.host_edit, self.port_edit = _add_input_row(
            "Host", "0.0.0.0", "Port", "8080")

        self.ctx_combo = _add_combo("Context", CONTEXT_CHOICES, "")
        self.ctx_combo.setEditable(True)
        self.gpu_layers_edit, self.mmproj_offload_combo = _add_input_combo_row(
            "GPU Layers", "", "Image Model Device", IMAGE_MODEL_DEVICE_CHOICES, "CPU",
            advanced=True)
        self.mtp_cache_edit = _add_input("MTP Cache", str(DEFAULT_MTP_CACHE), advanced=True)
        self.ctx_checkpoints_edit, self.checkpoint_every_nt_edit = _add_input_row(
            "Ctx Checkpoints", "", "Checkpoint Min Step", "", advanced=True)
        self.checkpoint_every_nt_edit.setToolTip(
            "Minimum spacing between context checkpoints in tokens. "
            "Passed as --checkpoint-min-step when the selected llama-server supports it."
        )

        # Row: Threads + Batch
        self.threads_edit, self.batch_edit = _add_input_row(
            "Threads", default_threads, "Batch", "2048")

        self.threads_batch_edit, self.ubatch_edit = _add_input_row(
            "Threads Batch", "", "UBatch", "", advanced=True)

        self.keep_edit = _add_input("Keep", "", advanced=True)

        # Row: KV Cache K + KV Cache V
        self.cache_type_k_combo, self.cache_type_v_combo = _add_combo_row(
            "KV Cache K", CACHE_TYPE_CHOICES, "q4_0",
            "KV Cache V", CACHE_TYPE_CHOICES, "q4_0")

        # Row: Flash Attention + Parallel
        self.parallel_edit, self.flash_attn_combo = _add_input_combo_row(
            "Parallel", "1",
            "Flash Attn", FLASH_ATTN_CHOICES, "on")

        self.device_edit, self.split_mode_combo = _add_input_combo_row(
            "Device", "",
            "Split Mode", ["default", "none", "layer", "row", "tensor"], "default",
            advanced=True)

        self.tensor_split_edit, self.main_gpu_edit = _add_input_row(
            "Tensor Split", "", "Main GPU", "", advanced=True)

        self.numa_combo, self.fit_combo = _add_combo_row(
            "NUMA", ["default", "distribute", "isolate", "numactl"], "default",
            "Fit VRAM", ["default", "on", "off"], "default", advanced=True)

        self.fit_target_edit, self.fit_ctx_edit = _add_input_row(
            "Fit Target", "", "Fit Min Ctx", "", advanced=True)

        self.mlock_combo, self.mmap_combo = _add_combo_row(
            "Mlock", DEFAULT_BOOL_CHOICES, "default",
            "Mmap", DEFAULT_ON_OFF_CHOICES, "default", advanced=True)

        self.kv_offload_combo, self.swa_full_combo = _add_combo_row(
            "KV Offload", DEFAULT_ON_OFF_CHOICES, "default",
            "SWA Full", DEFAULT_BOOL_CHOICES, "default", advanced=True)

        self.op_offload_combo, self.repack_combo = _add_combo_row(
            "Op Offload", DEFAULT_ON_OFF_CHOICES, "default",
            "Repack", DEFAULT_ON_OFF_CHOICES, "default", advanced=True)

        self.no_host_combo, self.cpu_moe_combo = _add_combo_row(
            "No Host", DEFAULT_BOOL_CHOICES, "default",
            "CPU MoE", DEFAULT_BOOL_CHOICES, "default", advanced=True)

        self.n_cpu_moe_edit, self.defrag_thold_edit = _add_input_row(
            "N CPU MoE", "", "Defrag Thold", "", advanced=True)

        # Row: Spec Type (combo) + Spec Draft N Max
        self.spec_type_combo, self.spec_draft_max_edit = _add_combo_row(
            "Spec Type", ["auto", "draft-mtp", "none", "draft-simple", "draft-eagle3", "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache"], "auto",
            "Draft N Max", ["0", "1", "2", "3", "4", "5", "6", "7", "8"], "3")

        # Row: Spec Draft N Min + Reasoning (combo)
        self.spec_draft_min_edit, self.reasoning_combo = _add_input_combo_row(
            "Draft N Min", "0",
            "Reasoning", ["off", "on", "auto"], "off")

        self.spec_draft_p_split_edit, self.spec_draft_p_min_edit = _add_input_row(
            "Draft P Split", "", "Draft P Min", "", advanced=True)

        self.spec_draft_cache_k_combo, self.spec_draft_cache_v_combo = _add_combo_row(
            "Draft Cache K", ["default", *CACHE_TYPE_CHOICES], "default",
            "Draft Cache V", ["default", *CACHE_TYPE_CHOICES], "default", advanced=True)

        self.spec_draft_ngl_edit, self.spec_draft_device_edit = _add_input_row(
            "Draft GPU Layers", "", "Draft Device", "", advanced=True)

        self.spec_draft_model_edit = _add_input("Draft Model", "", "(optional)", advanced=True)

        # Row: Reasoning Budget + No UI checkbox

        budget_row = QWidget()
        budget_rl = QHBoxLayout(budget_row)
        budget_rl.setContentsMargins(0, 0, 0, 0)
        budget_rl.setSpacing(8)
        budget_container, self.reasoning_budget_edit = _labeled_input("Reasoning Budget", "1024")
        budget_rl.addWidget(budget_container)
        self.no_ui_check = QCheckBox("Disable Web UI (API only)")
        self.no_ui_check.setFont(ui_font())
        budget_rl.addWidget(self.no_ui_check)
        self._params_layout.addWidget(budget_row)

        self.seed_edit, self.temp_edit = _add_input_row(
            "Seed", "", "Temperature", "")

        self.top_k_edit, self.top_p_edit = _add_input_row(
            "Top K", "", "Top P", "", advanced=True)

        self.min_p_edit, self.typical_p_edit = _add_input_row(
            "Min P", "", "Typical P", "", advanced=True)

        self.repeat_last_n_edit, self.repeat_penalty_edit = _add_input_row(
            "Repeat Last N", "", "Repeat Penalty", "", advanced=True)

        self.presence_penalty_edit, self.frequency_penalty_edit = _add_input_row(
            "Presence Penalty", "", "Frequency Penalty", "", advanced=True)

        self.dynatemp_range_edit, self.dynatemp_exp_edit = _add_input_row(
            "Dynatemp Range", "", "Dynatemp Exp", "", advanced=True)

        self.dry_multiplier_edit, self.dry_base_edit = _add_input_row(
            "DRY Multiplier", "", "DRY Base", "", advanced=True)

        self.dry_allowed_length_edit, self.dry_penalty_last_n_edit = _add_input_row(
            "DRY Allowed Len", "", "DRY Penalty N", "", advanced=True)

        self.samplers_edit, self.backend_sampling_combo = _add_input_combo_row(
            "Samplers", "",
            "Backend Sampling", DEFAULT_BOOL_CHOICES, "default", advanced=True)

        self.ignore_eos_combo, self.cache_prompt_combo = _add_combo_row(
            "Ignore EOS", DEFAULT_BOOL_CHOICES, "default",
            "Cache Prompt", DEFAULT_ON_OFF_CHOICES, "default", advanced=True)

        self.cache_reuse_edit, self.threads_http_edit = _add_input_row(
            "Cache Reuse", "", "HTTP Threads", "", advanced=True)

        self.timeout_edit, self.alias_edit = _add_input_row(
            "Timeout", "", "Alias", "", advanced=True)

        self.tags_edit, self.api_key_edit = _add_input_row(
            "Tags", "", "API Key", "", advanced=True)

        self.api_key_file_edit, self.tools_edit = _add_input_row(
            "API Key File", "", "Tools", "", advanced=True)

        self.metrics_combo, self.props_combo = _add_combo_row(
            "Metrics", DEFAULT_BOOL_CHOICES, "default",
            "Props Endpoint", DEFAULT_BOOL_CHOICES, "default", advanced=True)

        self.slots_combo, self.ui_mcp_proxy_combo = _add_combo_row(
            "Slots Endpoint", DEFAULT_ON_OFF_CHOICES, "default",
            "UI MCP Proxy", DEFAULT_ON_OFF_CHOICES, "default", advanced=True)

        self.static_path_edit, self.api_prefix_edit = _add_input_row(
            "Static Path", "", "API Prefix", "", advanced=True)

        self.media_path_edit, self.slot_save_path_edit = _add_input_row(
            "Media Path", "", "Slot Save Path", "", advanced=True)

        self.jinja_combo, self.skip_chat_parsing_combo = _add_combo_row(
            "Jinja", DEFAULT_ON_OFF_CHOICES, "default",
            "Skip Chat Parse", DEFAULT_ON_OFF_CHOICES, "default", advanced=True)

        self.prefill_assistant_combo, self.mmproj_auto_combo = _add_combo_row(
            "Prefill Assistant", DEFAULT_ON_OFF_CHOICES, "default",
            "MMProj Auto", DEFAULT_ON_OFF_CHOICES, "default", advanced=True)

        self.embedding_combo = _add_combo("Embeddings", DEFAULT_BOOL_CHOICES, "default", advanced=True)

        self.rerank_combo, self.pooling_combo = _add_combo_row(
            "Rerank", DEFAULT_BOOL_CHOICES, "default",
            "Pooling", ["default", "none", "mean", "cls", "last", "rank"], "default", advanced=True)

        # ── Full-width rows (long labels / hints) ────────────────────

        self.reasoning_format_combo = _add_combo("Reasoning Format", ["auto", "none", "deepseek", "deepseek-legacy"], "auto")
        self.reasoning_preserve_combo = _add_combo("Reasoning Preserve", ["default", "on", "off"], "default",
            advanced=True, hint="Off strips thinking from old messages (recommended for agent tools)")
        self.reasoning_budget_msg_edit = _add_input("Reasoning Budget Message", "", "(optional)", advanced=True)
        self.chat_template_edit = _add_input("Chat Template", "", "(built-in name or Jinja)", advanced=True)
        self.chat_template_file_edit = _add_input("Chat Template File", "", "(optional)", advanced=True)
        self.chat_template_preset_combo = _add_combo(
            "Template Preset", ["default", "qwen-fixed", "none"], "default", advanced=True)
        self.chat_template_preset_combo.currentTextChanged.connect(self._on_template_preset_changed)
        self.chat_template_kwargs_edit = _add_input("Template Kwargs", "", "(JSON object)", advanced=True)
        self.mmproj_edit = _add_input("MMProj", "", "(optional)")
        self.image_min_tokens_edit = _add_input("Image Min Tokens", "", "(optional)", advanced=True)
        self.image_max_tokens_edit = _add_input("Image Max Tokens", "", "(optional)", advanced=True)
        self.extra_args_edit = _add_input("Extra Args", "", "(advanced; shell-style)")
        self._params_layout.addStretch()

        # ---- Buttons row ----
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.start_btn = ClickableButton("\u25b6  Start")
        self.start_btn.setFont(ui_font(True))
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; padding: 6px 16px; }"
        )
        self.start_btn.setMinimumHeight(32)
        self.start_btn.clicked.connect(self._start_server)

        self.stop_btn = ClickableButton("\u23f9  Stop")
        self.stop_btn.setFont(ui_font(True))
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; padding: 6px 16px; }"
        )
        self.stop_btn.setMinimumHeight(32)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_server)

        open_browser_btn = ClickableButton("Open Browser")
        open_browser_btn.setFont(ui_font(True))
        open_browser_btn.setMinimumHeight(32)
        open_browser_btn.clicked.connect(self._open_browser)

        # --- New Buttons ---
        self.btn_always_on_top = ClickableButton("Always on Top")
        self.btn_always_on_top.setFont(ui_font(True))
        self.btn_always_on_top.setMinimumHeight(32)
        self.btn_always_on_top.clicked.connect(self._toggle_always_on_top)
        self.btn_always_on_top.setStyleSheet(
            "QPushButton { padding: 6px 16px; }"
        )

        self.btn_minimize_to_tray = ClickableButton("Min to Tray")
        self.btn_minimize_to_tray.setFont(ui_font(True))
        self.btn_minimize_to_tray.setMinimumHeight(32)
        self.btn_minimize_to_tray.clicked.connect(self._minimize_to_tray)
        self.btn_minimize_to_tray.setStyleSheet(
            "QPushButton { padding: 6px 16px; }"
        )
        # --- End New Buttons ---


        clear_log_btn = ClickableButton("Clear Log")
        clear_log_btn.setFont(ui_font(True))
        clear_log_btn.setMinimumHeight(32)
        clear_log_btn.clicked.connect(self._clear_log)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(open_browser_btn)
        
        # Add new buttons between Open Browser and Clear Log
        btn_row.addWidget(self.btn_always_on_top)
        btn_row.addWidget(self.btn_minimize_to_tray)
        
        btn_row.addStretch()
        btn_row.addWidget(clear_log_btn)
        main_layout.addLayout(btn_row)

        # ---- API tab content ----
        # Model name
        self._rosetta_layout.addWidget(self._make_copy_row(
            "Model:",
            "{model}",
            "model_name",
        ))

        # Chat Completions API (llama-server native)
        self._rosetta_layout.addWidget(self._make_copy_row(
            "Chat API:",
            "http://127.0.0.1:{llama_port}/v1/chat/completions",
            "chat_api",
        ))

        # Responses API (via gateway)
        self._rosetta_layout.addWidget(self._make_copy_row(
            "Responses API:",
            "http://127.0.0.1:{rosetta_port}/v1/responses",
            "responses_api",
        ))

        # Messages API (via gateway)
        self._rosetta_layout.addWidget(self._make_copy_row(
            "Messages API:",
            "http://127.0.0.1:{rosetta_port}/v1/messages",
            "messages_api",
        ))

        self._rosetta_layout.addStretch()

        # ---- Status bar ----
        status_bar = cast(QStatusBar, self.statusBar())
        status_bar.showMessage("Ready")

        # Persistent system resource labels on the right side of the status bar
        self._vram_label = QLabel("VRAM: ---")
        self._vram_label.setFont(log_font())
        self._ram_label = QLabel("RAM: ---")
        self._ram_label.setFont(log_font())
        status_bar.addPermanentWidget(self._vram_label)
        status_bar.addPermanentWidget(self._ram_label)

        # ---- Log area ----
        log_group = QGroupBox("Log")
        log_group.setFont(ui_font(True))
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(4, 4, 4, 4)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(log_font())
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group, 0)
        self.model_combo.currentIndexChanged.connect(self._on_model_selection_changed)
        self._launch_controls = [
            self._srv_row,
            self._dir_row,
            self.model_combo,
            self._params_widget,
        ]

    # ───────────────────────────── callbacks ──────────────────────────

    def _last_model_key(self) -> str:
        return "lastModel"

    def _normalize_model_path(self, model: str | Path) -> Path:
        model_path = Path(model).expanduser()
        try:
            return model_path.resolve(strict=False)
        except TypeError:
            return model_path.resolve()

    def _model_settings_prefix(self, model: str | Path) -> str:
        model_path = self._normalize_model_path(model)
        digest = hashlib.sha1(str(model_path).encode("utf-8")).hexdigest()[:16]
        stem = model_path.stem.lower()
        stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
        if not stem:
            stem = "model"
        return f"models/{stem}-{digest}"

    def _model_settings(self, settings: Any | None, model: str | Path) -> tuple[Any, str]:
        if settings is None:
            settings = QSettings("llamacpp", "llama-server-gui")
        return settings, self._model_settings_prefix(model)

    def _capture_parameter_settings(self) -> dict[str, str | bool]:
        captured: dict[str, str | bool] = {}
        for key, control, kind in self._parameter_settings():
            if kind == "line":
                captured[key] = control.text()
            elif kind == "combo":
                captured[key] = control.currentText()
            elif kind == "check":
                captured[key] = control.isChecked()
        return captured

    def _connect_parameter_autosave(self) -> None:
        for _key, control, kind in self._parameter_settings():
            if kind == "line":
                control.textChanged.connect(self._autosave_model_settings)
            elif kind == "combo":
                control.currentTextChanged.connect(self._autosave_model_settings)
            elif kind == "check":
                control.toggled.connect(self._autosave_model_settings)

    def _autosave_model_settings(self, *_args) -> None:
        if self._loading_model_settings:
            return
        self._save_model_settings()

    def _parameter_settings(self) -> list[tuple[str, Any, str]]:
        """Return runtime controls that should persist between GUI launches."""
        return [
            ("server", self.server_edit, "line"),
            ("modelDir", self.model_dir_edit, "line"),
            ("host", self.host_edit, "line"),
            ("port", self.port_edit, "line"),
            ("threads", self.threads_edit, "line"),
            ("threadsBatch", self.threads_batch_edit, "line"),
            ("batch", self.batch_edit, "line"),
            ("ubatch", self.ubatch_edit, "line"),
            ("predict", self.predict_edit, "line"),
            ("keep", self.keep_edit, "line"),
            ("cacheTypeK", self.cache_type_k_combo, "combo"),
            ("cacheTypeV", self.cache_type_v_combo, "combo"),
            ("parallel", self.parallel_edit, "line"),
            ("flashAttention", self.flash_attn_combo, "combo"),
            ("device", self.device_edit, "line"),
            ("splitMode", self.split_mode_combo, "combo"),
            ("tensorSplit", self.tensor_split_edit, "line"),
            ("mainGpu", self.main_gpu_edit, "line"),
            ("numa", self.numa_combo, "combo"),
            ("fit", self.fit_combo, "combo"),
            ("fitTarget", self.fit_target_edit, "line"),
            ("fitCtx", self.fit_ctx_edit, "line"),
            ("ctxCheckpoints", self.ctx_checkpoints_edit, "line"),
            ("checkpointEveryNTokens", self.checkpoint_every_nt_edit, "line"),
            ("mlock", self.mlock_combo, "combo"),
            ("mmap", self.mmap_combo, "combo"),
            ("kvOffload", self.kv_offload_combo, "combo"),
            ("swaFull", self.swa_full_combo, "combo"),
            ("opOffload", self.op_offload_combo, "combo"),
            ("repack", self.repack_combo, "combo"),
            ("noHost", self.no_host_combo, "combo"),
            ("cpuMoe", self.cpu_moe_combo, "combo"),
            ("nCpuMoe", self.n_cpu_moe_edit, "line"),
            ("defragThold", self.defrag_thold_edit, "line"),
            ("specType", self.spec_type_combo, "combo"),
            ("specDraftMax", self.spec_draft_max_edit, "combo"),
            ("specDraftMin", self.spec_draft_min_edit, "line"),
            ("specDraftPSplit", self.spec_draft_p_split_edit, "line"),
            ("specDraftPMin", self.spec_draft_p_min_edit, "line"),
            ("specDraftCacheK", self.spec_draft_cache_k_combo, "combo"),
            ("specDraftCacheV", self.spec_draft_cache_v_combo, "combo"),
            ("specDraftGpuLayers", self.spec_draft_ngl_edit, "line"),
            ("specDraftDevice", self.spec_draft_device_edit, "line"),
            ("specDraftModel", self.spec_draft_model_edit, "line"),
            ("reasoning", self.reasoning_combo, "combo"),
            ("reasoningBudget", self.reasoning_budget_edit, "line"),
            ("seed", self.seed_edit, "line"),
            ("temperature", self.temp_edit, "line"),
            ("topK", self.top_k_edit, "line"),
            ("topP", self.top_p_edit, "line"),
            ("minP", self.min_p_edit, "line"),
            ("typicalP", self.typical_p_edit, "line"),
            ("repeatLastN", self.repeat_last_n_edit, "line"),
            ("repeatPenalty", self.repeat_penalty_edit, "line"),
            ("presencePenalty", self.presence_penalty_edit, "line"),
            ("frequencyPenalty", self.frequency_penalty_edit, "line"),
            ("dynatempRange", self.dynatemp_range_edit, "line"),
            ("dynatempExp", self.dynatemp_exp_edit, "line"),
            ("dryMultiplier", self.dry_multiplier_edit, "line"),
            ("dryBase", self.dry_base_edit, "line"),
            ("dryAllowedLength", self.dry_allowed_length_edit, "line"),
            ("dryPenaltyLastN", self.dry_penalty_last_n_edit, "line"),
            ("samplers", self.samplers_edit, "line"),
            ("backendSampling", self.backend_sampling_combo, "combo"),
            ("ignoreEos", self.ignore_eos_combo, "combo"),
            ("cachePrompt", self.cache_prompt_combo, "combo"),
            ("cacheReuse", self.cache_reuse_edit, "line"),
            ("threadsHttp", self.threads_http_edit, "line"),
            ("timeout", self.timeout_edit, "line"),
            ("alias", self.alias_edit, "line"),
            ("tags", self.tags_edit, "line"),
            ("apiKey", self.api_key_edit, "line"),
            ("apiKeyFile", self.api_key_file_edit, "line"),
            ("tools", self.tools_edit, "line"),
            ("metrics", self.metrics_combo, "combo"),
            ("props", self.props_combo, "combo"),
            ("slots", self.slots_combo, "combo"),
            ("uiMcpProxy", self.ui_mcp_proxy_combo, "combo"),
            ("staticPath", self.static_path_edit, "line"),
            ("apiPrefix", self.api_prefix_edit, "line"),
            ("mediaPath", self.media_path_edit, "line"),
            ("slotSavePath", self.slot_save_path_edit, "line"),
            ("jinja", self.jinja_combo, "combo"),
            ("skipChatParsing", self.skip_chat_parsing_combo, "combo"),
            ("prefillAssistant", self.prefill_assistant_combo, "combo"),
            ("mmprojAuto", self.mmproj_auto_combo, "combo"),
            ("mmprojOffload", self.mmproj_offload_combo, "combo"),
            ("embedding", self.embedding_combo, "combo"),
            ("rerank", self.rerank_combo, "combo"),
            ("pooling", self.pooling_combo, "combo"),
            ("context", self.ctx_combo, "combo"),
            ("gpuLayers", self.gpu_layers_edit, "line"),
            ("reasoningFormat", self.reasoning_format_combo, "combo"),
            ("reasoningPreserve", self.reasoning_preserve_combo, "combo"),
            ("reasoningBudgetMessage", self.reasoning_budget_msg_edit, "line"),
            ("chatTemplate", self.chat_template_edit, "line"),
            ("chatTemplateFile", self.chat_template_file_edit, "line"),
            ("chatTemplatePreset", self.chat_template_preset_combo, "combo"),
            ("chatTemplateKwargs", self.chat_template_kwargs_edit, "line"),
            ("mmproj", self.mmproj_edit, "line"),
            ("imageMinTokens", self.image_min_tokens_edit, "line"),
            ("imageMaxTokens", self.image_max_tokens_edit, "line"),
            ("extraArgs", self.extra_args_edit, "line"),
            ("mtpCache", self.mtp_cache_edit, "line"),
            ("noUi", self.no_ui_check, "check"),
        ]

    def _save_parameter_settings(
        self,
        settings: Any | None = None,
        prefix: str | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        """Persist runtime parameter controls to QSettings."""
        if settings is None:
            settings = QSettings("llamacpp", "llama-server-gui")
        prefix = prefix or ""
        if prefix:
            settings.setValue(f"{prefix}/modelConfigVersion", MODEL_SETTINGS_VERSION)
            if model_path:
                settings.setValue(f"{prefix}/modelPath", str(self._normalize_model_path(model_path)))
        for key, control, kind in self._parameter_settings():
            setting_key = f"{prefix}/{key}" if prefix else key
            if kind == "line":
                settings.setValue(setting_key, control.text())
            elif kind == "combo":
                settings.setValue(setting_key, control.currentText())
            elif kind == "check":
                settings.setValue(setting_key, control.isChecked())
        if prefix:
            self._default_parameter_values = self._capture_parameter_settings()

    def _load_parameter_settings(self, settings: Any | None = None, prefix: str | None = None) -> None:
        """Restore runtime parameter controls from QSettings."""
        if settings is None:
            settings = QSettings("llamacpp", "llama-server-gui")
        prefix = prefix or ""
        for key, control, kind in self._parameter_settings():
            setting_key = f"{prefix}/{key}" if prefix else key
            if hasattr(settings, "contains") and not settings.contains(setting_key):
                continue
            if kind == "line":
                value = settings.value(setting_key, type=str, defaultValue=None)
                if value is not None and (value != "" or control.text() == ""):
                    control.setText(value)
            elif kind == "combo":
                value = settings.value(setting_key, type=str, defaultValue=None)
                if value is not None and (value != "" or control.currentText() == ""):
                    if key == "mmprojOffload":
                        value = normalize_image_model_device(value)
                    control.setCurrentText(value)
            elif kind == "check":
                value = settings.value(setting_key, type=bool, defaultValue=None)
                if value is not None:
                    control.setChecked(value)

    def _legacy_parameter_settings_values(self, settings: Any | None = None) -> dict[str, str | bool]:
        if settings is None:
            settings = QSettings("llamacpp", "llama-server-gui")
        values: dict[str, str | bool] = {}
        for key, _control, kind in self._parameter_settings():
            if hasattr(settings, "contains") and not settings.contains(key):
                continue
            if kind in {"line", "combo"}:
                value = settings.value(key, type=str, defaultValue=None)
                if value is not None:
                    values[key] = value
            elif kind == "check":
                value = settings.value(key, type=bool, defaultValue=None)
                if value is not None:
                    values[key] = value
        return values

    def _ensure_model_settings(self, settings: Any | None, model: str | Path) -> str:
        settings, prefix = self._model_settings(settings, model)
        assert settings is not None
        model_path = self._normalize_model_path(model)
        if settings.contains(f"{prefix}/modelConfigVersion"):
            return prefix

        legacy_values = self._legacy_parameter_settings_values(settings)
        source_values = self._capture_parameter_settings()
        try:
            metadata = inspect_gguf_metadata(model_path)
        except Exception:
            metadata = {}
        source_values.update(model_default_parameter_settings(model_path, metadata))
        source_values.update(legacy_values)
        for key, value in source_values.items():
            settings.setValue(f"{prefix}/{key}", value)
        settings.setValue(f"{prefix}/modelConfigVersion", MODEL_SETTINGS_VERSION)
        settings.setValue(f"{prefix}/modelPath", str(model_path))
        return prefix

    def _sanitize_existing_model_settings(self, settings: Any, prefix: str, model: str | Path) -> None:
        model_path = self._normalize_model_path(model)
        try:
            metadata = inspect_gguf_metadata(model_path)
        except Exception:
            metadata = {}
        if not _is_qwythos_9b_claude_mythos_1m(model_path, metadata):
            return
        keys = {"context", "ctxCheckpoints", "checkpointEveryNTokens", "mmproj", "mmprojAuto", "extraArgs"}
        defaults = model_default_parameter_settings(model_path, metadata)
        for key in keys:
            value = defaults.get(key)
            if value is not None:
                settings.setValue(f"{prefix}/{key}", value)

    def _save_model_settings(self, settings: Any | None = None, model: str | Path | None = None) -> None:
        model = model or self._active_model_path
        if not model:
            return
        settings, prefix = self._model_settings(settings, model)
        self._save_parameter_settings(settings, prefix, model)

    def _load_model_settings(self, settings: Any | None = None, model: str | Path | None = None) -> None:
        model = model or self._active_model_path
        if not model:
            return
        settings, prefix = self._model_settings(settings, model)
        self._ensure_model_settings(settings, model)
        self._sanitize_existing_model_settings(settings, prefix, model)
        self._loading_model_settings = True
        try:
            self._load_parameter_settings(settings, prefix)
        finally:
            self._loading_model_settings = False

    def _save_last_model(self) -> None:
        """Persist the currently selected model to QSettings."""
        model = self.model_combo.currentData()
        if model:
            settings = QSettings("llamacpp", "llama-server-gui")
            settings.setValue(self._last_model_key(), model)

    def _load_last_model(self) -> str | None:
        """Retrieve the previously used model from QSettings."""
        settings = QSettings("llamacpp", "llama-server-gui")
        return settings.value(self._last_model_key(), type=str, defaultValue=None)

    # ───────────────────────────── named presets ─────────────────────

    def _model_preset_key(self, model: str | Path) -> str:
        # Reuse the same identity algorithm as QSettings model settings.
        return self._model_settings_prefix(model).split("/", 1)[-1]

    def _load_presets_doc(self) -> dict[str, Any]:
        try:
            text = PRESETS_FILE.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"version": PRESETS_DOC_VERSION, "models": {}}
        except OSError as exc:
            cast(QStatusBar, self.statusBar()).showMessage(f"Presets file unreadable: {exc}")
            return {"version": PRESETS_DOC_VERSION, "models": {}}
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            cast(QStatusBar, self.statusBar()).showMessage("Presets file corrupted; starting empty.")
            return {"version": PRESETS_DOC_VERSION, "models": {}}
        if not isinstance(doc, dict) or not isinstance(doc.get("models"), dict):
            return {"version": PRESETS_DOC_VERSION, "models": {}}
        return doc

    def _save_presets_doc(self, doc: dict[str, Any]) -> None:
        tmp = PRESETS_FILE.with_suffix(PRESETS_FILE.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, PRESETS_FILE)
        except OSError as exc:
            cast(QStatusBar, self.statusBar()).showMessage(f"Failed to save presets: {exc}")

    def _model_presets(self, doc: dict[str, Any], model: str | Path) -> list[dict[str, Any]]:
        entry = doc["models"].get(self._model_preset_key(model))
        if not isinstance(entry, dict):
            return []
        presets = entry.get("presets")
        return presets if isinstance(presets, list) else []

    def _set_model_presets(
        self, doc: dict[str, Any], model: str | Path, presets: list[dict[str, Any]]
    ) -> None:
        key = self._model_preset_key(model)
        doc["models"][key] = {
            "modelPath": str(self._normalize_model_path(model)),
            "presets": presets,
        }

    def _preset_names(self, presets: list[dict[str, Any]]) -> list[str]:
        return [str(p.get("name", "")) for p in presets if isinstance(p, dict)]

    def _recommended_preset_name(self, model: str | Path) -> str:
        stem = self._normalize_model_path(model).stem
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"{stem} - {stamp}"

    def _apply_params_from_dict(self, params: dict[str, Any]) -> None:
        self._loading_model_settings = True
        try:
            for key, control, kind in self._parameter_settings():
                if key not in params:
                    continue
                value = params[key]
                if kind == "line":
                    if isinstance(value, str):
                        control.setText(value)
                elif kind == "combo":
                    if isinstance(value, str):
                        if key == "mmprojOffload":
                            value = normalize_image_model_device(value)
                        control.setCurrentText(value)
                elif kind == "check":
                    if isinstance(value, bool):
                        control.setChecked(value)
        finally:
            self._loading_model_settings = False
        # One explicit save so the loaded preset becomes the active config.
        self._save_model_settings()

    def _save_preset_as(self, name: str) -> bool:
        model = self._active_model_path
        if not model:
            return False
        name = name.strip()
        if not name:
            return False
        doc = self._load_presets_doc()
        presets = self._model_presets(doc, model)
        if name in self._preset_names(presets):
            reply = QMessageBox.question(
                self,
                "Overwrite preset",
                f"A preset named \"{name}\" already exists for this model. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False
            presets = [p for p in presets if p.get("name") != name]
        params = self._capture_parameter_settings()
        presets.append({
            "name": name,
            "savedAt": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "params": params,
        })
        self._set_model_presets(doc, model, presets)
        self._save_presets_doc(doc)
        return True

    def _load_preset(self, name: str) -> bool:
        model = self._active_model_path
        if not model:
            return False
        doc = self._load_presets_doc()
        for preset in self._model_presets(doc, model):
            if preset.get("name") == name:
                params = preset.get("params")
                if isinstance(params, dict):
                    self._apply_params_from_dict(params)
                    return True
        return False

    def _delete_preset(self, name: str) -> bool:
        model = self._active_model_path
        if not model:
            return False
        doc = self._load_presets_doc()
        presets = self._model_presets(doc, model)
        new_presets = [p for p in presets if p.get("name") != name]
        if len(new_presets) == len(presets):
            return False
        self._set_model_presets(doc, model, new_presets)
        self._save_presets_doc(doc)
        return True

    def _rename_preset(self, old_name: str, new_name: str) -> bool:
        model = self._active_model_path
        if not model:
            return False
        new_name = new_name.strip()
        if not new_name or new_name == old_name:
            return False
        doc = self._load_presets_doc()
        presets = self._model_presets(doc, model)
        names = self._preset_names(presets)
        if new_name in names:
            return False
        for preset in presets:
            if preset.get("name") == old_name:
                preset["name"] = new_name
                self._set_model_presets(doc, model, presets)
                self._save_presets_doc(doc)
                return True
        return False

    def _open_preset_dialog(self) -> None:
        model = self._active_model_path
        if not model:
            cast(QStatusBar, self.statusBar()).showMessage("Select a model first.")
            return
        dialog = PresetManagerDialog(self, model, self)
        dialog.exec()

    def _refresh_model_list(self):
        models = refresh_models(Path(self.model_dir_edit.text()))
        current = self.model_combo.currentData()
        self.model_combo.clear()
        if not models:
            self.model_combo.addItem("(no models found)", userData=None)
            self.model_combo.setEnabled(False)
            self._set_mtp_badge(False, visible=False)
            return
        self.model_combo.setEnabled(True)
        for m in models:
            self.model_combo.addItem(m.name, userData=str(m))

        # Prefer: previously selected model (if still exists), then last saved model, then first
        last_model = self._load_last_model()
        preferred = resolve_preferred_model(models, current, last_model)

        if preferred:
            idx = self.model_combo.findData(preferred)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
            elif self.model_combo.count() > 0:
                self.model_combo.setCurrentIndex(0)
        else:
            self.model_combo.setCurrentIndex(0)

        self._on_model_selection_changed(self.model_combo.currentIndex())

    def _on_template_preset_changed(self, preset: str) -> None:
        preset = preset.strip().lower()
        fixed_template_path = str(SCRIPT_DIR / "chat_template_qwen_fixed.jinja")
        if preset == "qwen-fixed":
            self.jinja_combo.setCurrentText("on")
            self.chat_template_edit.setText("")
            self.chat_template_file_edit.setText(fixed_template_path)
            self._log("Template preset: qwen-fixed (froggeric v21) - enables --jinja and --chat-template-file")
        elif preset == "none":
            self.jinja_combo.setCurrentText("off")
            self.chat_template_edit.setText("")
            self.chat_template_file_edit.setText("")
            self._log("Template preset: none - disables --jinja and clears template fields")
        else:
            self._log("Template preset: default (use model's built-in template)")

    def _on_model_selection_changed(self, _index: int) -> None:
        model = self.model_combo.currentData()
        if not model:
            self._set_mtp_badge(False, visible=False)
            return
        model_path = Path(model)
        if not model_path.is_file():
            self._set_mtp_badge(False, visible=False)
            return

        try:
            metadata = inspect_gguf_metadata(model_path)
        except Exception as exc:
            self._set_mtp_badge(False)
            cast(QStatusBar, self.statusBar()).showMessage(f"Model metadata inspection failed: {exc}")
            return

        if self._active_model_path and self._active_model_path != model:
            self._save_model_settings(model=self._active_model_path)

        self._active_model_path = model
        self._load_model_settings(model=model)

        self._update_mtp_badge(model_path)
        self._update_default_mmproj(model_path)
        self._update_default_draft_model(model_path)
        self._update_predict_default(model_path, metadata)
        self._update_rosetta_panel()

        server, notes = resolve_display_server_for_model(self.server_edit.text(), metadata)
        if server != self.server_edit.text():
            self.server_edit.setText(server)
        if notes:
            for note in notes:
                self._log("Auto: " + note)
            cast(QStatusBar, self.statusBar()).showMessage(notes[-1])
        self._save_model_settings(model=model)

    def _update_server_info(self, *_args) -> None:
        self.server_info_label.setText(format_server_build_info(self.server_edit.text()))

    def _set_mtp_badge(self, is_mtp: bool, visible: bool = True) -> None:
        text, style = resolve_mtp_badge(is_mtp)
        self.mtp_badge_label.setText(text if visible else "")
        self.mtp_badge_label.setStyleSheet(style if visible else "")
        self.mtp_badge_label.setVisible(visible)

    def _update_mtp_badge(self, model: Path) -> None:
        cache_path = Path(
            self.mtp_cache_edit.text().strip() or str(DEFAULT_MTP_CACHE)
        )
        try:
            is_mtp, _reason = detect_mtp_cached(model, cache_path)
        except Exception:
            is_mtp = False
        self._set_mtp_badge(is_mtp)

    def _update_default_mmproj(self, model: Path) -> None:
        if self.mmproj_auto_combo.currentText().strip() == "off":
            return

        current = self.mmproj_edit.text().strip()
        if current and Path(current).is_file():
            current_mmproj = Path(current)
            resolved_mmproj = resolve_default_mmproj(model)
            if resolved_mmproj is None or current_mmproj.resolve() == resolved_mmproj.resolve():
                return

        mmproj = resolve_default_mmproj(model)
        if mmproj is None:
            return

        self.mmproj_edit.setText(str(mmproj))

    def _update_default_draft_model(self, model: Path) -> None:
        current = self.spec_draft_model_edit.text().strip()
        if current and Path(current).is_file():
            return

        cache_path = Path(
            self.mtp_cache_edit.text().strip() or str(DEFAULT_MTP_CACHE)
        )
        draft_model = resolve_draft_mtp_model(model, cache_path)
        if draft_model is None:
            return

        self.spec_draft_model_edit.setText(str(draft_model))

    def _update_predict_default(self, model: Path, metadata: dict[str, object]) -> None:
        """Update the Predict field tooltip with model context info if not set by user."""
        current = self.predict_edit.text().strip()
        
        # If user has already set a value, don't override
        if current:
            return
        
        # Get the model's context length from metadata
        train_ctx = model_context_length(metadata)
        
        if train_ctx is not None:
            self.predict_edit.setToolTip(
                f"Maximum tokens to generate. "
                f"Default: unlimited (-1). "
                f"Model context: {train_ctx:,} tokens. "
                f"Set a value to limit output length."
            )
        else:
            self.predict_edit.setToolTip(
                "Maximum tokens to generate. "
                "Default: unlimited (-1). "
                "Set a value to limit output length."
            )

    def _resolve_spec_type(self, model: Path) -> tuple[str, str]:
        spec = self.spec_type_combo.currentText()
        if spec != "auto":
            return spec, "manual"
        cache_path = Path(
            self.mtp_cache_edit.text().strip() or str(DEFAULT_MTP_CACHE)
        )
        is_mtp, reason = detect_mtp_cached(model, cache_path)
        if is_mtp:
            return "draft-mtp", reason

        draft_model = resolve_draft_mtp_model(model, cache_path)
        if draft_model is not None:
            return "draft-mtp", f"companion MTP draft model: {draft_model}"
        return "none", reason

    def _log(self, msg: str, analyze: bool = True) -> None:
        # Add timestamp to log message
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        timestamped_msg = f"[{timestamp}] {msg}"
        # Always write to file first (thread-safe, never throws on cross-thread)
        _append_log_file(timestamped_msg)
        # Qt UI updates must happen on the main thread; skip if called from
        # a worker thread (WebSocket handler, HTTP handler, etc.)
        if threading.current_thread() is not threading.main_thread():
            return
        self.log_text.appendPlainText(timestamped_msg)
        if not analyze:
            return
        for diagnostic in self._log_analyzer.analyze_line(msg):
            self._log(diagnostic, analyze=False)
            if diagnostic.startswith(("WARNING:", "ERROR:")):
                cast(QStatusBar, self.statusBar()).showMessage(diagnostic)

    def _clear_log(self) -> None:
        self.log_text.clear()

    def _open_browser(self) -> None:
        port = self._running_port or self.port_edit.text() or "8080"
        webbrowser.open(f"http://127.0.0.1:{port}")

    # ───────────────────────────── responses panel ───────────────────

    def _make_copy_row(self, label_text: str, template: str, key: str) -> QWidget:
        """Build a label + read-only text field + copy button row."""
        row = QHBoxLayout()
        row.setSpacing(4)
        lbl = QLabel(label_text)
        lbl.setFont(ui_font(True))
        lbl.setMinimumWidth(130)
        row.addWidget(lbl, 0)

        text_field = QPlainTextEdit()
        text_field.setReadOnly(True)
        text_field.setFont(log_font())
        text_field.setMaximumHeight(48)
        row.addWidget(text_field, 1)

        copy_btn = ClickableButton("Copy")
        copy_btn.setFont(ui_font(True))
        copy_btn.setMinimumHeight(28)
        copy_btn.setMinimumWidth(60)
        copy_btn.clicked.connect(lambda _=False, f=text_field: self._copy_to_clipboard(f))
        row.addWidget(copy_btn, 0)

        container = QWidget()
        container.setLayout(row)
        if not hasattr(self, "_rosetta_templates"):
            self._rosetta_templates = {}
        self._rosetta_templates[key] = (template, text_field)
        return container

    def _copy_to_clipboard(self, text_field: QPlainTextEdit) -> None:
        text = text_field.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            cast(QStatusBar, self.statusBar()).showMessage("Copied to clipboard", 2000)

    def _update_rosetta_panel(self) -> None:
        """Refresh the API tab with current model and ports."""
        alias = self.alias_edit.text().strip()
        model = alias or self.model_combo.currentText() or "model"
        llama_port = self._running_port or self.port_edit.text() or "8080"
        if hasattr(self, "_rosetta_templates"):
            for template, text_field in self._rosetta_templates.values():
                text_field.setPlainText(template.format(
                    rosetta_port=ROSETTA_GATEWAY_PORT, llama_port=llama_port, model=model))

    # ───────────────────────────── window controls ────────────────────

    def _toggle_advanced_settings(self, checked: bool) -> None:
        """Show or hide advanced parameter rows; persist the preference."""
        self._show_advanced = checked
        self._apply_advanced_visibility()
        settings = QSettings("llamacpp", "llama-server-gui")
        settings.setValue("showAdvanced", checked)

    def _apply_advanced_visibility(self) -> None:
        for widget in self._advanced_widgets:
            widget.setVisible(self._show_advanced)

    def _toggle_always_on_top(self) -> None:
        """Toggle the window's 'always on top' flag."""
        current = self.windowFlags()
        if current & Qt.WindowType.WindowStaysOnTopHint:
            self.setWindowFlags(current & ~Qt.WindowType.WindowStaysOnTopHint)
            self.show()
            self.btn_always_on_top.setText("Always on Top")
            self.btn_always_on_top.setStyleSheet(
                "QPushButton { padding: 6px 16px; }"
            )
        else:
            self.setWindowFlags(current | Qt.WindowType.WindowStaysOnTopHint)
            self.show()
            self.btn_always_on_top.setText("Always on Top ✓")
            self.btn_always_on_top.setStyleSheet(
                "QPushButton { background-color: #2196F3; color: white; padding: 6px 16px; }"
            )

    def _minimize_to_tray(self) -> None:
        """Minimize the window to the system tray."""
        self.showMinimized()

    # ───────────────────────────── single-instance local socket ────────

    def _setup_local_server(self) -> None:
        """Create a QLocalServer so other instances can notify us."""
        self._local_server = QLocalServer()
        # Remove leftover socket if previous run crashed
        QLocalServer.removeServer(_GUI_SOCKET_NAME)
        if not self._local_server.listen(_GUI_SOCKET_NAME):
            # Cannot listen — likely another instance is still alive.
            # This is okay; it just means the duplicate check will catch it.
            self._local_server = None
            return
        self._local_server.newConnection.connect(self._on_local_socket_connection)

    def _on_local_socket_connection(self) -> None:
        """Handle incoming notification from a duplicate instance."""
        if self._local_server is None:
            return
        socket = self._local_server.nextPendingConnection()
        if socket is None:
            return
        if not socket.waitForReadyRead(3000):
            socket.disconnectFromServer()
            return
        data = socket.readAll().data().decode("utf-8", errors="replace").strip()
        socket.disconnectFromServer()

        # Parse "RAISE <port>" message
        parts = data.split()
        if parts and parts[0] == "RAISE":
            # Raise the window to front
            self.show()
            self.raise_()
            self.activateWindow()
            # Also make sure window state is normal (not minimized)
            if self.windowState() & Qt.WindowState.WindowMinimized:
                self.showNormal()
            self.setWindowTitle(f"llama-server Web UI Launcher  —  already running")

    # ───────────────────────────── server lifecycle ───────────────────

    def _build_command(self) -> list[str]:
        server = self.server_edit.text()
        model = self.model_combo.currentData()
        self._launch_env_overrides = {}
        self._launch_notes = []

        if not model:
            raise ValueError("No model selected. Please choose a model.")
        if not Path(model).is_file():
            raise ValueError(f"Model file not found: {model}")

        metadata: dict[str, object] = {}
        try:
            metadata = inspect_gguf_metadata(Path(model))
        except Exception as exc:
            self._launch_notes.append(f"GGUF metadata inspection failed: {exc}")

        server, server_notes = resolve_server_for_model(server, metadata)
        self._launch_notes.extend(server_notes)
        if not Path(server).is_file():
            raise ValueError(f"llama-server not found: {server}\nBuild it first.")

        ctx_raw = parse_context_choice(self.ctx_combo.currentText())
        ctx, ctx_notes = resolve_auto_context(ctx_raw, metadata)
        launch_defaults = build_model_launch_defaults(metadata, ctx)
        self._launch_env_overrides = launch_defaults.env
        self._launch_notes.extend(ctx_notes)
        self._launch_notes.extend(launch_defaults.notes)

        gpu_layers_raw = self.gpu_layers_edit.text().strip()
        gpu_layers = gpu_layers_raw if gpu_layers_raw else "all"

        command = [
            server,
            "-m", model,
            "--host", self.host_edit.text() or "0.0.0.0",
            "--port", self.port_edit.text() or "8080",
            "-c", str(ctx),
            "-t", self.threads_edit.text() or str(max((os.cpu_count() or 8) // 2, 1)),
            "-b", self.batch_edit.text() or "2048",
            "-ngl", gpu_layers,
            "-fa", self.flash_attn_combo.currentText() or "on",
            "-ctk", self.cache_type_k_combo.currentText() or "q4_0",
            "-ctv", self.cache_type_v_combo.currentText() or "q4_0",
            "-np", self.parallel_edit.text() or "1",
            "--reasoning", self.reasoning_combo.currentText(),
            "--reasoning-format", self.reasoning_format_combo.currentText(),
            "--reasoning-budget", self.reasoning_budget_edit.text() or "1024",
        ]

        def flag_supported(flag: str) -> bool:
            if server_supports_flag(server, flag):
                return True
            self._launch_notes.append(f"Skipped unsupported llama-server flag: {flag}")
            return False

        def add_text(flag: str, edit: QLineEdit) -> None:
            value = edit.text().strip()
            if value and flag_supported(flag):
                command.extend([flag, value])

        def add_combo_value(flag: str, combo: QComboBox) -> None:
            value = combo.currentText().strip()
            if value and value != "default" and flag_supported(flag):
                command.extend([flag, value])

        def add_on_off(combo: QComboBox, on_flag: str, off_flag: str | None = None) -> None:
            value = combo.currentText().strip()
            if value == "on" and flag_supported(on_flag):
                command.append(on_flag)
            elif value == "off" and off_flag and flag_supported(off_flag):
                command.append(off_flag)

        add_text("-tb", self.threads_batch_edit)
        add_text("-ub", self.ubatch_edit)
        add_text("-n", self.predict_edit)
        add_text("--keep", self.keep_edit)
        add_text("-dev", self.device_edit)
        add_combo_value("-sm", self.split_mode_combo)
        add_text("-ts", self.tensor_split_edit)
        add_text("-mg", self.main_gpu_edit)
        add_combo_value("--numa", self.numa_combo)
        add_on_off(self.mlock_combo, "--mlock")
        add_on_off(self.mmap_combo, "--mmap", "--no-mmap")
        add_on_off(self.kv_offload_combo, "--kv-offload", "--no-kv-offload")
        add_on_off(self.swa_full_combo, "--swa-full")
        add_on_off(self.op_offload_combo, "--op-offload", "--no-op-offload")
        add_on_off(self.repack_combo, "--repack", "--no-repack")
        add_on_off(self.no_host_combo, "--no-host")
        add_on_off(self.cpu_moe_combo, "--cpu-moe")
        add_text("-ncmoe", self.n_cpu_moe_edit)
        add_combo_value("--fit", self.fit_combo)
        add_text("-fitt", self.fit_target_edit)
        add_text("-fitc", self.fit_ctx_edit)
        add_text("-dt", self.defrag_thold_edit)
        add_text("--ctx-checkpoints", self.ctx_checkpoints_edit)
        add_text("--checkpoint-min-step", self.checkpoint_every_nt_edit)

        add_text("-s", self.seed_edit)
        add_text("--temp", self.temp_edit)
        add_text("--top-k", self.top_k_edit)
        add_text("--top-p", self.top_p_edit)
        add_text("--min-p", self.min_p_edit)
        add_text("--typical", self.typical_p_edit)
        add_text("--repeat-last-n", self.repeat_last_n_edit)
        add_text("--repeat-penalty", self.repeat_penalty_edit)
        add_text("--presence-penalty", self.presence_penalty_edit)
        add_text("--frequency-penalty", self.frequency_penalty_edit)
        add_text("--dynatemp-range", self.dynatemp_range_edit)
        add_text("--dynatemp-exp", self.dynatemp_exp_edit)
        add_text("--dry-multiplier", self.dry_multiplier_edit)
        add_text("--dry-base", self.dry_base_edit)
        add_text("--dry-allowed-length", self.dry_allowed_length_edit)
        add_text("--dry-penalty-last-n", self.dry_penalty_last_n_edit)
        add_text("--samplers", self.samplers_edit)
        add_on_off(self.backend_sampling_combo, "--backend-sampling")
        add_on_off(self.ignore_eos_combo, "--ignore-eos")

        add_text("-a", self.alias_edit)
        add_text("--tags", self.tags_edit)
        add_text("--api-key", self.api_key_edit)
        add_text("--api-key-file", self.api_key_file_edit)
        add_text("--threads-http", self.threads_http_edit)
        add_text("-to", self.timeout_edit)
        add_on_off(self.cache_prompt_combo, "--cache-prompt", "--no-cache-prompt")
        add_text("--cache-reuse", self.cache_reuse_edit)
        add_on_off(self.metrics_combo, "--metrics")
        add_on_off(self.props_combo, "--props")
        add_on_off(self.slots_combo, "--slots", "--no-slots")
        add_on_off(self.ui_mcp_proxy_combo, "--ui-mcp-proxy", "--no-ui-mcp-proxy")
        add_text("--tools", self.tools_edit)
        add_text("--path", self.static_path_edit)
        add_text("--api-prefix", self.api_prefix_edit)
        add_text("--media-path", self.media_path_edit)
        add_text("--slot-save-path", self.slot_save_path_edit)

        add_on_off(self.jinja_combo, "--jinja", "--no-jinja")
        add_text("--chat-template", self.chat_template_edit)
        add_text("--chat-template-file", self.chat_template_file_edit)
        add_text("--chat-template-kwargs", self.chat_template_kwargs_edit)
        add_on_off(self.skip_chat_parsing_combo, "--skip-chat-parsing", "--no-skip-chat-parsing")
        add_on_off(self.prefill_assistant_combo, "--prefill-assistant", "--no-prefill-assistant")

        mmproj_mode = self.mmproj_auto_combo.currentText().strip().lower()
        mmproj_path = self.mmproj_edit.text().strip()
        if mmproj_path:
            try:
                model_metadata = inspect_gguf_metadata(Path(model))
                mmproj_metadata = inspect_gguf_metadata(Path(mmproj_path))
                if not _mmproj_matches_text_model(model_metadata, mmproj_metadata):
                    mmproj_path = ""
                    mmproj_mode = "off"
            except Exception:
                pass
        if mmproj_path:
            command.extend(["-mm", mmproj_path])
        if mmproj_mode == "on":
            command.append("--mmproj-auto")
        elif mmproj_mode == "off" or not mmproj_path:
            command.append("--no-mmproj")
        if normalize_image_model_device(self.mmproj_offload_combo.currentText()) == "GPU":
            command.append("--mmproj-offload")
        else:
            command.append("--no-mmproj-offload")
        add_text("--image-min-tokens", self.image_min_tokens_edit)
        add_text("--image-max-tokens", self.image_max_tokens_edit)
        add_on_off(self.embedding_combo, "--embedding")
        add_on_off(self.rerank_combo, "--rerank")
        add_combo_value("--pooling", self.pooling_combo)

        spec, _spec_reason = self._resolve_spec_type(Path(model))
        if spec != "none":
            command.extend([
                "--spec-type", spec,
                "--spec-draft-n-max", self.spec_draft_max_edit.currentText() or "3",
                "--spec-draft-n-min", self.spec_draft_min_edit.text() or "0",
            ])
            add_text("--spec-draft-p-split", self.spec_draft_p_split_edit)
            add_text("--spec-draft-p-min", self.spec_draft_p_min_edit)
            add_combo_value("--spec-draft-type-k", self.spec_draft_cache_k_combo)
            add_combo_value("--spec-draft-type-v", self.spec_draft_cache_v_combo)
            add_text("--spec-draft-ngl", self.spec_draft_ngl_edit)
            add_text("--spec-draft-device", self.spec_draft_device_edit)
            draft_model = self.spec_draft_model_edit.text().strip()
            if not draft_model and spec == "draft-mtp":
                cache_path = Path(
                    self.mtp_cache_edit.text().strip() or str(DEFAULT_MTP_CACHE)
                )
                resolved_draft_model = resolve_draft_mtp_model(Path(model), cache_path)
                if resolved_draft_model is not None:
                    draft_model = str(resolved_draft_model)
            if draft_model and Path(draft_model).is_file():
                command.extend(["--spec-draft-model", draft_model])

        rbm = self.reasoning_budget_msg_edit.text().strip()
        if rbm:
            command.extend(["--reasoning-budget-message", rbm])

        rp = self.reasoning_preserve_combo.currentText()
        if rp == "on" and flag_supported("--reasoning-preserve"):
            command.append("--reasoning-preserve")
        elif rp == "off" and flag_supported("--no-reasoning-preserve"):
            command.append("--no-reasoning-preserve")

        if self.no_ui_check.isChecked():
            command.append("--no-ui")
        else:
            command.append("--ui")

        extra_args = self.extra_args_edit.text().strip()
        if extra_args:
            try:
                command.extend(shlex.split(extra_args))
            except ValueError as exc:
                raise ValueError(f"Invalid Extra Args: {exc}") from exc

        # Backend sampling (-bs/--backend-sampling) runs a vocab-wide softmax via a
        # cooperative CUDA kernel. It crashes warmup with "SOFT_MAX failed /
        # invalid argument" when CUDA graphs are ON (graph capture can't launch
        # the cooperative kernel) and also on newer builds even with graphs OFF
        # for qwen35's large padded vocab (248,320). Strip it in both cases so
        # MTP + warmup keep working; CPU sampling is the safe fallback.
        strip_backend_sampling = False
        strip_reason = ""
        if server_cuda_graphs_enabled(Path(server)) is True:
            strip_backend_sampling = True
            strip_reason = (
                f"incompatible with CUDA graphs (server build {server} has "
                "GGML_CUDA_GRAPHS=ON; it crashes warmup with SOFT_MAX/invalid argument). "
                "Use a no-graphs server build to keep it."
            )
        elif _model_architecture(metadata) == "qwen35":
            strip_backend_sampling = True
            strip_reason = (
                "incompatible with qwen35 large padded vocab on this server build "
                "(crashes warmup with SOFT_MAX/invalid argument). CPU sampling is used instead."
            )

        if strip_backend_sampling:
            backend_sampling_flags = {"--backend-sampling", "-bs"}
            if any(arg in backend_sampling_flags for arg in command):
                command = [arg for arg in command if arg not in backend_sampling_flags]
                self._launch_notes.append(f"Removed --backend-sampling: {strip_reason}")

        return command

    def _start_server_core(self) -> tuple[bool, str | None]:
        """Start the llama-server subprocess. Returns (success, error_message)."""
        try:
            command = self._build_command()
        except ValueError as exc:
            return False, str(exc)

        if self._server_process and self._server_process.poll() is None:
            return False, "server is already running"

        self._log("=" * 50)
        self._log_analyzer = LlamaLogAnalyzer()
        self._running_port = self.port_edit.text() or "8080"

        self._log("Command: " + shlex.join(command))
        try:
            spec_type, spec_reason = self._resolve_spec_type(
                Path(self.model_combo.currentData())
            )
            self._log(f"Spec Type: {spec_type} ({spec_reason})")
        except Exception as exc:
            self._log(f"Spec Type: failed to resolve ({exc})")
        for note in self._launch_notes:
            self._log("Auto: " + note)
        for key, value in self._launch_env_overrides.items():
            self._log(f"Env: {key}={value}")
        self._log("=" * 50)

        self._save_last_model()
        self._save_model_settings(model=self.model_combo.currentData())
        cast(QStatusBar, self.statusBar()).showMessage("Starting server\u2026")
        self._set_running_controls(True)

        try:
            env = os.environ.copy()
            env.update(self._launch_env_overrides)
            self._server_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env=env,
            )
        except Exception as exc:
            self._set_idle()
            return False, str(exc)

        self._stop_event.clear()
        self._log_thread = ServerLogThread(self._server_process)
        self._log_thread.line_received.connect(self._log)
        self._log_thread.finished_with_code.connect(self._on_server_exit)
        self._log_thread.start()

        self._log(f"Server PID: {self._server_process.pid}")
        port = self.port_edit.text() or "8080"
        self._log(f"UI available at http://127.0.0.1:{port}")
        cast(QStatusBar, self.statusBar()).showMessage(f"Running \u2014 PID {self._server_process.pid}")

        # Start Rosetta gateway for Responses + Messages API
        self._start_rosetta_gateway(port)
        return True, None

    def _start_rosetta_gateway(self, llama_port: str) -> None:
        """Generate config and launch the Rosetta gateway subprocess."""
        alias = self.alias_edit.text().strip()
        model_name = alias or self.model_combo.currentText() or "model"
        try:
            config_path = _write_rosetta_config(model_name, llama_port)
        except Exception as exc:
            self._log(f"Rosetta gateway: failed to write config: {exc}")
            return
        # Stop tracked thread if running
        if self._rosetta_thread and self._rosetta_thread.isRunning():
            self._log("Rosetta gateway: stopping old instance")
            self._rosetta_thread.stop()
            self._rosetta_thread.wait(5000)
        # Kill any orphaned gateway process on the port (e.g. from a crashed session)
        _kill_process_on_port(ROSETTA_GATEWAY_PORT)
        self._rosetta_thread = RosettaGatewayThread(config_path)
        self._rosetta_thread.line_received.connect(self._log)
        self._rosetta_thread.finished_with_code.connect(self._on_rosetta_exit)
        self._rosetta_thread.start()
        self._log(f"Rosetta gateway: http://127.0.0.1:{ROSETTA_GATEWAY_PORT} "
                  f"model='{model_name}' "
                  f"(Responses: /v1/responses, Messages: /v1/messages)")
        self._update_rosetta_panel()

    def _on_rosetta_exit(self, code: int) -> None:
        if code != 0:
            self._log(f"[Rosetta gateway exited with code {code}]")

    def _start_server(self):
        success, error = self._start_server_core()
        if not success and error:
            QMessageBox.critical(self, "Error", error)

    def _on_server_exit(self, code: int) -> None:
        if not self._stop_event.is_set():
            self._log(f"[Server exited with code {code}]")
            # Stop Rosetta gateway since upstream is gone
            if self._rosetta_thread and self._rosetta_thread.isRunning():
                self._rosetta_thread.stop()
                self._rosetta_thread.wait(5000)
                self._rosetta_thread = None
            self._set_idle()

    def _stop_server(self):
        # Stop Rosetta gateway first
        if self._rosetta_thread and self._rosetta_thread.isRunning():
            self._log("Stopping Rosetta gateway")
            self._rosetta_thread.stop()
            self._rosetta_thread.wait(5000)
            self._rosetta_thread = None

        if self._server_process and self._server_process.poll() is None:
            self._log("Stopping server\u2026")
            self._stop_event.set()
            if self._log_thread:
                self._log_thread.stop()
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
                self._log("Server stopped.")
            except subprocess.TimeoutExpired:
                self._log("Server did not stop gracefully; killing\u2026")
                self._server_process.kill()
                self._server_process.wait()
                self._log("Server killed.")
        self._set_idle()

    def _set_idle(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_launch_controls_locked(False)
        self._running_port = None
        cast(QStatusBar, self.statusBar()).showMessage("Ready")

    def _set_running_controls(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self._set_launch_controls_locked(running)

    def _set_launch_controls_locked(self, locked: bool) -> None:
        if locked:
            if self._launch_control_states is None:
                self._launch_control_states = [
                    (control, control.isEnabled()) for control in self._launch_controls
                ]
            for control in self._launch_controls:
                control.setEnabled(False)
            return

        if self._launch_control_states is None:
            return

        for control, was_enabled in self._launch_control_states:
            control.setEnabled(was_enabled)
        self._launch_control_states = None

    # ───────────────────────────── HTTP control API ──────────────────

    def _start_api_server(self) -> None:
        """Start a localhost HTTP server for external control (e.g. Hermes)."""
        base = int(os.environ.get("LLAMA_GUI_API_PORT", GUI_API_DEFAULT_PORT))
        for offset in range(GUI_API_PORT_RANGE):
            port = base + offset
            try:
                server = http.server.ThreadingHTTPServer(
                    ("127.0.0.1", port), GuiApiHandler
                )
                server.api_bridge = self  # type: ignore[attr-defined]
                break
            except OSError:
                continue
        else:
            self._log("Failed to start GUI API server: no free port in range.")
            return

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._api_server = server
        self._api_port = port
        self._log(f"GUI API: http://127.0.0.1:{port}")
        self._update_rosetta_panel()

    def _stop_api_server(self) -> None:
        if self._api_server is not None:
            self._api_server.shutdown()
            self._api_server.server_close()
            self._api_server = None

    def api_execute(self, method_name: str, args: tuple = ()) -> Any:
        """Thread-safe dispatch: run a method on the Qt main thread and return result."""
        self._api_result = None
        self._api_result_event.clear()
        self._api_execute_signal.emit(method_name, args)
        if not self._api_result_event.wait(timeout=30):
            return {"error": "timeout waiting for main thread"}
        return self._api_result

    def _on_api_execute(self, method_name: str, args: object) -> None:
        try:
            method = getattr(self, method_name)
            self._api_result = method(*args)  # type: ignore[arg-type]
        except Exception as exc:
            self._api_result = {"error": str(exc)}
        finally:
            self._api_result_event.set()

    def api_get_status(self) -> dict:
        running = (
            self._server_process is not None
            and self._server_process.poll() is None
        )
        return {
            "running": running,
            "pid": self._server_process.pid if running else None,
            "port": self._running_port or self.port_edit.text() or "8080",
            "model": self.model_combo.currentData() or None,
            "model_name": self.model_combo.currentText()
            if self.model_combo.currentData() else None,
            "api_port": getattr(self, "_api_port", None),
        }

    def api_start(self) -> dict:
        success, error = self._start_server_core()
        if success:
            return {
                "started": True,
                "pid": self._server_process.pid if self._server_process else None,
            }
        return {"error": error}

    def api_stop(self) -> dict:
        if not (
            self._server_process and self._server_process.poll() is None
        ):
            return {"error": "server is not running"}
        self._stop_server()
        return {"stopped": True}

    def api_get_models(self) -> dict:
        models = []
        for i in range(self.model_combo.count()):
            models.append({
                "name": self.model_combo.itemText(i),
                "path": self.model_combo.itemData(i),
            })
        return {
            "models": models,
            "current": self.model_combo.currentData() or None,
            "current_name": self.model_combo.currentText()
            if self.model_combo.currentData() else None,
        }

    def api_set_model(self, model: str) -> dict:
        for i in range(self.model_combo.count()):
            data = self.model_combo.itemData(i)
            text = self.model_combo.itemText(i)
            if data == model or text == model or (data and Path(data).name == model):
                self.model_combo.setCurrentIndex(i)
                return {"selected": text, "path": data}
        return {"error": f"model not found: {model}"}

    def api_get_params(self) -> dict:
        result: dict[str, str | bool] = {}
        for key, control, kind in self._parameter_settings():
            if kind == "line":
                result[key] = control.text()
            elif kind == "combo":
                result[key] = control.currentText()
            elif kind == "check":
                result[key] = control.isChecked()
        return result

    def api_set_params(self, params: dict) -> dict:
        applied: dict[str, Any] = {}
        errors: dict[str, str] = {}
        param_map = {
            key: (control, kind) for key, control, kind in self._parameter_settings()
        }
        for key, value in params.items():
            if key not in param_map:
                errors[key] = "unknown parameter"
                continue
            control, kind = param_map[key]
            try:
                if kind == "line":
                    control.setText(str(value))
                elif kind == "combo":
                    control.setCurrentText(str(value))
                elif kind == "check":
                    control.setChecked(bool(value))
                applied[key] = value
            except Exception as exc:
                errors[key] = str(exc)
        return {"applied": applied, "errors": errors}

    def api_refresh_models(self) -> dict:
        self._refresh_model_list()
        return {"refreshed": True}

    def api_get_log(self) -> dict:
        return {"log": self.log_text.toPlainText()}

    # ───────────────────────────── system monitor ───────────────────

    def _start_status_monitor(self) -> None:
        """Start a 5-second timer that refreshes VRAM/RAM labels."""
        self._update_status_once()  # immediate first read
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(5000)
        self._status_timer.timeout.connect(self._update_status_once)
        self._status_timer.start()

    def _update_status_once(self) -> None:
        try:
            vram_str, vram_pct = self._query_vram() or ("", None)
            if vram_str and self._vram_label:
                self._vram_label.setText(vram_str)
                if vram_pct is not None and vram_pct >= 95:
                    self._vram_label.setStyleSheet(
                        "QLabel { background-color: #d32f2f; color: #ffffff; padding: 2px 6px; border-radius: 3px; }"
                    )
                else:
                    self._vram_label.setStyleSheet("")
        except Exception:
            pass
        try:
            ram_str, ram_pct = self._query_ram() or ("", None)
            if ram_str and self._ram_label:
                self._ram_label.setText(ram_str)
                if ram_pct is not None and ram_pct >= 95:
                    self._ram_label.setStyleSheet(
                        "QLabel { background-color: #d32f2f; color: #ffffff; padding: 2px 6px; border-radius: 3px; }"
                    )
                else:
                    self._ram_label.setStyleSheet("")
        except Exception:
            pass

    @staticmethod
    def _query_vram() -> tuple[str, int | None] | None:
        """Return ('VRAM: used/total (pct%)', pct) from nvidia-smi, or None."""
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return None
            # Sum across all GPUs
            total_used = 0
            total_cap = 0
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    total_used += int(parts[0])
                    total_cap += int(parts[1])
            if total_cap == 0:
                return None
            used_gb = total_used / 1024
            cap_gb = total_cap / 1024
            pct = total_used * 100 // total_cap
            return f"VRAM: {used_gb:.1f} / {cap_gb:.1f} GB ({pct}%)", pct
        except FileNotFoundError:
            return None

    @staticmethod
    def _query_ram() -> tuple[str, int | None] | None:
        """Return ('RAM: used/total (pct%)', pct) from /proc/meminfo, or None."""
        try:
            meminfo = Path("/proc/meminfo").read_text()
            vals: dict[str, int] = {}
            for line in meminfo.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    vals[parts[0].rstrip(":")] = int(parts[1])
            total = vals.get("MemTotal", 0)
            available = vals.get("MemAvailable", 0)
            if total == 0:
                return None
            used = total - available
            used_gb = used / 1024 / 1024
            total_gb = total / 1024 / 1024
            pct = used * 100 // total
            return f"RAM: {used_gb:.1f} / {total_gb:.1f} GB ({pct}%)", pct
        except (FileNotFoundError, OSError):
            return None

    def closeEvent(self, a0) -> None:
        """Ask for confirmation if server is running; close silently otherwise."""
        self._save_model_settings()
        if self._status_timer:
            self._status_timer.stop()
        self._stop_api_server()
        if self._server_process and self._server_process.poll() is None:
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Server is still running. Stop it and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                if a0 is not None:
                    a0.ignore()
                return
            self._stop_event.set()
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
                self._server_process.wait()
        _close_log_file()
        if a0 is not None:
            a0.accept()


# ── Single-instance mechanism (Qt LocalSocket) ───────────────────────

_GUI_SOCKET_NAME = "llama-server-gui-instance-check"


def _find_existing_port() -> int | None:
    """Check if llama-server is already running on the default port range.

    Returns the port number if an existing instance is detected, else None.
    """
    for port in range(8080, 8090):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return port
    return None


def _notify_existing_instance(port: int) -> bool:
    """Connect to the existing instance's local socket and send the port.

    Returns True if the message was sent successfully.
    """
    try:
        sock = QLocalSocket()
        sock.connectToServer(_GUI_SOCKET_NAME)
        if not sock.waitForConnected(3000):
            return False
        # Send the port as a simple string
        data = f"RAISE {port}\n".encode("utf-8")
        sock.write(data)
        sock.flush()
        return sock.waitForBytesWritten(3000)
    except Exception:
        return False


def _check_duplicate() -> int | None:
    """Check for an existing GUI instance via local socket.

    Returns the port of the existing instance if found, else None.
    """
    port = _find_existing_port()
    if port is None:
        return None

    # Existing instance detected — try to notify it via local socket
    if _notify_existing_instance(port):
        print(f"Notified existing instance (port {port}) to raise its window.\n")
    else:
        print(f"Another instance is already running (port {port}). "
              f"Could not notify it — the GUI window may have been closed.\n")

    return port


def main():
    # Enable High-DPI / fractional scaling on Wayland.
    # This must be set BEFORE QApplication is created.
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    # Check for duplicate instances before creating QApplication.
    # If a duplicate is found, try to restore the existing instance and exit.
    dup_port = _check_duplicate()
    if dup_port is not None:
        sys.exit(0)

    app = QApplication(sys.argv)
    app.setApplicationName("llama-server Web UI Launcher")
    app.setDesktopFileName("llama-server-GUI.desktop")

    _init_fonts()

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
