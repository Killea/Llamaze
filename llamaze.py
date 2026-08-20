#!/usr/bin/env python3
"""Start a local GGUF model in llama-server Web UI mode."""

from __future__ import annotations

import argparse
import configparser
import os
import struct
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR.parent / "llama_model"
DEFAULT_MTP_CACHE = SCRIPT_DIR / "run_qwen36_27b_mtp.ini"
DEFAULT_CTX = 256000
DEFAULT_LLAMA_DIR = SCRIPT_DIR.parent / "llama.cpp"
DEFAULT_SERVER_CANDIDATES = (
    Path("build-qwen38-nographs/bin/llama-server"),
    Path("build-b9813-cuda13-nographs/bin/llama-server"),
    Path("build-cuda/bin/llama-server"),
    Path("build/bin/llama-server"),
)
GGUF_SCALAR_SIZES = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}
GGUF_SCALAR_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a local GGUF model with llama-server Web UI.")
    parser.add_argument("--model", type=Path, default=None, help="Model file. If omitted, choose from --model-dir.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory containing .gguf models.")
    parser.add_argument("--mtp-cache", type=Path, default=DEFAULT_MTP_CACHE, help="INI cache for MTP detection results.")
    parser.add_argument("--server", type=Path, default=None)
    parser.add_argument("--llama-dir", type=Path, default=DEFAULT_LLAMA_DIR,
                        help="Base directory containing llama.cpp build dirs (default: sibling llama.cpp).")
    parser.add_argument("--launch-dir", type=Path, default=None,
                        help="Working directory for llama-server. Defaults to the server binary's directory.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ctx", type=int, default=None, help=f"Context size. If omitted, capped by model metadata (max {DEFAULT_CTX}).")
    parser.add_argument("--threads", type=int, default=max((os.cpu_count() or 8) // 2, 1))
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--gpu-layers", default=None, help="GPU layers to offload. If omitted, all layers are loaded on GPU.")
    parser.add_argument("--cache-type-kv", default="q4_0", help="KV cache type for K and V (default: q4_0)")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--spec-type", default="auto", help="Speculative decoding type. Use auto to enable MTP only when detected.")
    parser.add_argument("--spec-draft-n-max", type=int, default=3)
    parser.add_argument("--spec-draft-n-min", type=int, default=0)
    parser.add_argument("--reasoning", choices=("on", "off", "auto"), default="off")
    parser.add_argument("--reasoning-format", choices=("auto", "none", "deepseek", "deepseek-legacy"), default="auto")
    parser.add_argument("--reasoning-budget", type=int, default=1024)
    parser.add_argument("--reasoning-budget-message", default=None)
    parser.add_argument("--reasoning-preserve", choices=("default", "on", "off"), default="default",
                        help="Preserve reasoning in full history (on), strip from old messages (off), or template default")
    parser.add_argument("--no-ui", action="store_true", help="Disable the web UI and expose API only.")
    return parser.parse_args()


def read_exact(file, size: int) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise EOFError("unexpected end of GGUF file")
    return data


def read_gguf_string(file) -> str:
    length = struct.unpack("<Q", read_exact(file, 8))[0]
    if length > 16 * 1024 * 1024:
        raise ValueError(f"invalid GGUF string length: {length}")
    return read_exact(file, length).decode("utf-8", "replace")


def read_gguf_value(file, value_type: int):
    if value_type == 8:  # STRING
        return read_gguf_string(file)
    if value_type == 9:  # ARRAY
        element_type = struct.unpack("<I", read_exact(file, 4))[0]
        length = struct.unpack("<Q", read_exact(file, 8))[0]
        if element_type == 8:
            return [read_gguf_string(file) for _ in range(length)]
        if element_type in GGUF_SCALAR_SIZES:
            file.seek(GGUF_SCALAR_SIZES[element_type] * length, os.SEEK_CUR)
            return None
        raise ValueError(f"unsupported GGUF array element type: {element_type}")
    if value_type in GGUF_SCALAR_FORMATS:
        return struct.unpack(GGUF_SCALAR_FORMATS[value_type], read_exact(file, GGUF_SCALAR_SIZES[value_type]))[0]
    raise ValueError(f"unsupported GGUF value type: {value_type}")


def inspect_gguf_for_mtp(model: Path) -> tuple[bool, str]:
    with model.open("rb") as file:
        if read_exact(file, 4) != b"GGUF":
            raise ValueError("not a GGUF file")
        struct.unpack("<I", read_exact(file, 4))
        tensor_count, kv_count = struct.unpack("<QQ", read_exact(file, 16))

        for _ in range(kv_count):
            key = read_gguf_string(file)
            value_type = struct.unpack("<I", read_exact(file, 4))[0]
            value = read_gguf_value(file, value_type)
            if key.endswith(".nextn_predict_layers") and isinstance(value, int) and value > 0:
                return True, f"{key}={value}"
            if "mtp" in key.lower():
                return True, f"metadata key {key}"

        for _ in range(tensor_count):
            name = read_gguf_string(file)
            dims = struct.unpack("<I", read_exact(file, 4))[0]
            file.seek(8 * dims + 4 + 8, os.SEEK_CUR)
            name_lower = name.lower()
            if ".nextn." in name_lower or "mtp" in name_lower:
                return True, f"tensor {name}"

    return False, "no nextn/MTP metadata or tensors"


def inspect_gguf_metadata(model: Path) -> dict[str, object]:
    metadata: dict[str, object] = {}
    with model.open("rb") as file:
        if read_exact(file, 4) != b"GGUF":
            raise ValueError("not a GGUF file")
        struct.unpack("<I", read_exact(file, 4))
        tensor_count, kv_count = struct.unpack("<QQ", read_exact(file, 16))

        for _ in range(kv_count):
            key = read_gguf_string(file)
            value_type = struct.unpack("<I", read_exact(file, 4))[0]
            metadata[key] = read_gguf_value(file, value_type)

        # Skip tensor infos; callers only need metadata.
        for _ in range(tensor_count):
            read_gguf_string(file)
            dims = struct.unpack("<I", read_exact(file, 4))[0]
            file.seek(8 * dims + 4 + 8, os.SEEK_CUR)

    return metadata


def model_context_length(metadata: dict[str, object]) -> int | None:
    architecture = metadata.get("general.architecture")
    if not isinstance(architecture, str):
        return None
    value = metadata.get(f"{architecture}.context_length")
    return value if isinstance(value, int) and value > 0 else None


def apply_model_defaults(args: argparse.Namespace) -> None:
    try:
        metadata = inspect_gguf_metadata(args.model)
    except Exception as exc:
        print(f"Warning: failed to inspect GGUF metadata: {exc}", file=sys.stderr)
        metadata = {}

    train_ctx = model_context_length(metadata)
    args.model_train_ctx = train_ctx
    if args.ctx is None:
        args.ctx = min(DEFAULT_CTX, train_ctx) if train_ctx is not None else DEFAULT_CTX

    if args.gpu_layers is None:
        args.gpu_layers = "all"


def load_mtp_cache(cache_path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(cache_path)
    return config


def save_mtp_cache(cache_path: Path, config: configparser.ConfigParser) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as file:
        config.write(file)


def detect_mtp_cached(model: Path, cache_path: Path) -> tuple[bool, str]:
    model = model.resolve()
    stat = model.stat()
    section = str(model)
    config = load_mtp_cache(cache_path)

    if config.has_section(section):
        cached = config[section]
        if (
            cached.get("size") == str(stat.st_size)
            and cached.get("mtime_ns") == str(stat.st_mtime_ns)
            and cached.get("is_mtp") in {"true", "false"}
        ):
            return cached.getboolean("is_mtp"), f"cache: {cached.get('reason', 'cached')}"

    try:
        is_mtp, reason = inspect_gguf_for_mtp(model)
    except Exception as exc:
        is_mtp, reason = False, f"detection failed: {exc}"

    if not config.has_section(section):
        config.add_section(section)
    config[section]["size"] = str(stat.st_size)
    config[section]["mtime_ns"] = str(stat.st_mtime_ns)
    config[section]["is_mtp"] = "true" if is_mtp else "false"
    config[section]["reason"] = reason
    save_mtp_cache(cache_path, config)
    return is_mtp, reason


def select_model(model_dir: Path) -> Path:
    if not model_dir.is_dir():
        raise SystemExit(f"Model directory not found: {model_dir}")

    models = sorted(model_dir.glob("*.gguf"))
    if not models:
        raise SystemExit(f"No .gguf models found in: {model_dir}")

    print("Available models:")
    for index, model in enumerate(models, start=1):
        print(f"  {index}) {model.name}")
    print()

    choice = input("Select model [1]: ").strip() or "1"
    if not choice.isdigit():
        raise SystemExit(f"Invalid model selection: {choice}")
    choice_index = int(choice)
    if choice_index < 1 or choice_index > len(models):
        raise SystemExit(f"Invalid model selection: {choice}")

    return models[choice_index - 1]


def resolve_server_path(server_path: Path | None, llama_dir: Path) -> Path:
    if server_path is not None:
        return server_path
    for candidate in DEFAULT_SERVER_CANDIDATES:
        resolved = candidate if candidate.is_absolute() else (llama_dir / candidate)
        if resolved.is_file():
            return resolved
    return llama_dir / DEFAULT_SERVER_CANDIDATES[0]


def ensure_files(args: argparse.Namespace) -> None:
    args.server = resolve_server_path(args.server, args.llama_dir)
    if not args.server.is_file():
        raise SystemExit(
            f"llama-server not found: {args.server}\n"
            "Build it first with the bootstrap script or:\n"
            "  cmake -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON\n"
            "  cmake --build build-cuda -j 8 --target llama-server"
        )
    if not args.model.is_file():
        raise SystemExit(f"Model file not found: {args.model}")
    # Resolve to absolute paths so a non-default launch dir does not break lookups.
    args.model = args.model.resolve()
    args.mtp_cache = args.mtp_cache.resolve()
    if args.launch_dir is None:
        args.launch_dir = args.server.parent
    else:
        args.launch_dir = args.launch_dir.resolve()
        if not args.launch_dir.is_dir():
            raise SystemExit(f"Launch directory not found: {args.launch_dir}")


def resolve_spec_type(args: argparse.Namespace) -> tuple[str, str]:
    if args.spec_type != "auto":
        return args.spec_type, "manual"
    is_mtp, reason = detect_mtp_cached(args.model, args.mtp_cache)
    return ("draft-mtp" if is_mtp else "none"), reason


def run_server(args: argparse.Namespace) -> int:
    spec_type, spec_reason = resolve_spec_type(args)

    command = [
        str(args.server),
        "-m",
        str(args.model),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "-c",
        str(args.ctx),
        "-t",
        str(args.threads),
        "-b",
        str(args.batch),
        "-ngl",
        str(args.gpu_layers),
        "-ctk",
        args.cache_type_kv,
        "-ctv",
        args.cache_type_kv,
        "-np",
        str(args.parallel),
        "--reasoning",
        args.reasoning,
        "--reasoning-format",
        args.reasoning_format,
        "--reasoning-budget",
        str(args.reasoning_budget),
    ]

    if args.reasoning_preserve == "on":
        command.append("--reasoning-preserve")
    elif args.reasoning_preserve == "off":
        command.append("--no-reasoning-preserve")

    if spec_type != "none":
        command.extend([
            "--spec-type",
            spec_type,
            "--spec-draft-n-max",
            str(args.spec_draft_n_max),
            "--spec-draft-n-min",
            str(args.spec_draft_n_min),
        ])

    if args.reasoning_budget_message is not None:
        command.extend(["--reasoning-budget-message", args.reasoning_budget_message])

    if args.no_ui:
        command.append("--no-ui")
    else:
        command.append("--ui")

    print("============================================")
    print(" llama-server Web UI")
    print("============================================")
    print(f" Model:      {args.model}")
    print(f" Context:    {args.ctx}")
    print(f" KV Cache:   {args.cache_type_kv}")
    print(f" GPU Layers: {args.gpu_layers}")
    print(f" Threads:    {args.threads}")
    print(f" Batch:      {args.batch}")
    print(f" Spec Type:  {spec_type} ({spec_reason})")
    print(f" Reasoning:  {args.reasoning}")
    print(f" UI:         {'off (API only)' if args.no_ui else 'on'}")
    print(f" Launch Dir: {args.launch_dir}")
    print(f" URL:        http://{args.host}:{args.port}")
    print("============================================")
    print()
    sys.stdout.flush()
    return subprocess.call(command, cwd=str(args.launch_dir))


def main() -> int:
    args = parse_args()
    if args.model is None:
        args.model = select_model(args.model_dir)
    ensure_files(args)
    apply_model_defaults(args)
    return run_server(args)


if __name__ == "__main__":
    raise SystemExit(main())
