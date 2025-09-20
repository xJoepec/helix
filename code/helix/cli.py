from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from time import time as _time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from textwrap import dedent

import numpy as np

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parents[1]
    package_str = str(package_root)
    if package_str not in sys.path:
        sys.path.insert(0, package_str)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as _e:  # pragma: no cover
    torch = None
    nn = None
    F = None

from helix import (
    CapacityLossMetrics,
    PersistentHomologySummary,
    build_V_from_incidence,
    compute_capacity_loss,
    compute_persistent_homology,
    extract_partitions,
    mass_consistency_errors,
    region_counts,
    sanity_check_ucp,
    spectral_gap,
    ulam_pf,
)
from helix.plotting import (
    plot_cp_errors,
    plot_mass_consistency,
    plot_region_counts,
    plot_ulam_spectrum,
)


def _ensure_repo_root_on_path() -> Path:
    """Guarantee the repository root is importable for env packages."""

    repo_root = Path(__file__).resolve().parents[2]
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.append(repo_str)
    return repo_root


def _import_af_partition_env():
    try:
        from environments.helixenv.af_partition.env import AFPartitionEnv
        return AFPartitionEnv
    except ModuleNotFoundError as exc:
        _ensure_repo_root_on_path()
        try:
            from environments.helixenv.af_partition.env import AFPartitionEnv
            return AFPartitionEnv
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Helix environments not found. Ensure the 'environments' package "
                "is available or run the Helix CLI from the repository root."
            ) from exc


class HelixEnvFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Formatter that shows defaults while preserving deliberate line breaks."""


def _build_helix_env_parser(prog: str = "helix helixenv") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=dedent(
            """
            Run the Helix operator-algebra environment on a model or dataset.

            Provide your own artefacts (weights, dataset, builder) or fall back to the
            built-in two-moons generator. Tune diagnostics and reward weighting to
            mirror Helix Verifiers integrations.
            """
        ),
        epilog=dedent(
            """
            Examples:
              helix helixenv --samples 3000 --noise 0.05 --widths 32,16,16 --epochs 80
              helix helixenv --data-x data.npy --model-module my_model.py --weights model.pt
              helix helixenv --ask-api-key --api-key-var HELIX_API_KEY --show-config
            """
        ),
        formatter_class=HelixEnvFormatter,
    )

    source_group = parser.add_argument_group("Data & model inputs")
    source_group.add_argument(
        "--data-x",
        type=str,
        default="",
        metavar="PATH",
        help="Feature array (.npy/.npz/.csv); omit to generate synthetic two-moons",
    )
    source_group.add_argument(
        "--data-y",
        type=str,
        default="",
        metavar="PATH",
        help="Optional labels array aligning with data-x",
    )
    source_group.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Synthetic sample count when data-x is omitted",
    )
    source_group.add_argument(
        "--noise",
        type=float,
        default=0.08,
        help="Noise level for synthetic two-moons data",
    )
    source_group.add_argument("--seed", type=int, default=1, help="Random seed for demo data")
    source_group.add_argument(
        "--model-module",
        type=str,
        default="",
        metavar="PATH",
        help="Python file path that defines a model builder",
    )
    source_group.add_argument(
        "--model-func",
        type=str,
        default="build_model",
        metavar="NAME",
        help="Factory function name within the model module",
    )
    source_group.add_argument(
        "--model-kwargs",
        type=str,
        default="",
        metavar="JSON",
        help="JSON dict of keyword arguments for the builder",
    )
    source_group.add_argument(
        "--weights",
        type=str,
        default="",
        metavar="PATH",
        help="Optional state_dict (.pt/.pth) to load before evaluation",
    )
    source_group.add_argument(
        "--width",
        type=int,
        default=None,
        help="Legacy single hidden width; overrides --widths if provided",
    )
    source_group.add_argument(
        "--widths",
        type=str,
        default="16,16,16,16",
        metavar="CSV",
        help="Comma-separated hidden widths applied per ReLU layer",
    )
    source_group.add_argument("--d-out", type=int, default=2, help="Output width for built-in MLP")

    train_group = parser.add_argument_group("Training & optimisation")
    train_group.add_argument("--epochs", type=int, default=60, help="Training epochs when enabled")
    train_group.add_argument(
        "--no-train",
        action="store_true",
        help="Skip the lightweight training loop (recommended with pretrained weights)",
    )

    diag_group = parser.add_argument_group("Diagnostics & search")
    diag_group.add_argument(
        "--max-depth",
        type=int,
        default=0,
        help="Limit AF depth traversal (0 processes the full partition tower)",
    )
    diag_group.add_argument(
        "--no-ulam",
        action="store_true",
        help="Disable Ulam spectral gap diagnostics (faster runtime)",
    )
    diag_group.add_argument(
        "--ulam-bins",
        type=int,
        default=28,
        help="Grid bins per dimension for the Ulam PF operator",
    )
    diag_group.add_argument(
        "--ulam-samples-per-cell",
        type=int,
        default=1,
        help="Random samples per grid cell when estimating the PF operator",
    )
    diag_group.add_argument(
        "--ulam-eps",
        type=float,
        default=0.4,
        help="Residual block epsilon used in the synthetic PF map",
    )

    rewards_group = parser.add_argument_group("Reward weights (Helix Verifiers)")
    rewards_group.add_argument(
        "--mass-weight",
        type=float,
        default=1.0,
        help="Weight applied to mass consistency error",
    )
    rewards_group.add_argument(
        "--wasted-weight",
        type=float,
        default=0.1,
        help="Weight for penalising wasted AF regions",
    )
    rewards_group.add_argument(
        "--entropy-weight",
        type=float,
        default=0.05,
        help="Weight assigned to combinatorial entropy",
    )
    rewards_group.add_argument(
        "--cp-weight",
        type=float,
        default=1.0,
        help="Weight applied to CP map violations",
    )
    rewards_group.add_argument(
        "--gap-weight",
        type=float,
        default=0.5,
        help="Weight applied to Ulam spectral gap improvements",
    )

    api_group = parser.add_argument_group("API & integrations")
    api_group.add_argument(
        "--api-key",
        type=str,
        default="",
        metavar="KEY",
        help="API key for LLM-based judges; stored in the configured environment variable",
    )
    api_group.add_argument(
        "--api-key-var",
        type=str,
        default="OPENAI_API_KEY",
        metavar="VAR",
        help="Environment variable used to expose the provided API key",
    )
    api_group.add_argument(
        "--api-key-file",
        type=str,
        default="",
        metavar="PATH",
        help="Read API key from a file (first non-empty line wins)",
    )
    api_group.add_argument(
        "--ask-api-key",
        action="store_true",
        help="Prompt for an API key interactively when none is supplied",
    )

    util_group = parser.add_argument_group("Utility")
    util_group.add_argument(
        "--show-config",
        action="store_true",
        help="Print a sanitized summary of resolved arguments before execution",
    )
    util_group.add_argument(
        "--no-layer-summary",
        action="store_true",
        help="Skip detailed module/layer breakdown for the loaded model",
    )

    security_group = parser.add_argument_group("Security")
    security_group.add_argument(
        "--allow-pickled-arrays",
        action="store_true",
        help="Permit loading .npy/.npz files that require pickle (disabled by default)",
    )
    security_group.add_argument(
        "--allow-pickled-weights",
        action="store_true",
        help="Permit torch.load of pickled checkpoints when you trust the source",
    )
    return parser


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    stream = getattr(sys, "stdout", None)
    return bool(stream) and stream.isatty()


def _style(text: str, *, color: Optional[str] = None, bold: bool = False) -> str:
    if not _supports_color():
        return text
    codes: List[str] = []
    if color:
        codes.append(color)
    if bold:
        codes.append("1")
    if not codes:
        return text
    prefix = "\033[" + ";".join(codes) + "m"
    return f"{prefix}{text}\033[0m"


_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY_ENV_VALUES


def _load_state_dict(path: str, *, allow_pickled: bool = False):
    if torch is None:
        raise RuntimeError("PyTorch is not available; cannot load weights.")

    load_kwargs = {"map_location": "cpu"}
    if not allow_pickled:
        try:
            return torch.load(path, weights_only=True, **load_kwargs)
        except TypeError as exc:
            raise RuntimeError(
                "Safe checkpoint loading requires torch>=2.0. "
                "Upgrade PyTorch or rerun with --allow-pickled-weights (or HELIX_ALLOW_PICKLED_WEIGHTS=1) if you trust the checkpoint."
            ) from exc
        except RuntimeError as exc:
            message = str(exc).lower()
            if "weights_only" in message or "state_dict" in message:
                raise RuntimeError(
                    "Checkpoint does not expose a plain state_dict. "
                    "Rerun with --allow-pickled-weights or set HELIX_ALLOW_PICKLED_WEIGHTS=1 if you trust the source."
                ) from exc
            raise
    return torch.load(path, **load_kwargs)


TRIM_TOP_LINES = 4
TRIM_MAX_LINES = 18
TRIM_BOTTOM_LINES = 1  # trim last row which is static/non-animating


def _trim_frame(
    frame: str,
    top: int = TRIM_TOP_LINES,
    max_lines: int = TRIM_MAX_LINES,
    bottom: int = TRIM_BOTTOM_LINES,
) -> str:
    lines = frame.splitlines()
    if top > 0:
        lines = lines[top:] if len(lines) > top else []
    helix_idx = None
    for idx, line in enumerate(lines):
        if "HELIX" in line.upper():
            helix_idx = idx
            break
    if helix_idx is not None:
        lines = lines[:helix_idx]
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines]
    if bottom > 0:
        lines = lines[: len(lines) - bottom] if len(lines) > bottom else []
    return "\n".join(lines)


def _normalize_frames(frames: List[str]) -> Tuple[List[str], int, int]:
    """Pad/truncate frames so they all share a common width and height.

    Returns (normalized_frames, width, height).
    """
    split_frames: List[List[str]] = [f.splitlines() for f in frames]
    if not split_frames:
        return [], 0, 0

    max_height = max(len(lines) for lines in split_frames)
    max_width = 0
    for lines in split_frames:
        for line in lines:
            if len(line) > max_width:
                max_width = len(line)

    normalized: List[str] = []
    for lines in split_frames:
        padded_lines = [line + (" " * (max_width - len(line))) for line in lines]
        if len(padded_lines) < max_height:
            padded_lines.extend([" " * max_width] * (max_height - len(padded_lines)))
        normalized.append("\n".join(padded_lines))

    return normalized, max_width, max_height


def _load_external_helix_frames() -> List[str]:
    root = Path(__file__).resolve().parents[2]
    candidate = root / "ascii-animations" / "post-compile.txt"
    frames: List[str] = []
    try:
        with candidate.open("r", encoding="utf-8") as f:
            current: List[str] = []
            for raw in f:
                line = raw.rstrip("\n")
                if not line:
                    # allow empty rows within frames to pass through
                    if current:
                        current.append(line)
                    continue
                if "," in line and line.split(",", 1)[0].isdigit():
                    # frame header like "12,1"
                    continue
                if line == "*":
                    current = []
                    continue
                if line == "**":
                    if current:
                        frames.append("\n".join(current))
                    current = []
                    continue
                current.append(line)
    except FileNotFoundError:
        return []
    except OSError:
        return []

    if not frames:
        return []

    # Append label centered to each frame for brand consistency
    first_lines = frames[0].splitlines()
    width = len(first_lines[0]) if first_lines else 64
    label = "<<  HELIX  >>".center(width)
    return [f"{frame}\n{label}" for frame in frames]


def _build_helix_frames() -> List[str]:
    width = 64
    height = 18
    frames: List[str] = []
    total_frames = 8
    left_center = width // 2 - 12
    right_center = width // 2 + 12
    amplitude = 8
    margin = 4

    for frame_idx in range(total_frames):
        rows: List[str] = []
        phase_shift = (2 * math.pi * frame_idx) / total_frames
        for row in range(height):
            angle = (2 * math.pi * row / height) + phase_shift
            offset = math.sin(angle) * amplitude
            left = int(round(left_center + offset))
            right = int(round(right_center - offset))
            left = max(margin, min(width - margin - 1, left))
            right = max(left + 2, min(width - margin - 1, right))

            depth = math.cos(angle)
            line = [" "] * width

            if depth >= 0:
                if left - 1 >= 0:
                    line[left - 1] = "/"
                line[left] = "("
                if right + 1 < width:
                    line[right + 1] = "\\"
                line[right] = ")"
            else:
                if left - 1 >= 0:
                    line[left - 1] = "'"
                line[left] = "/"
                line[right] = "\\"
                if right + 1 < width:
                    line[right + 1] = "'"

            rung_visible = (row + frame_idx) % 3 == 0
            if rung_visible and (right - left) > 4:
                rung_char = "=" if depth >= 0 else "~"
                glow_char = "*" if depth >= 0 else "."
                inner_left = left + 1
                inner_right = right - 1
                line[inner_left] = rung_char
                line[inner_right] = rung_char
                mid = (inner_left + inner_right) // 2
                line[mid] = glow_char
                if (inner_right - inner_left) > 4:
                    if mid - 2 > inner_left:
                        line[mid - 2] = rung_char
                    if mid + 2 < inner_right:
                        line[mid + 2] = rung_char

            rows.append("".join(line))

        label = "<<  HELIX  >>"
        rows.append(" " * max(0, (width - len(label)) // 2) + label)
        frames.append("\n".join(rows))

    return frames


ASCII_HELIX_FRAMES = _load_external_helix_frames() or _build_helix_frames()

# Build a normalized set of frames for reliable terminal rendering
NORMALIZED_HELIX_FRAMES, NORMALIZED_HELIX_WIDTH, NORMALIZED_HELIX_HEIGHT = (
    _normalize_frames([_trim_frame(f) for f in ASCII_HELIX_FRAMES])
    if ASCII_HELIX_FRAMES
    else ([], 0, 0)
)


def _colorize_frame(frame: str) -> str:
    lines = frame.splitlines()
    if not _supports_color():
        return "\n".join(lines)

    accent = "95"  # bright magenta
    shadow = "94"  # bright blue
    styled: List[str] = []
    cutoff = len(lines) // 2
    for idx, line in enumerate(lines):
        base_color = accent if idx <= cutoff else shadow
        if "HELIX" in line:
            left, right = line.split("HELIX", 1)
            left_colored = _style(left, color=base_color, bold=True)
            core = _style("HELIX", color="95", bold=True)
            right_colored = _style(right, color=base_color, bold=True)
            styled.append(f"{left_colored}{core}{right_colored}")
        else:
            styled.append(_style(line, color=base_color, bold=True))
    return "\n".join(styled)


def _render_banner() -> str:
    if not NORMALIZED_HELIX_FRAMES:
        return ""
    frame = NORMALIZED_HELIX_FRAMES[int(_time()) % len(NORMALIZED_HELIX_FRAMES)]
    return _colorize_frame(frame)


def _is_tty() -> bool:
    stream = getattr(sys, "stdout", None)
    return bool(stream) and stream.isatty()


def _animate_banner(loops: int = 1, fps: float = 15.0) -> None:
    if not ASCII_HELIX_FRAMES:
        return

    if not _is_tty() or fps <= 0:
        banner = _render_banner()
        if banner:
            print(banner)
        return

    delay = 1.0 / max(fps, 1.0)
    hide_cursor = "\x1b[?25l"
    show_cursor = "\x1b[?25h"
    clear_home = "\x1b[2J\x1b[H"
    move_home = "\x1b[H"

    loops = max(1, loops)
    frames = (
        NORMALIZED_HELIX_FRAMES
        if NORMALIZED_HELIX_FRAMES
        else [_trim_frame(frame) for frame in ASCII_HELIX_FRAMES]
    )

    try:
        sys.stdout.write(hide_cursor)
        sys.stdout.write(clear_home)
        sys.stdout.flush()
        for _ in range(loops):
            for frame in frames:
                sys.stdout.write(move_home)
                sys.stdout.write(_colorize_frame(frame))
                sys.stdout.write("\n")
                sys.stdout.flush()
                time.sleep(delay)
        sys.stdout.write(clear_home)
        sys.stdout.write(_colorize_frame(frames[0]))
        sys.stdout.write("\n")
        sys.stdout.flush()
    finally:
        sys.stdout.write(show_cursor)
        sys.stdout.flush()


def _animate_banner_continuous(
    stop_event: threading.Event,
    lines_after_frame: int,
    frame_height: int,
    fps: float = 15.0,
) -> None:
    if not ASCII_HELIX_FRAMES or not _is_tty():
        return

    delay = 1.0 / max(fps, 1.0)
    frames = (
        NORMALIZED_HELIX_FRAMES
        if NORMALIZED_HELIX_FRAMES
        else [_trim_frame(frame) for frame in ASCII_HELIX_FRAMES]
    )
    total_up = max(0, frame_height + lines_after_frame)
    move_up = f"\x1b[{total_up}A" if total_up else ""
    frame_idx = 0

    while not stop_event.is_set():
        sys.stdout.write("\x1b[u")  # restore to prompt/input position
        sys.stdout.write("\x1b[s")  # immediately resave for return after drawing
        if move_up:
            sys.stdout.write(move_up)
        sys.stdout.write("\x1b[0G")
        colored_frame = _colorize_frame(frames[frame_idx])
        frame_lines = colored_frame.splitlines()
        for idx, line in enumerate(frame_lines):
            sys.stdout.write("\x1b[2K")
            sys.stdout.write(line)
            if idx < len(frame_lines) - 1:
                sys.stdout.write("\n")
                sys.stdout.write("\x1b[0G")
        sys.stdout.write("\x1b[u")  # return to prompt/input
        sys.stdout.write("\x1b[s")  # keep prompt location saved for next loop
        sys.stdout.flush()

        frame_idx = (frame_idx + 1) % len(frames)
        if stop_event.wait(delay):
            break


def _divider(char: str = "=") -> str:
    columns = shutil.get_terminal_size((72, 0)).columns
    width = max(48, min(columns, 96))
    return char * width


def _format_metric(label: str, value: str, icon: str = "->") -> str:
    label_txt = _style(label, color="96", bold=True) if _supports_color() else label
    return f"  {icon} {label_txt}: {value}"


def _short_path(path: str) -> str:
    if not path:
        return "<none>"
    try:
        return os.path.basename(path) or path
    except Exception:
        return path


def _print_helixenv_config(
    args: argparse.Namespace,
    *,
    api_key_var: str,
    api_key_source: str,
    api_key_active: bool,
) -> None:
    """Pretty-print a summary of the resolved Helix environment arguments."""

    demo_summary = (
        f"demo | samples={args.samples}, noise={args.noise:g}, seed={args.seed}"
        if not getattr(args, "data_x", "")
        else f"dataset={_short_path(args.data_x)}"
    )
    if getattr(args, "data_y", ""):
        demo_summary += f" (labels={_short_path(args.data_y)})"

    if getattr(args, "model_module", ""):
        model_summary = (
            f"builder={_short_path(args.model_module)}::{args.model_func}"
            + (f" kwargs={args.model_kwargs}" if args.model_kwargs else "")
        )
    else:
        widths_display = getattr(args, "widths", "") or "16,16,16,16"
        if getattr(args, "width", None) is not None:
            widths_display = f"{args.width} (legacy override)"
        model_summary = f"builtin MLP widths={widths_display}, d_out={args.d_out}"
    if getattr(args, "weights", ""):
        model_summary += f" | weights={_short_path(args.weights)}"

    train_summary = "skipped" if getattr(args, "no_train", False) else f"epochs={args.epochs}"

    ulam_summary = "disabled" if getattr(args, "no_ulam", False) else (
        f"bins={args.ulam_bins}, samples/cell={args.ulam_samples_per_cell}, eps={args.ulam_eps}"
    )

    rewards_summary = (
        f"mass={args.mass_weight}, wasted={args.wasted_weight}, entropy={args.entropy_weight}, "
        f"cp={args.cp_weight}, gap={args.gap_weight}"
    )

    api_summary = (
        f"active via {api_key_var} ({api_key_source})"
        if api_key_active
        else f"not set (env var {api_key_var})"
    )

    print(_subtle(_divider()))
    print(_headline("Helix Environment Configuration"))
    print(_format_metric("data", demo_summary, icon="[D]"))
    print(_format_metric("model", model_summary, icon="[M]"))
    print(_format_metric("training", train_summary, icon="[T]"))
    print(_format_metric("ulam", ulam_summary, icon="[U]"))
    print(_format_metric("rewards", rewards_summary, icon="[R]"))
    print(_format_metric("max depth", str(args.max_depth or "auto"), icon="[∂]"))
    print(_format_metric("api", api_summary, icon="[API]"))
    print(_subtle(_divider("-")))


AF_LAYER_CARD_METRICS = (
    ("mass_err", "mass L1", "log", ".2e"),
    ("trace_linf", "trace L∞", "log", ".2e"),
    ("entropy", "entropy", "linear", ".3f"),
    ("wasted", "wasted", "linear", "int"),
    ("cp_unital", "cp unital", "log", ".2e"),
    ("cp_coiso", "cp coiso", "log", ".2e"),
    ("cp_psd", "cp psd", "log", ".2e"),
    ("gap", "spectral gap", "linear", ".4f"),
    ("reward", "reward", "linear", ".4f"),
)


AF_SUMMARY_PLOT_CONFIG = (
    ("mass_err", "Mass residuals", "log", ".2e"),
    ("trace_linf", "Trace residuals", "log", ".2e"),
    ("entropy", "Entropy progression", "linear", ".3f"),
    ("cp_unital", "CP unital slack", "log", ".2e"),
    ("cp_psd", "CP PSD slack", "log", ".2e"),
    ("reward", "Reward trajectory", "linear", ".4f"),
)


def _metric_bounds(values: Sequence[float], *, scale: str) -> Optional[Tuple[float, float]]:
    numbers: List[float] = []
    for val in values:
        if val is None:
            continue
        try:
            float_val = float(val)
        except (TypeError, ValueError):
            continue
        if math.isnan(float_val):
            continue
        numbers.append(float_val)

    if not numbers:
        return None

    if scale == "log":
        positives = [abs(v) for v in numbers if abs(v) > 0]
        if not positives:
            return (0.0, 0.0)
        min_log = min(math.log10(v) for v in positives)
        max_log = max(math.log10(v) for v in positives)
        return (min_log, max_log)

    min_v = min(numbers)
    max_v = max(numbers)
    return (min_v, max_v)


def _metric_ratio(value: float, bounds: Tuple[float, float], *, scale: str) -> float:
    if scale == "log":
        min_log, max_log = bounds
        magnitude = abs(float(value))
        if magnitude <= 0:
            return 0.0
        if math.isclose(max_log, min_log):
            return 0.5
        log_v = math.log10(magnitude)
        ratio = (log_v - min_log) / (max_log - min_log)
    else:
        min_v, max_v = bounds
        if math.isclose(max_v, min_v):
            return 0.0 if math.isclose(max_v, 0.0) else 0.5
        ratio = (float(value) - min_v) / (max_v - min_v)
    return max(0.0, min(1.0, ratio))


def _render_bar(ratio: float, *, width: int = 14) -> str:
    clamped = max(0.0, min(1.0, ratio))
    filled = int(round(clamped * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _format_layer_metric_value(value: float, fmt: str) -> str:
    if fmt == "int":
        return str(int(round(float(value))))
    try:
        return f"{float(value):{fmt}}"
    except (ValueError, TypeError):
        return str(value)


def _render_layer_cards(
    rows: Sequence[Dict[str, Any]],
    series_map: Dict[str, Sequence[float]],
    *,
    entropy_max: float,
) -> None:
    if not rows:
        return

    bounds_map: Dict[str, Tuple[float, float]] = {}
    for key, _, scale, _ in AF_LAYER_CARD_METRICS:
        values = series_map.get(key, [])
        bounds = _metric_bounds(values, scale=scale) if values else None
        if bounds is not None:
            bounds_map[key] = bounds

    if not bounds_map:
        return

    print(_headline("AF Layer Profiles"))
    for row in rows:
        print(_subtle(_divider(".")))
        heat = _heatmap_for_row(row, entropy_max=entropy_max)
        header = f"depth {row['depth']} • regions={row['regions']}"
        header += f" • reward={row['reward']:.4f}"
        if heat:
            header += f" • heat {heat}"
        print(_headline(header))

        for key, label, scale, fmt in AF_LAYER_CARD_METRICS:
            if key not in bounds_map:
                continue
            value = row.get(key)
            if value is None:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(numeric_value):
                continue
            ratio = _metric_ratio(numeric_value, bounds_map[key], scale=scale)
            bar = _render_bar(ratio)
            value_str = _format_layer_metric_value(numeric_value, fmt)
            print(f"  {label:<12} {value_str:>12} |{bar}|")

    print(_subtle(_divider(".")))


def _format_betti(summary: PersistentHomologySummary) -> str:
    return ", ".join(f"β{idx}={val}" for idx, val in enumerate(summary.betti_numbers))


def _print_persistent_homology(summary: Optional[PersistentHomologySummary]) -> None:
    print(_headline("Persistent Homology"))
    if summary is None:
        print(_subtle("No persistent homology metrics available."))
        return
    precision = "exact" if summary.computed else "heuristic"
    print(
        _format_metric(
            "backend",
            f"{summary.backend} ({precision})",
            icon="[PH]",
        )
    )
    print(_format_metric("betti", _format_betti(summary), icon="[PH]"))
    lifetimes = ", ".join(
        f"β{idx}:{summary.average_lifetimes[idx]:.3g}/{summary.max_lifetimes[idx]:.3g}"
        for idx in range(len(summary.average_lifetimes))
    )
    print(_format_metric("lifetimes", lifetimes or "-", icon="[PH]"))
    if summary.notes:
        for note in summary.notes:
            print(_subtle(f"  note: {note}"))


def _format_capacity_layer_scores(metrics: CapacityLossMetrics) -> str:
    if not metrics.layer_scores:
        return "-"
    items = sorted(metrics.layer_scores.items(), key=lambda kv: kv[0])
    return ", ".join(f"{name}:{score:.2f}" for name, score in items)


def _print_capacity_metrics(metrics: Optional[CapacityLossMetrics]) -> None:
    print(_headline("Capacity Loss"))
    if metrics is None:
        print(_subtle("Capacity diagnostics unavailable."))
        return
    if not metrics.computed:
        print(_subtle("Capacity metrics unavailable (" + "; ".join(metrics.notes) + ")"))
        return
    print(_format_metric("mean", f"{metrics.mean_loss:.3f}", icon="[CL]"))
    print(_format_metric("max", f"{metrics.max_loss:.3f}", icon="[CL]"))
    if metrics.layer_scores:
        print(_format_metric("layers", _format_capacity_layer_scores(metrics), icon="[CL]"))
    if metrics.notes:
        for note in metrics.notes:
            print(_subtle(f"  note: {note}"))


def _print_cli_dashboard(
    *,
    persistent: Optional[PersistentHomologySummary],
    capacity: Optional[CapacityLossMetrics],
) -> None:
    print(_subtle(_divider(".")))
    print(_headline("Topology & Capacity Dashboard"))
    _print_persistent_homology(persistent)
    print()
    _print_capacity_metrics(capacity)
    print(_subtle(_divider(".")))


def _render_cli_barchart(
    title: str,
    depths: Sequence[int],
    values: Sequence[float],
    *,
    scale: str = "linear",
    fmt: str = ".3f",
    width: int = 30,
    flagged_depths: Optional[Sequence[int]] = None,
) -> None:
    if not values:
        return

    cleaned: List[Tuple[int, float]] = []
    for depth, raw_val in zip(depths, values):
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            continue
        if math.isnan(val):
            continue
        cleaned.append((depth, val))

    if not cleaned:
        return

    if scale == "log":
        eps = 1e-18
        logs = [math.log10(max(abs(val), eps)) for _, val in cleaned]
        lo = min(logs)
        hi = max(logs)
        if math.isclose(lo, hi):
            lo -= 1e-6
            hi += 1e-6
        ratios = []
        for (_, val), log_val in zip(cleaned, logs):
            ratio = (log_val - lo) / (hi - lo)
            ratios.append((val, ratio))
    else:
        vals = [val for _, val in cleaned]
        lo = min(vals)
        hi = max(vals)
        if math.isclose(lo, hi):
            lo -= 1e-6
            hi += 1e-6
        ratios = []
        for _, val in cleaned:
            ratio = (val - lo) / (hi - lo)
            ratios.append((val, ratio))

    print(_headline(title))
    flagged_set = set(flagged_depths or [])

    for (depth, _), (value, ratio) in zip(cleaned, ratios):
        filled = int(round(max(0.0, min(1.0, ratio)) * width))
        bar = "█" * filled + "░" * (width - filled)
        line = f"  d{depth:<3} |{bar}| {value:{fmt}}"
        if depth in flagged_set:
            line = _highlight_line(line)
        print(line)


def _render_summary_plots(rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return

    flagged_depths = [row["depth"] for row in rows if _row_is_flagged(row)]
    print(_headline("Summary Plots"))
    for key, title, scale, fmt in AF_SUMMARY_PLOT_CONFIG:
        values: List[float] = []
        depths: List[int] = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            try:
                float_val = float(value)
            except (TypeError, ValueError):
                continue
            values.append(float_val)
            depths.append(row["depth"])
        if not values:
            continue
        print(_subtle(_divider(".")))
        _render_cli_barchart(
            title,
            depths,
            values,
            scale=scale,
            fmt=fmt,
            flagged_depths=flagged_depths,
        )
    print(_subtle(_divider(".")))


def _print_af_summary(rows: Sequence[Dict[str, Any]], *, computed_gap: Optional[float]) -> None:
    if not rows:
        return

    flagged = sum(1 for row in rows if _row_is_flagged(row))

    mass_worst = max(rows, key=lambda r: r["mass_err"])
    trace_worst = max(rows, key=lambda r: r["trace_linf"])
    entropy_peak = max(rows, key=lambda r: r["entropy"])
    cp_unital_worst = max(rows, key=lambda r: r["cp_unital"])
    cp_psd_worst = max(rows, key=lambda r: r["cp_psd"])
    reward_best = max(rows, key=lambda r: r["reward"])
    reward_worst = min(rows, key=lambda r: r["reward"])

    gap_values = [
        float(row["gap"])
        for row in rows
        if isinstance(row["gap"], (int, float)) and not math.isnan(float(row["gap"]))
    ]

    print(_headline("Summary"))
    print(
        _format_metric(
            "mass residual",
            f"max {mass_worst['mass_err']:.2e} @ depth {mass_worst['depth']}",
            icon="[Σ]",
        )
    )
    print(
        _format_metric(
            "trace residual",
            f"max {trace_worst['trace_linf']:.2e} @ depth {trace_worst['depth']}",
            icon="[Σ]",
        )
    )
    print(
        _format_metric(
            "cp slack",
            (
                f"unital {cp_unital_worst['cp_unital']:.2e} @ depth {cp_unital_worst['depth']} | "
                f"psd {cp_psd_worst['cp_psd']:.2e} @ depth {cp_psd_worst['depth']}"
            ),
            icon="[Σ]",
        )
    )
    print(
        _format_metric(
            "entropy peak",
            f"{entropy_peak['entropy']:.3f} @ depth {entropy_peak['depth']}",
            icon="[Σ]",
        )
    )
    print(
        _format_metric(
            "reward span",
            f"best {reward_best['reward']:.4f} / worst {reward_worst['reward']:.4f}",
            icon="[Σ]",
        )
    )
    if gap_values:
        print(
            _format_metric(
                "spectral gap",
                f"range {min(gap_values):.4f}-{max(gap_values):.4f}",
                icon="[Σ]",
            )
        )
    elif computed_gap is not None:
        print(
            _format_metric(
                "spectral gap",
                f"{computed_gap:.4f} (global)",
                icon="[Σ]",
            )
        )
    print(
        _format_metric(
            "flags",
            f"{flagged} depth(s) over tolerance",  # intentionally pluralised
            icon="[Σ]",
        )
    )


def _format_module_name(name: str, *, max_width: int = 44) -> str:
    if not name:
        return "<root>"
    parts = name.split(".")
    indent = "  " * (len(parts) - 1)
    label = parts[-1]
    display = f"{indent}{label}"
    if len(display) <= max_width:
        return display
    return display[: max_width - 1] + "…"


def _format_human_count(value: int) -> str:
    abs_val = abs(value)
    if abs_val < 1000:
        return f"{value}"
    for unit, denom in (("K", 1_000), ("M", 1_000_000), ("B", 1_000_000_000), ("T", 1_000_000_000_000)):
        if abs_val < denom * 1000:
            return f"{value / denom:.2f}{unit}"
    return f"{value}"  # fallback


def _collect_model_layers(model: nn.Module) -> List[Dict[str, Any]]:
    layers: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        if name == "":
            continue

        direct_params = list(module.named_parameters(recurse=False))
        param_count = sum(param.numel() for _, param in direct_params)
        trainable_count = sum(param.numel() for _, param in direct_params if param.requires_grad)
        buffers = list(module.named_buffers(recurse=False))
        buffer_count = sum(buf.numel() for _, buf in buffers)
        is_leaf = not any(True for _ in module.children())

        if param_count == 0 and buffer_count == 0 and not is_leaf:
            continue

        layers.append(
            {
                "name": name,
                "type": module.__class__.__name__,
                "params": param_count,
                "trainable": trainable_count,
                "buffers": buffer_count,
                "is_leaf": is_leaf,
            }
        )
    return layers


def _print_model_layers(model: nn.Module) -> None:
    layers = _collect_model_layers(model)
    if not layers:
        print(_headline("Model Layers"))
        print(_subtle("No parameterised layers found."))
        return

    total_params = sum(layer["params"] for layer in layers)
    total_trainable = sum(layer["trainable"] for layer in layers)
    total_buffers = sum(layer["buffers"] for layer in layers)

    print(_headline("Model Layers"))
    summary_bits = [
        f"layers={len(layers)}",
        f"trainable={_format_human_count(total_trainable)}",
        f"params={_format_human_count(total_params)}",
    ]
    if total_buffers:
        summary_bits.append(f"buffers={_format_human_count(total_buffers)}")
    print(_subtle(" | ".join(summary_bits)))

    header = (
        f"{'#':>4} {'module':<46} {'type':<24} "
        f"{'trainable':>12} {'params':>12} {'buffers':>10} {'pct':>6}"
    )
    print(header)
    print(_subtle("-" * len(header)))

    for idx, layer in enumerate(layers, start=1):
        trainable = layer["trainable"]
        params = layer["params"]
        buffers = layer["buffers"]
        pct = (trainable / total_trainable * 100.0) if total_trainable else 0.0
        name_fmt = _format_module_name(layer["name"], max_width=46)
        type_fmt = layer["type"][:24]
        line = (
            f"{idx:>4} {name_fmt:<46} {type_fmt:<24} "
            f"{_format_human_count(trainable):>12} {_format_human_count(params):>12} "
            f"{_format_human_count(buffers):>10} {pct:>6.2f}"
        )
        print(line)

    print(_subtle(_divider("-")))


def _prepare_sequence(seq: Union[Sequence[float], np.ndarray]) -> str:
    if isinstance(seq, np.ndarray):
        arr = seq.tolist()
    else:
        arr = list(seq)
    return ", ".join(f"{x:.3g}" if isinstance(x, (int, float)) else str(x) for x in arr)


def _sparkline(values: Sequence[float], *, eps: float = 1e-18) -> str:
    glyphs = "▁▂▃▄▅▆▇█"
    if not values:
        return ""
    arr = np.asarray(values, dtype=np.float64)
    if np.all(arr <= 0):
        return glyphs[0] * len(values)
    safe = np.where(arr > 0, arr, eps)
    log_vals = np.log10(safe)
    min_v = np.min(log_vals)
    max_v = np.max(log_vals)
    if np.isclose(max_v, min_v):
        idx = int((len(glyphs) - 1) / 2)
        return glyphs[idx] * len(values)
    norm = (log_vals - min_v) / (max_v - min_v)
    indices = np.clip(np.round(norm * (len(glyphs) - 1)), 0, len(glyphs) - 1).astype(int)
    return "".join(glyphs[i] for i in indices)


def _sequence_with_sparkline(values: Sequence[float]) -> str:
    if not values:
        return "-"
    return f"{_prepare_sequence(values)} | {_sparkline(values)}"


def _shade_metric(value: float, thresholds: Sequence[float], *, normalized: bool = False) -> str:
    glyphs = "·░▒▓"
    val = float(value)
    if normalized:
        val = max(0.0, min(1.0, val))
    for idx, threshold in enumerate(thresholds):
        if val <= threshold:
            return glyphs[idx]
    return glyphs[-1]


def _heatmap_for_row(row: Dict[str, Any], *, entropy_max: float) -> str:
    denom = entropy_max if entropy_max > 0 else 1.0
    entropy_norm = row["entropy"] / denom
    return "".join(
        [
            _shade_metric(row["mass_err"], (1e-12, 1e-9, 1e-6)),
            _shade_metric(row["trace_linf"], (1e-12, 1e-9, 1e-6)),
            _shade_metric(entropy_norm, (0.25, 0.5, 0.75), normalized=True),
            _shade_metric(row["cp_unital"], (1e-4, 1e-2, 1e-1)),
            _shade_metric(row["cp_psd"], (1e-12, 1e-9, 1e-6)),
        ]
    )


def _highlight_line(text: str) -> str:
    if _supports_color():
        return _style(text, color="91", bold=True)
    return f"! {text}"


def _row_is_flagged(row: Dict[str, Any]) -> bool:
    return bool(
        row["mass_err"] > 1e-6
        or row["trace_linf"] > 1e-6
        or row["cp_unital"] > 1e-1
        or row["cp_psd"] > 1e-6
    )


def _prompt_text(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    if raw:
        return raw
    return default or ""


def _prompt_secret(prompt: str) -> str:
    try:
        return getpass.getpass(f"{prompt}: ")
    except Exception:
        return input(f"{prompt}: ").strip()


def _prompt_int(prompt: str, default: int) -> int:
    while True:
        raw = _prompt_text(prompt, str(default))
        try:
            return int(raw)
        except ValueError:
            print("Please enter a valid integer.")


def _prompt_float(prompt: str, default: float) -> float:
    while True:
        raw = _prompt_text(prompt, str(default))
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")


def _prompt_bool(prompt: str, default: bool = False) -> bool:
    default_str = "y" if default else "n"
    while True:
        raw = _prompt_text(f"{prompt} [y/n]", default_str).lower()
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer with y or n.")


def _parse_widths_csv(raw: str) -> Tuple[int, ...]:
    entries = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            entries.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid width '{token}' in --widths") from exc
    return tuple(entries)


def _resolve_hidden_widths(widths_csv: str, legacy_width: Optional[int]) -> Tuple[int, ...]:
    widths = _parse_widths_csv(widths_csv)
    if not widths:
        if legacy_width is None:
            # preserve historical 2-layer default when nothing provided
            return (16, 16)
        return (legacy_width, legacy_width)
    if legacy_width is not None:
        return tuple(legacy_width for _ in widths)
    return widths


def _hidden_widths_from_args(args: argparse.Namespace) -> Tuple[int, ...]:
    widths_csv = getattr(args, "widths", "") or ""
    legacy_width = getattr(args, "width", None)
    return _resolve_hidden_widths(widths_csv, legacy_width)


def _estimate_ulam_spectral_gap(
    dim: int,
    *,
    bins: int,
    samples_per_cell: int,
    eps: float,
) -> Optional[float]:
    if torch is None or nn is None:
        return None

    h2 = nn.Sequential(nn.Linear(dim, 16), nn.ReLU(), nn.Linear(16, dim))
    with torch.no_grad():
        for param in h2.parameters():
            param.mul_(0.1)

    def F_block(x_np: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(x_np.astype(np.float32))
        y_ = x_t + eps * h2(x_t)
        return y_.detach().numpy().astype(np.float64)

    lo = np.full((dim,), -2.0)
    hi = np.full((dim,), 2.5)
    if dim >= 2:
        lo[:2] = np.array([-2.0, -1.5])
        hi[:2] = np.array([3.0, 2.5])

    try:
        P, _ = ulam_pf(
            F_block,
            (lo, hi),
            bins_per_dim=bins,
            samples_per_cell=samples_per_cell,
        )
    except Exception:
        return None
    return float(spectral_gap(P))


def _interactive_collect_helix_args() -> argparse.Namespace:
    print(_headline("Operator Algebra Environment Setup"))
    print(_subtle("Leave entries blank to fall back to demo defaults."))
    data_x = _prompt_text("Dataset path (.npy/.npz/.csv)")
    data_y = _prompt_text("Labels path (optional)") if data_x else ""
    model_module = _prompt_text("Model builder module path")
    if model_module:
        model_func = _prompt_text("Builder function name", "build_model") or "build_model"
        model_kwargs = _prompt_text("Model kwargs JSON (optional)")
    else:
        model_func = "build_model"
        model_kwargs = ""
    weights = _prompt_text("Weights path (.pt/.pth)")
    samples = _prompt_int("Demo samples (used when no dataset)", 2000)
    noise = _prompt_float("Demo noise level", 0.08)
    seed = _prompt_int("Random seed", 1)
    while True:
        widths_csv = _prompt_text(
            "Hidden layer widths (comma separated)", "16,16,16,16"
        ).strip()
        try:
            parsed_widths = _parse_widths_csv(widths_csv)
        except ValueError as exc:
            print(_error(str(exc)))
            continue
        if not parsed_widths:
            widths_csv = "16,16,16,16"
            parsed_widths = _parse_widths_csv(widths_csv)
        break
    width = parsed_widths[0] if parsed_widths else 16
    d_out = _prompt_int("Demo output width", 2)
    epochs = _prompt_int("Training epochs (demo or labelled data)", 60)
    max_depth = _prompt_int("Max AF depth (0 = all)", 0)
    mass_weight = _prompt_float("Reward weight: mass error", 1.0)
    wasted_weight = _prompt_float("Reward weight: wasted regions", 0.1)
    entropy_weight = _prompt_float("Reward weight: entropy", 0.05)
    cp_weight = _prompt_float("Reward weight: CP violations", 1.0)
    gap_weight = _prompt_float("Reward weight: spectral gap", 0.5)
    run_ulam = _prompt_bool("Run Ulam spectral gap diagnostic", False)
    if run_ulam:
        ulam_bins = _prompt_int("Ulam bins per dimension", 28)
        ulam_samples_per_cell = _prompt_int("Ulam samples per cell", 1)
        ulam_eps = _prompt_float("Residual block epsilon (Ulam)", 0.4)
    else:
        ulam_bins = 28
        ulam_samples_per_cell = 1
        ulam_eps = 0.4
    api_key = _prompt_secret("LLM judge API key (leave blank to skip)").strip()
    api_key_var = _prompt_text("API key env var", "OPENAI_API_KEY") or "OPENAI_API_KEY"
    show_config = _prompt_bool("Show configuration summary before running", True)
    show_layers = _prompt_bool("Display model layer breakdown", True)
    no_train_default = bool(weights or (data_x and not data_y))
    no_train = _prompt_bool("Skip training (set true if model already trained)", no_train_default)
    return argparse.Namespace(
        data_x=data_x,
        data_y=data_y,
        model_module=model_module,
        model_func=model_func,
        model_kwargs=model_kwargs,
        weights=weights,
        samples=samples,
        noise=noise,
        seed=seed,
        width=width,
        widths=widths_csv,
        epochs=epochs,
        no_train=no_train,
        max_depth=max_depth,
        mass_weight=mass_weight,
        wasted_weight=wasted_weight,
        entropy_weight=entropy_weight,
        cp_weight=cp_weight,
        gap_weight=gap_weight,
        no_ulam=not run_ulam,
        ulam_bins=ulam_bins,
        ulam_samples_per_cell=ulam_samples_per_cell,
        ulam_eps=ulam_eps,
        d_out=d_out,
        api_key=api_key,
        api_key_var=api_key_var,
        api_key_file="",
        ask_api_key=False,
        show_config=show_config,
        no_layer_summary=not show_layers,
        allow_pickled_arrays=False,
        allow_pickled_weights=False,
    )


def run_helixenv_overview(args: Optional[argparse.Namespace] = None) -> int:
    """Display a summary of the Helix diagnostics environment dataset."""

    _ensure_repo_root_on_path()
    from environments.helixenv.af_partition.dataset import build_af_examples

    examples = build_af_examples()
    count = len(examples)
    label_counts: dict[str, int] = {"A": 0, "B": 0, "C": 0}
    for ex in examples:
        data = json.loads(ex["answer"])
        letter = data.get("label", "?")
        if letter in label_counts:
            label_counts[letter] += 1

    _animate_banner(loops=1, fps=18.0)
    print(_subtle(_divider()))
    print(_headline("Helix Diagnostics Environment"))
    print(
        _subtle(
            f"{count} synthesized scenarios with labelled outcomes: "
            f"stable (A) {label_counts['A']}, capacity (B) {label_counts['B']}, collapsed (C) {label_counts['C']}."
        )
    )

    if examples:
        sample = examples[0]
        data = json.loads(sample["answer"])
        print(_headline("Sample Scenario"))
        lines = sample["question"].splitlines()
        for line in lines[:16]:
            print(line)
        if len(lines) > 16:
            print("...")
        print(_subtle(f"Expected label: {data['label']} ({data['label_name']})"))

    print(_subtle(_divider("-")))
    print(
        _format_metric(
            "verifiers loader",
            "from environments.helixenv.af_partition import load_environment",
            icon="[HX]",
        )
    )
    print(
        _format_metric(
            "system prompt",
            "Use load_verifiers_environment() for helix/af_partition:v0",
            icon="[HX]",
        )
    )
    return 0


def _headline(text: str) -> str:
    return _style(text, color="92", bold=True) if _supports_color() else text


def _subtle(text: str) -> str:
    return _style(text, color="90") if _supports_color() else text


def _error(text: str) -> str:
    return _style(text, color="91", bold=True) if _supports_color() else text


def run_interactive() -> int:
    """Run the interactive menu system."""
    try:
        while True:
            # Clear screen for better presentation
            print("\033[2J\033[H" if _supports_color() else "\n" * 50)

            frame_height = 0
            if NORMALIZED_HELIX_FRAMES:
                first_frame = NORMALIZED_HELIX_FRAMES[0]
                frame_text = _colorize_frame(first_frame)
                sys.stdout.write(frame_text)
                sys.stdout.write("\n")
                sys.stdout.flush()
                frame_height = NORMALIZED_HELIX_HEIGHT

            lines_after_frame = 0

            def _print_menu_line(text: str = "") -> None:
                nonlocal lines_after_frame
                print(text)
                lines_after_frame += text.count("\n") + 1

            _print_menu_line(_subtle(_divider()))
            _print_menu_line(_headline("Helix Interactive Menu"))
            _print_menu_line(_subtle("Select an option to continue:"))
            _print_menu_line()
            _print_menu_line("  [1] Run Helix environment (model analytics)")
            _print_menu_line("  [2] Helix environment summary")
            _print_menu_line("  [3] Run demo (two moons dataset)")
            _print_menu_line("  [4] Analyze custom model/data")
            _print_menu_line("  [5] Interactive TUI mode")
            _print_menu_line("  [6] Exit")
            _print_menu_line()

            prompt = (
                _style("Your choice: ", color="96", bold=True)
                if _supports_color()
                else "Your choice: "
            )
            sys.stdout.write(prompt)
            sys.stdout.flush()

            stop_event: Optional[threading.Event] = None
            anim_thread: Optional[threading.Thread] = None

            if ASCII_HELIX_FRAMES and _is_tty():
                stop_event = threading.Event()
                sys.stdout.write("\x1b[s")
                sys.stdout.flush()
                anim_thread = threading.Thread(
                    target=_animate_banner_continuous,
                    args=(stop_event, lines_after_frame, frame_height, 18.0),
                    daemon=True,
                )
                anim_thread.start()

            try:
                choice = input()
            finally:
                if stop_event is not None:
                    stop_event.set()
                    if anim_thread is not None:
                        anim_thread.join(timeout=0.5)
                    sys.stdout.write("\x1b[u")
                    sys.stdout.write("\x1b[s")
                    sys.stdout.flush()

            if choice == "1":
                print()
                helix_args = _interactive_collect_helix_args()
                print()
                return run_helix_env(helix_args)
            elif choice == "2":
                print()
                result = run_helixenv_overview()
                return result
            elif choice == "3":
                args = argparse.Namespace(
                    samples=4000,
                    noise=0.07,
                    seed=1,
                    width=16,
                    epochs=100,
                    no_train=False,
                    ulam_bins=25,
                    ulam_samples_per_cell=1,
                    plot=False,
                    no_show=True,
                    save_prefix="",
                )
                print()
                return run_demo(args)
            elif choice == "4":
                print()
                print(_headline("Custom Analysis Setup"))

                data_x = input(
                    "Path to features array (.npy/.npz/.csv) [press Enter to use demo data]: "
                ).strip()
                model_module = input(
                    "Path to model module [press Enter for built-in MLP]: "
                ).strip()

                args = argparse.Namespace(
                    model_module=model_module,
                    model_func="build_model",
                    model_kwargs="",
                    weights="",
                    data_x=data_x,
                    data_y="",
                    d_out=2,
                    width=16,
                    epochs=50,
                    no_train=False,
                    ulam_bins=25,
                    ulam_samples_per_cell=1,
                    no_ulam=False,
                    seed=1,
                    samples=4000,
                    noise=0.07,
                )
                print()
                return run_analyze(args)
            elif choice == "5":
                try:
                    from .tui import run_tui

                    return run_tui([])
                except Exception as e:
                    print(
                        _error(
                            "\nHelix TUI not available. Install TUI extras: pip install '.[tui]'"
                        )
                    )
                    print(_subtle(f"Details: {e}"))
                    input("\nPress Enter to continue...")
                    continue
            elif choice == "6":
                print(_subtle("\nExiting Helix. Thank you!"))
                return 0
            else:
                print(_error("\nInvalid choice. Please select 1-6."))
                input("Press Enter to continue...")
    except (KeyboardInterrupt, EOFError):
        print(_subtle("\n\nExiting Helix. Thank you!"))
        return 0


class MLP(nn.Module):
    def __init__(self, d_in=2, widths=(16, 16), d_out=2):
        super().__init__()
        layers = []
        last = d_in
        for w in widths:
            layers += [nn.Linear(last, w), nn.ReLU(inplace=False)]
            last = w
        layers += [nn.Linear(last, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_moons(n=2000, noise=0.08, seed=0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, math.pi, n // 2)
    x1 = np.c_[np.cos(theta), np.sin(theta)]
    x2 = np.c_[1 - np.cos(theta), 1 - np.sin(theta)] + np.array([0.1, -0.2])
    X = np.vstack([x1, x2]).astype(np.float32)
    X += noise * rng.standard_normal(X.shape).astype(np.float32)
    y = np.r_[np.zeros(n // 2, dtype=np.int64), np.ones(n // 2, dtype=np.int64)]
    return X, y


def _seed_torch(seed: int) -> None:
    try:
        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:  # pragma: no cover
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass


def _load_array(path: str, *, allow_pickle: bool = False) -> np.ndarray:
    p = str(path)
    lower = p.lower()

    def _raise_pickle_hint(loader: str, exc: ValueError) -> None:
        if not allow_pickle and "pickled" in str(exc).lower():
            raise ValueError(
                f"{loader} '{p}' requires pickle deserialisation. "
                "Rerun with --allow-pickled-arrays or set HELIX_ALLOW_PICKLED_ARRAYS=1 if you trust the file."
            ) from exc

    if lower.endswith(".npy"):
        try:
            return np.load(p, allow_pickle=allow_pickle)
        except ValueError as exc:
            _raise_pickle_hint("Array", exc)
            raise
    if lower.endswith(".npz"):
        try:
            with np.load(p, allow_pickle=allow_pickle) as npz:
                # Prefer common keys
                for k in ("X", "x", "data", "array"):
                    if k in npz:
                        return np.array(npz[k])
                # Fallback to first array
                for k in npz.files:
                    return np.array(npz[k])
                raise ValueError(f"No arrays found in npz: {p}")
        except ValueError as exc:
            _raise_pickle_hint("Archive", exc)
            raise
    if lower.endswith(".csv") or lower.endswith(".txt"):
        return np.loadtxt(p, delimiter=",")
    raise ValueError(f"Unsupported array file type: {p}")


def _dynamic_import_builder(module_path: str, func_name: str = "build_model"):
    import importlib.util

    spec = importlib.util.spec_from_file_location("helix_user_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)
    fn = getattr(mod, func_name, None)
    if not callable(fn):
        raise AttributeError(f"Function {func_name} not found in {module_path}")
    return fn


def run_demo(args: argparse.Namespace) -> int:
    if torch is None:
        raise RuntimeError("PyTorch required for the demo. pip install torch")

    # Deterministic seeding for reproducibility
    _seed_torch(args.seed)

    X, y = make_moons(n=args.samples, noise=args.noise, seed=args.seed)
    try:
        persistent_summary = compute_persistent_homology(X, maxdim=2)
    except Exception as exc:
        persistent_summary = None
        print(_subtle(f"[warn] Persistent homology unavailable: {exc}"))
    hidden_widths = _hidden_widths_from_args(args)
    model = MLP(d_in=2, widths=hidden_widths, d_out=2)

    if not args.no_train:
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        for _ in range(args.epochs):
            opt.zero_grad()
            logits = model(X_t)
            loss = F.cross_entropy(logits, y_t)
            loss.backward()
            opt.step()

    try:
        capacity_metrics = compute_capacity_loss(model)
    except Exception as exc:
        capacity_metrics = None
        print(_subtle(f"[warn] Capacity metrics unavailable: {exc}"))

    af = extract_partitions(model, X)
    n_list = region_counts(af.B_list)
    errs = mass_consistency_errors(af.B_list, af.tau_list)

    cp_stats = []
    for k, B in enumerate(af.B_list, start=1):
        tau_prev = np.array([1.0]) if k == 1 else af.tau_list[k - 2]
        tau_cur = af.tau_list[k - 1]
        V = build_V_from_incidence(B, tau_prev, tau_cur)
        cp_stats.append(sanity_check_ucp(V, trials=6))

    # residual block for Ulam
    h2 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
    with torch.no_grad():
        for p in h2.parameters():
            p.mul_(0.1)

    def F_block(x_np: np.ndarray, eps: float = 0.3) -> np.ndarray:
        x_t = torch.from_numpy(x_np.astype(np.float32))
        y_ = x_t + eps * h2(x_t)
        return y_.detach().numpy().astype(np.float64)

    lo = np.array([-2.0, -1.5])
    hi = np.array([3.0, 2.5])
    P, _ = ulam_pf(
        lambda z: F_block(z, eps=0.4),
        (lo, hi),
        bins_per_dim=args.ulam_bins,
        samples_per_cell=args.ulam_samples_per_cell,
    )
    gap = spectral_gap(P)

    _animate_banner(loops=1, fps=18.0)
    print(_subtle(_divider()))
    print(_headline("Helix Diagnostics Demo"))
    meta = " | ".join(
        [
            f"samples={args.samples}",
            f"noise={args.noise:.3f}",
            f"seed={args.seed}",
            f"widths={','.join(map(str, hidden_widths))}",
        ]
    )
    print(_subtle(meta))
    print(_headline("Quick Actions"))
    print(_format_metric("custom data", "helix analyze --help", icon="[?]"))
    print(_format_metric("interactive mode", "helix tui", icon="[?]"))
    print(_format_metric("plots", "add --plot for live figures", icon="[?]"))
    print(_subtle(_divider("-")))

    print(_headline("AF Partitions"))
    print(_format_metric("region counts", _prepare_sequence(n_list), icon="[*]"))
    print(_format_metric("mass L1 error", _prepare_sequence(errs), icon="[*]"))

    print(_headline("CP Diagnostics"))
    for depth, stats in enumerate(cp_stats, start=1):
        value = " | ".join(
            [
                f"unital={stats['unital_err_fro']:.2e}",
                f"coiso={stats['coisometry_err_fro']:.2e}",
                f"psd={stats['psd_min_eig_violation']:.2e}",
            ]
        )
        print(_format_metric(f"depth {depth}", value, icon="[C]"))

    print(_headline("Ulam Transfer"))
    print(_format_metric("spectral gap", f"{gap:.4f}", icon="[U]"))

    if args.plot:
        fig1, _ = plot_region_counts(n_list)
        fig2, _ = plot_mass_consistency(errs)
        fig3, _ = plot_cp_errors(cp_stats)
        fig4, _ = plot_ulam_spectrum(P)
        import matplotlib.pyplot as plt

        if args.save_prefix:
            fig1.savefig(f"{args.save_prefix}_regions.png", dpi=150, bbox_inches="tight")
            fig2.savefig(f"{args.save_prefix}_mass_consistency.png", dpi=150, bbox_inches="tight")
            fig3.savefig(f"{args.save_prefix}_cp.png", dpi=150, bbox_inches="tight")
            fig4.savefig(f"{args.save_prefix}_ulam.png", dpi=150, bbox_inches="tight")
        if not args.no_show:
            plt.show()

    return 0


def run_helix_env(args: argparse.Namespace) -> int:
    """Run the operator-algebra AF environment with optional custom data/model."""

    if torch is None:
        raise RuntimeError("PyTorch required. Install with `pip install torch`.")

    AFPartitionEnv = _import_af_partition_env()

    cli_pickled_arrays = getattr(args, "allow_pickled_arrays", False)
    cli_pickled_weights = getattr(args, "allow_pickled_weights", False)
    env_pickled_arrays = _env_flag("HELIX_ALLOW_PICKLED_ARRAYS")
    env_pickled_weights = _env_flag("HELIX_ALLOW_PICKLED_WEIGHTS")
    allow_pickled_arrays = cli_pickled_arrays or env_pickled_arrays
    allow_pickled_weights = cli_pickled_weights or env_pickled_weights

    if cli_pickled_arrays:
        print(
            _subtle(
                "--allow-pickled-arrays: numpy pickle deserialisation enabled for this run."
            )
        )
    elif env_pickled_arrays:
        print(
            _subtle(
                "HELIX_ALLOW_PICKLED_ARRAYS=1: numpy pickle deserialisation enabled for this run."
            )
        )
    if cli_pickled_weights:
        print(
            _subtle(
                "--allow-pickled-weights: pickled checkpoint loading enabled for this run."
            )
        )
    elif env_pickled_weights:
        print(
            _subtle(
                "HELIX_ALLOW_PICKLED_WEIGHTS=1: pickled checkpoint loading enabled for this run."
            )
        )

    api_key_var = getattr(args, "api_key_var", "OPENAI_API_KEY") or "OPENAI_API_KEY"
    api_key_source = "cli flag"
    resolved_api_key = (getattr(args, "api_key", "") or "").strip()

    if not resolved_api_key and getattr(args, "api_key_file", ""):
        api_key_source = f"file:{_short_path(args.api_key_file)}"
        path = Path(args.api_key_file).expanduser()
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    candidate = line.strip()
                    if candidate:
                        resolved_api_key = candidate
                        break
            if resolved_api_key:
                print(_subtle(f"Loaded API key from {path}."))
        except FileNotFoundError:
            print(_error(f"API key file not found: {path}"))
            api_key_source = "missing file"
        except OSError as exc:
            print(_error(f"Unable to read API key file {path}: {exc}"))
            api_key_source = "unreadable file"

    if not resolved_api_key and getattr(args, "ask_api_key", False):
        if _is_tty():
            resolved_api_key = _prompt_secret("Enter API key").strip()
            api_key_source = "interactive prompt"
        else:
            print(_error("Cannot prompt for API key because stdout is not a TTY."))
            api_key_source = "prompt unavailable"

    api_key_active = False
    if resolved_api_key:
        os.environ[api_key_var] = resolved_api_key
        api_key_active = True
        print(
            _subtle(
                f"Stored API key in environment variable {api_key_var} for Helix Verifiers integrations."
            )
        )
    elif os.environ.get(api_key_var):
        api_key_source = "existing environment"
        api_key_active = True
        print(_subtle(f"Using API key from environment variable {api_key_var}."))
    else:
        api_key_source = "not provided"

    if getattr(args, "show_config", False):
        _print_helixenv_config(
            args,
            api_key_var=api_key_var,
            api_key_source=api_key_source,
            api_key_active=api_key_active,
        )

    _seed_torch(args.seed)

    persistent_summary: Optional[PersistentHomologySummary] = None
    capacity_metrics: Optional[CapacityLossMetrics] = None

    if getattr(args, "data_x", ""):
        X = _load_array(args.data_x, allow_pickle=allow_pickled_arrays)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = np.asarray(X, dtype=np.float32)
        y = None
        if getattr(args, "data_y", ""):
            y_arr = _load_array(args.data_y, allow_pickle=allow_pickled_arrays)
            y_flat = np.asarray(y_arr, dtype=np.int64).reshape(-1)
            if y_flat.shape[0] != X.shape[0]:
                raise ValueError("data_y length must match number of rows in data_x")
            y = y_flat
    else:
        X, y = make_moons(n=args.samples, noise=args.noise, seed=args.seed)

    try:
        persistent_summary = compute_persistent_homology(X, maxdim=2)
    except Exception as exc:
        print(_subtle(f"[warn] Persistent homology unavailable: {exc}"))

    hidden_widths = _hidden_widths_from_args(args)

    if args.model_module:
        builder = _dynamic_import_builder(args.model_module, args.model_func)
        kwargs: dict[str, Any] = {}
        if getattr(args, "model_kwargs", ""):
            import json as _json

            try:
                kwargs = _json.loads(args.model_kwargs)
            except Exception as e:
                raise ValueError(f"Invalid JSON for --model-kwargs: {e}")
        model = builder(**kwargs)
        if not isinstance(model, nn.Module):
            raise TypeError("Builder must return a torch.nn.Module")
    else:
        d_in = int(X.shape[1]) if X.ndim == 2 else 1
        model = MLP(d_in=d_in, widths=hidden_widths, d_out=args.d_out)

    if getattr(args, "weights", ""):
        state = _load_state_dict(args.weights, allow_pickled=allow_pickled_weights)
        try:
            model.load_state_dict(state)
        except Exception:
            model.load_state_dict(state, strict=False)

    if (not args.no_train) and (y is not None):
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(args.epochs):
            opt.zero_grad()
            logits = model(X_t)
            loss = F.cross_entropy(logits, y_t)
            loss.backward()
            opt.step()

    try:
        capacity_metrics = compute_capacity_loss(model)
    except Exception as exc:
        print(_subtle(f"[warn] Capacity metrics unavailable: {exc}"))

    max_depth = args.max_depth if getattr(args, "max_depth", 0) and args.max_depth > 0 else None
    env = AFPartitionEnv(
        model,
        X,
        max_depth=max_depth,
        mass_weight=args.mass_weight,
        wasted_weight=args.wasted_weight,
        entropy_weight=args.entropy_weight,
        cp_weight=args.cp_weight,
        gap_weight=args.gap_weight,
    )

    if persistent_summary is not None:
        env.update_persistent_homology(persistent_summary)
    if capacity_metrics is not None:
        env.update_capacity_metrics(capacity_metrics)

    computed_gap: Optional[float] = None
    if not getattr(args, "no_ulam", False):
        dim = int(X.shape[1]) if X.ndim == 2 else 2
        computed_gap = _estimate_ulam_spectral_gap(
            dim,
            bins=args.ulam_bins,
            samples_per_cell=args.ulam_samples_per_cell,
            eps=args.ulam_eps,
        )
        if computed_gap is not None:
            env.update_spectral_gaps([computed_gap] * len(env.levels))
        else:
            print(
                _subtle(
                    "[warn] Ulam spectral gap unavailable; continuing without gap reward"
                )
            )

    _animate_banner(loops=1, fps=18.0)
    print(_subtle(_divider()))
    print(_headline("Helix Environment"))
    meta_bits = [
        f"samples={X.shape[0]}",
        f"dim={int(X.shape[1]) if X.ndim == 2 else 1}",
        f"seed={args.seed}",
        f"depth_cap={max_depth or 'all'}",
    ]
    if getattr(args, "data_x", ""):
        meta_bits.append(f"data={os.path.basename(args.data_x)}")
    else:
        meta_bits.append(f"demo_noise={args.noise:.3f}")
    if args.model_module:
        meta_bits.append(f"model={os.path.basename(args.model_module)}")
    else:
        widths_display = getattr(args, "widths", "") or ",".join(map(str, hidden_widths))
        meta_bits.append(f"demo_widths={widths_display}")
    if getattr(args, "weights", ""):
        meta_bits.append("weights=loaded")
    if args.no_train or (y is None):
        meta_bits.append("training=skipped")
    meta_bits.append(f"ulam={'off' if getattr(args, 'no_ulam', False) else 'on'}")
    print(_subtle(" | ".join(meta_bits)))
    print(
        _format_metric(
            "reward weights",
            (
                f"mass={args.mass_weight:.2f}, waste={args.wasted_weight:.2f}, "
                f"entropy={args.entropy_weight:.2f}, cp={args.cp_weight:.2f}, "
                f"gap={args.gap_weight:.2f}"
            ),
            icon="[HX]",
        )
    )
    if not getattr(args, "no_layer_summary", False):
        _print_model_layers(model)
        print()
    print(_headline("AF levels"))

    rows = []
    step = env.reset()
    while True:
        obs = step.obs
        rows.append(
            {
                "depth": obs["depth"],
                "regions": obs["n_regions"],
                "mass_err": float(obs["mass_error"]),
                "trace_linf": float(obs.get("trace_residual_linf", 0.0)),
                "wasted": obs["wasted_regions"],
                "entropy": float(obs.get("combinatorial_entropy", 0.0)),
                "cp_unital": float(obs.get("cp_unital_err", 0.0)),
                "cp_coiso": float(obs.get("cp_coisometry_err", 0.0)),
                "cp_psd": float(obs.get("cp_psd_violation", 0.0)),
                "gap": obs.get("spectral_gap"),
                "reward": step.reward,
            }
        )
        if step.done:
            break
        step = env.step()

    entropy_max = max((row["entropy"] for row in rows), default=0.0)

    header = (
        f"{'depth':>5} {'regions':>9} {'mass_L1':>12} {'trace_inf':>11} "
        f"{'wasted':>8} {'entropy':>9} {'cp_uni':>9} {'cp_psd':>9} {'gap':>8} {'reward':>10} {'heat':>6}"
    )
    print(header)
    print(_subtle("-" * len(header)))
    for row in rows:
        gap_val = row["gap"]
        if isinstance(gap_val, (int, float)):
            gap_float = float(gap_val)
            gap_field = f"{gap_float:>8.4f}" if not math.isnan(gap_float) else f"{'-':>8}"
        else:
            gap_field = f"{'-':>8}"
        heat_field = _heatmap_for_row(row, entropy_max=entropy_max)
        line = (
            f"{row['depth']:>5} "
            f"{row['regions']:>9} "
            f"{row['mass_err']:>12.4e} "
            f"{row['trace_linf']:>11.4e} "
            f"{row['wasted']:>8} "
            f"{row['entropy']:>9.4f} "
            f"{row['cp_unital']:>9.2e} "
            f"{row['cp_psd']:>9.2e} "
            f"{gap_field} "
            f"{row['reward']:>10.4f} "
            f"{heat_field:>6}"
        )
        if _row_is_flagged(row):
            line = _highlight_line(line)
        print(line)

    mass_series = [row["mass_err"] for row in rows]
    trace_series = [row["trace_linf"] for row in rows]
    entropy_series = [row["entropy"] for row in rows]
    cp_unital_series = [row["cp_unital"] for row in rows]
    cp_coiso_series = [row["cp_coiso"] for row in rows]
    cp_psd_series = [row["cp_psd"] for row in rows]
    wasted_series = [row["wasted"] for row in rows]
    reward_series = [row["reward"] for row in rows]
    gap_series = [
        float(row["gap"])
        for row in rows
        if isinstance(row["gap"], (int, float)) and not math.isnan(float(row["gap"]))
    ]
    print(_format_metric("mass L1", _sequence_with_sparkline(mass_series), icon="[M]"))
    print(_format_metric("trace L∞", _sequence_with_sparkline(trace_series), icon="[T]"))
    print(_format_metric("entropy", _sequence_with_sparkline(entropy_series), icon="[E]"))
    print(_format_metric("cp unital", _sequence_with_sparkline(cp_unital_series), icon="[C]"))
    print(_format_metric("cp coiso", _sequence_with_sparkline(cp_coiso_series), icon="[C]"))
    print(_format_metric("cp psd", _sequence_with_sparkline(cp_psd_series), icon="[C]"))
    if computed_gap is not None:
        print(_format_metric("spectral gap", f"{computed_gap:.4f}", icon="[U]"))

    series_map = {
        "mass_err": mass_series,
        "trace_linf": trace_series,
        "entropy": entropy_series,
        "wasted": wasted_series,
        "cp_unital": cp_unital_series,
        "cp_coiso": cp_coiso_series,
        "cp_psd": cp_psd_series,
        "gap": gap_series,
        "reward": reward_series,
    }
    _render_layer_cards(rows, series_map, entropy_max=entropy_max)
    _print_af_summary(rows, computed_gap=computed_gap)
    _render_summary_plots(rows)

    _print_cli_dashboard(persistent=persistent_summary, capacity=capacity_metrics)

    print(_subtle(_divider("-")))
    print(
        _format_metric(
            "verifiers loader",
            "from environments.helixenv.af_partition import load_environment",
            icon="[HX]",
        )
    )
    print(
        _format_metric(
            "prime integration",
            "register_helix_envs(lambda env_id, fn: ...)",
            icon="[HX]",
        )
    )

    return 0


def run_analyze(args: argparse.Namespace) -> int:
    if torch is None:
        raise RuntimeError("PyTorch required. pip install torch")
    cli_pickled_arrays = getattr(args, "allow_pickled_arrays", False)
    cli_pickled_weights = getattr(args, "allow_pickled_weights", False)
    env_pickled_arrays = _env_flag("HELIX_ALLOW_PICKLED_ARRAYS")
    env_pickled_weights = _env_flag("HELIX_ALLOW_PICKLED_WEIGHTS")
    allow_pickled_arrays = cli_pickled_arrays or env_pickled_arrays
    allow_pickled_weights = cli_pickled_weights or env_pickled_weights
    if cli_pickled_arrays:
        print(
            _subtle(
                "--allow-pickled-arrays: numpy pickle deserialisation enabled for this run."
            )
        )
    elif env_pickled_arrays:
        print(
            _subtle(
                "HELIX_ALLOW_PICKLED_ARRAYS=1: numpy pickle deserialisation enabled for this run."
            )
        )
    if cli_pickled_weights:
        print(
            _subtle(
                "--allow-pickled-weights: pickled checkpoint loading enabled for this run."
            )
        )
    elif env_pickled_weights:
        print(
            _subtle(
                "HELIX_ALLOW_PICKLED_WEIGHTS=1: pickled checkpoint loading enabled for this run."
            )
        )
    _seed_torch(args.seed)

    persistent_summary: Optional[PersistentHomologySummary] = None
    capacity_metrics: Optional[CapacityLossMetrics] = None

    # Load data or synthesize
    if args.data_x:
        X = _load_array(args.data_x, allow_pickle=allow_pickled_arrays)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = X.astype(np.float32)
    else:
        X, _ = make_moons(n=args.samples, noise=args.noise, seed=args.seed)

    try:
        persistent_summary = compute_persistent_homology(X, maxdim=2)
    except Exception as exc:
        print(_subtle(f"[warn] Persistent homology unavailable: {exc}"))

    y = None
    if args.data_y:
        y_arr = _load_array(args.data_y, allow_pickle=allow_pickled_arrays)
        y = y_arr.astype(np.int64).reshape(-1)
        if y.shape[0] != X.shape[0]:
            raise ValueError("data_y length must match number of rows in data_x")

    # Build or import model
    hidden_widths = _hidden_widths_from_args(args)

    if args.model_module:
        builder = _dynamic_import_builder(args.model_module, args.model_func)
        kwargs = {}
        if args.model_kwargs:
            import json as _json

            try:
                kwargs = _json.loads(args.model_kwargs)
            except Exception as e:
                raise ValueError(f"Invalid JSON for --model-kwargs: {e}")
        model = builder(**kwargs)
        if not isinstance(model, nn.Module):
            raise TypeError("Builder must return a torch.nn.Module")
    else:
        d_in = int(X.shape[1]) if X.ndim == 2 else 2
        model = MLP(d_in=d_in, widths=hidden_widths, d_out=args.d_out)

    # Optional weights
    if args.weights:
        state = _load_state_dict(args.weights, allow_pickled=allow_pickled_weights)
        try:
            model.load_state_dict(state)
        except Exception:
            # allow non-strict if user state dict has extras
            model.load_state_dict(state, strict=False)

    # Optional training if labels provided and not disabled
    if (not args.no_train) and (y is not None):
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        try:
            opt = torch.optim.Adam(model.parameters(), lr=1e-2)
            for _ in range(args.epochs):
                opt.zero_grad()
                logits = model(X_t)
                loss = F.cross_entropy(logits, y_t)
                loss.backward()
                opt.step()
        except Exception as e:
            print(f"[warn] Training skipped due to error: {e}")

    try:
        capacity_metrics = compute_capacity_loss(model)
    except Exception as exc:
        print(_subtle(f"[warn] Capacity metrics unavailable: {exc}"))

    # Metrics
    af = extract_partitions(model, X)
    n_list = region_counts(af.B_list)
    errs = mass_consistency_errors(af.B_list, af.tau_list)

    cp_stats = []
    for k, B in enumerate(af.B_list, start=1):
        tau_prev = np.array([1.0]) if k == 1 else af.tau_list[k - 2]
        tau_cur = af.tau_list[k - 1]
        V = build_V_from_incidence(B, tau_prev, tau_cur)
        cp_stats.append(sanity_check_ucp(V, trials=6))

    # Optional Ulam
    gap = None
    if not args.no_ulam:
        d = int(X.shape[1]) if X.ndim == 2 else 2
        # Default box; for d>2, broadcast typical 2D extents
        lo = np.full((d,), -2.0)
        hi = np.full((d,), 2.5)
        if d >= 2:
            lo[:2] = np.array([-2.0, -1.5])
            hi[:2] = np.array([3.0, 2.5])
        # Use a simple residual MLP block unless user disables
        h2 = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, d))
        with torch.no_grad():
            for p in h2.parameters():
                p.mul_(0.1)

        def F_block(x_np: np.ndarray, eps: float = 0.3) -> np.ndarray:
            x_t = torch.from_numpy(x_np.astype(np.float32))
            y_ = x_t + eps * h2(x_t)
            return y_.detach().numpy().astype(np.float64)

        P, _ = ulam_pf(
            lambda z: F_block(z, eps=0.4),
            (lo, hi),
            bins_per_dim=args.ulam_bins,
            samples_per_cell=args.ulam_samples_per_cell,
        )
        gap = spectral_gap(P)

    _animate_banner(loops=1, fps=18.0)
    print(_subtle(_divider()))
    print(_headline("Helix Analyze"))
    meta_bits = [
        f"samples={X.shape[0]}",
        f"dim={int(X.shape[1]) if X.ndim == 2 else 1}",
        f"seed={args.seed}",
    ]
    if args.model_module:
        meta_bits.append(f"model={os.path.basename(args.model_module)}")
    if args.data_x:
        meta_bits.append(f"data={os.path.basename(args.data_x)}")
    print(_subtle(" | ".join(meta_bits)))
    print(_headline("Quick Actions"))
    print(_format_metric("skip training", "--no-train", icon="[?]"))
    print(_format_metric("ulam control", "--no-ulam or --ulam-bins N", icon="[?]"))
    print(_format_metric("share visuals", "--plot --save-prefix run", icon="[?]"))
    print(_subtle(_divider("-")))

    print(_headline("AF Partitions"))
    print(_format_metric("region counts", _prepare_sequence(n_list), icon="[*]"))
    print(_format_metric("mass L1 error", _prepare_sequence(errs), icon="[*]"))

    print(_headline("CP Diagnostics"))
    for depth, stats in enumerate(cp_stats, start=1):
        value = " | ".join(
            [
                f"unital={stats['unital_err_fro']:.2e}",
                f"coiso={stats['coisometry_err_fro']:.2e}",
                f"psd={stats['psd_min_eig_violation']:.2e}",
            ]
        )
        print(_format_metric(f"depth {depth}", value, icon="[C]"))

    if gap is not None:
        print(_headline("Ulam Transfer"))
        print(_format_metric("spectral gap", f"{gap:.4f}", icon="[U]"))

    _print_cli_dashboard(persistent=persistent_summary, capacity=capacity_metrics)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    # If no arguments provided, run interactive mode
    if argv is None or len(argv) == 0:
        # When called without arguments, enter interactive mode
        return run_interactive()

    # Quick subcommand dispatch for `helix tui` without breaking legacy flags-only usage
    if argv and len(argv) > 0 and argv[0] == "tui":
        try:
            from .tui import run_tui
        except Exception as e:  # pragma: no cover
            print(
                f"Helix TUI not available. Install TUI extras: pip install '.[tui]'.\nDetails: {e}"
            )
            return 1
        return run_tui(argv[1:])
    # New: custom analysis subcommand
    if argv and len(argv) > 0 and argv[0] == "analyze":
        pz = argparse.ArgumentParser(description="Helix CLI: analyze a custom model/dataset")
        pz.add_argument(
            "--model-module",
            type=str,
            default="",
            help="Path to Python file that defines a model builder",
        )
        pz.add_argument(
            "--model-func",
            type=str,
            default="build_model",
            help="Builder function name in the module",
        )
        pz.add_argument(
            "--model-kwargs",
            type=str,
            default="",
            help="JSON dict of kwargs to pass to the builder",
        )
        pz.add_argument(
            "--weights", type=str, default="", help="Optional path to a state_dict .pt/.pth file"
        )
        pz.add_argument(
            "--data-x", type=str, default="", help="Path to features array (.npy/.npz/.csv)"
        )
        pz.add_argument(
            "--data-y", type=str, default="", help="Optional path to labels array for training"
        )
        pz.add_argument("--d-out", type=int, default=2, help="Output width if using built-in MLP")
        pz.add_argument(
            "--width", type=int, default=16, help="Hidden width for built-in MLP if used"
        )
        pz.add_argument("--epochs", type=int, default=50)
        pz.add_argument("--no-train", action="store_true")
        pz.add_argument("--ulam-bins", type=int, default=25)
        pz.add_argument("--ulam-samples-per-cell", type=int, default=1)
        pz.add_argument("--no-ulam", action="store_true")
        pz.add_argument("--seed", type=int, default=1)
        pz.add_argument(
            "--allow-pickled-arrays",
            action="store_true",
            help="Permit loading .npy/.npz files that require pickle (disabled by default)",
        )
        pz.add_argument(
            "--allow-pickled-weights",
            action="store_true",
            help="Permit torch.load of pickled checkpoints when you trust the source",
        )
        az = pz.parse_args(argv[1:])
        return run_analyze(az)

    if argv and len(argv) > 0 and argv[0] in {"helixenv", "helix-env", "oa-env", "oa"}:
        po = _build_helix_env_parser(prog="helix helixenv")
        helix_args = po.parse_args(argv[1:])
        return run_helix_env(helix_args)

    if argv and len(argv) > 0 and argv[0] == "env":
        if len(argv) == 1:
            return run_interactive()
        env_cmd = argv[1]
        rest = argv[2:]
        if env_cmd in {"operator", "oa", "analytics", "helix"}:
            po = _build_helix_env_parser(prog="helix env helix")
            helix_args = po.parse_args(rest)
            return run_helix_env(helix_args)
        if env_cmd in {"summary", "overview"}:
            return run_helixenv_overview()
        if env_cmd in {"interactive", "menu"}:
            return run_interactive()
        print(f"Unknown environment option: {env_cmd}")
        print("Use 'operator' or 'helix'.")
        return 1

    p = argparse.ArgumentParser(description="Helix CLI: run diagnostics and plots")
    p.add_argument(
        "demo",
        nargs="?",
        default=None,
        help="run demo (use 'demo' explicitly or use --non-interactive)",
    )
    p.add_argument(
        "--non-interactive", action="store_true", help="Skip interactive menu and run directly"
    )
    p.add_argument("--samples", type=int, default=4000)
    p.add_argument("--noise", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="Legacy single hidden width; overrides --widths when provided",
    )
    p.add_argument(
        "--widths",
        type=str,
        default="16,16,16,16",
        help="Comma-separated hidden widths for the demo network",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--ulam-bins", type=int, default=25, help="Ulam bins per dimension")
    p.add_argument(
        "--ulam-samples-per-cell",
        type=int,
        default=1,
        help="Number of random samples per grid cell for Ulam (barycentric)",
    )
    p.add_argument("--plot", action="store_true")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--save-prefix", type=str, default="")
    args = p.parse_args(argv)

    # If --non-interactive flag is set or 'demo' positional arg, run demo directly
    if args.non_interactive or args.demo == "demo":
        return run_demo(args)

    # Back-compat: allow `helix tui` to pass through even if parsed as positional
    if getattr(args, "demo", None) == "tui":
        try:
            from .tui import run_tui
        except Exception as e:  # pragma: no cover
            print(
                f"Helix TUI not available. Install TUI extras: pip install '.[tui]'.\nDetails: {e}"
            )
            return 1
        return run_tui([])

    # If we have other flags but no explicit command, run demo with those flags
    if any(
        [
            args.plot,
            args.no_train,
            args.save_prefix,
            args.no_show,
            args.samples != 4000,
            args.noise != 0.07,
            args.width is not None,
            args.widths != "16,16,16,16",
            args.epochs != 100,
            args.ulam_bins != 25,
            args.ulam_samples_per_cell != 1,
        ]
    ):
        return run_demo(args)

    # Otherwise, run interactive mode
    return run_interactive()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
