from __future__ import annotations

import argparse
import getpass
import json
import locale
import math
import os
import re
import shutil
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from textwrap import dedent
from time import time as _time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:
    import termios
    import tty

    _HAS_TERMIOS = True
except ImportError:  # pragma: no cover
    _HAS_TERMIOS = False

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
    try:
        if bool(stream) and stream.isatty():
            return True
    except Exception:
        pass
    alt_stream = getattr(sys, "__stdout__", None)
    if alt_stream is not stream:
        try:
            if bool(alt_stream) and alt_stream.isatty():
                return True
        except Exception:
            pass
    return _force_interactive_output()


def _oklch_to_rgb(lightness: float, chroma: float, hue: float) -> Tuple[int, int, int]:
    """
    Convert OKLCH color to RGB (0-255 range).
    
    OKLCH is LCH in the OKLab color space, which is perceptually uniform.
    Conversion: OKLCH -> OKLab -> Linear RGB -> sRGB
    """
    # Convert OKLCH to OKLab
    # lightness is lightness (0-1), chroma is chroma, hue is hue (0-360 degrees)
    h_rad = math.radians(hue)
    a = chroma * math.cos(h_rad)
    b = chroma * math.sin(h_rad)
    
    # OKLab to linear RGB matrix (inverse of OKLab transformation)
    # Using the standard OKLab to linear RGB conversion
    l_ = lightness + 0.3963377774 * a + 0.2158037573 * b
    m_ = lightness - 0.1055613458 * a - 0.0638541728 * b
    s_ = lightness - 0.0894841775 * a - 1.2914855480 * b
    
    # Apply non-linearity
    l_ = l_ * l_ * l_
    m_ = m_ * m_ * m_
    s_ = s_ * s_ * s_
    
    # Linear RGB
    r = +4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    b_val = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_
    
    # Apply sRGB gamma correction
    def srgb_gamma(x: float) -> float:
        if x <= 0.0031308:
            return 12.92 * x
        return 1.055 * (x ** (1.0 / 2.4)) - 0.055
    
    r = srgb_gamma(r)
    g = srgb_gamma(g)
    b_val = srgb_gamma(b_val)
    
    # Clamp and convert to 0-255 range
    r = max(0, min(255, int(round(r * 255))))
    g = max(0, min(255, int(round(g * 255))))
    b_val = max(0, min(255, int(round(b_val * 255))))
    
    return (r, g, b_val)


def _rgb_to_ansi24(r: int, g: int, b: int) -> str:
    """Convert RGB values (0-255) to ANSI 24-bit color code."""
    return f"38;2;{r};{g};{b}"


# OKLCH color definitions
# Format: (L, C, H) where L is lightness (0-1), C is chroma, H is hue (0-360)
COLORS = {
    "accent": (0.65, 0.20, 320.0),  # Bright magenta/pink
    "shadow": (0.55, 0.18, 240.0),  # Bright blue
    "headline": (0.70, 0.15, 140.0),  # Bright green
    "subtle": (0.45, 0.05, 0.0),  # Dark gray
    "error": (0.60, 0.22, 15.0),  # Bright red
    "highlight": (0.75, 0.18, 200.0),  # Cyan
    "hover": (0.72, 0.16, 290.0),  # Light purple for hover/highlight
}


def _oklch_to_ansi(oklch: Tuple[float, float, float]) -> str:
    """Convert OKLCH color tuple to ANSI 24-bit color code."""
    r, g, b = _oklch_to_rgb(oklch[0], oklch[1], oklch[2])
    return _rgb_to_ansi24(r, g, b)


def _style(text: str, *, color: Optional[str] = None, bold: bool = False) -> str:
    if not _supports_color():
        return text
    codes: List[str] = []
    if color:
        # If color is an OKLCH tuple, convert it
        if isinstance(color, tuple) and len(color) == 3:
            color = _oklch_to_ansi(color)
        codes.append(color)
    if bold:
        codes.append("1")
    if not codes:
        return text
    prefix = "\033[" + ";".join(codes) + "m"
    return f"{prefix}{text}\033[0m"


_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY_ENV_VALUES


_FALSEY_FORCE_VALUES = {"", "0", "false", "no", "off"}


def _force_interactive_output() -> bool:
    bool_flags = (
        "HELIX_FORCE_ASCII",
        "HELIX_FORCE_TTY",
        "HELIX_FORCE_COLOR",
        "PYCHARM_HOSTED",
        "PYCHARM_ACTIVE",
    )
    if any(_env_flag(name) for name in bool_flags):
        return True
    force_color = os.environ.get("FORCE_COLOR")
    if force_color and force_color.strip().lower() not in _FALSEY_FORCE_VALUES:
        return True
    if os.environ.get("VSCODE_PID"):
        return True
    return False


def _supports_unicode_drawing() -> bool:
    if _env_flag("HELIX_ASCII_GRAPH"):
        return False
    stream = getattr(sys, "stdout", None)
    encoding = ""
    if stream is not None:
        encoding = getattr(stream, "encoding", "") or ""
    if not encoding:
        encoding = locale.getpreferredencoding(False) or ""
    encoding = encoding.upper()
    if any(token in encoding for token in ("UTF", "UNICODE")):
        return True
    term = os.environ.get("TERM", "").lower()
    if "utf" in term or "xterm" in term:
        return True
    return False


def _load_state_dict(path: str, *, allow_pickled: bool = False):
    if torch is None:
        raise RuntimeError("PyTorch is not available; cannot load weights.")

    load_kwargs = {"map_location": "cpu"}
    if not allow_pickled:
        try:
            return torch.load(path, weights_only=True, **load_kwargs)
        except TypeError as exc:
            suggestion = (
                "Upgrade PyTorch or rerun with --allow-pickled-weights "
                "(or HELIX_ALLOW_PICKLED_WEIGHTS=1) if you trust the checkpoint."
            )
            raise RuntimeError(
                "Safe checkpoint loading requires torch>=2.0. " + suggestion
            ) from exc
        except RuntimeError as exc:
            message = str(exc).lower()
            if "weights_only" in message or "state_dict" in message:
                hint = (
                    "Rerun with --allow-pickled-weights or set "
                    "HELIX_ALLOW_PICKLED_WEIGHTS=1 if you trust the source."
                )
                raise RuntimeError(
                    "Checkpoint does not expose a plain state_dict. " + hint
                ) from exc
            raise
    return torch.load(path, **load_kwargs)


TRIM_TOP_LINES = 4
TRIM_MAX_LINES = 18
TRIM_BOTTOM_LINES = 0  # keep bottom row so the full helix animates


def _trim_frame(
    frame: str,
    top: int = TRIM_TOP_LINES,
    max_lines: int = TRIM_MAX_LINES,
    bottom: int = TRIM_BOTTOM_LINES,
) -> str:
    lines = frame.splitlines()
    should_trim_top = top > 0 and (max_lines <= 0 or len(lines) > max_lines)
    if should_trim_top:
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

    # Drop blank rows from the top and bottom so the helix hugs the origin.
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]

    # Remove any common leading whitespace so frames render flush-left.
    min_indent: Optional[int] = None
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            continue
        indent = len(line) - len(stripped)
        if min_indent is None or indent < min_indent:
            min_indent = indent
    if min_indent:
        lines = [line[min_indent:] if len(line) >= min_indent else "" for line in lines]

    # Trim trailing whitespace from each line so we only keep the animated core.
    lines = [line.rstrip() for line in lines]
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
            pad_count = max_height - len(padded_lines)
            padded_lines = padded_lines + ([" " * max_width] * pad_count)
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

    return frames


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

        frames.append("\n".join(rows))

    return frames


ASCII_HELIX_FRAMES = _load_external_helix_frames() or _build_helix_frames()

# Build a normalized set of frames for reliable terminal rendering
NORMALIZED_HELIX_FRAMES, NORMALIZED_HELIX_WIDTH, NORMALIZED_HELIX_HEIGHT = (
    _normalize_frames([_trim_frame(f) for f in ASCII_HELIX_FRAMES])
    if ASCII_HELIX_FRAMES
    else ([], 0, 0)
)


def _layer_blueprint_from_widths(d_in: int, widths: Sequence[int], d_out: int) -> List[int]:
    blueprint: List[int] = []
    for value in [d_in, *widths, d_out]:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = 1
        blueprint.append(max(1, numeric))
    return blueprint


def _layer_labels_from_blueprint(blueprint: Sequence[int]) -> List[str]:
    labels: List[str] = []
    total = len(blueprint)
    for idx in range(total):
        if idx == 0:
            labels.append("in")
        elif idx == total - 1:
            labels.append("out")
        else:
            labels.append(f"h{idx}")
    return labels


def _nodes_for_layer(size: int) -> int:
    return max(3, min(8, int(round(math.sqrt(float(max(size, 1)))) + 2)))


def _normalize_activity_sequence(
    values: Sequence[float], *, target_len: int, fallback: float = 0.5
) -> List[float]:
    if target_len <= 0:
        return []
    cleaned = [max(0.0, float(v)) for v in values if isinstance(v, (int, float))]
    if not cleaned:
        cleaned = [fallback]
    while len(cleaned) < target_len:
        cleaned.append(cleaned[-1])
    if len(cleaned) > target_len:
        cleaned = cleaned[:target_len]
    max_val = max(cleaned)
    if max_val <= 0:
        return [fallback for _ in range(target_len)]
    return [min(1.0, val / max_val) for val in cleaned]


def _activity_char(level: float, *, highlight: bool = False, use_unicode: bool = False) -> str:
    if use_unicode:
        palette = ["·", "∙", "○", "◍", "●"] if not highlight else ["·", "○", "◎", "◉", "●"]
    else:
        palette = [" ", ".", "o", "O", "@"] if not highlight else [" ", ".", "o", "@", "*"]
    clamped = max(0.0, min(1.0, level))
    idx = min(len(palette) - 1, int(round(clamped * (len(palette) - 1))))
    return palette[idx]


def _colorize_activation(char: str, level: float, *, highlight: bool = False) -> str:
    if not _supports_color():
        return char
    if highlight:
        color = "95"
        bold = True
    elif level >= 0.75:
        color = "91"
        bold = True
    elif level >= 0.5:
        color = "93"
        bold = False
    elif level >= 0.25:
        color = "96"
        bold = False
    else:
        color = "90"
        bold = False
    return _style(char, color=color, bold=bold)


def _colorize_edge(char: str, strength: float) -> str:
    if not _supports_color():
        return char
    if strength >= 0.75:
        color = "94"
    elif strength >= 0.5:
        color = "36"
    else:
        color = "90"
    return _style(char, color=color)


def _visible_len(text: str) -> int:
    return len(_ANSI_ESCAPE_RE.sub("", text))


def _set_auto_wrap(enabled: bool) -> None:
    """Enable or disable terminal auto-wrap so layouts stay fixed-width."""
    seq = "\x1b[?7h" if enabled else "\x1b[?7l"
    try:
        sys.stdout.write(seq)
        sys.stdout.flush()
    except Exception:
        pass


def _truncate_display_width(text: str, max_width: int) -> str:
    """Clip ANSI-colored text so it fits within the provided column width."""
    if max_width <= 0:
        return ""
    if not text:
        return ""

    def _clip_line(line: str) -> str:
        visible = 0
        idx = 0
        pieces: List[str] = []
        saw_escape = False
        while idx < len(line) and visible < max_width:
            if line[idx] == "\x1b":
                match = _ANSI_ESCAPE_RE.match(line, idx)
                if match:
                    pieces.append(match.group(0))
                    saw_escape = True
                    idx = match.end()
                    continue
            pieces.append(line[idx])
            visible += 1
            idx += 1
        truncated = idx < len(line)
        if truncated and saw_escape and _supports_color():
            pieces.append("\x1b[0m")
        return "".join(pieces)

    lines = text.split("\n")
    truncated_lines: List[str] = []
    any_truncated = False
    for line in lines:
        if _visible_len(line) <= max_width:
            truncated_lines.append(line)
            continue
        any_truncated = True
        truncated_lines.append(_clip_line(line))
    if not any_truncated:
        return text
    return "\n".join(truncated_lines)


def _animate_forward_pass(
    layer_labels: Sequence[str],
    activities: Sequence[float],
    *,
    delay: float = 0.28,
) -> None:
    if not _is_tty():
        return
    labels = list(layer_labels)
    scores = _normalize_activity_sequence(activities, target_len=len(labels) or 1)
    unicode_ok = _supports_unicode_drawing()
    bar_char = "█" if unicode_ok else "#"
    max_width = 26
    lines = len(labels) or 1
    sys.stdout.write("\n" * (lines + 2))
    sys.stdout.write(f"\x1b[{lines + 2}A")
    sys.stdout.flush()
    print(_subtle("Forward activation sweep"))
    for idx, label in enumerate(labels):
        level = scores[idx]
        bar_len = max(1, int(round(level * max_width)))
        color = "92" if level >= 0 else "91"
        if level >= 0.66:
            color = "92"
        elif level >= 0.33:
            color = "96"
        else:
            color = "90"
        bar = bar_char * bar_len
        strength = _style(bar, color=color, bold=level >= 0.66)
        score_text = _style(f"{level:.2f}", color="93")
        line = f"{label:<6} {strength} {score_text}"
        sys.stdout.write("\x1b[2K")
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        time.sleep(delay)
    print()


def _infer_summary_input_shape(array: Optional[np.ndarray], d_in: int) -> Optional[Tuple[int, ...]]:
    if array is None:
        if d_in > 0:
            return (1, d_in)
        return None
    if array.ndim == 1:
        return (1, array.shape[0])
    shape = array.shape[1:]
    if not shape:
        return (1, int(array.shape[0]))
    return (1,) + tuple(int(dim) for dim in shape)


def _maybe_print_model_summary(model: nn.Module, input_shape: Optional[Tuple[int, ...]]) -> bool:
    if not _is_tty():
        return False
    if _env_flag("HELIX_SKIP_MODEL_SUMMARY"):
        return False
    if torch is None or not isinstance(model, nn.Module):
        return False

    summary_fn = None
    summary_kind = ""
    try:
        from torchinfo import summary as torchinfo_summary  # type: ignore
        summary_fn = torchinfo_summary
        summary_kind = "torchinfo"
    except Exception:
        try:
            from torchsummary import summary as torchsummary_summary  # type: ignore
            summary_fn = torchsummary_summary
            summary_kind = "torchsummary"
        except Exception:
            return False

    if input_shape is None:
        return False

    try:
        print(_headline("Model Summary"))
        if summary_kind == "torchinfo":
            result = summary_fn(
                model,
                input_size=input_shape,
                col_names=("input_size", "output_size", "num_params"),
                verbose=0,
            )
            print(result)
        else:
            params = list(model.parameters())
            orig_device = params[0].device if params else torch.device("cpu")
            target_model = model if orig_device.type == "cpu" else model.to("cpu")
            try:
                with torch.no_grad():
                    summary_fn(target_model, input_size=input_shape[1:])
            finally:
                if target_model is not model and orig_device:
                    model.to(orig_device)
        return True
    except Exception as exc:  # pragma: no cover
        print(_subtle(f"[warn] model summary unavailable: {exc}"))
    return False


def _render_neuron_panel(
    layer_sizes: Sequence[int],
    layer_labels: Sequence[str],
    activities: Sequence[float],
    *,
    caption: str,
    metrics: Optional[Sequence[str]] = None,
    pulse_phase: float = 0.0,
    highlight: bool = False,
    extra_lines: Optional[Sequence[str]] = None,
) -> str:
    if not layer_sizes:
        return ""
    use_unicode = _supports_unicode_drawing()
    num_layers = len(layer_sizes)
    metrics = list(metrics or [])
    while len(metrics) < num_layers:
        metrics.append("")
    activities = _normalize_activity_sequence(activities, target_len=num_layers)
    nodes_per_layer = [_nodes_for_layer(size) for size in layer_sizes]
    column_step = 8 if use_unicode else 6
    height = max(11, max(nodes_per_layer) * 2 + 3)
    width = column_step * max(1, num_layers - 1) + 9
    top_margin = 2
    bottom_margin = 2
    grid: List[List[str]] = [[" " for _ in range(width)] for _ in range(height)]
    positions: List[List[Dict[str, float]]] = []

    def _compute_levels(layer_idx: int, node_idx: int, base: float) -> float:
        phase = pulse_phase + layer_idx * 0.9 + node_idx * 0.4
        return max(0.05, min(1.0, 0.55 * base + 0.45 * (0.5 + 0.5 * math.sin(phase))))

    for layer_idx, node_count in enumerate(nodes_per_layer):
        x_pos = 3 + layer_idx * column_step
        if node_count <= 1:
            y_positions = [height // 2]
        else:
            span = height - top_margin - bottom_margin - 1
            if span <= 0:
                span = max(2, height - 2)
            y_positions = [
                top_margin + int(round(idx * span / (node_count - 1)))
                for idx in range(node_count)
            ]
        layer_nodes: List[Dict[str, float]] = []
        base_activity = activities[layer_idx]
        for node_idx, y_pos in enumerate(y_positions):
            level = _compute_levels(layer_idx, node_idx, base_activity)
            layer_nodes.append({"x": x_pos, "y": y_pos, "level": level})
        positions.append(layer_nodes)

    def _plot_edge(x0: int, y0: int, x1: int, y1: int, strength: float, phase_seed: float) -> None:
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy), 1)
        for step in range(1, steps):
            x = x0 + round(dx * step / steps)
            y = y0 + round(dy * step / steps)
            if not (0 <= x < width and 0 <= y < height):
                continue
            vertical_char = "│" if use_unicode else "|"
            vertical_light = "╎" if use_unicode else "|"
            horizontal_char = "═" if use_unicode else "="
            horizontal_light = "─" if use_unicode else "-"
            diag_up = "╱" if use_unicode else "/"
            diag_down = "╲" if use_unicode else "\\"
            if abs(dx) < abs(dy) * 0.45:
                char = vertical_char if strength > 0.6 else vertical_light
            elif abs(dy) < abs(dx) * 0.45:
                char = horizontal_char if strength > 0.6 else horizontal_light
            elif dy > 0:
                char = diag_up
            else:
                char = diag_down
            signal_phase = (phase_seed + step / steps) % 1.0
            if strength >= 0.45 and signal_phase < 0.18:
                pulse_char = "•" if use_unicode else "*"
                char = _colorize_activation(pulse_char, strength, highlight=False)
            else:
                char = _colorize_edge(char, strength)
            grid[y][x] = char

    for layer_idx in range(num_layers - 1):
        src_nodes = positions[layer_idx]
        dst_nodes = positions[layer_idx + 1]
        for src_idx, src in enumerate(src_nodes):
            sorted_targets = sorted(dst_nodes, key=lambda node: abs(node["y"] - src["y"]))
            target_count = max(1, min(len(sorted_targets), 4))
            targets = sorted_targets[:target_count]
            for dst_idx, dst in enumerate(targets):
                strength = 0.5 * (src["level"] + dst["level"])
                phase_seed = (
                    pulse_phase + 0.2 * layer_idx + 0.07 * src_idx + 0.03 * dst_idx
                )
                _plot_edge(
                    int(src["x"]),
                    int(src["y"]),
                    int(dst["x"]),
                    int(dst["y"]),
                    strength=strength,
                    phase_seed=phase_seed,
                )

    glow_char = "·" if use_unicode else "."
    for layer_idx, layer_nodes in enumerate(positions):
        for node_idx, node in enumerate(layer_nodes):
            x = int(node["x"])
            y = int(node["y"])
            if 0 <= x < width and 0 <= y < height:
                firing = node["level"] > 0.78 or highlight
                char = _activity_char(node["level"], highlight=firing, use_unicode=use_unicode)
                char = _colorize_activation(char, node["level"], highlight=firing)
                grid[y][x] = char
                if firing:
                    for dx_offset, dy_offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx_offset, y + dy_offset
                        if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == " ":
                            grid[ny][nx] = _colorize_activation(
                                glow_char,
                                node["level"] * 0.6,
                                highlight=False,
                            )

    grid_lines = ["".join(row).rstrip() for row in grid]
    label_line: List[Tuple[int, str]] = []
    for idx, label in enumerate(layer_labels):
        x_pos = 3 + idx * column_step
        token = label[:3].upper()
        label_line.append((x_pos, token))

    if grid_lines:
        label_row = [" "] * width
        for x_pos, token in label_line:
            start = max(0, x_pos - len(token) // 2)
            for j, ch in enumerate(token):
                pos = start + j
                if 0 <= pos < width:
                    label_row[pos] = ch
        label_text = "".join(label_row).rstrip()
        if label_text.strip():
            grid_lines.append(label_text)

    info_lines = [f"{label:<6} {metrics[idx]}" for idx, label in enumerate(layer_labels)]
    if extra_lines:
        info_lines.extend(line for line in extra_lines if line)

    raw_lines = grid_lines + [""] + info_lines if info_lines else grid_lines
    inner_width = max(
        64,
        min(
            140,
            max((_visible_len(line) for line in raw_lines + [caption]), default=64),
        ),
    )

    if use_unicode:
        border_char = "═" if highlight else "─"
        divider_char = "─"
        border = f"╔{border_char * (inner_width + 2)}╗"
        sep = f"╟{divider_char * (inner_width + 2)}╢"
        bottom = f"╚{border_char * (inner_width + 2)}╝"
    else:
        border_char = "=" if highlight else "-"
        border = "+" + border_char * (inner_width + 2) + "+"
        sep = "+" + "-" * (inner_width + 2) + "+"
        bottom = border

    def _pad(line: str) -> str:
        padding = max(0, inner_width - _visible_len(line))
        return f"| {line}{' ' * padding} |" if not use_unicode else f"║ {line}{' ' * padding} ║"

    output_lines = [border, _pad(caption.center(inner_width)), sep]
    output_lines.extend(_pad(line) for line in raw_lines)
    output_lines.append(bottom)
    return "\n".join(output_lines)


class NeuronAnimator:
    def __init__(
        self,
        layer_sizes: Sequence[int],
        *,
        caption: str = "Neuron activity",
        linger_seconds: float = 1.5,
    ):
        sanitized: List[int] = []
        for value in layer_sizes:
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                continue
            if numeric > 0:
                sanitized.append(max(1, numeric))
        self.layer_sizes = sanitized
        self.layer_labels = _layer_labels_from_blueprint(self.layer_sizes)
        self._caption = caption
        self._metrics = [f"width={size}" for size in self.layer_sizes]
        self._activity = _normalize_activity_sequence(
            self.layer_sizes,
            target_len=len(self.layer_sizes),
        )
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._panel_lines = max(1, len(self.layer_sizes) + 6)
        self._anchor_saved = False
        self.inline_summary_displayed = False
        self._last_panel_text = ""
        self._linger_seconds = max(0.0, float(linger_seconds))
        self._extra_lines: Optional[List[str]] = None
        self._highlight_panel = False

    @property
    def last_panel_text(self) -> str:
        return self._last_panel_text

    def start(self, caption: str = "Neuron activity", fps: float = 12.0) -> None:
        if not self.layer_sizes or not _is_tty():
            return
        with self._lock:
            self._caption = caption
            self._last_panel_text = ""
            self._extra_lines = None
            self._highlight_panel = False
        first_frame = _render_neuron_panel(
            self.layer_sizes,
            self.layer_labels,
            self._activity,
            caption=self._caption,
            metrics=self._metrics,
            pulse_phase=0.0,
            highlight=False,
        )
        self._panel_lines = max(1, len(first_frame.splitlines()))
        if not self._prepare_anchor():
            return
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, args=(fps,), daemon=True)
        self._thread.start()

    def stop(self, *, erase: bool = True) -> None:
        self._join_thread()
        self._release_anchor(erase=erase)
        self._last_panel_text = ""
        self._extra_lines = None
        self._highlight_panel = False

    def update_status(self, caption: str) -> None:
        if not caption:
            return
        with self._lock:
            self._caption = caption

    def update_activity_hint(self, values: Sequence[float]) -> None:
        with self._lock:
            self._activity = _normalize_activity_sequence(values, target_len=len(self.layer_sizes))

    def finish(
        self,
        *,
        activities: Optional[Sequence[float]] = None,
        metrics: Optional[Sequence[str]] = None,
        caption: str = "Neuron diagnostics",
        extra_lines: Optional[Sequence[str]] = None,
        highlight: bool = True,
        linger_seconds: Optional[float] = None,
    ) -> str:
        linger_for = self._linger_seconds if linger_seconds is None else max(0.0, linger_seconds)
        self._linger_animation(linger_for)
        self._join_thread()
        if not self.layer_sizes:
            self._last_panel_text = ""
            self._release_anchor(erase=True)
            return ""
        final_metrics = metrics or self._metrics
        activity_profile = (
            activities
            if activities is not None
            else _normalize_activity_sequence(self.layer_sizes, target_len=len(self.layer_sizes))
        )
        panel_text = _render_neuron_panel(
            self.layer_sizes,
            self.layer_labels,
            activity_profile,
            caption=caption,
            metrics=final_metrics,
            pulse_phase=0.0,
            highlight=highlight,
            extra_lines=extra_lines,
        )
        self._last_panel_text = panel_text
        if self._anchor_saved and _is_tty():
            self._write_frame(panel_text)
            self.inline_summary_displayed = True
            self._release_anchor(erase=False)
            return ""
        self.inline_summary_displayed = False
        self._release_anchor(erase=True)
        return panel_text

    def _run_loop(self, fps: float) -> None:
        if self._stop_event is None:
            return
        delay = 1.0 / max(1.0, fps)
        phase = 0.0

        while not self._stop_event.is_set():
            with self._lock:
                frame = _render_neuron_panel(
                    self.layer_sizes,
                    self.layer_labels,
                    self._activity,
                    caption=self._caption,
                    metrics=self._metrics,
                    pulse_phase=phase,
                    highlight=self._highlight_panel,
                    extra_lines=self._extra_lines,
                )
            self._write_frame(frame)
            phase += 0.6
            if self._stop_event.wait(delay):
                break

    def _prepare_anchor(self) -> bool:
        if not _is_tty():
            return False
        reserve_lines = self._panel_lines + 1
        sys.stdout.write("\n" * reserve_lines)
        sys.stdout.write(f"\x1b[{reserve_lines}A")
        sys.stdout.write("\x1b[s")
        sys.stdout.flush()
        self._anchor_saved = True
        return True

    def _write_frame(self, frame: str) -> None:
        if not self._anchor_saved:
            return
        lines = frame.splitlines()
        self._panel_lines = max(self._panel_lines, len(lines))
        total = self._panel_lines
        sys.stdout.write("\x1b[u")
        sys.stdout.write("\x1b[s")
        sys.stdout.write("\x1b[0G")
        for idx in range(total):
            sys.stdout.write("\x1b[2K")
            if idx < len(lines):
                sys.stdout.write(lines[idx])
            if idx < total - 1:
                sys.stdout.write("\n")
                sys.stdout.write("\x1b[0G")
        sys.stdout.write("\x1b[u")
        sys.stdout.write("\x1b[s")
        sys.stdout.flush()

    def _release_anchor(self, erase: bool) -> None:
        if not self._anchor_saved:
            return
        sys.stdout.write("\x1b[u")
        sys.stdout.write("\x1b[0G")
        if erase:
            for _ in range(self._panel_lines):
                sys.stdout.write("\x1b[2K\n")
            sys.stdout.write("\x1b[2K")
        else:
            sys.stdout.write(f"\x1b[{self._panel_lines}B")
            sys.stdout.write("\x1b[0G")
        sys.stdout.flush()
        self._anchor_saved = False

    def _join_thread(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=0.5)
        self._stop_event = None
        self._thread = None

    def _linger_animation(self, seconds: float) -> None:
        if self._stop_event is None or seconds <= 0:
            return
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            time.sleep(min(0.05, remaining))
            if self._stop_event is None:
                break


def _activity_profile_from_stats(
    blueprint: Sequence[int],
    region_counts: Sequence[int],
    spectral_gap: Optional[float],
) -> List[float]:
    total_layers = len(blueprint)
    if total_layers == 0:
        return []
    hidden_count = max(0, total_layers - 2)
    activities: List[float] = [1.0]
    relevant_regions = list(region_counts[:hidden_count])
    max_regions = max(relevant_regions, default=1)
    for idx in range(hidden_count):
        rc = region_counts[idx] if idx < len(region_counts) else 0
        if max_regions <= 0:
            activities.append(0.0)
        else:
            activities.append(min(1.0, rc / max(1, max_regions)))
    while len(activities) < total_layers - 1:
        activities.append(activities[-1] if activities else 0.5)
    gap_val = 0.0
    if isinstance(spectral_gap, (int, float)):
        gap_float = float(spectral_gap)
        if not math.isnan(gap_float):
            gap_val = max(0.0, min(1.0, gap_float))
    activities.append(gap_val)
    return activities[:total_layers]


def _neuron_metric_lines(
    blueprint: Sequence[int],
    region_counts: Sequence[int],
    mass_errors: Sequence[float],
    cp_stats: Sequence[Dict[str, float]],
    spectral_gap: Optional[float],
) -> List[str]:
    if not blueprint:
        return []
    lines: List[str] = [f"d={blueprint[0]}"]
    hidden_count = max(0, len(blueprint) - 2)
    for idx in range(hidden_count):
        rc = region_counts[idx] if idx < len(region_counts) else 0
        mass = mass_errors[idx] if idx < len(mass_errors) else 0.0
        stats = cp_stats[idx] if idx < len(cp_stats) else {}
        bits = [f"R={rc}"]
        bits.append(f"M={mass:.1e}")
        if "unital_err_fro" in stats:
            bits.append(f"U={stats['unital_err_fro']:.1e}")
        if "coisometry_err_fro" in stats:
            bits.append(f"C={stats['coisometry_err_fro']:.1e}")
        if "psd_min_eig_violation" in stats:
            bits.append(f"P={stats['psd_min_eig_violation']:.1e}")
        lines.append(" ".join(bits))
    if len(blueprint) >= 2:
        if (
            spectral_gap is None
            or not isinstance(spectral_gap, (int, float))
            or math.isnan(float(spectral_gap))
        ):
            lines.append("gap=--")
        else:
            lines.append(f"gap={float(spectral_gap):.4f}")
    return lines


def _neuron_extra_lines(
    mass_errors: Sequence[float],
    cp_stats: Sequence[Dict[str, float]],
    spectral_gap: Optional[float],
) -> List[str]:
    extras: List[str] = []
    if mass_errors:
        extras.append(f"max mass L1={max(abs(err) for err in mass_errors):.2e}")
    if cp_stats:
        max_unital = max(stats.get("unital_err_fro", 0.0) for stats in cp_stats)
        extras.append(f"max unital={max_unital:.2e}")
    if isinstance(spectral_gap, (int, float)):
        gap_float = float(spectral_gap)
        if not math.isnan(gap_float):
            extras.append(f"spectral gap={gap_float:.4f}")
    return extras
def _colorize_frame(frame: str) -> str:
    lines = frame.splitlines()
    if not _supports_color():
        return "\n".join(lines)

    accent = COLORS["accent"]
    shadow = COLORS["shadow"]
    styled: List[str] = []
    cutoff = len(lines) // 2
    for idx, line in enumerate(lines):
        base_color = accent if idx <= cutoff else shadow
        if "HELIX" in line:
            left, right = line.split("HELIX", 1)
            left_colored = _style(left, color=base_color, bold=True)
            core = _style("HELIX", color=COLORS["accent"], bold=True)
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
    try:
        if bool(stream) and stream.isatty():
            return True
    except Exception:
        pass
    alt_stream = getattr(sys, "__stdout__", None)
    if alt_stream is not stream:
        try:
            if bool(alt_stream) and alt_stream.isatty():
                return True
        except Exception:
            pass
    return _force_interactive_output()


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
    animation_paused: Optional[threading.Event] = None,
    ready_event: Optional[threading.Event] = None,
    refresh_callback: Optional[Callable[[], None]] = None,
    screen_lock: Optional[threading.Lock] = None,
    max_cols: Optional[int] = None,
) -> None:
    if not ASCII_HELIX_FRAMES or not _is_tty():
        return

    # Always anchor the animation to the terminal origin so window resizes
    # can't desynchronise the banner from the menu below.
    # Wait for main thread to signal it's ready
    if ready_event:
        ready_event.wait()
        # Add initial delay to ensure main thread has finished all output
        time.sleep(0.1)

    delay = 1.0 / max(fps, 1.0)
    frames = (
        NORMALIZED_HELIX_FRAMES
        if NORMALIZED_HELIX_FRAMES
        else [_trim_frame(frame) for frame in ASCII_HELIX_FRAMES]
    )
    move_home = "\x1b[H"
    frame_idx = 0
    try:
        initial_size = shutil.get_terminal_size((80, 24))
        last_size = (initial_size.columns, initial_size.lines)
    except Exception:
        last_size = (80, 24)

    while not stop_event.is_set():
        # Get terminal size for bounds checking and resize detection
        try:
            size = shutil.get_terminal_size((80, 24))
            term_cols = size.columns
            current_size = (size.columns, size.lines)
        except Exception:
            term_cols = 80
            current_size = last_size

        size_changed = current_size != last_size
        last_size = current_size

        # Always draw with the requested width so layout stays identical regardless of resize.
        if max_cols is not None:
            draw_cols = max(1, min(term_cols, max_cols))
        else:
            draw_cols = max(1, term_cols)

        paused = bool(animation_paused and animation_paused.is_set())
        if paused:
            if size_changed and refresh_callback:
                try:
                    refresh_callback()
                except Exception:
                    pass
            if stop_event.wait(delay):
                break
            continue

        if screen_lock:
            screen_lock.acquire()
        try:
            sys.stdout.write("\x1b[u")  # restore to prompt/input position
            sys.stdout.write("\x1b[s")  # immediately resave for return after drawing
            sys.stdout.write(move_home)  # always draw from the top-left corner
            sys.stdout.write("\x1b[0G")
            colored_frame = _colorize_frame(frames[frame_idx])
            frame_lines = colored_frame.splitlines()

            # Ensure we have exactly the expected number of lines
            while len(frame_lines) < frame_height:
                frame_lines.append("")

            for idx, line in enumerate(frame_lines[:frame_height]):
                sys.stdout.write("\x1b[2K")  # Clear entire line
                # Clip using visible width to avoid wrapping when resized
                clipped = _truncate_display_width(line, draw_cols)
                sys.stdout.write(clipped)
                if idx < frame_height - 1:
                    sys.stdout.write("\n")
                    sys.stdout.write("\x1b[0G")
            sys.stdout.write("\x1b[u")  # return to prompt/input
            sys.stdout.flush()
        finally:
            if screen_lock:
                screen_lock.release()

        if size_changed and refresh_callback:
            try:
                refresh_callback()
            except Exception:
                pass

        frame_idx = (frame_idx + 1) % len(frames)
        if stop_event.wait(delay):
            break


def _divider(char: str = "=", width: Optional[int] = None) -> str:
    if width is None:
        columns = shutil.get_terminal_size((72, 0)).columns
        span = max(48, min(columns, 96))
    else:
        span = max(1, min(width, 96))
    return char * span


MENU_CANVAS_WIDTH = 96


def _format_metric(label: str, value: str, icon: str = "->") -> str:
    label_txt = _style(label, color=COLORS["highlight"], bold=True) if _supports_color() else label
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
    unit_scales = (
        ("K", 1_000),
        ("M", 1_000_000),
        ("B", 1_000_000_000),
        ("T", 1_000_000_000_000),
    )
    for unit, denom in unit_scales:
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
        return _style(text, color=COLORS["error"], bold=True)
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

    try:
        from environments.helixenv.af_partition.dataset import build_af_examples
        examples = build_af_examples()
    except RuntimeError as e:
        # Handle missing PyTorch gracefully
        _animate_banner(loops=1, fps=18.0)
        print(_subtle(_divider()))
        print(_headline("Helix Diagnostics Environment"))
        print()
        print(_error("PyTorch is required for the environment summary."))
        print(_subtle(str(e)))
        print()
        print(_subtle("Install PyTorch with one of:"))
        print(_subtle("  pip install torch"))
        print(_subtle("  pip install -e .[torch]"))
        print(_subtle("  pip install -e .[full]"))
        return 1

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
    counts_summary = (
        f"{count} synthesized scenarios with labelled outcomes: "
        f"stable (A) {label_counts['A']}, "
        f"capacity (B) {label_counts['B']}, "
        f"collapsed (C) {label_counts['C']}."
    )
    print(_subtle(counts_summary))

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
    return _style(text, color=COLORS["headline"], bold=True) if _supports_color() else text


def _ascii_bar_rows(values: Sequence[float], width: int = 18) -> str:
    if not values:
        return "  (no data)"
    max_val = max(values)
    if max_val <= 0:
        max_val = 1.0
    rows: List[str] = []
    for idx, val in enumerate(values):
        ratio = val / max_val if max_val else 0.0
        filled = int(round(ratio * width))
        if val > 0 and filled == 0:
            filled = 1
        empty = max(0, width - filled)
        bar = "#" * filled + "." * empty
        rows.append(f"  {idx}: {bar} {val:0.3f}")
    return "\n".join(rows)


def _residual_bar(value: float, scale: float = 1e-2, width: int = 12) -> str:
    if scale <= 0:
        scale = 1.0
    if value <= 0:
        filled = 0
    else:
        ratio = min(1.0, value / scale)
        filled = max(1, int(round(ratio * width)))
    empty = max(0, width - filled)
    return "#" * filled + "." * empty


def _render_relu_partition_ascii(grid: int, seed: int) -> Tuple[str, Dict[str, float]]:
    grid = max(8, grid)
    rng = np.random.default_rng(seed)
    hidden = max(4, grid // 4)
    weights = rng.normal(scale=1.1, size=(2, hidden))
    biases = rng.normal(scale=0.3, size=(hidden,))
    xs = np.linspace(-1.0, 1.0, grid)
    ys = np.linspace(-1.0, 1.0, grid)
    palette = " .:-=+*#%@"
    codes: List[int] = []
    rows: List[str] = []
    for y in ys[::-1]:
        row_chars: List[str] = []
        for x in xs:
            activations = np.dot(np.array([x, y]), weights) + biases
            mask = activations > 0
            code = 0
            for idx, bit in enumerate(mask):
                if bit:
                    code |= 1 << idx
            row_chars.append(palette[code % len(palette)])
            codes.append(int(code))
        rows.append("".join(row_chars))
    counts = Counter(codes)
    total = sum(counts.values()) or 1
    probs = np.array(list(counts.values()), dtype=np.float64) / total
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    ascii_art = "\n".join(rows)
    return ascii_art, {"regions": len(counts), "entropy": entropy}


def _render_cp_flow_ascii(seed: int) -> Tuple[str, Dict[str, float]]:
    rng = np.random.default_rng(seed + 1)
    dim = 3
    kraus_count = 3
    kraus_ops = rng.normal(scale=0.6, size=(kraus_count, dim, dim))
    kraus_ops /= np.linalg.norm(kraus_ops, axis=(1, 2), keepdims=True) + 1e-9
    input_mass = rng.dirichlet(np.ones(dim))
    rho = np.diag(input_mass)
    raw_output = np.zeros_like(rho)
    for op in kraus_ops:
        raw_output += op @ rho @ op.T
    output_mass = np.clip(np.diag(raw_output), 0.0, None)
    trace_gap = float(abs(output_mass.sum() - 1.0))
    if output_mass.sum() > 0:
        output_mass = output_mass / output_mass.sum()
    out_entropy = float(-np.sum(output_mass * np.log2(output_mass + 1e-12)))
    kraus_energy = [float(np.linalg.norm(k, ord="fro")) for k in kraus_ops]

    lines = [
        "input mass:",
        _ascii_bar_rows(input_mass.tolist()),
        "channel energy:",
        _ascii_bar_rows(kraus_energy),
        "output mass:",
        _ascii_bar_rows(output_mass.tolist()),
    ]
    return "\n".join(lines), {"trace_gap": trace_gap, "entropy_out": out_entropy}


def _render_spectral_gap_ascii(seed: int) -> Tuple[str, Dict[str, float]]:
    rng = np.random.default_rng(seed + 2)
    dims = [2, 3, 4, 5]
    gaps: List[float] = []
    for bins in dims:
        P = rng.dirichlet(np.ones(bins), size=bins)
        eigvals = np.linalg.eigvals(P.T)
        eigvals = np.sort(np.abs(eigvals))[::-1]
        gap = 0.0
        if len(eigvals) > 1:
            gap = float(max(0.0, min(1.0, 1.0 - eigvals[1])))
        gaps.append(gap)
    max_gap = max(gaps) if gaps else 1.0
    rows: List[str] = []
    for bins, gap in zip(dims, gaps):
        ratio = gap / max_gap if max_gap > 0 else 0.0
        filled = int(round(ratio * 24))
        if gap > 0 and filled == 0:
            filled = 1
        bar = "#" * filled + "." * (24 - filled)
        rows.append(f"  d={bins:<2} [{bar}] gap={gap:.3f}")
    stats = {"min_gap": min(gaps) if gaps else 0.0, "max_gap": max(gaps) if gaps else 0.0}
    return "\n".join(rows), stats


def _render_residual_ascii(seed: int) -> Tuple[str, Dict[str, float]]:
    rng = np.random.default_rng(seed + 3)
    depths = 3
    prev_mass = rng.dirichlet(np.ones(4))
    mass_residuals: List[float] = []
    trace_residuals: List[float] = []
    rows: List[str] = []
    for depth in range(1, depths + 1):
        next_dim = int(rng.integers(3, 6))
        incidence = (rng.random((prev_mass.size, next_dim)) > 0.4).astype(float)
        incidence = np.where(incidence.sum(axis=1, keepdims=True) == 0, 1.0, incidence)
        tau_next = rng.dirichlet(np.ones(next_dim))
        lifted = incidence @ tau_next
        lifted_sum = lifted.sum()
        if lifted_sum > 0:
            lifted = lifted / lifted_sum
        mass_res = float(np.linalg.norm(prev_mass - lifted, ord=1))
        trace_res = float(abs(prev_mass.sum() - tau_next.sum()))
        mass_residuals.append(mass_res)
        trace_residuals.append(trace_res)
        rows.append(
            f"  depth {depth}: mass {_residual_bar(mass_res, scale=0.3)} {mass_res:.2e} | "
            f"trace {_residual_bar(trace_res, scale=0.05)} {trace_res:.2e}"
        )
        prev_mass = tau_next
    stats = {
        "mass_residual_avg": float(np.mean(mass_residuals)),
        "trace_residual_avg": float(np.mean(trace_residuals)),
    }
    return "\n".join(rows), stats


# Interpretation thresholds for reveals command
# ReLU Partition thresholds
_ENTROPY_LOW = 1.5
_ENTROPY_GOOD = 2.0
_ENTROPY_HIGH = 3.5
_REGIONS_LOW = 5
_REGIONS_MODERATE = 15
_REGIONS_HIGH = 50

# CP Map thresholds
_TRACE_GAP_EXCELLENT = 0.001
_TRACE_GAP_GOOD = 0.01
_TRACE_GAP_WARNING = 0.05
_TRACE_GAP_CRITICAL = 0.1

# Spectral Gap thresholds
_MIXING_EXCELLENT = 0.7
_MIXING_GOOD = 0.5
_MIXING_WARNING = 0.3
_MIXING_CRITICAL = 0.15

# Numerical Health thresholds
_MASS_RESIDUAL_EXCELLENT = 0.1
_MASS_RESIDUAL_GOOD = 0.5
_MASS_RESIDUAL_WARNING = 1.0
_MASS_RESIDUAL_CRITICAL = 2.0


def _interpret_relu_partitions(regions: int, entropy: float) -> Dict[str, str]:
    """Interpret ReLU partition metrics and provide actionable guidance."""
    status = "good"
    messages = []
    actions = []

    # Analyze entropy
    if entropy < _ENTROPY_LOW:
        status = "warning"
        messages.append(f"Low partition entropy ({entropy:.2f} < {_ENTROPY_LOW})")
        messages.append("Network capacity is underutilized - regions are imbalanced")
        actions.append("Consider: Prune unused neurons or reduce layer width")
    elif entropy < _ENTROPY_GOOD:
        messages.append(f"Moderate entropy ({entropy:.2f})")
        messages.append("Some imbalance in region utilization")
        actions.append("Monitor: Check if all neurons contribute meaningfully")
    elif entropy >= _ENTROPY_HIGH:
        messages.append(f"Excellent entropy ({entropy:.2f} >= {_ENTROPY_HIGH})")
        messages.append("Very balanced partitioning - capacity well-distributed")
        actions.append("Current architecture is well-utilized")
    else:
        messages.append(f"Balanced partitioning (entropy = {entropy:.2f})")
        actions.append("Current architecture size is appropriate")

    # Analyze region count
    if regions < _REGIONS_LOW:
        status = "warning"
        messages.append(f"Few regions ({regions} < {_REGIONS_LOW})")
        messages.append("Network may be too simple for complex tasks")
        actions.append("Consider: Add more layers or increase width if underfitting")
    elif regions > _REGIONS_HIGH:
        if entropy < _ENTROPY_GOOD:
            status = "warning"
            messages.append(f"Many regions ({regions}) but low utilization")
            actions.append("Consider: Network may be over-parameterized - simplify or regularize")
        else:
            messages.append(f"High complexity ({regions} regions) with good utilization")
            actions.append("Architecture handles complex decision boundaries well")

    return {
        "status": status,
        "message": " • ".join(messages) if messages else "Partition analysis complete",
        "action": " • ".join(actions) if actions else "No changes recommended"
    }


def _interpret_cp_flow(trace_gap: float, entropy_out: float) -> Dict[str, str]:
    """Interpret CP map information flow and provide actionable guidance."""
    status = "good"
    messages = []
    actions = []

    # Analyze trace gap (information preservation)
    if trace_gap >= _TRACE_GAP_CRITICAL:
        status = "critical"
        messages.append(f"Severe information loss (gap = {trace_gap:.3f} >= {_TRACE_GAP_CRITICAL})")
        messages.append("Critical bottleneck detected")
        actions.append("URGENT: Add skip connections or significantly widen layers")
        actions.append("Check for vanishing activations")
    elif trace_gap >= _TRACE_GAP_WARNING:
        status = "warning"
        messages.append(f"Moderate information loss (gap = {trace_gap:.3f} >= {_TRACE_GAP_WARNING})")
        messages.append("Noticeable compression through this layer")
        actions.append("Consider: Add residual connections to preserve information")
        actions.append("Or: Widen bottleneck layers")
    elif trace_gap >= _TRACE_GAP_GOOD:
        messages.append(f"Minor information loss (gap = {trace_gap:.3f})")
        messages.append("Acceptable compression level")
        actions.append("Monitor: Ensure task doesn't require lost information")
    else:
        messages.append(f"Excellent information preservation (gap = {trace_gap:.3f} < {_TRACE_GAP_GOOD})")
        actions.append("Information flow is optimal")

    # Analyze output entropy (compression quality)
    if entropy_out < 0.5:
        status = max(status, "warning", key=lambda x: ["good", "warning", "critical"].index(x))
        messages.append(f"Very low output entropy ({entropy_out:.2f})")
        messages.append("Extreme compression - may be losing critical nuance")
        actions.append("Verify: Task allows for this level of abstraction")
    elif entropy_out < 1.0:
        messages.append(f"Low output entropy ({entropy_out:.2f}) - strong compression")
    else:
        messages.append(f"Output entropy = {entropy_out:.2f} bits")

    return {
        "status": status,
        "message": " • ".join(messages) if messages else "Information flow analysis complete",
        "action": " • ".join(actions) if actions else "No changes recommended"
    }


def _interpret_spectral_gaps(min_gap: float, max_gap: float) -> Dict[str, str]:
    """Interpret spectral gap metrics and provide actionable guidance."""
    status = "good"
    messages = []
    actions = []

    avg_gap = (min_gap + max_gap) / 2.0
    gap_variance = max_gap - min_gap

    # Analyze mixing speed
    if avg_gap >= _MIXING_EXCELLENT:
        messages.append(f"Strong mixing (avg gap = {avg_gap:.3f} >= {_MIXING_EXCELLENT})")
        messages.append("Fast convergence expected")
        actions.append("Safe to use higher learning rates (1.5-2× default)")
        actions.append("Network should train stably")
    elif avg_gap >= _MIXING_GOOD:
        messages.append(f"Good mixing (avg gap = {avg_gap:.3f})")
        messages.append("Moderate convergence speed")
        actions.append("Standard learning rates appropriate")
    elif avg_gap >= _MIXING_WARNING:
        status = "warning"
        messages.append(f"Slow mixing (avg gap = {avg_gap:.3f} < {_MIXING_GOOD})")
        messages.append("Convergence may be slow")
        actions.append("Consider: Add batch normalization or layer normalization")
        actions.append("Or: Add residual connections to improve flow")
        actions.append("Use lower learning rates")
    else:
        status = "critical"
        messages.append(f"Very slow mixing (avg gap = {avg_gap:.3f} < {_MIXING_WARNING})")
        messages.append("Training instability likely")
        actions.append("URGENT: Add normalization layers (BatchNorm/LayerNorm)")
        actions.append("Add skip connections")
        actions.append("Significantly reduce learning rate")

    # Analyze variance across dimensions
    if gap_variance > 0.3:
        messages.append(f"High variance in gaps ({gap_variance:.3f})")
        messages.append("Inconsistent mixing across dimensions")
        actions.append("May need dimension-specific conditioning")

    return {
        "status": status,
        "message": " • ".join(messages) if messages else "Spectral gap analysis complete",
        "action": " • ".join(actions) if actions else "No changes recommended"
    }


def _interpret_numerical_health(mass_avg: float, trace_avg: float) -> Dict[str, str]:
    """Interpret numerical health metrics and provide actionable guidance."""
    status = "good"
    messages = []
    actions = []

    # Analyze mass residual
    if mass_avg >= _MASS_RESIDUAL_CRITICAL:
        status = "critical"
        messages.append(f"Critical mass residual ({mass_avg:.2e} >= {_MASS_RESIDUAL_CRITICAL})")
        messages.append("Severe numerical instability")
        actions.append("URGENT: Check weight initialization (use He/Xavier)")
        actions.append("Add gradient clipping (max_norm=1.0)")
        actions.append("Reduce learning rate significantly")
    elif mass_avg >= _MASS_RESIDUAL_WARNING:
        status = "warning"
        messages.append(f"Elevated mass residual ({mass_avg:.2e} >= {_MASS_RESIDUAL_WARNING})")
        messages.append("Moderate probability leakage detected")
        actions.append("Consider: Gradient clipping (max_norm=1.0-5.0)")
        actions.append("Review weight initialization scheme")
        actions.append("May indicate need for normalization")
    elif mass_avg >= _MASS_RESIDUAL_GOOD:
        messages.append(f"Acceptable mass residual ({mass_avg:.2e})")
        actions.append("Monitor during training")
    else:
        messages.append(f"Excellent mass conservation ({mass_avg:.2e} < {_MASS_RESIDUAL_GOOD})")
        actions.append("Probability flow is numerically stable")

    # Analyze trace residual (precision)
    if trace_avg < 1e-10:
        messages.append(f"Perfect trace precision ({trace_avg:.2e} < 1e-10)")
    elif trace_avg < 1e-6:
        messages.append(f"Excellent trace precision ({trace_avg:.2e})")
    elif trace_avg < 1e-3:
        messages.append(f"Good trace precision ({trace_avg:.2e})")
    else:
        status = max(status, "warning", key=lambda x: ["good", "warning", "critical"].index(x))
        messages.append(f"Trace precision concerns ({trace_avg:.2e} >= 1e-3)")
        actions.append("Check for accumulation of numerical errors")

    return {
        "status": status,
        "message": " • ".join(messages) if messages else "Numerical health check complete",
        "action": " • ".join(actions) if actions else "System is numerically stable"
    }


def _format_interpretation(status: str, message: str, action: str) -> str:
    """Format interpretation with status symbols and proper indentation."""
    symbol_map = {
        "good": "✓",
        "warning": "⚠",
        "critical": "✗"
    }
    symbol = symbol_map.get(status, "•")

    lines = []
    if message:
        lines.append(f"{symbol} {message}")
    if action:
        lines.append(f"→ {action}")

    return "\n".join(lines)


def _subtle(text: str) -> str:
    return _style(text, color=COLORS["subtle"]) if _supports_color() else text


def _error(text: str) -> str:
    return _style(text, color=COLORS["error"], bold=True) if _supports_color() else text


def _get_key() -> str:
    """Read a single keypress, supporting arrow-key navigation when possible."""
    stream = getattr(sys, "stdin", None)
    try:
        stdin_is_tty = bool(stream) and stream.isatty()
    except Exception:
        stdin_is_tty = False

    if not (_HAS_TERMIOS and stdin_is_tty):
        return input()

    fd = stream.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = stream.read(1)

        if ch == "\x03":  # Ctrl+C
            raise KeyboardInterrupt
        if ch == "\x04":  # Ctrl+D (EOF)
            raise EOFError
        if ch in ("\r", "\n"):
            return "ENTER"
        if ch == "\x1b":
            ch2 = stream.read(1)
            if ch2 == "[":
                ch3 = stream.read(1)
                mapping = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}
                return mapping.get(ch3, "")
            if ch2 == "O":
                ch3 = stream.read(1)
                mapping = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}
                return mapping.get(ch3, "")
            return ch + ch2
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def run_interactive() -> int:
    """Run the interactive menu system."""
    # Suppress warnings during interactive mode to prevent display corruption
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="urllib3")

    autowrap_applied = False

    try:
        _set_auto_wrap(False)
        autowrap_applied = True
        while True:
            frame_height = 0
            lines_after_frame = 0
            menu_start_line = 0
            prompt_line = 0
            menu_block_lines = 0
            menu_extra_lines: List[str] = []
            screen_lock = threading.RLock()

            menu_items = [
                "  [1] Run Helix environment (model analytics)",
                "  [2] Helix environment summary",
                "  [3] Run demo (two moons dataset)",
                "  [4] Analyze custom model/data",
                "  [5] Interactive TUI mode",
                "  [6] Exit",
            ]
            selected_idx = 0
            stdin_stream = getattr(sys, "stdin", None)
            try:
                stdin_is_tty = bool(stdin_stream) and stdin_stream.isatty()
            except Exception:
                stdin_is_tty = False
            use_arrow_keys = stdin_is_tty and _HAS_TERMIOS
            menu_width = max(MENU_CANVAS_WIDTH, NORMALIZED_HELIX_WIDTH or 0)

            def _current_draw_width() -> int:
                """Clamp draw width to fit within the current terminal width."""
                try:
                    columns = shutil.get_terminal_size((menu_width, 24)).columns
                except Exception:
                    columns = menu_width
                return max(1, min(menu_width, columns))


            stop_event: Optional[threading.Event] = None
            animation_paused: Optional[threading.Event] = None
            ready_event: Optional[threading.Event] = None
            anim_thread: Optional[threading.Thread] = None

            min_cols_required = menu_width
            min_rows_required = max(NORMALIZED_HELIX_HEIGHT or 0, 12) + 6
            layout_guard_active = False

            def _current_terminal_size(
                target_rows: Optional[int] = None, target_cols: Optional[int] = None
            ) -> Tuple[int, int]:
                rows_target = target_rows or min_rows_required
                cols_target = target_cols or min_cols_required
                try:
                    size = shutil.get_terminal_size((cols_target, rows_target))
                    return size.columns, size.lines
                except Exception:
                    return cols_target, rows_target

            def _render_guard_notice(
                required_cols: int,
                required_rows: int,
                actual_cols: int,
                actual_rows: int,
            ) -> None:
                nonlocal layout_guard_active
                layout_guard_active = True
                if animation_paused:
                    animation_paused.set()
                with screen_lock:
                    sys.stdout.write("\x1b[2J\x1b[H")
                    draw_width = max(1, actual_cols)
                    notice_lines = [
                        _headline("Helix Interactive Menu"),
                        _error("Terminal window too small for interactive mode."),
                    ]
                    notice_lines.append(
                        _subtle(
                            f"Need at least {required_cols} cols x {required_rows} rows."
                        )
                    )
                    notice_lines.append(
                        _subtle(f"Current size: {actual_cols} cols x {actual_rows} rows.")
                    )
                    notice_lines.append(_subtle("Resize the window larger to resume..."))
                    for line in notice_lines:
                        sys.stdout.write(_truncate_display_width(line, draw_width))
                        sys.stdout.write("\n")
                    sys.stdout.flush()

            def _check_terminal_space(*, show_notice: bool = False) -> bool:
                nonlocal layout_guard_active
                required_rows = max(min_rows_required, (NORMALIZED_HELIX_HEIGHT or 0) or 1)
                required_cols = max(min_cols_required, 1)
                actual_cols, actual_rows = _current_terminal_size(required_rows, required_cols)
                if actual_cols < required_cols or actual_rows < required_rows:
                    if show_notice or not layout_guard_active:
                        _render_guard_notice(required_cols, required_rows, actual_cols, actual_rows)
                    return False
                if layout_guard_active:
                    layout_guard_active = False
                    if animation_paused:
                        animation_paused.clear()
                return True

            def _wait_for_safe_space() -> None:
                while not _check_terminal_space(show_notice=True):
                    time.sleep(0.1)


            def _format_menu_item(idx: int) -> str:
                item = menu_items[idx]
                if not (use_arrow_keys and idx == selected_idx):
                    return item
                if _supports_color():
                    return _style(item, color=COLORS["hover"], bold=True)
                return f"> {item.strip()}"

            def _render_layout() -> None:
                nonlocal frame_height, lines_after_frame, menu_start_line
                nonlocal prompt_line, menu_block_lines, menu_extra_lines, min_rows_required
                menu_extra_lines = []
                with screen_lock:
                    draw_width = _current_draw_width()
                    # Redraw our block in place without clearing the whole screen
                    row = 1
                    def _pos(r: int) -> None:
                        sys.stdout.write(f"\x1b[{r};1H")
                        sys.stdout.write("\x1b[2K")

                    frame_height = 0
                    if NORMALIZED_HELIX_FRAMES:
                        first_frame = NORMALIZED_HELIX_FRAMES[0]
                        frame_text = _colorize_frame(first_frame)
                        frame_height = NORMALIZED_HELIX_HEIGHT
                        frame_lines = frame_text.splitlines()
                        for idx in range(frame_height):
                            _pos(row)
                            line = frame_lines[idx] if idx < len(frame_lines) else ""
                            sys.stdout.write(_truncate_display_width(line, draw_width))
                            row += 1
                        sys.stdout.flush()

                    lines_after_frame = 0

                    def _write_line(text: str = "") -> None:
                        nonlocal lines_after_frame, row
                        _pos(row)
                        truncated = _truncate_display_width(text, draw_width) if text else ""
                        sys.stdout.write(truncated)
                        row += 1
                        lines_after_frame += 1

                    _write_line(_subtle(_divider(width=draw_width)))
                    _write_line(_headline("Helix Interactive Menu"))
                    _write_line(_subtle("Select an option to continue:"))
                    _write_line()

                    menu_start_line = frame_height + lines_after_frame + 1
                    for idx, _ in enumerate(menu_items):
                        _write_line(_format_menu_item(idx))
                    extra_lines: List[str] = []
                    if use_arrow_keys:
                        instructions_text = _subtle(
                            "Use ↑/↓ arrow keys or type 1-6, then press Enter:"
                        )
                        extra_lines.append(instructions_text)
                    # Always leave a blank row before the prompt
                    extra_lines.append("")
                    for line in extra_lines:
                        _write_line(line)
                    menu_extra_lines = extra_lines
                    menu_block_lines = len(menu_items) + len(menu_extra_lines)
                    prompt_line = menu_start_line + menu_block_lines
                    min_rows_required = max(prompt_line + 2, frame_height + 1, 12)
                    # Leave prompt line blank; it will be populated after rendering.
                    sys.stdout.flush()

            prompt = (
                _style("Your choice: ", color=COLORS["highlight"], bold=True)
                if _supports_color()
                else "Your choice: "
            )

            def _refresh_menu(*, pause_animation: bool = False) -> None:
                if not use_arrow_keys:
                    return

                # Pause animation before refreshing menu if requested
                if pause_animation and animation_paused:
                    animation_paused.set()
                    time.sleep(0.05)  # Give animation time to pause

                with screen_lock:
                    draw_width = _current_draw_width()
                    for idx, _ in enumerate(menu_items):
                        line_row = menu_start_line + idx
                        sys.stdout.write(f"\x1b[{line_row};1H")
                        sys.stdout.write("\x1b[2K")
                        line = _truncate_display_width(_format_menu_item(idx), draw_width)
                        sys.stdout.write(line)
                    extra_start = menu_start_line + len(menu_items)
                    for offset, extra in enumerate(menu_extra_lines):
                        line_row = extra_start + offset
                        sys.stdout.write(f"\x1b[{line_row};1H")
                        sys.stdout.write("\x1b[2K")
                        sys.stdout.write(_truncate_display_width(extra, draw_width))
                    sys.stdout.write(f"\x1b[{prompt_line};1H")
                    sys.stdout.write("\x1b[2K")
                    sys.stdout.write(_truncate_display_width(prompt, draw_width))
                    sys.stdout.flush()

                # Resume animation after menu refresh
                if pause_animation and animation_paused:
                    animation_paused.clear()

            def _repaint_screen(*, pause_animation: bool = False) -> None:
                _render_layout()
                if use_arrow_keys:
                    _refresh_menu(pause_animation=pause_animation)
                else:
                    with screen_lock:
                        draw_width = _current_draw_width()
                        sys.stdout.write(f"\x1b[{prompt_line};1H")
                        sys.stdout.write("\x1b[2K")
                        sys.stdout.write(_truncate_display_width(prompt, draw_width))
                        sys.stdout.flush()
                with screen_lock:
                    sys.stdout.write("\x1b[s")
                    sys.stdout.flush()

            def _handle_resize_event() -> None:
                if not _check_terminal_space(show_notice=True):
                    return
                _repaint_screen(pause_animation=True)

            # Initial paint before launching the animation thread
            _repaint_screen()
            _check_terminal_space(show_notice=True)

            # Create animation with pause support for arrow key mode
            if ASCII_HELIX_FRAMES and _is_tty():
                stop_event = threading.Event()
                ready_event = threading.Event()
                animation_paused = threading.Event()
                if layout_guard_active:
                    animation_paused.set()

                anim_thread = threading.Thread(
                    target=_animate_banner_continuous,
                    args=(
                        stop_event,
                        lines_after_frame,
                        frame_height,
                        18.0,
                        animation_paused,
                        ready_event,
                        _handle_resize_event,
                        screen_lock,
                        menu_width,
                    ),
                    daemon=True,
                )
                anim_thread.start()

                # Signal that main thread is ready for animation
                ready_event.set()

            choice = ""
            try:
                if use_arrow_keys:
                    while True:
                        _wait_for_safe_space()
                        key = _get_key()
                        if key == "UP":
                            selected_idx = (selected_idx - 1) % len(menu_items)
                            _refresh_menu()
                            continue
                        if key == "DOWN":
                            selected_idx = (selected_idx + 1) % len(menu_items)
                            _refresh_menu()
                            continue
                        if key == "ENTER":
                            choice = str(selected_idx + 1)
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                            break
                        if key and key.isdigit() and 1 <= int(key) <= len(menu_items):
                            selected_idx = int(key) - 1
                            _refresh_menu()
                            sys.stdout.write(f"{key}\n")
                            sys.stdout.flush()
                            choice = key
                            break
                        _refresh_menu()
                else:
                    _wait_for_safe_space()
                    choice = input()
            finally:
                if stop_event is not None:
                    stop_event.set()
                    if anim_thread is not None:
                        anim_thread.join(timeout=0.5)
                    with screen_lock:
                        sys.stdout.write("\x1b[u")
                        sys.stdout.write("\x1b[s")
                        sys.stdout.flush()

            def _prompt_return_to_menu() -> bool:
                prompt_msg = "\nPress Enter to return to the menu, or type 'q' to exit: "
                display_prompt = _subtle(prompt_msg) if _supports_color() else prompt_msg
                try:
                    _wait_for_safe_space()
                    response = input(display_prompt)
                except (KeyboardInterrupt, EOFError):
                    return False
                return response.strip().lower() not in {"q", "quit", "exit"}

            def _clear_menu_screen() -> None:
                if not _is_tty():
                    return
                sys.stdout.write("\x1b[2J\x1b[H")
                sys.stdout.flush()

            if choice == "1":
                print()
                helix_args = _interactive_collect_helix_args()
                print()
                try:
                    result = run_helix_env(helix_args)
                except RuntimeError as e:
                    print(_error(f"\n{e}"))
                    input("\nPress Enter to continue...")
                    _clear_menu_screen()
                    continue
                if _prompt_return_to_menu():
                    _clear_menu_screen()
                    continue
                return result
            elif choice == "2":
                print()
                result = run_helixenv_overview()
                if _prompt_return_to_menu():
                    _clear_menu_screen()
                    continue
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
                    animate_forward=False,
                )
                print()
                try:
                    result = run_demo(args)
                except RuntimeError as e:
                    print(_error(f"\n{e}"))
                    input("\nPress Enter to continue...")
                    _clear_menu_screen()
                    continue
                if _prompt_return_to_menu():
                    _clear_menu_screen()
                    continue
                return result
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
                try:
                    result = run_analyze(args)
                except RuntimeError as e:
                    print(_error(f"\n{e}"))
                    input("\nPress Enter to continue...")
                    _clear_menu_screen()
                    continue
                if _prompt_return_to_menu():
                    _clear_menu_screen()
                    continue
                return result
            elif choice == "5":
                try:
                    from .tui import run_tui

                    result = run_tui([])
                except Exception as e:
                    print(
                        _error(
                            "\nHelix TUI not available. Install TUI extras: pip install '.[tui]'"
                        )
                    )
                    print(_subtle(f"Details: {e}"))
                    input("\nPress Enter to continue...")
                    continue
                if _prompt_return_to_menu():
                    _clear_menu_screen()
                    continue
                return result
            elif choice == "6":
                print(_subtle("\nExiting Helix. Thank you!"))
                return 0
            else:
                print(_error("\nInvalid choice. Please select 1-6."))
                input("Press Enter to continue...")
    except (KeyboardInterrupt, EOFError):
        print(_subtle("\n\nExiting Helix. Thank you!"))
        return 0
    finally:
        if autowrap_applied:
            _set_auto_wrap(True)


if nn is not None:

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

else:

    class MLP:  # type: ignore[misc]
        """Placeholder when PyTorch is unavailable."""

        def __init__(self, *_, **__) -> None:
            raise RuntimeError(
                "PyTorch is required for the built-in MLP. Install torch or supply "
                "--model-module/--weights inputs."
            )


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
            hint = (
                "Rerun with --allow-pickled-arrays or set "
                "HELIX_ALLOW_PICKLED_ARRAYS=1 if you trust the file."
            )
            raise ValueError(
                f"{loader} '{p}' requires pickle deserialisation. {hint}"
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


def _load_array_from_path(path: str, *, allow_pickle: bool = False) -> np.ndarray:
    """Compatibility wrapper for subcommands that predated _load_array."""

    return _load_array(path, allow_pickle=allow_pickle)


def _load_model_from_module(
    module_path: str,
    func_name: str = "build_model",
    kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Load a user-provided module builder and instantiate a model."""

    builder = _dynamic_import_builder(module_path, func_name)
    return builder(**(kwargs or {}))


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
        raise RuntimeError(
            "PyTorch is required for the demo command.\n"
            "Install with one of:\n"
            "  pip install torch\n"
            "  pip install -e .[torch]\n"
            "  pip install -e .[full]"
        )

    # Deterministic seeding for reproducibility
    _seed_torch(args.seed)
    hidden_widths = _hidden_widths_from_args(args)
    layer_blueprint = _layer_blueprint_from_widths(2, hidden_widths, 2)
    neuron_anim = NeuronAnimator(layer_blueprint, caption="Neuron flow (live)")
    neuron_anim.update_activity_hint(layer_blueprint)
    neuron_anim.start("Preparing Helix diagnostics")
    neuron_panel_summary = ""
    summary_activity: List[float] = []

    X, y = make_moons(n=args.samples, noise=args.noise, seed=args.seed)
    try:
        neuron_anim.update_status("Persistent homology")
        compute_persistent_homology(X, maxdim=2)
    except Exception as exc:
        print(_subtle(f"[warn] Persistent homology unavailable: {exc}"))
    model = MLP(d_in=2, widths=hidden_widths, d_out=2)

    if not args.no_train:
        neuron_anim.update_status("Training network")
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        for _ in range(args.epochs):
            opt.zero_grad()
            logits = model(X_t)
            loss = F.cross_entropy(logits, y_t)
            loss.backward()
            opt.step()

    _maybe_print_model_summary(model, _infer_summary_input_shape(X, layer_blueprint[0]))
    try:
        compute_capacity_loss(model)
    except Exception as exc:
        print(_subtle(f"[warn] Capacity metrics unavailable: {exc}"))

    neuron_anim.update_status("Extracting AF partitions")
    af = extract_partitions(model, X)
    n_list = region_counts(af.B_list)
    errs = mass_consistency_errors(af.B_list, af.tau_list)
    neuron_anim.update_activity_hint([layer_blueprint[0], *n_list, layer_blueprint[-1]])

    cp_stats = []
    neuron_anim.update_status("CP diagnostics")
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
    neuron_anim.update_status("Ulam transfer sampling")
    P, _ = ulam_pf(
        lambda z: F_block(z, eps=0.4),
        (lo, hi),
        bins_per_dim=args.ulam_bins,
        samples_per_cell=args.ulam_samples_per_cell,
    )
    gap = spectral_gap(P)
    neuron_anim.update_activity_hint([layer_blueprint[0], *n_list, max(gap, 0.001)])

    try:
        summary_activity = _activity_profile_from_stats(layer_blueprint, n_list, gap)
        summary_metrics = _neuron_metric_lines(layer_blueprint, n_list, errs, cp_stats, gap)
        summary_extras = _neuron_extra_lines(errs, cp_stats, gap)
        neuron_anim.finish(
            activities=summary_activity,
            metrics=summary_metrics,
            caption="Neuron activity summary",
            extra_lines=summary_extras,
        )
    except Exception:
        neuron_anim.stop()
        raise

    if getattr(args, "animate_forward", False) and summary_activity:
        labels_for_anim = neuron_anim.layer_labels or _layer_labels_from_blueprint(layer_blueprint)
        _animate_forward_pass(labels_for_anim, summary_activity)

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
    panel_block = neuron_panel_summary or neuron_anim.last_panel_text
    if panel_block:
        print(panel_block)
        print()

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

    neuron_anim.stop(erase=False)
    return 0


def run_helix_env(args: argparse.Namespace) -> int:
    """Run the operator-algebra AF environment with optional custom data/model."""

    if torch is None:
        raise RuntimeError(
            "PyTorch is required for the helixenv command.\n"
            "Install with one of:\n"
            "  pip install torch\n"
            "  pip install -e .[torch]\n"
            "  pip install -e .[full]"
        )

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
        info_msg = (
            f"Stored API key in environment variable {api_key_var} for "
            "Helix Verifiers integrations."
        )
        print(_subtle(info_msg))
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

    hidden_widths = _hidden_widths_from_args(args)
    feature_dim = int(X.shape[1]) if X.ndim == 2 else 1
    layer_blueprint = _layer_blueprint_from_widths(feature_dim, hidden_widths, args.d_out)
    neuron_anim = NeuronAnimator(layer_blueprint, caption="Neuron flow (live)")
    neuron_anim.update_activity_hint(layer_blueprint)
    neuron_anim.start("Preparing Helix diagnostics")

    try:
        neuron_anim.update_status("Persistent homology")
        persistent_summary = compute_persistent_homology(X, maxdim=2)
    except Exception as exc:
        print(_subtle(f"[warn] Persistent homology unavailable: {exc}"))

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
        model = MLP(d_in=feature_dim, widths=hidden_widths, d_out=args.d_out)

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

    header_fields = [
        f"{'depth':>5}",
        f"{'regions':>9}",
        f"{'mass_L1':>12}",
        f"{'trace_inf':>11}",
        f"{'wasted':>8}",
        f"{'entropy':>9}",
        f"{'cp_uni':>9}",
        f"{'cp_psd':>9}",
        f"{'gap':>8}",
        f"{'reward':>10}",
        f"{'heat':>6}",
    ]
    header = " ".join(header_fields)
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
        raise RuntimeError(
            "PyTorch is required for the analyze command.\n"
            "Install with one of:\n"
            "  pip install torch\n"
            "  pip install -e .[torch]\n"
            "  pip install -e .[full]"
        )
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

    hidden_widths = _hidden_widths_from_args(args)
    feature_dim = int(X.shape[1]) if X.ndim == 2 else 2
    layer_blueprint = _layer_blueprint_from_widths(feature_dim, hidden_widths, args.d_out)
    neuron_anim = NeuronAnimator(layer_blueprint, caption="Neuron flow (live)")
    neuron_anim.update_activity_hint(layer_blueprint)
    neuron_anim.start("Preparing Helix diagnostics")
    neuron_panel_summary = ""

    try:
        neuron_anim.update_status("Persistent homology")
        persistent_summary = compute_persistent_homology(X, maxdim=2)
    except Exception as exc:
        print(_subtle(f"[warn] Persistent homology unavailable: {exc}"))

    y = None
    if args.data_y:
        y_arr = _load_array(args.data_y, allow_pickle=allow_pickled_arrays)
        y = y_arr.astype(np.int64).reshape(-1)
        if y.shape[0] != X.shape[0]:
            raise ValueError("data_y length must match number of rows in data_x")

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
        model = MLP(d_in=feature_dim, widths=hidden_widths, d_out=args.d_out)

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
            neuron_anim.update_status("Training network")
            opt = torch.optim.Adam(model.parameters(), lr=1e-2)
            for _ in range(args.epochs):
                opt.zero_grad()
                logits = model(X_t)
                loss = F.cross_entropy(logits, y_t)
                loss.backward()
                opt.step()
        except Exception as e:
            print(f"[warn] Training skipped due to error: {e}")

    _maybe_print_model_summary(model, _infer_summary_input_shape(X, feature_dim))
    try:
        capacity_metrics = compute_capacity_loss(model)
    except Exception as exc:
        print(_subtle(f"[warn] Capacity metrics unavailable: {exc}"))

    # Metrics
    neuron_anim.update_status("Extracting AF partitions")
    af = extract_partitions(model, X)
    n_list = region_counts(af.B_list)
    errs = mass_consistency_errors(af.B_list, af.tau_list)
    neuron_anim.update_activity_hint([layer_blueprint[0], *n_list, layer_blueprint[-1]])

    cp_stats = []
    neuron_anim.update_status("CP diagnostics")
    for k, B in enumerate(af.B_list, start=1):
        tau_prev = np.array([1.0]) if k == 1 else af.tau_list[k - 2]
        tau_cur = af.tau_list[k - 1]
        V = build_V_from_incidence(B, tau_prev, tau_cur)
        cp_stats.append(sanity_check_ucp(V, trials=6))

    # Optional Ulam
    gap = None
    if not args.no_ulam:
        neuron_anim.update_status("Ulam transfer sampling")
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

    gap_hint = 0.001
    if isinstance(gap, (int, float)) and not math.isnan(float(gap)):
        gap_hint = float(gap)
    neuron_anim.update_activity_hint([layer_blueprint[0], *n_list, gap_hint])

    try:
        summary_activity = _activity_profile_from_stats(layer_blueprint, n_list, gap)
        summary_metrics = _neuron_metric_lines(layer_blueprint, n_list, errs, cp_stats, gap)
        summary_extras = _neuron_extra_lines(errs, cp_stats, gap)
        neuron_panel_summary = neuron_anim.finish(
            activities=summary_activity,
            metrics=summary_metrics,
            caption="Neuron activity summary",
            extra_lines=summary_extras,
        )
    except Exception:
        neuron_anim.stop()
        raise

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
    panel_block = neuron_panel_summary or neuron_anim.last_panel_text
    if panel_block:
        print(panel_block)
        print()

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
    neuron_anim.stop(erase=False)
    return 0


def run_reveals_showcase(argv: List[str]) -> int:
    """Render lightweight CLI visuals for Helix invariants with actionable interpretations."""
    parser = argparse.ArgumentParser(description="Helix CLI: What Helix reveals")
    parser.add_argument("--grid", type=int, default=24, help="Grid resolution for ReLU partitions")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible visuals")
    args = parser.parse_args(argv)

    grid = max(8, args.grid)
    seed = int(args.seed)

    print(_headline("WHAT HELIX REVEALS"))

    # Section 1: ReLU Partitions
    relu_ascii, relu_stats = _render_relu_partition_ascii(grid, seed)
    print("\n[ReLU partitions — AF refinement]")
    print(relu_ascii)
    print(
        f"regions: {relu_stats['regions']} • partition entropy: {relu_stats['entropy']:.2f} bits"
    )
    # Add interpretation
    interp = _interpret_relu_partitions(relu_stats['regions'], relu_stats['entropy'])
    print(_format_interpretation(interp['status'], interp['message'], interp['action']))

    # Section 2: CP Maps
    cp_ascii, cp_stats = _render_cp_flow_ascii(seed)
    print("\n[Information flow — CP maps]")
    print(cp_ascii)
    print(
        f"trace gap: {cp_stats['trace_gap']:.2e} • output entropy: {cp_stats['entropy_out']:.2f} bits"
    )
    # Add interpretation
    interp = _interpret_cp_flow(cp_stats['trace_gap'], cp_stats['entropy_out'])
    print(_format_interpretation(interp['status'], interp['message'], interp['action']))

    # Section 3: Spectral Gaps
    gap_ascii, gap_stats = _render_spectral_gap_ascii(seed)
    print("\n[Mixing — spectral gaps]")
    print(gap_ascii)
    print(f"min gap: {gap_stats['min_gap']:.3f} • max gap: {gap_stats['max_gap']:.3f}")
    # Add interpretation
    interp = _interpret_spectral_gaps(gap_stats['min_gap'], gap_stats['max_gap'])
    print(_format_interpretation(interp['status'], interp['message'], interp['action']))

    # Section 4: Numerical Health
    resid_ascii, resid_stats = _render_residual_ascii(seed)
    print("\n[Numerical health — mass & trace residuals]")
    print(resid_ascii)
    print(
        f"avg mass residual: {resid_stats['mass_residual_avg']:.2e} • "
        f"avg trace residual: {resid_stats['trace_residual_avg']:.2e}"
    )
    # Add interpretation
    interp = _interpret_numerical_health(
        resid_stats['mass_residual_avg'],
        resid_stats['trace_residual_avg']
    )
    print(_format_interpretation(interp['status'], interp['message'], interp['action']))

    return 0


def run_ktheory_analysis(argv: List[str]) -> int:
    """Run K-theory analysis of a neural network."""
    parser = argparse.ArgumentParser(description="Helix K-theory analysis")

    # Model specification
    parser.add_argument("--model-module", type=str, default="",
                       help="Path to Python file that defines a model builder")
    parser.add_argument("--model-func", type=str, default="build_model",
                       help="Builder function name in the module")
    parser.add_argument("--weights", type=str, default="",
                       help="Optional path to a state_dict .pt/.pth file")

    # Data specification
    parser.add_argument("--data-x", type=str, required=True,
                       help="Path to features array (.npy/.npz/.csv)")
    parser.add_argument("--width", type=int, default=16,
                       help="Hidden width for built-in MLP if used")

    # K-theory options
    parser.add_argument("--method", type=str, default="hodge",
                       choices=["hodge", "smith", "spectral"],
                       help="K-theory computation method")
    parser.add_argument("--tolerance", type=float, default=1e-10,
                       help="Numerical tolerance for computation")
    parser.add_argument("--max-depth", type=int, default=None,
                       help="Maximum AF partition depth to analyze")

    # Output options
    parser.add_argument("--save-report", type=str, default="",
                       help="Save detailed report to JSON file")
    parser.add_argument("--plot", action="store_true",
                       help="Generate plots of K-theory evolution")

    args = parser.parse_args(argv)

    try:
        # Load data
        X = _load_array_from_path(args.data_x)
        print(f"Loaded data: {X.shape}")

        # Load or build model
        if args.model_module:
            model = _load_model_from_module(args.model_module, args.model_func, {})
        else:
            # Use default MLP
            import torch.nn as nn
            model = nn.Sequential(
                nn.Linear(X.shape[1], args.width),
                nn.ReLU(),
                nn.Linear(args.width, args.width),
                nn.ReLU(),
                nn.Linear(args.width, 2)
            )

        if args.weights:
            import torch
            model.load_state_dict(torch.load(args.weights))

        print(f"Model: {model}")

        # Run K-theory analysis
        _ensure_repo_root_on_path()
        from environments.ktheory.env import load_k_theory_environment

        # Create K-theory environment
        env = load_k_theory_environment(
            model, X,
            max_depth=args.max_depth,
            tolerance=args.tolerance
        )

        print("\n" + "="*60)
        print("K-THEORY ANALYSIS RESULTS")
        print("="*60)

        # Run analysis
        env.reset()
        step_count = 0

        while step_count < env._max_steps:
            step_result = env.step({"type": "advance"})

            if step_result.done:
                break

            # Print current analysis
            k_analysis = step_result.info.get("k_theory_analysis", {})
            if k_analysis:
                depth = k_analysis.get("depth", step_count + 1)
                print(f"\nDepth {depth}:")
                print(f"  Rank: {k_analysis.get('rank', 'N/A')}")
                print(f"  Nullity: {k_analysis.get('nullity', 'N/A')}")
                print(f"  Torsion orders: {k_analysis.get('torsion_orders', [])}")
                print(f"  Spectral gap: {k_analysis.get('spectral_gap', 'N/A'):.4f}")

                physics_interp = step_result.info.get("physics_interpretation", {})
                if physics_interp:
                    print(f"  Topology: {physics_interp.get('topology', 'N/A')}")
                    print(f"  Regime: {physics_interp.get('regime', 'N/A')}")

            step_count += 1

        # Print summary
        persistence_summary = env.get_persistence_summary()
        if persistence_summary:
            print("\nPersistent K-theory Summary:")
            for key, value in persistence_summary.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value}")
                elif isinstance(value, str):
                    print(f"  {key}: {value}")

        # Save report if requested
        if args.save_report:
            import json
            report = {
                "analysis_history": env.get_history(),
                "persistence_summary": persistence_summary,
                "configuration": {
                    "method": args.method,
                    "tolerance": args.tolerance,
                    "max_depth": args.max_depth
                }
            }

            with open(args.save_report, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nReport saved to: {args.save_report}")

        return 0

    except Exception as e:
        print(f"Error in K-theory analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    # Early warning for missing PyTorch
    if torch is None:
        commands_requiring_torch = ['demo', 'helixenv', 'analyze']
        if argv is None or len(argv) == 0 or (len(argv) > 0 and argv[0] in commands_requiring_torch):
            print("⚠️  Warning: PyTorch not found. Most Helix commands require PyTorch.")
            print("   Install with: pip install torch or pip install -e .[torch]")
            print()

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

    if argv and len(argv) > 0 and argv[0] == "reveals":
        return run_reveals_showcase(argv[1:])

    # New: K-theory analysis subcommand
    if argv and len(argv) > 0 and argv[0] == "ktheory":
        return run_ktheory_analysis(argv[1:])

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
            "--animate-forward",
            action="store_true",
            help="Play a layer-by-layer activation animation before the textual summary",
        )
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
    p.add_argument(
        "--animate-forward",
        action="store_true",
        help="Play an activation sweep animation before printing summaries",
    )
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
