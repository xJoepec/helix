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
from typing import Any, List, Optional, Sequence, Tuple, Union

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
    build_V_from_incidence,
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


def _build_helix_env_parser(prog: str = "helix helixenv") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Helix operator-algebra environment demo (AF partitions)",
    )
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--d-out", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--data-x", type=str, default="")
    parser.add_argument("--data-y", type=str, default="")
    parser.add_argument("--model-module", type=str, default="")
    parser.add_argument("--model-func", type=str, default="build_model")
    parser.add_argument("--model-kwargs", type=str, default="")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--max-depth", type=int, default=0, help="Limit depth traversal (0 = all)")
    parser.add_argument("--mass-weight", type=float, default=1.0)
    parser.add_argument("--wasted-weight", type=float, default=0.1)
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key for LLM-based judges (exported to the environment for downstream tools)",
    )
    parser.add_argument(
        "--api-key-var",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name to store the provided API key",
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


def _prepare_sequence(seq: Union[Sequence[float], np.ndarray]) -> str:
    if isinstance(seq, np.ndarray):
        arr = seq.tolist()
    else:
        arr = list(seq)
    return ", ".join(f"{x:.3g}" if isinstance(x, (int, float)) else str(x) for x in arr)


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
    width = _prompt_int("Demo hidden width", 16)
    d_out = _prompt_int("Demo output width", 2)
    epochs = _prompt_int("Training epochs (demo or labelled data)", 60)
    max_depth = _prompt_int("Max AF depth (0 = all)", 0)
    mass_weight = _prompt_float("Reward weight: mass error", 1.0)
    wasted_weight = _prompt_float("Reward weight: wasted regions", 0.1)
    api_key = _prompt_secret("LLM judge API key (leave blank to skip)").strip()
    api_key_var = _prompt_text("API key env var", "OPENAI_API_KEY") or "OPENAI_API_KEY"
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
        epochs=epochs,
        no_train=no_train,
        max_depth=max_depth,
        mass_weight=mass_weight,
        wasted_weight=wasted_weight,
        d_out=d_out,
        api_key=api_key,
        api_key_var=api_key_var,
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


def _load_array(path: str) -> np.ndarray:
    p = str(path)
    lower = p.lower()
    if lower.endswith(".npy"):
        return np.load(p)
    if lower.endswith(".npz"):
        npz = np.load(p)
        # Prefer common keys
        for k in ("X", "x", "data", "array"):
            if k in npz:
                return np.array(npz[k])
        # Fallback to first array
        for k in npz.files:
            return np.array(npz[k])
        raise ValueError(f"No arrays found in npz: {p}")
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
    model = MLP(d_in=2, widths=(args.width, args.width), d_out=2)

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
            f"width={args.width}",
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

    api_key = getattr(args, "api_key", "")
    api_key_var = getattr(args, "api_key_var", "OPENAI_API_KEY") or "OPENAI_API_KEY"
    if api_key:
        os.environ[api_key_var] = api_key
        print(
            _subtle(
                f"Stored API key in environment variable {api_key_var} for Helix Verifiers integrations."
            )
        )
    elif os.environ.get(api_key_var):
        print(_subtle(f"Using API key from environment variable {api_key_var}."))

    _seed_torch(args.seed)

    if getattr(args, "data_x", ""):
        X = _load_array(args.data_x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = np.asarray(X, dtype=np.float32)
        y = None
        if getattr(args, "data_y", ""):
            y_arr = _load_array(args.data_y)
            y_flat = np.asarray(y_arr, dtype=np.int64).reshape(-1)
            if y_flat.shape[0] != X.shape[0]:
                raise ValueError("data_y length must match number of rows in data_x")
            y = y_flat
    else:
        X, y = make_moons(n=args.samples, noise=args.noise, seed=args.seed)

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
        widths = (args.width, args.width)
        model = MLP(d_in=d_in, widths=widths, d_out=args.d_out)

    if getattr(args, "weights", ""):
        state = torch.load(args.weights, map_location="cpu")
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

    max_depth = args.max_depth if getattr(args, "max_depth", 0) and args.max_depth > 0 else None
    env = AFPartitionEnv(
        model,
        X,
        max_depth=max_depth,
        mass_weight=args.mass_weight,
        wasted_weight=args.wasted_weight,
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
        meta_bits.append(f"demo_width={args.width}")
    if getattr(args, "weights", ""):
        meta_bits.append("weights=loaded")
    if args.no_train or (y is None):
        meta_bits.append("training=skipped")
    print(_subtle(" | ".join(meta_bits)))
    print(_headline("AF levels"))

    rows = []
    step = env.reset()
    while True:
        obs = step.obs
        rows.append(
            {
                "depth": obs["depth"],
                "regions": obs["n_regions"],
                "mass_err": obs["mass_error"],
                "wasted": obs["wasted_regions"],
                "reward": step.reward,
            }
        )
        if step.done:
            break
        step = env.step()

    header = f"{'depth':>5} {'regions':>9} {'mass_err':>12} {'wasted':>8} {'reward':>10}"
    print(header)
    print(_subtle("-" * len(header)))
    for row in rows:
        print(
            f"{row['depth']:>5} {row['regions']:>9} {row['mass_err']:>12.4e} {row['wasted']:>8} {row['reward']:>10.4f}"
        )

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
    _seed_torch(args.seed)

    # Load data or synthesize
    if args.data_x:
        X = _load_array(args.data_x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = X.astype(np.float32)
    else:
        X, _ = make_moons(n=args.samples, noise=args.noise, seed=args.seed)

    y = None
    if args.data_y:
        y_arr = _load_array(args.data_y)
        y = y_arr.astype(np.int64).reshape(-1)
        if y.shape[0] != X.shape[0]:
            raise ValueError("data_y length must match number of rows in data_x")

    # Build or import model
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
        widths = (args.width, args.width)
        model = MLP(d_in=d_in, widths=widths, d_out=args.d_out)

    # Optional weights
    if args.weights:
        state = torch.load(args.weights, map_location="cpu")
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
    p.add_argument("--width", type=int, default=16)
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
            args.width != 16,
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
