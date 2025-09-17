#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path
from typing import List

# Simple CLI player for garbtronix_animation post-compile files
# Plays frames in the terminal using ANSI escape codes.
# Usage:
#   python3 play_postcompile_cli.py --file post-compile.txt --fps 15 --loop


def parse_post_compile(file_path: Path) -> List[List[str]]:
    """Parse a garbtronix post-compile file into a list of frames (list of lines)."""
    frames: List[List[str]] = []
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    capturing = False
    copies = 1
    current: List[str] = []
    expecting_header = True

    # Read without stripping spaces; only strip newlines
    with file_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if expecting_header:
                # Header should be like: "0,1" (drawing index, copies count)
                try:
                    parts = line.split(",")
                    # drawing_index = int(parts[0])  # unused
                    copies = int(parts[1]) if len(parts) > 1 else 1
                except Exception:
                    # If header malformed, default to 1 copy
                    copies = 1
                expecting_header = False
                continue

            if line == "*":
                capturing = True
                current = []
                continue

            if line == "**":
                # End of one frame block
                capturing = False
                # Duplicate per copies
                for _ in range(max(1, copies)):
                    frames.append(list(current))
                expecting_header = True
                continue

            if capturing:
                current.append(line)

    return frames


def hide_cursor() -> None:
    sys.stdout.write("\x1b[?25l")
    sys.stdout.flush()


def show_cursor() -> None:
    sys.stdout.write("\x1b[?25h")
    sys.stdout.flush()


def clear_screen() -> None:
    # Clear and move cursor to home
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()


def render_frame(lines: List[str]) -> None:
    # Move to home and print lines
    sys.stdout.write("\x1b[H")
    sys.stdout.write("\n".join(lines))
    sys.stdout.flush()


def play_cli(frames: List[List[str]], fps: float, loop: bool) -> None:
    if fps <= 0:
        fps = 15.0
    delay = 1.0 / fps

    print("Press Ctrl+C to exit.")
    hide_cursor()
    try:
        first = True
        while True:
            for frame in frames:
                if first:
                    clear_screen()
                    first = False
                render_frame(frame)
                time.sleep(delay)
            if not loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        show_cursor()
        # Move to next line so the shell prompt appears cleanly
        sys.stdout.write("\n")
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description="CLI player for garbtronix post-compile animations")
    ap.add_argument("--file", "-f", type=Path, default=Path("post-compile.txt"), help="Path to post-compile.txt")
    ap.add_argument("--fps", type=float, default=15.0, help="Frames per second")
    ap.add_argument("--loop", action="store_true", help="Loop playback (default: false)")
    args = ap.parse_args()

    frames = parse_post_compile(args.file)
    if not frames:
        print("No frames parsed. Is the file in the correct post-compile format?", file=sys.stderr)
        sys.exit(1)

    play_cli(frames, args.fps, loop=args.loop)


if __name__ == "__main__":
    main()
