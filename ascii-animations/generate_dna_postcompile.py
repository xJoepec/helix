#!/usr/bin/env python3
import math
from pathlib import Path
from typing import List

# Rotating DNA helix generator for garbtronix_animation post-compile format
# Produces a file named `post-compile.txt` in the same directory.

# You can tweak these values to change the look/feel
WIDTH = 80          # characters wide
HEIGHT = 32         # characters tall
FRAMES = 64         # total frames in the loop
WAVELENGTH = 14.0   # vertical rows per full sine wave cycle
AMPLITUDE = int(WIDTH * 0.27)  # horizontal swing of the strands
RUNG_GAP = 2        # vertical spacing between base-pair rungs (rows)

OUTPUT_FILE = Path(__file__).parent / "post-compile.txt"


def depth_char(frontness: float) -> str:
    """Map a frontness value in [-1, 1] to a character for depth shading."""
    s = (frontness + 1.0) / 2.0  # 0..1
    if s > 0.85:
        return "@"
    if s > 0.65:
        return "#"
    if s > 0.45:
        return "O"
    if s > 0.25:
        return "o"
    return "."


def rung_char(frontness: float) -> str:
    s = (frontness + 1.0) / 2.0
    if s > 0.66:
        return "="
    if s > 0.33:
        return "-"
    return "."


def generate_frame(t: int) -> List[str]:
    grid = [[" "] * WIDTH for _ in range(HEIGHT)]

    omega_t = 2.0 * math.pi * (t / FRAMES)  # rotation over time
    cx = WIDTH // 2

    for y in range(HEIGHT):
        phi = 2.0 * math.pi * (y / WAVELENGTH) + omega_t
        # Left/Right strand x-positions
        xL = cx + int(AMPLITUDE * math.sin(phi))
        xR = cx - int(AMPLITUDE * math.sin(phi))
        if xL < 0: xL = 0
        if xL >= WIDTH: xL = WIDTH - 1
        if xR < 0: xR = 0
        if xR >= WIDTH: xR = WIDTH - 1

        # Rungs (base pairs) every RUNG_GAP rows
        if y % RUNG_GAP == 0:
            rc = rung_char(math.cos(phi))  # use cos(phi) for apparent front/back
            x0, x1 = sorted((xL, xR))
            for x in range(x0 + 1, x1):
                grid[y][x] = rc

        # Strands with depth shading
        frontL = math.cos(phi)
        frontR = math.cos(phi + math.pi)  # opposite phase
        grid[y][xL] = depth_char(frontL)
        grid[y][xR] = depth_char(frontR)

    # Optional: add a subtle center line for reference (comment out if undesired)
    # for y in range(HEIGHT):
    #     if grid[y][cx] == " ":
    #         grid[y][cx] = ":"

    return ["".join(row) for row in grid]


def write_post_compile(path: Path):
    with path.open("w", encoding="utf-8") as f:
        for t in range(FRAMES):
            # metadata: drawing number (t) and copies (1)
            f.write(f"{t},1\n")
            f.write("*\n")
            for line in generate_frame(t):
                f.write(line + "\n")
            f.write("**\n")


def main():
    write_post_compile(OUTPUT_FILE)
    print(f"Wrote rotating DNA helix to: {OUTPUT_FILE}")
    print("Copy this post-compile.txt into the root of garbtronix_animation and run main.py.")


if __name__ == "__main__":
    main()
