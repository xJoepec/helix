"""
Helix TUI Theme System

Provides consistent theming and color system for the TUI,
matching the CLI's sophisticated OKLCH color palette.
"""

from typing import Optional, Tuple

# OKLCH Color Palette (matching CLI)
# Format: (lightness, chroma, hue)
COLORS = {
    "accent": (0.65, 0.20, 320.0),    # Bright magenta/pink
    "shadow": (0.55, 0.18, 240.0),    # Bright blue
    "headline": (0.70, 0.15, 140.0),  # Bright green
    "subtle": (0.45, 0.05, 0.0),      # Dark gray
    "error": (0.60, 0.22, 15.0),      # Bright red
    "highlight": (0.75, 0.18, 200.0), # Cyan
    "success": (0.65, 0.18, 140.0),   # Green (for success messages)
    "warning": (0.70, 0.20, 60.0),    # Orange (for warnings)
    "info": (0.60, 0.15, 200.0),      # Light blue (for info)
    "background": (0.15, 0.02, 0.0),  # Very dark background
    "surface": (0.20, 0.02, 0.0),     # Dark surface
    "border": (0.35, 0.05, 0.0),      # Border color
}

def oklch_to_rgb(lightness: float, chroma: float, hue: float) -> Tuple[int, int, int]:
    """
    Convert OKLCH color to RGB.
    Simplified conversion for TUI use.
    """
    # Convert to HSL-like values (approximation)
    h = hue / 360.0
    s = min(1.0, chroma * 2.0)  # Approximate saturation from chroma
    # Use a different variable name to avoid E741
    lightness_val = lightness

    # Convert to RGB
    if s == 0:
        r = g = b = lightness_val
    else:
        def hue_to_rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1/6:
                return p + (q - p) * 6 * t
            if t < 1/2:
                return q
            if t < 2/3:
                return p + (q - p) * (2/3 - t) * 6
            return p

        q = lightness_val * (1 + s) if lightness_val < 0.5 else lightness_val + s - lightness_val * s
        p = 2 * lightness_val - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)

    return (int(r * 255), int(g * 255), int(b * 255))

def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex color string."""
    return f"#{r:02x}{g:02x}{b:02x}"

def get_color_hex(color_name: str) -> str:
    """Get hex color value for a named color."""
    if color_name not in COLORS:
        return "#808080"  # Default gray
    lightness, c, h = COLORS[color_name]
    r, g, b = oklch_to_rgb(lightness, c, h)
    return rgb_to_hex(r, g, b)

def interpolate_color(color1: str, color2: str, t: float) -> str:
    """
    Interpolate between two colors for gradient effects.
    t should be between 0 (color1) and 1 (color2).
    """
    if color1 not in COLORS or color2 not in COLORS:
        return get_color_hex(color1)

    lightness1, c1, h1 = COLORS[color1]
    lightness2, c2, h2 = COLORS[color2]

    # Interpolate in OKLCH space
    lightness = lightness1 + (lightness2 - lightness1) * t
    c = c1 + (c2 - c1) * t

    # Handle hue interpolation (shortest path)
    h_diff = h2 - h1
    if h_diff > 180:
        h_diff -= 360
    elif h_diff < -180:
        h_diff += 360
    h = (h1 + h_diff * t) % 360

    r, g, b = oklch_to_rgb(lightness, c, h)
    return rgb_to_hex(r, g, b)

# CSS Template for Textual
def generate_css() -> str:
    """Generate CSS with the full color palette for Textual."""
    css_lines = [
        "/* Helix TUI Theme - Matching CLI OKLCH Palette */",
        ""
    ]

    # Define CSS variables at root level (Textual syntax)
    for name in COLORS:
        hex_color = get_color_hex(name)
        css_lines.append(f"${name}: {hex_color};")
    css_lines.append("")

    # Component-specific styling
    css_lines.extend([
        "/* Header styling */",
        "Header {",
        "    background: $accent;",
        "    color: white;",
        "    height: 3;",
        "}",
        "",
        "/* Button styling */",
        "Button {",
        "    border: solid $accent;",
        "    background: transparent;",
        "}",
        "",
        "Button:hover {",
        "    background: $accent;",
        "    color: white;",
        "    border: solid white;",
        "}",
        "",
        "Button.primary {",
        "    background: $accent;",
        "    color: white;",
        "}",
        "",
        "Button.primary:hover {",
        "    background: $highlight;",
        "}",
        "",
        "Button.secondary {",
        "    border: solid $shadow;",
        "}",
        "",
        "Button.danger {",
        "    border: solid $error;",
        "}",
        "",
        "/* DataTable styling */",
        "DataTable {",
        "    border: round $shadow;",
        "    background: $surface;",
        "}",
        "",
        "DataTable > .datatable--header {",
        "    background: $accent;",
        "    color: white;",
        "    text-style: bold;",
        "}",
        "",
        "DataTable > .datatable--cursor {",
        "    background: $highlight 20%;",
        "}",
        "",
        "DataTable > .datatable--hover {",
        "    background: $shadow 10%;",
        "}",
        "",
        "/* Input styling */",
        "Input {",
        "    border: solid $border;",
        "    background: $surface;",
        "}",
        "",
        "Input:focus {",
        "    border: solid $accent;",
        "}",
        "",
        "/* Container styling */",
        "Container {",
        "    padding: 0 1;",
        "}",
        "",
        "Horizontal {",
        "    height: auto;",
        "}",
        "",
        "/* Progress bar styling */",
        "ProgressBar {",
        "    height: 1;",
        "}",
        "",
        "ProgressBar > .progress--bar {",
        "    color: $accent;",
        "}",
        "",
        "ProgressBar > .progress--percentage {",
        "    color: $highlight;",
        "}",
        "",
        "/* Log styling */",
        "Log {",
        "    background: $surface;",
        "    border: solid $border;",
        "    padding: 0 1;",
        "}",
        "",
        "/* Static text styling */",
        "Static.headline {",
        "    color: $headline;",
        "    text-style: bold;",
        "}",
        "",
        "Static.subtle {",
        "    color: $subtle;",
        "}",
        "",
        "Static.error {",
        "    color: $error;",
        "    text-style: bold;",
        "}",
        "",
        "Static.success {",
        "    color: $success;",
        "}",
        "",
        "Static.warning {",
        "    color: $warning;",
        "}",
        "",
        "Static.info {",
        "    color: $info;",
        "}",
        "",
        "Static.accent {",
        "    color: $accent;",
        "    text-style: bold;",
        "}",
        "",
        "/* Help panel styling */",
        ".help-panel {",
        "    border: round $accent;",
        "    background: $surface;",
        "    padding: 1 2;",
        "    height: 12;",
        "}",
        "",
        "/* Section headers */",
        ".section-header {",
        "    color: $headline;",
        "    text-style: bold;",
        "    padding: 1 0;",
        "}",
        "",
        "/* Divider styling */",
        ".divider {",
        "    color: $border;",
        "    height: 1;",
        "}",
        "",
        "/* Footer styling */",
        "Footer {",
        "    background: $surface;",
        "    color: $subtle;",
        "}",
        "",
        "/* Command input styling */",
        ".command-input {",
        "    background: $background;",
        "    border: solid $accent;",
        "    padding: 0 1;",
        "}",
        "",
        "/* DNA Helix Header */",
        ".dna-helix-header {",
        "    background: $background;",
        "    padding: 1;",
        "    height: auto;",
        "    align: center middle;",
        "}",
        "",
        "/* Results section */",
        ".results-container {",
        "    border: solid $border;",
        "    padding: 1;",
        "    margin: 1 0;",
        "}",
    ])

    return "\n".join(css_lines)

# Message type colors for log output
MESSAGE_COLORS = {
    "error": "error",
    "warning": "warning",
    "success": "success",
    "info": "info",
    "debug": "subtle",
    "headline": "headline",
    "normal": None,  # No special color
}

def get_message_style(message: str) -> Optional[str]:
    """
    Determine the style class for a log message based on its content.
    Returns the appropriate style class name or None for default styling.
    """
    message_lower = message.lower()

    if any(word in message_lower for word in ["error", "failed", "exception", "traceback"]):
        return "error"
    elif any(word in message_lower for word in ["warning", "warn", "caution"]):
        return "warning"
    elif any(word in message_lower for word in ["success", "complete", "done", "finished"]):
        return "success"
    elif any(word in message_lower for word in ["info", "note", "processing"]):
        return "info"
    elif any(word in message_lower for word in ["debug", "verbose", "trace"]):
        return "subtle"
    elif message.startswith("="):  # Divider lines
        return "subtle"
    elif message.startswith("#"):  # Headers
        return "headline"

    return None

# DNA Helix frame colors (for gradient effect)
def get_dna_frame_colors(frame_index: int, total_frames: int) -> Tuple[str, str]:
    """
    Get the gradient colors for a DNA helix frame.
    Returns (primary_color, secondary_color) as hex strings.
    """
    # Create a gradient from accent to shadow color
    t = frame_index / max(1, total_frames - 1)
    primary = interpolate_color("accent", "shadow", t)
    secondary = interpolate_color("shadow", "highlight", t)
    return primary, secondary

# Export key theme elements
__all__ = [
    "COLORS",
    "get_color_hex",
    "interpolate_color",
    "generate_css",
    "MESSAGE_COLORS",
    "get_message_style",
    "get_dna_frame_colors",
]