"""
Custom Textual Widgets for Helix TUI

Provides styled widgets that match the CLI's visual design,
including animated headers, color-coded logs, and themed components.
"""

import os
from pathlib import Path
from typing import List, Optional

from rich.console import RenderableType
from rich.text import Text
from textual import work
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Button, DataTable, Log, Static

try:
    from .tui_theme import (
        COLORS,
        get_color_hex,
        get_message_style,
        interpolate_color,
    )
except ImportError:
    # Fallback for direct imports
    from tui_theme import (
        COLORS,
        get_color_hex,
        get_message_style,
        interpolate_color,
    )


class DNAHelixHeader(Static):
    """
    Animated DNA helix header matching the CLI's signature banner.
    Displays animated frames from ascii-animations/post-compile.txt.
    """

    MAX_HEADER_HEIGHT = 20
    MIN_HEADER_HEIGHT = 8
    MIN_BODY_SPACE = 6  # Minimum rows we want to leave for the rest of the UI

    DEFAULT_CSS = """
    DNAHelixHeader {
        min-height: 8;
        max-height: 20;
        width: 100%;
        padding: 1 2;
        align: center middle;
        text-align: center;
        overflow: hidden;
    }
    """

    frame_index = reactive(0)
    frames: List[str] = []
    colorized_frames: List[Text] = []  # Cache for colorized frames
    animation_timer: Optional[Timer] = None
    resize_timer: Optional[Timer] = None  # For debouncing resize events

    def __init__(self, animation_file: Optional[str] = None, **kwargs):
        """
        Initialize the DNA helix header.

        Args:
            animation_file: Path to the animation frames file.
                          Defaults to ascii-animations/post-compile.txt.
        """
        super().__init__("", **kwargs)

        # Find the animation file
        if animation_file is None:
            # Try to find it relative to the module
            module_dir = Path(__file__).parent.parent.parent
            animation_file = module_dir / "ascii-animations" / "post-compile.txt"

        self.animation_file = animation_file
        # Defer frame loading to on_mount to avoid blocking I/O

    def _update_responsive_height(self, screen_height: Optional[int] = None) -> None:
        """
        Resize the header so it doesn't consume the entire window height.

        When the terminal height shrinks we clip the animation but keep
        a few rows free so the rest of the UI never disappears entirely.
        """
        if screen_height is None:
            screen = getattr(self, "screen", None)
            if screen and getattr(screen, "size", None):
                screen_height = screen.size.height

        if not screen_height:
            return

        available = screen_height - self.MIN_BODY_SPACE
        target_height = min(
            self.MAX_HEADER_HEIGHT,
            max(self.MIN_HEADER_HEIGHT, available),
        )
        target_height = min(target_height, screen_height)
        self.styles.height = target_height

        # Drop vertical padding when extremely compact so the ASCII art
        # remains the only visible portion.
        if target_height <= self.MIN_HEADER_HEIGHT + 1:
            self.styles.padding = (0, 2)
        else:
            self.styles.padding = (1, 2)

    @work(exclusive=True, thread=True)
    def _load_frames_worker(self) -> List[str]:
        """Load animation frames from the file in a worker thread."""
        if not os.path.exists(self.animation_file):
            # Fallback to a simple static header if file not found
            return [self._create_static_helix()]

        try:
            with open(self.animation_file, 'r') as f:
                content = f.read()

            # Parse frames (format: frame_number,frame_index\n*\nframe_content\n**)
            raw_frames = content.split('**')
            loaded_frames = []

            for frame_data in raw_frames:
                if not frame_data.strip():
                    continue

                lines = frame_data.strip().split('\n')
                if len(lines) > 2 and lines[1] == '*':
                    # Extract frame content (skip metadata and markers)
                    frame_lines = lines[2:]
                    # Normalize frame to exactly 18 lines
                    normalized_lines = []
                    for line in frame_lines[:18]:  # Take up to 18 lines
                        normalized_lines.append(self._normalize_frame_line(line))
                    # Pad with empty lines if needed
                    while len(normalized_lines) < 18:
                        normalized_lines.append(' ' * 46)
                    frame = '\n'.join(normalized_lines)
                    loaded_frames.append(frame)

            if not loaded_frames:
                return [self._create_static_helix()]
            return loaded_frames
        except Exception:
            # Fallback on any error
            return [self._create_static_helix()]

    def _on_frames_loaded(self, loaded_frames: List[str]) -> None:
        """Callback for when frames have been loaded by the worker."""
        self.frames = loaded_frames
        # Pre-colorize all frames for better performance
        self.colorized_frames = [self._colorize_frame(frame) for frame in self.frames]

        if len(self.frames) > 1:
            # Start animation timer (15 FPS like CLI)
            if self.animation_timer is None:
                self.animation_timer = self.set_interval(1/15, self.advance_frame)
        else:
            # Show static frame
            self.update(self.colorized_frames[0])

    def _normalize_frame_line(self, line: str) -> str:
        """Normalize a frame line to fixed width."""
        # Ensure fixed width of 46 characters
        line = line.rstrip()  # Remove trailing spaces
        return line.ljust(46)[:46]  # Pad or truncate to exactly 46 chars

    def _create_static_helix(self) -> str:
        """Create a simple static helix for fallback."""
        return """
╔═══════════════════════════════════════════════════════════════════════╗
║                           H E L I X                                  ║
║                   Neural Network Analyzer                            ║
╚═══════════════════════════════════════════════════════════════════════╝
        """.strip()

    def _colorize_frame(self, frame: str) -> Text:
        """Apply gradient coloring to a frame."""
        text = Text()
        lines = frame.split('\n')
        total_lines = len(lines)

        for i, line in enumerate(lines):
            # Calculate gradient position
            t = i / max(1, total_lines - 1)

            # Get interpolated color
            if "HELIX" in line:
                # Special highlighting for HELIX text
                color = get_color_hex("accent")
                text.append(line, style=f"bold {color}")
            else:
                # Gradient from accent to shadow
                color = interpolate_color("accent", "shadow", t)
                text.append(line, style=color)

            if i < total_lines - 1:
                text.append("\n")

        return text

    async def on_mount(self) -> None:
        """Start animation when widget is mounted."""
        self._update_responsive_height()
        worker = self._load_frames_worker()
        frames = await worker.wait()
        self._on_frames_loaded(frames)

    def advance_frame(self) -> None:
        """Advance to the next animation frame."""
        if self.colorized_frames:
            self.frame_index = (self.frame_index + 1) % len(self.colorized_frames)
            self.update(self.colorized_frames[self.frame_index])

    def on_resize(self, event) -> None:
        """Handle window resize events."""
        self._update_responsive_height()
        # Pause animation during resize
        if self.animation_timer:
            self.animation_timer.pause()

        # Cancel existing resize timer
        if self.resize_timer:
            self.resize_timer.stop()

        # Start debounce timer to resume animation after resize stabilizes
        self.resize_timer = self.set_timer(0.3, self._resume_animation)

    def _resume_animation(self) -> None:
        """Resume animation after resize stabilizes."""
        if self.animation_timer and len(self.colorized_frames) > 1:
            self.animation_timer.resume()
        # Clear resize timer reference
        self.resize_timer = None

    def on_unmount(self) -> None:
        """Stop animation when widget is unmounted."""
        if self.animation_timer:
            self.animation_timer.stop()
        if self.resize_timer:
            self.resize_timer.stop()


class RichLog(Log):
    """
    Enhanced log widget with color-coded output based on message type.
    Automatically detects and styles errors, warnings, success messages, etc.
    """

    DEFAULT_CSS = """
    RichLog {
        padding: 0 1;
    }
    """

    def write_line(self, line: str) -> None:
        """
        Write a line with automatic color coding based on content.

        Args:
            line: The line to write to the log.
        """
        # Determine message style
        style = get_message_style(line)

        if style:
            # Create styled text
            text = Text(line)
            color = get_color_hex(style)
            text.stylize(color)

            # Add bold for certain styles
            if style in ["error", "headline", "success"]:
                text.stylize("bold")

            super().write(text.markup)
        else:
            # Default styling
            super().write_line(line)

    def write_error(self, message: str) -> None:
        """Write an error message with appropriate styling."""
        text = Text(f"✗ {message}")
        text.stylize(f"bold {get_color_hex('error')}")
        super().write(text.markup)

    def write_success(self, message: str) -> None:
        """Write a success message with appropriate styling."""
        text = Text(f"✓ {message}")
        text.stylize(f"bold {get_color_hex('success')}")
        super().write(text.markup)

    def write_warning(self, message: str) -> None:
        """Write a warning message with appropriate styling."""
        text = Text(f"⚠ {message}")
        text.stylize(f"bold {get_color_hex('warning')}")
        super().write(text.markup)

    def write_info(self, message: str) -> None:
        """Write an info message with appropriate styling."""
        text = Text(f"ℹ {message}")
        text.stylize(get_color_hex('info'))
        super().write(text.markup)

    def write_headline(self, message: str) -> None:
        """Write a headline message with appropriate styling."""
        text = Text(message)
        text.stylize(f"bold {get_color_hex('headline')}")
        super().write(text.markup)


class StyledText(Static):
    """
    Static text widget with semantic styling options.
    Supports headline, subtle, error, success, warning, info, and accent styles.
    """

    def __init__(
        self,
        renderable: RenderableType = "",
        *,
        style_type: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize styled text.

        Args:
            renderable: The text to display.
            style_type: One of "headline", "subtle", "error", "success",
                       "warning", "info", "accent", or None for default.
        """
        super().__init__(renderable, **kwargs)

        if style_type and style_type in COLORS:
            self.add_class(style_type)
        elif style_type:
            # Handle special style types
            style_map = {
                "headline": "headline",
                "subtle": "subtle",
                "error": "error",
                "success": "success",
                "warning": "warning",
                "info": "info",
                "accent": "accent",
            }
            if style_type in style_map:
                self.add_class(style_map[style_type])


class ThemedButton(Button):
    """
    Themed button with support for different button types.
    Supports primary, secondary, danger, and default styles.
    """

    def __init__(
        self,
        label: str = "Button",
        *,
        button_type: str = "default",
        **kwargs
    ):
        """
        Initialize themed button.

        Args:
            label: Button label text.
            button_type: One of "primary", "secondary", "danger", or "default".
        """
        super().__init__(label, **kwargs)

        if button_type == "primary":
            self.add_class("primary")
        elif button_type == "secondary":
            self.add_class("secondary")
        elif button_type == "danger":
            self.add_class("danger")


class SectionHeader(Static):
    """
    Section header with consistent styling and optional divider.
    """

    DEFAULT_CSS = """
    SectionHeader {
        height: auto;
        padding: 1 0;
    }
    """

    def __init__(
        self,
        title: str,
        *,
        show_divider: bool = True,
        **kwargs
    ):
        """
        Initialize section header.

        Args:
            title: The section title.
            show_divider: Whether to show a divider line below the title.
        """
        # Create the header text
        text = Text(title.upper())
        text.stylize(f"bold {get_color_hex('headline')}")

        if show_divider:
            # Add divider line
            divider = "─" * len(title)
            text.append(f"\n{divider}", style=get_color_hex('border'))

        super().__init__(text, **kwargs)
        self.add_class("section-header")


class ThemedDataTable(DataTable):
    """
    Enhanced DataTable with automatic theming and color coding.
    """

    def on_mount(self) -> None:
        """Apply theming when mounted."""
        super().on_mount()

        # Set zebra striping
        self.zebra_stripes = True

        # Set cursor style
        self.cursor_type = "row"

    def add_column_with_style(
        self,
        label: str,
        *,
        style_type: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Add a column with optional style type.

        Args:
            label: Column label.
            style_type: Style type for the column (e.g., "error", "success").
        """
        # Create styled label
        text = Text(label)
        if style_type:
            color = get_color_hex(style_type)
            text.stylize(f"bold {color}")

        self.add_column(text, **kwargs)

    def add_row_with_styles(
        self,
        *values,
        style_types: Optional[List[Optional[str]]] = None
    ) -> None:
        """
        Add a row with optional per-cell styling.

        Args:
            values: Cell values.
            style_types: List of style types for each cell.
        """
        styled_values = []

        for i, value in enumerate(values):
            if style_types and i < len(style_types) and style_types[i]:
                text = Text(str(value))
                color = get_color_hex(style_types[i])
                text.stylize(color)
                styled_values.append(text)
            else:
                styled_values.append(value)

        self.add_row(*styled_values)


class ButtonGroup(Horizontal):
    """
    Group of buttons with semantic organization and spacing.
    """

    DEFAULT_CSS = """
    ButtonGroup {
        height: auto;
        margin: 1 0;
        padding: 0;
    }

    ButtonGroup > Button {
        margin: 0 1;
    }

    ButtonGroup.actions {
        align: left middle;
    }

    ButtonGroup.export {
        align: center middle;
    }

    ButtonGroup.config {
        align: right middle;
    }
    """

    def __init__(
        self,
        *buttons: Button,
        group_type: str = "actions",
        **kwargs
    ):
        """
        Initialize button group.

        Args:
            buttons: Buttons to include in the group.
            group_type: One of "actions", "export", or "config".
        """
        super().__init__(*buttons, **kwargs)
        self.add_class(group_type)


class Divider(Static):
    """
    Visual divider with consistent styling.
    """

    DEFAULT_CSS = """
    Divider {
        height: 1;
        padding: 0;
        margin: 1 0;
    }
    """

    def __init__(self, char: str = "─", **kwargs):
        """
        Initialize divider.

        Args:
            char: Character to use for the divider line.
        """
        super().__init__("", **kwargs)
        self.char = char
        self.add_class("divider")

    def on_mount(self) -> None:
        """Create divider line on mount."""
        width = self.size.width if self.size else 80
        line = self.char * width
        text = Text(line, style=get_color_hex('border'))
        self.update(text)

    def on_resize(self) -> None:
        """Update divider on resize."""
        if self.size:
            line = self.char * self.size.width
            text = Text(line, style=get_color_hex('border'))
            self.update(text)


# Export all custom widgets
__all__ = [
    "DNAHelixHeader",
    "RichLog",
    "StyledText",
    "ThemedButton",
    "SectionHeader",
    "ThemedDataTable",
    "ButtonGroup",
    "Divider",
]
