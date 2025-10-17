"""Styling utilities for entity resolution evaluation UI."""

GROUP_STYLES = {
    "a": ("■", "red"),
    "b": ("●", "blue"),
    "c": ("▲", "green"),
    "d": ("◆", "yellow"),
    "e": ("★", "magenta"),
    "f": ("⬢", "cyan"),
    "g": ("♦", "bright_red"),
    "h": ("▼", "bright_blue"),
    "i": ("○", "bright_green"),
    "j": ("△", "bright_yellow"),
    "k": ("◇", "bright_magenta"),
    "l": ("☆", "bright_cyan"),
    "m": ("⬡", "white"),
    "n": ("✦", "bright_white"),
    "o": ("✧", "red"),
    "p": ("⟐", "blue"),
    "q": ("■", "green"),
    "r": ("●", "yellow"),
    "s": ("▲", "magenta"),
    "t": ("◆", "cyan"),
    "u": ("★", "bright_red"),
    "v": ("⬢", "bright_blue"),
    "w": ("♦", "bright_green"),
    "x": ("▼", "bright_yellow"),
    "y": ("○", "bright_magenta"),
    "z": ("△", "bright_cyan"),
}


def get_group_style(group: str) -> tuple[str, str]:
    """Get symbol and colour for a group letter.

    Args:
        group: Single letter group identifier (a-z)

    Returns:
        Tuple of (symbol, colour)
    """
    return GROUP_STYLES.get(group.lower(), ("■", "white"))


def get_display_text(group: str, count: int) -> tuple[str, str]:
    """Get formatted display text with colour and symbol.

    Args:
        group: Single letter group identifier (a-z) or "unassigned"
        count: Number of items in this group

    Returns:
        Tuple of (formatted_text, colour)
    """
    if group == "unassigned":
        return f"UNASSIGNED ({count})", "dim"

    symbol, colour = get_group_style(group)
    text = f"{symbol} {group.upper()} ({count})"
    return text, colour


def generate_css_classes() -> str:
    """Generate CSS classes for all groups.

    Returns:
        CSS string with all group styling classes
    """
    lines = []
    for group, (_symbol, colour) in GROUP_STYLES.items():
        lines.append(f"ComparisonTable .group-{group} {{ background: {colour}; }}")
    return "\n".join(lines)
