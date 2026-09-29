"""Write paper numbers as reproducible LaTeX macros."""

import re
from numbers import Integral, Real
from pathlib import Path


def write_macros(path, values: dict) -> None:
    if any(not re.fullmatch(r"[A-Za-z]+", name) for name in values):
        raise ValueError("Macro names must contain ASCII letters only")
    lines = []
    for name, value in sorted(values.items()):
        if isinstance(value, Integral):
            value = f"{value:,}"
        elif isinstance(value, Real):
            value = f"{value:.2f}"
            # A hyphen prints too short in text; a real minus sign works in text and math.
            if value.startswith("-"):
                value = "\\ensuremath{-}" + value[1:]
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}\n")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
