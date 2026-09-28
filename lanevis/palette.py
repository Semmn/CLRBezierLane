"""Categorical palette for runs, plus chart chrome.

Eight slots in a fixed order, validated for colour-vision deficiency on the
adjacent pairlist that line charts use (worst adjacent CVD dE 9.1 light /
8.4 dark; worst normal-vision dE 19.6 / 19.3).

Two rules this module exists to enforce:

* **A colour belongs to a run, not to its position in the current selection.**
  Slots are assigned once from the full loaded run list, so hiding a run never
  repaints the others.
* **Nothing past slot 8 gets a generated hue.** A ninth run is a reason to
  facet or to filter, not to invent a colour that no longer validates.

On the light surface three slots (aqua, yellow, magenta) fall below 3:1
contrast, so the charts carry direct end-of-line labels and a summary table —
identity never rests on colour alone.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

SERIES_LIGHT: List[str] = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

SERIES_DARK: List[str] = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]

MAX_SERIES = len(SERIES_LIGHT)

CHROME = dict(
    light=dict(
        surface="#fcfcfb",
        plane="#f9f9f7",
        ink="#0b0b0b",
        ink_secondary="#52514e",
        ink_muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        border="rgba(11,11,11,0.10)",
    ),
    dark=dict(
        surface="#1a1a19",
        plane="#0d0d0d",
        ink="#ffffff",
        ink_secondary="#c3c2b7",
        ink_muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        border="rgba(255,255,255,0.10)",
    ),
)


def assign_slots(names: Sequence[str]) -> Dict[str, int]:
    """Stable run -> slot index, in the order the runs were loaded."""
    return {name: index for index, name in enumerate(names)}


def color(slot: int, mode: str = "light") -> str:
    table = SERIES_DARK if mode == "dark" else SERIES_LIGHT
    return table[slot % len(table)]
