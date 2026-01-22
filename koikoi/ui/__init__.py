"""
User interface module for Koi-Koi game.

Contains GUI components built with FreeSimpleGUI.
"""

from koikoi.ui.gui import (
    KoiKoiGUI,
    InitGUI,  # Legacy export
    classify_cards,
)

__all__ = [
    "KoiKoiGUI",
    "InitGUI",
    "classify_cards",
]
