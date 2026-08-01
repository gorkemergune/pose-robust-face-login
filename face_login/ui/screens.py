"""Full-screen application screens and scan feedback (Pillow-rendered).

Draws the menu, name-entry, and message screens plus the live scan feedback
banner, returning BGR frames for display through the OpenCV window. Pillow is
used for crisp Unicode/Turkish text. This module is stateless and holds only
visual configuration; it contains no camera, pipeline, or business logic. The
in-frame recognition overlays live in ``overlay.py`` and ``coverage_bar.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Color = tuple[int, int, int]  # RGB

_REGULAR = ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
_BOLD = ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
         "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


@dataclass(frozen=True, slots=True)
class Button:
    """A clickable/keyable region on a screen."""

    action: str
    label: str
    rect: tuple[int, int, int, int]  # x1, y1, x2, y2 (frame coordinates)
    key: str


class ScreenRenderer:
    """Render application screens and scan banners as BGR frames.

    All colors and sizes are constructor-configurable; a single instance is
    stateless and reusable across frames and threads.
    """

    def __init__(self, width: int = 960, height: int = 540) -> None:
        """Store screen geometry and the palette; build no window."""
        self.width = width
        self.height = height
        self._bg_top: Color = (8, 10, 16)       # near-black
        self._bg_bottom: Color = (14, 20, 36)   # deep navy
        self._card: Color = (18, 26, 44)
        self._text: Color = (230, 236, 246)
        self._muted: Color = (110, 124, 152)
        self._green: Color = (34, 132, 116)     # teal (register / success)
        self._blue: Color = (44, 104, 194)      # deep blue (login / info)
        self._red: Color = (184, 66, 66)
        self._gray: Color = (40, 48, 68)        # dark slate (quit / back)
        self._tones = {"positive": self._green, "negative": self._red,
                       "info": self._blue, "success": self._green,
                       "error": self._red}
        self._font_cache: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}

    # -- public screens ----------------------------------------------------

    def menu(self) -> tuple[np.ndarray, list[Button]]:
        """Main menu with Register / Login / Quit choices."""
        img = self._blank()
        d = ImageDraw.Draw(img)
        self._center(d, self.height * 0.16, "Pose-Robust Face Login", 44, True, self._text)
        self._center(d, self.height * 0.28, "Main Menu", 24, False, self._muted)
        buttons = [
            self._make_button(d, 0.42, "Register  [R]", "register", "r", self._green),
            self._make_button(d, 0.58, "Login  [L]", "login", "l", self._blue),
            self._make_button(d, 0.74, "Quit  [Q]", "quit", "q", self._gray),
        ]
        self._center(d, self.height * 0.90, "Click a button or press the shortcut key",
                     18, False, self._muted)
        return self._to_bgr(img), buttons

    def name_input(self, name: str) -> tuple[np.ndarray, list[Button]]:
        """Name-entry screen showing the typed text and Submit / Back actions."""
        img = self._blank()
        d = ImageDraw.Draw(img)
        self._center(d, self.height * 0.18, "Register — Enter Name", 38, True, self._text)
        bx1, by1, bx2, by2 = int(self.width*0.18), int(self.height*0.38), \
            int(self.width*0.82), int(self.height*0.50)
        d.rounded_rectangle([bx1, by1, bx2, by2], radius=12, fill=self._card,
                            outline=self._blue, width=2)
        shown = name if name else "Type your name…"
        color = self._text if name else self._muted
        d.text((bx1 + 20, by1 + (by2 - by1) // 2 - 18), shown + ("|" if name else ""),
               font=self._font(30, False), fill=color)
        buttons = [
            self._make_button(d, 0.62, "Submit  [Enter]", "submit", "", self._green,
                              cx_frac=0.68, w=280),
            self._make_button(d, 0.62, "Back  [Esc]", "back", "", self._gray,
                              cx_frac=0.32, w=280),
        ]
        self._center(d, self.height * 0.86,
                     "[Enter] Submit     [Esc] Back     [Backspace] Delete", 18, False, self._muted)
        return self._to_bgr(img), buttons

    def message(self, title: str, subtitle: str, tone: str) -> np.ndarray:
        """A centered message screen (welcome / success / info / error)."""
        img = self._blank()
        d = ImageDraw.Draw(img)
        accent = self._tones.get(tone, self._blue)
        cx = self.width // 2
        cy = int(self.height * 0.34)
        d.ellipse([cx-46, cy-46, cx+46, cy+46], outline=accent, width=5)
        if tone in ("positive", "success"):
            d.line([(cx-22, cy+2), (cx-6, cy+20)], fill=accent, width=6)
            d.line([(cx-6, cy+20), (cx+24, cy-18)], fill=accent, width=6)
        self._center(d, self.height * 0.56, title, 46, True, self._text)
        if subtitle:
            self._center(d, self.height * 0.68, subtitle, 26, False, self._muted)
        self._center(d, self.height * 0.90, "Press any key to continue", 18, False, self._muted)
        return self._to_bgr(img)

    def scan_banner(self, bgr_frame: np.ndarray, text: str, tone: str) -> np.ndarray:
        """Draw a feedback banner along the bottom of a live scan frame (cv2).

        Uses OpenCV drawing (not Pillow) so it is cheap enough to run on every
        camera frame. English text renders fine with the Hershey font.
        """
        h, w = bgr_frame.shape[:2]
        r, g, b = self._tones.get(tone, self._blue)
        overlay = bgr_frame.copy()
        cv2.rectangle(overlay, (0, h - 64), (w, h), (b, g, r), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, bgr_frame, 0.3, 0.0, dst=bgr_frame)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        cv2.putText(bgr_frame, text, ((w - tw) // 2, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        return bgr_frame

    # -- primitives --------------------------------------------------------

    def _make_button(self, d: ImageDraw.ImageDraw, cy_frac: float, label: str,
                     action: str, key: str, color: Color,
                     cx_frac: float = 0.5, w: int = 380) -> Button:
        """Draw one button and return its clickable region."""
        cx, cy = int(self.width * cx_frac), int(self.height * cy_frac)
        h = 62
        x1, y1, x2, y2 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
        d.rounded_rectangle([x1, y1, x2, y2], radius=14, fill=color)
        f = self._font(26, True)
        tw = d.textlength(label, font=f)
        d.text((cx - tw / 2, cy - 18), label, font=f, fill=(255, 255, 255))
        return Button(action=action, label=label, rect=(x1, y1, x2, y2), key=key)

    def _center(self, d: ImageDraw.ImageDraw, y: float, text: str, size: int,
                bold: bool, fill: Color) -> None:
        """Draw horizontally-centered text at vertical position ``y``."""
        f = self._font(size, bold)
        tw = d.textlength(text, font=f)
        d.text((self.width / 2 - tw / 2, y), text, font=f, fill=fill)

    def _blank(self) -> Image.Image:
        """Return a fresh vertical-gradient background image (RGB)."""
        grad = np.zeros((self.height, self.width, 3), np.uint8)
        top, bot = np.array(self._bg_top), np.array(self._bg_bottom)
        for y in range(self.height):
            grad[y, :] = (top + (bot - top) * (y / self.height)).astype(np.uint8)
        return Image.fromarray(grad)

    def _font(self, size: int, bold: bool) -> ImageFont.FreeTypeFont:
        """Return a cached TrueType font, falling back to Pillow's default."""
        key = (size, bold)
        if key not in self._font_cache:
            for path in (_BOLD if bold else _REGULAR):
                try:
                    self._font_cache[key] = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue
            else:
                self._font_cache[key] = ImageFont.load_default()
        return self._font_cache[key]

    @staticmethod
    def _to_bgr(img: Image.Image) -> np.ndarray:
        """Convert an RGB Pillow image to a contiguous BGR numpy array."""
        return np.ascontiguousarray(np.array(img)[:, :, ::-1])
