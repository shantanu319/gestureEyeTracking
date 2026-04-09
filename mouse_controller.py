from __future__ import annotations

from typing import Optional


class MouseController:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.available = False
        self.disabled_reason: Optional[str] = None
        self.dragging = False
        self._pyautogui = None
        self.screen_size = (1280, 720)

        try:
            import pyautogui

            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0
            self._pyautogui = pyautogui
            size = pyautogui.size()
            self.screen_size = (int(size[0]), int(size[1]))
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on desktop session
            self.disabled_reason = str(exc)

    @property
    def is_enabled(self) -> bool:
        return self.enabled and self.available

    def set_enabled(self, enabled: bool) -> None:
        if not enabled:
            self.sync_drag(False)
        self.enabled = enabled

    def status_message(self) -> str:
        if self.is_enabled:
            return "Mouse control: ON"
        if self.available:
            return "Mouse control: OFF"
        if self.disabled_reason:
            return f"Mouse control unavailable: {self.disabled_reason}"
        return "Mouse control unavailable"

    def _call(self, method_name: str, *args, **kwargs) -> bool:
        if not self.is_enabled:
            return False

        method = getattr(self._pyautogui, method_name)
        try:
            method(*args, **kwargs)
            return True
        except Exception as exc:  # pragma: no cover - depends on OS permissions
            self.available = False
            self.disabled_reason = str(exc)
            return False

    def _clamp_point(self, x: float, y: float) -> tuple[int, int]:
        max_x = max(self.screen_size[0] - 1, 0)
        max_y = max(self.screen_size[1] - 1, 0)
        return (
            int(min(max(round(x), 0), max_x)),
            int(min(max(round(y), 0), max_y)),
        )

    def move_to(self, x: float, y: float) -> bool:
        target = self._clamp_point(x, y)
        return self._call("moveTo", *target)

    def click(self) -> bool:
        return self._call("click")

    def right_click(self) -> bool:
        return self._call("click", button="right")

    def scroll(self, delta: int) -> bool:
        return self._call("scroll", int(delta))

    def mouse_down(self) -> bool:
        if self.dragging:
            return True
        success = self._call("mouseDown", button="left")
        if success:
            self.dragging = True
        return success

    def mouse_up(self) -> bool:
        if not self.dragging:
            return True
        if not self.available or not self.enabled:
            self.dragging = False
            return False
        success = self._call("mouseUp", button="left")
        if success:
            self.dragging = False
        return success

    def sync_drag(self, active: bool) -> None:
        if active:
            self.mouse_down()
        else:
            self.mouse_up()
