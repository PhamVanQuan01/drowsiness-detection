from __future__ import annotations

from pathlib import Path

import pygame


class AlarmPlayer:
    def __init__(self, alarm_path: str | Path):
        self.alarm_path = Path(alarm_path)
        self.available = False
        self.sound = None
        self.is_playing = False

        if not self.alarm_path.exists():
            print(f"[WARN] Không tìm thấy file âm thanh: {self.alarm_path}")
            return

        try:
            pygame.mixer.init()
            self.sound = pygame.mixer.Sound(str(self.alarm_path))
            self.available = True
        except Exception as e:
            print(f"[WARN] Không khởi tạo được audio: {e}")
            self.available = False

    def play(self) -> None:
        if not self.available or self.sound is None:
            return
        if not self.is_playing:
            self.sound.play(loops=-1)
            self.is_playing = True

    def stop(self) -> None:
        if not self.available or self.sound is None:
            return
        self.sound.stop()
        self.is_playing = False

    def close(self) -> None:
        try:
            self.stop()
            if self.available:
                pygame.mixer.quit()
        except Exception:
            pass
