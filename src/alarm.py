from __future__ import annotations

from pathlib import Path


class AlarmPlayer:
    """
    Bộ quản lý trạng thái cảnh báo âm thanh.

    Lưu ý:
    - Trong môi trường local desktop, bạn có thể mở rộng để phát thật.
    - Trong môi trường Streamlit/cloud, việc phát âm thanh server-side thường
      không có ý nghĩa với người dùng cuối.
    - Vì vậy class này được thiết kế an toàn: không làm crash app nếu audio backend
      không khả dụng.

    Hiện tại class tập trung vào:
    - giữ trạng thái alarm_on / alarm_off
    - xác thực file âm thanh có tồn tại hay không
    - cung cấp interface play/stop/close nhất quán
    """

    def __init__(self, alarm_path: str | Path | None = None) -> None:
        self.alarm_path = Path(alarm_path) if alarm_path else None
        self.is_playing = False
        self.is_available = False
        self.backend_name = "none"

        if self.alarm_path is not None and self.alarm_path.exists():
            self.is_available = True

        # Có thể mở rộng backend local ở đây sau này
        # Ví dụ: pygame / simpleaudio / playsound
        # Nhưng để deploy Streamlit an toàn, mặc định không ép dùng backend.
        self.backend_name = "safe_noop"

    def play(self) -> None:
        """
        Bật trạng thái cảnh báo.

        Ở bản an toàn này:
        - không phát âm thanh thật trên server
        - chỉ chuyển trạng thái sang đang cảnh báo
        """
        self.is_playing = True

    def stop(self) -> None:
        """
        Tắt trạng thái cảnh báo.
        """
        self.is_playing = False

    def close(self) -> None:
        """
        Giải phóng tài nguyên nếu có.
        """
        self.is_playing = False

    def status(self) -> dict:
        """
        Trả về thông tin trạng thái alarm.
        """
        return {
            "is_playing": self.is_playing,
            "is_available": self.is_available,
            "backend_name": self.backend_name,
            "alarm_path": str(self.alarm_path) if self.alarm_path else None,
        }
    