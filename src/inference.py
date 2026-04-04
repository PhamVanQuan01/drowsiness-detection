from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
from tensorflow.keras.models import load_model

from src.preprocessing import prepare_eye_crop_for_model


class EyeStateClassifier:
    """
    Classifier dùng model Keras để phân loại trạng thái mắt.

    Kỳ vọng:
    - model nhận input ảnh crop mắt
    - label_map.json ánh xạ index -> tên nhãn

    Ví dụ label_map.json:
    {
        "0": "Closed_Eyes",
        "1": "Open_Eyes"
    }
    """

    def __init__(
        self,
        model_path: str | Path,
        label_map_path: str | Path,
        input_size: tuple[int, int] = (64, 64),
        closed_label_name: str = "Closed_Eyes",
    ) -> None:
        self.model_path = Path(model_path)
        self.label_map_path = Path(label_map_path)
        self.input_size = input_size
        self.closed_label_name = closed_label_name

        self.model = self._load_model()
        self.label_map = self._load_label_map()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy model: {self.model_path}")

        try:
            # Load model an toàn: compile=False tránh lệch version Keras khi deserialize
            model = load_model(self.model_path, compile=False, safe_mode=False)
            return model
        except Exception as exc:
            raise RuntimeError(f"Lỗi khi load model Keras từ {self.model_path}: {exc}") from exc

    def _load_label_map(self) -> dict[int, str]:
        if not self.label_map_path.exists():
            raise FileNotFoundError(f"Không tìm thấy label map: {self.label_map_path}")

        try:
            with open(self.label_map_path, "r", encoding="utf-8") as f:
                raw_map = json.load(f)

            # Chuẩn hóa key về int
            label_map: dict[int, str] = {}
            for k, v in raw_map.items():
                label_map[int(k)] = str(v)

            if not label_map:
                raise ValueError("label_map.json rỗng.")

            return label_map
        except Exception as exc:
            raise RuntimeError(f"Lỗi khi đọc label map từ {self.label_map_path}: {exc}") from exc

    def preprocess(self, eye_crop_bgr: np.ndarray) -> np.ndarray:
        """
        Chuẩn bị input cho model.
        """
        return prepare_eye_crop_for_model(
            eye_crop_bgr=eye_crop_bgr,
            input_size=self.input_size,
            convert_to_rgb=True,
            normalize=True,
        )

    def predict_proba(self, eye_crop_bgr: np.ndarray) -> np.ndarray:
        """
        Trả về vector xác suất cho toàn bộ lớp.

        Returns
        -------
        np.ndarray
            Shape (num_classes,)
        """
        x = self.preprocess(eye_crop_bgr)

        preds = self.model.predict(x, verbose=0)
        preds = np.asarray(preds)

        if preds.ndim == 2 and preds.shape[0] == 1:
            preds = preds[0]

        if preds.ndim != 1:
            raise RuntimeError(f"Output model không đúng shape kỳ vọng. Nhận được shape={preds.shape}")

        return preds.astype(float)

    def predict(self, eye_crop_bgr: np.ndarray | None) -> dict[str, Any]:
        """
        Dự đoán trạng thái mắt từ ảnh crop.

        Returns
        -------
        dict[str, Any]
            {
                "label": "Closed_Eyes",
                "confidence": 0.92,
                "probs": {
                    "Closed_Eyes": 0.92,
                    "Open_Eyes": 0.08
                }
            }
        """
        if eye_crop_bgr is None or eye_crop_bgr.size == 0:
            return {
                "label": "Unknown",
                "confidence": 0.0,
                "probs": {},
            }

        probs_vec = self.predict_proba(eye_crop_bgr)

        probs: dict[str, float] = {}
        for idx, prob in enumerate(probs_vec):
            label_name = self.label_map.get(idx, f"class_{idx}")
            probs[label_name] = float(prob)

        best_index = int(np.argmax(probs_vec))
        best_label = self.label_map.get(best_index, f"class_{best_index}")
        best_conf = float(probs_vec[best_index])

        return {
            "label": best_label,
            "confidence": best_conf,
            "probs": probs,
        }

    def get_closed_prob(self, pred: dict[str, Any]) -> float:
        """
        Lấy xác suất class Closed_Eyes.
        """
        probs = pred.get("probs", {})
        return float(probs.get(self.closed_label_name, 0.0))

    def is_closed(
        self,
        pred: dict[str, Any],
        threshold: float = 0.75,
    ) -> bool:
        """
        Kiểm tra mắt có được dự đoán là nhắm không.
        """
        return self.get_closed_prob(pred) >= threshold
    