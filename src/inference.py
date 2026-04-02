from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.preprocessing import preprocess_eye_image


class EyeStateClassifier:
    def __init__(self, model_path: str | Path, label_map_path: str | Path):
        self.model_path = Path(model_path)
        self.label_map_path = Path(label_map_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy model: {self.model_path}")
        if not self.label_map_path.exists():
            raise FileNotFoundError(f"Không tìm thấy label map: {self.label_map_path}")

        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self.idx_to_label = self._load_label_map(self.label_map_path)

    @staticmethod
    def _load_label_map(label_map_path: Path) -> dict[int, str]:
        with open(label_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        idx_to_label: dict[int, str] = {}

        if all(str(k).isdigit() for k in data.keys()):
            for k, v in data.items():
                idx_to_label[int(k)] = str(v)
        else:
            for label, idx in data.items():
                idx_to_label[int(idx)] = str(label)

        return idx_to_label

    def predict(self, eye_bgr: np.ndarray) -> dict[str, Any]:
        x = preprocess_eye_image(eye_bgr)
        preds = self.model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_label = self.idx_to_label.get(pred_idx, str(pred_idx))
        confidence = float(preds[pred_idx])

        probs_by_label: dict[str, float] = {}
        for idx, prob in enumerate(preds):
            label = self.idx_to_label.get(idx, str(idx))
            probs_by_label[label] = float(prob)

        return {
            "label": pred_label,
            "confidence": confidence,
            "probs": probs_by_label,
        }
