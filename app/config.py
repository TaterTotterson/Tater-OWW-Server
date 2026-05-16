from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _text(value: Any) -> str:
    return str(value or "").strip()


def _bool_env(name: str, default: bool) -> bool:
    token = _text(os.getenv(name)).lower()
    if not token:
        return bool(default)
    if token in {"1", "true", "yes", "on", "enabled"}:
        return True
    if token in {"0", "false", "no", "off", "disabled"}:
        return False
    return bool(default)


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(float(_text(os.getenv(name)) or default))
    except Exception:
        value = int(default)
    return max(int(minimum), min(int(maximum), value))


def _float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(_text(os.getenv(name)) or default)
    except Exception:
        value = float(default)
    return max(float(minimum), min(float(maximum), value))


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    model_source: str
    framework: str
    device: str
    threshold: float
    patience: int
    debounce_s: float
    vad_threshold: float
    model_dir: Path
    idle_ttl_s: float
    cleanup_interval_s: float
    max_chunk_bytes: int
    prefer_hint: bool
    reset_on_detect: bool
    warmup: bool
    log_level: str

    @property
    def engine_key(self) -> str:
        return "|".join(
            [
                self.model_source,
                self.framework,
                self.device,
                f"{self.threshold:.4f}",
                str(self.patience),
                f"{self.debounce_s:.3f}",
                f"{self.vad_threshold:.3f}",
                str(self.prefer_hint),
            ]
        )


def load_settings() -> Settings:
    framework = _text(os.getenv("TATER_OWW_FRAMEWORK") or "onnx").lower()
    if framework not in {"onnx", "tflite"}:
        framework = "onnx"

    device = _text(os.getenv("TATER_OWW_DEVICE") or "auto").lower()
    if device not in {"auto", "cpu", "gpu", "cuda"}:
        device = "auto"
    if device == "cuda":
        device = "gpu"

    model_source = _text(os.getenv("TATER_OWW_MODEL") or "hey_jarvis")

    return Settings(
        host=_text(os.getenv("TATER_OWW_HOST") or "0.0.0.0"),
        port=_int_env("TATER_OWW_PORT", 8502, minimum=1, maximum=65535),
        model_source=model_source,
        framework=framework,
        device=device,
        threshold=_float_env("TATER_OWW_THRESHOLD", 0.50, minimum=0.01, maximum=0.99),
        patience=_int_env("TATER_OWW_PATIENCE", 2, minimum=1, maximum=10),
        debounce_s=_float_env("TATER_OWW_DEBOUNCE_S", 2.0, minimum=0.0, maximum=30.0),
        vad_threshold=_float_env("TATER_OWW_VAD_THRESHOLD", 0.0, minimum=0.0, maximum=0.99),
        model_dir=Path(_text(os.getenv("TATER_OWW_MODEL_DIR") or "/models")).expanduser(),
        idle_ttl_s=_float_env("TATER_OWW_IDLE_TTL_S", 3600.0, minimum=60.0, maximum=86400.0),
        cleanup_interval_s=_float_env("TATER_OWW_CLEANUP_INTERVAL_S", 60.0, minimum=10.0, maximum=3600.0),
        max_chunk_bytes=_int_env("TATER_OWW_MAX_CHUNK_BYTES", 512 * 1024, minimum=1024, maximum=4 * 1024 * 1024),
        prefer_hint=_bool_env("TATER_OWW_PREFER_HINT", True),
        reset_on_detect=_bool_env("TATER_OWW_RESET_ON_DETECT", True),
        warmup=_bool_env("TATER_OWW_WARMUP", True),
        log_level=_text(os.getenv("TATER_OWW_LOG_LEVEL") or "info").lower(),
    )
