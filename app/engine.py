from __future__ import annotations

import audioop
import contextlib
import logging
import threading
import time
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from urllib.parse import urlparse

from .config import Settings


logger = logging.getLogger("tater_oww")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _safe_filename(value: Any, *, fallback: str = "openwakeword.onnx") -> str:
    token = Path(_text(value).split("?", 1)[0]).name.strip()
    if not token:
        return fallback
    clean = "".join(ch for ch in token if ch.isalnum() or ch in {".", "-", "_"}).strip("._")
    return clean or fallback


def _download_model(url: str, model_dir: Path, framework: str) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{framework}"
    filename = _safe_filename(url, fallback=f"openwakeword{suffix}")
    if not filename.lower().endswith(suffix):
        filename = f"{filename}{suffix}"
    target = model_dir / filename
    if target.exists() and target.stat().st_size > 0:
        return target
    logger.info("downloading openWakeWord model url=%s target=%s", url, target)
    with urllib_request.urlopen(url, timeout=60) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"model download returned no data: {url}")
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(target)
    return target


def _pretrained_model_path(openwakeword_mod: Any, model_source: str, framework: str) -> Path:
    models = getattr(openwakeword_mod, "MODELS", {}) or {}
    entry = models.get(model_source)
    if isinstance(entry, dict):
        value = entry.get(framework) or entry.get(framework.upper()) or entry.get("model")
    else:
        value = entry
    if not value:
        available = ", ".join(sorted(str(key) for key in models.keys())[:30])
        raise RuntimeError(f"unknown openWakeWord model '{model_source}'. Available examples: {available}")
    path = Path(str(value)).expanduser()
    if not path.exists():
        with contextlib.suppress(Exception):
            from openwakeword.utils import download_models

            logger.info("downloading bundled openWakeWord models")
            download_models()
    return path


def _feature_model_path(openwakeword_mod: Any, key: str, framework: str) -> Path:
    features = getattr(openwakeword_mod, "FEATURE_MODELS", {}) or {}
    entry = features.get(key)
    if isinstance(entry, dict):
        value = entry.get(framework) or entry.get(framework.upper()) or entry.get("model")
    else:
        value = entry
    if not value:
        raise RuntimeError(f"openWakeWord feature model '{key}' is unavailable for {framework}")
    path = Path(str(value)).expanduser()
    if not path.exists():
        with contextlib.suppress(Exception):
            from openwakeword.utils import download_models

            logger.info("downloading bundled openWakeWord feature models")
            download_models()
    return path


def _resolve_model_path(openwakeword_mod: Any, settings: Settings) -> Path:
    source = settings.model_source
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        return _download_model(source, settings.model_dir, settings.framework)
    path = Path(source).expanduser()
    if path.exists():
        return path
    return _pretrained_model_path(openwakeword_mod, source, settings.framework)


def _requested_device(settings: Settings) -> str:
    if settings.device == "gpu":
        return "gpu"
    if settings.device == "cpu":
        return "cpu"
    with contextlib.suppress(Exception):
        import onnxruntime as ort

        providers = set(ort.get_available_providers() or [])
        if "CUDAExecutionProvider" in providers:
            return "gpu"
    return "cpu"


def _exception_chain_text(exc: BaseException) -> str:
    parts: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " | ".join(part for part in parts if part)


def _looks_like_cuda_error(exc: BaseException) -> bool:
    token = _exception_chain_text(exc).lower()
    return any(marker in token for marker in ("cuda", "cublas", "cudnn", "cudaexecutionprovider"))


class DetectorState:
    def __init__(self, settings: Settings, selector: str) -> None:
        import openwakeword
        from openwakeword.model import Model

        model_path = _resolve_model_path(openwakeword, settings)
        melspec_path = _feature_model_path(openwakeword, "melspectrogram", settings.framework)
        embedding_path = _feature_model_path(openwakeword, "embedding", settings.framework)
        device = _requested_device(settings)

        self.model = Model(
            wakeword_models=[str(model_path)],
            inference_framework=settings.framework,
            vad_threshold=float(settings.vad_threshold),
            melspec_model_path=str(melspec_path),
            embedding_model_path=str(embedding_path),
            device=device,
        )
        self.selector = selector
        self.model_source = settings.model_source
        self.model_path = str(model_path)
        self.framework = settings.framework
        self.device = device
        self.lock = threading.Lock()
        self.ratecv_state: Any = None
        self.counts: dict[str, int] = {}
        self.last_detection_ts = 0.0
        self.last_seen_ts = time.time()

    def reset(self) -> None:
        with contextlib.suppress(Exception):
            self.model.reset()
        self.ratecv_state = None
        self.counts = {}


class OpenWakeWordEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._detectors: dict[str, DetectorState] = {}
        self._lock = threading.Lock()
        self._last_cleanup_ts = 0.0
        self._cuda_fallback_until_ts = 0.0
        self._cuda_fallback_reason = ""
        self._last_error = ""

    def warmup(self) -> None:
        self._ensure_detector("__warmup__")

    def reset(self) -> None:
        with self._lock:
            self._detectors.clear()
            self._last_error = ""

    def status(self) -> dict[str, Any]:
        with self._lock:
            detectors = {
                key: {
                    "last_seen_ts": detector.last_seen_ts,
                    "last_detection_ts": detector.last_detection_ts,
                    "device": detector.device,
                    "framework": detector.framework,
                    "model_source": detector.model_source,
                    "model_path": detector.model_path,
                }
                for key, detector in self._detectors.items()
            }
            return {
                "ok": True,
                "available": not bool(self._last_error),
                "error": self._last_error,
                "settings": {
                    "model_source": self.settings.model_source,
                    "framework": self.settings.framework,
                    "device": self.settings.device,
                    "threshold": self.settings.threshold,
                    "patience": self.settings.patience,
                    "debounce_s": self.settings.debounce_s,
                    "vad_threshold": self.settings.vad_threshold,
                    "prefer_hint": self.settings.prefer_hint,
                },
                "detector_count": len(detectors),
                "detectors": detectors,
                "cuda_fallback_active": time.time() < self._cuda_fallback_until_ts,
                "cuda_fallback_reason": self._cuda_fallback_reason,
            }

    def _cleanup_locked(self) -> None:
        now_ts = time.time()
        if (now_ts - self._last_cleanup_ts) < self.settings.cleanup_interval_s:
            return
        self._last_cleanup_ts = now_ts
        expired = [
            key
            for key, detector in self._detectors.items()
            if key != "__warmup__" and (now_ts - detector.last_seen_ts) >= self.settings.idle_ttl_s
        ]
        for key in expired:
            self._detectors.pop(key, None)
        if expired:
            logger.info("cleaned idle detectors selectors=%s", ",".join(sorted(expired)))

    def _new_detector(self, selector: str) -> DetectorState:
        try:
            return DetectorState(self.settings, selector)
        except Exception as exc:
            if self.settings.device != "cpu" and _looks_like_cuda_error(exc):
                self._cuda_fallback_until_ts = time.time() + 900.0
                self._cuda_fallback_reason = _exception_chain_text(exc)[:1000]
                logger.warning("GPU openWakeWord failed, falling back to CPU for this process: %s", self._cuda_fallback_reason)
                fallback = Settings(
                    **{
                        **self.settings.__dict__,
                        "device": "cpu",
                    }
                )
                return DetectorState(fallback, selector)
            raise

    def _ensure_detector(self, selector: str) -> DetectorState:
        token = _text(selector) or "remote"
        with self._lock:
            self._cleanup_locked()
            detector = self._detectors.get(token)
            if detector is not None:
                detector.last_seen_ts = time.time()
                return detector
        try:
            detector = self._new_detector(token)
        except Exception as exc:
            self._last_error = _exception_chain_text(exc) or str(exc)
            raise
        with self._lock:
            self._last_error = ""
            existing = self._detectors.get(token)
            if existing is not None:
                return existing
            self._detectors[token] = detector
            logger.info(
                "loaded openWakeWord detector selector=%s model=%s framework=%s device=%s",
                token,
                detector.model_source,
                detector.framework,
                detector.device,
            )
            return detector

    def _pcm_to_pcm16_mono_16k(self, detector: DetectorState, audio_bytes: bytes, audio_format: dict[str, int]) -> bytes:
        data = bytes(audio_bytes or b"")
        if not data:
            return b""

        rate = int(audio_format.get("rate") or 16000)
        width = int(audio_format.get("width") or 2)
        channels = int(audio_format.get("channels") or 1)

        if width != 2:
            data = audioop.lin2lin(data, width, 2)
            width = 2
        if channels > 1:
            data = audioop.tomono(data, width, 0.5, 0.5)
            channels = 1
        if rate != 16000:
            data, detector.ratecv_state = audioop.ratecv(data, width, channels, rate, 16000, detector.ratecv_state)
        return data

    def process_audio(
        self,
        *,
        selector: str,
        audio_bytes: bytes,
        audio_format: dict[str, int],
        wake_word_hint: str = "",
    ) -> dict[str, Any]:
        detector = self._ensure_detector(selector)
        detector.last_seen_ts = time.time()
        with detector.lock:
            pcm = self._pcm_to_pcm16_mono_16k(detector, audio_bytes, audio_format)
            if not pcm:
                return {"ok": True, "detected": False}

            import numpy as np

            samples = np.frombuffer(pcm, dtype=np.int16)
            if samples.size <= 0:
                return {"ok": True, "detected": False}

            predictions = detector.model.predict(samples)
            if not isinstance(predictions, dict) or not predictions:
                return {"ok": True, "detected": False}

            best_label, best_score = max(
                ((_text(label), float(score or 0.0)) for label, score in predictions.items()),
                key=lambda item: item[1],
            )
            now_ts = time.time()
            for label in list(detector.counts.keys()):
                if label != best_label:
                    detector.counts[label] = 0
            if best_score >= self.settings.threshold:
                detector.counts[best_label] = int(detector.counts.get(best_label, 0)) + 1
            else:
                detector.counts[best_label] = 0

            detected = (
                best_score >= self.settings.threshold
                and int(detector.counts.get(best_label, 0)) >= self.settings.patience
                and (now_ts - detector.last_detection_ts) >= self.settings.debounce_s
            )
            if not detected:
                return {
                    "ok": True,
                    "detected": False,
                    "score": best_score,
                    "best_label": best_label,
                }

            detector.last_detection_ts = now_ts
            if self.settings.reset_on_detect:
                detector.reset()
            wake_word = _text(wake_word_hint) if self.settings.prefer_hint and _text(wake_word_hint) else best_label
            return {
                "ok": True,
                "detected": True,
                "wake_word": wake_word or "openwakeword",
                "score": best_score,
                "engine": "openwakeword",
                "model_source": detector.model_source,
                "model_label": best_label,
            }
