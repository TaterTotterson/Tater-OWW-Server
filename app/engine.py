from __future__ import annotations

import audioop
import contextlib
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from urllib.parse import unquote, urlparse

from .config import Settings


logger = logging.getLogger("tater_oww")
_MODEL_SUFFIXES = {".onnx", ".tflite"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _safe_filename(value: Any, *, fallback: str = "openwakeword.onnx") -> str:
    token = Path(unquote(_text(value).split("?", 1)[0])).name.strip()
    if not token:
        return fallback
    clean = "".join(ch for ch in token if ch.isalnum() or ch in {".", "-", "_"}).strip("._")
    return clean or fallback


def _slug(value: Any) -> str:
    token = _text(value).lower().replace("-", "_").replace(" ", "_")
    return "_".join(part for part in token.split("_") if part)


def _framework_from_source(source: Any, framework: str = "") -> str:
    suffix = Path(_text(source).split("?", 1)[0]).suffix.lower().lstrip(".")
    if suffix in {"onnx", "tflite"}:
        return suffix
    framework_token = _text(framework).lower()
    return framework_token if framework_token in {"onnx", "tflite"} else ""


def _looks_like_model_path(value: Any) -> bool:
    token = _text(value)
    if not token:
        return False
    suffix = Path(token.split("?", 1)[0]).suffix.lower()
    return suffix in _MODEL_SUFFIXES or "/" in token or "\\" in token


def _download_file(url: str, target: Path, *, timeout: float = 120.0) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    logger.info("downloading openWakeWord model url=%s target=%s", url, target)
    with urllib_request.urlopen(url, timeout=timeout) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"model download returned no data: {url}")
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(target)
    return target


def _download_model(url: str, model_dir: Path, framework: str) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{framework}"
    filename = _safe_filename(url, fallback=f"openwakeword{suffix}")
    if not filename.lower().endswith(suffix):
        filename = f"{filename}{suffix}"
    return _download_file(url, model_dir / "custom" / filename)


def _resource_url_for_framework(url: str, framework: str) -> str:
    token = _text(url)
    if framework == "onnx":
        return token.replace(".tflite", ".onnx")
    return token.replace(".onnx", ".tflite")


def _resource_path_for_framework(path: str, framework: str) -> str:
    token = _text(path)
    if framework == "onnx":
        return token.replace(".tflite", ".onnx")
    return token.replace(".onnx", ".tflite")


def _local_model_matches(filename: str, settings: Settings, *, framework: str = "") -> list[Path]:
    name = _text(filename)
    if not name or not settings.model_dir.exists():
        return []
    framework_token = _framework_from_source(name, framework)
    matches: list[Path] = []
    for path in sorted(settings.model_dir.rglob(name)):
        if not path.is_file() or path.suffix.lower() not in _MODEL_SUFFIXES:
            continue
        if framework_token and path.suffix.lower().lstrip(".") != framework_token:
            continue
        matches.append(path)
    return matches


def _model_dir_alias(path_value: Any, settings: Settings) -> Path | None:
    token = _text(path_value)
    if not token:
        return None
    parts = Path(token).parts
    markers = [
        ("models",),
        ("agent_lab", "models", "openwakeword"),
    ]
    for marker in markers:
        for idx in range(0, max(0, len(parts) - len(marker) + 1)):
            if tuple(parts[idx : idx + len(marker)]) != marker:
                continue
            tail = parts[idx + len(marker) :]
            if not tail:
                continue
            candidate = settings.model_dir.joinpath(*tail)
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


def _copy_external_model_to_dir(path: Path, settings: Settings) -> Path:
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"openWakeWord model path is not accessible: {path}")
    if path.suffix.lower() not in _MODEL_SUFFIXES:
        raise ValueError("openWakeWord model path must end in .onnx or .tflite")
    with contextlib.suppress(Exception):
        path.relative_to(settings.model_dir)
        return path
    target = settings.model_dir / "custom" / (_slug(path.stem) or "custom") / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != path.stat().st_size:
        shutil.copy2(path, target)
    return target


def _normalize_model_path(source: str, settings: Settings) -> Path:
    framework = _framework_from_source(source, settings.framework)
    candidate = Path(source).expanduser()
    if candidate.exists() and candidate.is_file():
        return _copy_external_model_to_dir(candidate, settings)

    alias = _model_dir_alias(source, settings)
    if alias is not None:
        return alias

    matches = _local_model_matches(Path(source).name, settings, framework=framework)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        labels = ", ".join(str(path) for path in matches[:3])
        raise RuntimeError(f"multiple local openWakeWord models match {Path(source).name}: {labels}")
    raise RuntimeError(
        "openWakeWord model path is not accessible. "
        "Mount the model under TATER_OWW_MODEL_DIR or use a prebuilt model name/HTTP URL."
    )


def _pretrained_model_path(openwakeword_mod: Any, model_source: str, settings: Settings) -> Path:
    framework = settings.framework
    model_key = _slug(model_source)
    models = getattr(openwakeword_mod, "MODELS", {}) or {}
    if model_key not in models and model_source == "current_weather":
        model_key = "weather"
    entry = models.get(model_key)
    if isinstance(entry, dict):
        url = _resource_url_for_framework(_text(entry.get("download_url")), framework)
        source_path = Path(_resource_path_for_framework(_text(entry.get("model_path")), framework))
        if url and source_path.name:
            return _download_file(url, settings.model_dir / "pretrained" / source_path.name)
        value = entry.get(framework) or entry.get(framework.upper()) or entry.get("model")
    else:
        value = entry
    if not value:
        available = ", ".join(sorted(str(key) for key in models.keys())[:30])
        raise RuntimeError(f"unknown openWakeWord model '{model_source}'. Available examples: {available}")
    parsed = urlparse(_text(value))
    if parsed.scheme in {"http", "https"}:
        return _download_model(_text(value), settings.model_dir, framework)
    path = Path(str(value)).expanduser()
    if not path.exists():
        with contextlib.suppress(Exception):
            from openwakeword.utils import download_models

            logger.info("downloading bundled openWakeWord models")
            download_models()
    return path


def _feature_model_path(openwakeword_mod: Any, key: str, settings: Settings) -> Path:
    framework = settings.framework
    features = getattr(openwakeword_mod, "FEATURE_MODELS", {}) or {}
    entry = features.get(key)
    if isinstance(entry, dict):
        url = _resource_url_for_framework(_text(entry.get("download_url")), framework)
        source_path = Path(_resource_path_for_framework(_text(entry.get("model_path")), framework))
        if url and source_path.name:
            return _download_file(url, settings.model_dir / "features" / source_path.name)
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
    if _looks_like_model_path(source):
        return _normalize_model_path(source, settings)
    return _pretrained_model_path(openwakeword_mod, source, settings)


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
        melspec_path = _feature_model_path(openwakeword, "melspectrogram", settings)
        embedding_path = _feature_model_path(openwakeword, "embedding", settings)
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
        self._warm_detector: DetectorState | None = None
        self._lock = threading.Lock()
        self._last_cleanup_ts = 0.0
        self._cuda_fallback_until_ts = 0.0
        self._cuda_fallback_reason = ""
        self._last_error = ""

    def warmup(self) -> None:
        with self._lock:
            if self._warm_detector is not None:
                return
        detector = self._ensure_detector("__warmup__")
        with self._lock:
            self._warm_detector = detector

    def reset(self) -> None:
        with self._lock:
            self._detectors.clear()
            self._warm_detector = None
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
                "warm_detector_loaded": self._warm_detector is not None,
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
            if self._warm_detector is not None and not token.startswith("__"):
                detector = self._warm_detector
                for warm_key, warm_detector in list(self._detectors.items()):
                    if warm_detector is detector:
                        self._detectors.pop(warm_key, None)
                detector.selector = token
                detector.reset()
                detector.last_seen_ts = time.time()
                self._detectors[token] = detector
                self._warm_detector = None
                logger.info(
                    "assigned warm openWakeWord detector selector=%s model=%s framework=%s device=%s",
                    token,
                    detector.model_source,
                    detector.framework,
                    detector.device,
                )
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

        if width not in {1, 2, 3, 4}:
            return b""
        if width != 2:
            with contextlib.suppress(Exception):
                data = audioop.lin2lin(data, width, 2)
                width = 2
        if width != 2:
            return b""
        if channels <= 0:
            channels = 1
        if channels > 1:
            with contextlib.suppress(Exception):
                data = audioop.tomono(data, width, 0.5, 0.5)
                channels = 1
        if channels != 1:
            return b""
        if rate != 16000:
            with contextlib.suppress(Exception):
                data, detector.ratecv_state = audioop.ratecv(data, width, channels, rate, 16000, detector.ratecv_state)
                rate = 16000
        if rate != 16000:
            return b""
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
                    "threshold": self.settings.threshold,
                    "patience": self.settings.patience,
                    "hit_count": int(detector.counts.get(best_label, 0)),
                    "model_source": detector.model_source,
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
