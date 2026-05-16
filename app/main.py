from __future__ import annotations

import asyncio
import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from . import __version__
from .config import Settings, load_settings
from .engine import OpenWakeWordEngine


settings: Settings = load_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("tater_oww")
engine = OpenWakeWordEngine(settings)
app = FastAPI(title="Tater OWW Server", version=__version__)


def _header_int(request: Request, header_name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = str(request.headers.get(header_name) or "").strip()
    try:
        parsed = int(raw) if raw else int(default)
    except Exception:
        parsed = int(default)
    return max(int(minimum), min(int(maximum), parsed))


@app.on_event("startup")
async def startup() -> None:
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    if not settings.warmup:
        return
    try:
        await asyncio.to_thread(engine.warmup)
    except Exception as exc:
        logger.warning("openWakeWord warmup failed; first detection will retry: %s", exc)


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "ok": True,
        "name": "tater-oww-server",
        "version": __version__,
        "detect_endpoint": "/api/openwakeword/detect",
    }


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    status = engine.status()
    return {
        "ok": True,
        "version": __version__,
        "available": bool(status.get("available")),
        "detector_count": int(status.get("detector_count") or 0),
    }


@app.get("/api/openwakeword/status")
async def openwakeword_status() -> dict[str, Any]:
    return engine.status()


@app.post("/api/openwakeword/reset")
async def openwakeword_reset() -> dict[str, Any]:
    engine.reset()
    return {"ok": True}


@app.post("/api/openwakeword/detect")
async def detect_openwakeword(request: Request) -> dict[str, Any]:
    audio_bytes = await request.body()
    if not audio_bytes:
        return {"ok": True, "detected": False}
    if len(audio_bytes) > settings.max_chunk_bytes:
        raise HTTPException(status_code=413, detail="openWakeWord audio chunk is too large")

    audio_bits = _header_int(request, "X-Audio-Bits", 16, minimum=8, maximum=32)
    audio_format = {
        "rate": _header_int(request, "X-Audio-Rate", 16000, minimum=8000, maximum=48000),
        "width": max(1, audio_bits // 8),
        "channels": _header_int(request, "X-Audio-Channels", 1, minimum=1, maximum=2),
    }
    client_host = getattr(request.client, "host", "") if request.client is not None else ""
    selector = str(
        request.headers.get("X-Source-Device")
        or request.query_params.get("selector")
        or client_host
        or "remote"
    ).strip()
    wake_word_hint = str(request.headers.get("X-Wake-Word") or "").strip()

    try:
        result = await asyncio.to_thread(
            engine.process_audio,
            selector=selector,
            audio_bytes=audio_bytes,
            audio_format=audio_format,
            wake_word_hint=wake_word_hint,
        )
    except Exception as exc:
        logger.warning("openWakeWord detection failed selector=%s error=%s", selector, exc)
        raise HTTPException(status_code=503, detail=str(exc) or "openWakeWord detection is unavailable") from exc

    if not bool(result.get("detected")):
        return {"ok": True, "detected": False}
    return {
        "ok": True,
        "detected": True,
        "wake_word": str(result.get("wake_word") or wake_word_hint or "openwakeword"),
        "score": float(result.get("score") or 0.0),
        "engine": "openwakeword",
        "model_source": str(result.get("model_source") or settings.model_source),
        "model_label": str(result.get("model_label") or ""),
    }


def main() -> None:
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
