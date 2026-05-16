from __future__ import annotations

import asyncio
import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

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


def _query_int(websocket: WebSocket, name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = str(websocket.query_params.get(name) or "").strip()
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
        "stream_endpoint": "/api/openwakeword/stream",
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


@app.websocket("/api/openwakeword/stream")
async def stream_openwakeword(websocket: WebSocket) -> None:
    client_host = getattr(websocket.client, "host", "") if websocket.client is not None else ""
    selector = str(
        websocket.query_params.get("selector")
        or websocket.query_params.get("source_device")
        or client_host
        or "remote"
    ).strip()
    wake_word_hint = str(websocket.query_params.get("wake_word") or "").strip()
    audio_bits = _query_int(websocket, "bits", 16, minimum=8, maximum=32)
    audio_format = {
        "rate": _query_int(websocket, "rate", 16000, minimum=8000, maximum=48000),
        "width": max(1, audio_bits // 8),
        "channels": _query_int(websocket, "channels", 1, minimum=1, maximum=2),
    }
    await websocket.accept()
    frame_count = 0
    logger.info("openWakeWord stream started selector=%s client=%s", selector, client_host or "-")
    try:
        while True:
            message = await websocket.receive()
            if str(message.get("type") or "") == "websocket.disconnect":
                break
            audio_bytes = message.get("bytes")
            if audio_bytes is None:
                continue
            frame_count += 1
            if not audio_bytes:
                await websocket.send_json({"ok": True, "detected": False})
                continue
            if len(audio_bytes) > settings.max_chunk_bytes:
                await websocket.send_json({"ok": False, "error": "openWakeWord audio chunk is too large"})
                await websocket.close(code=1009)
                return

            try:
                result = await asyncio.to_thread(
                    engine.process_audio,
                    selector=selector,
                    audio_bytes=bytes(audio_bytes),
                    audio_format=audio_format,
                    wake_word_hint=wake_word_hint,
                )
            except Exception as exc:
                detail = str(exc) or "openWakeWord detection is unavailable"
                logger.warning("openWakeWord stream detection failed selector=%s error=%s", selector, detail)
                await websocket.send_json({"ok": False, "error": detail})
                continue

            if not bool(result.get("detected")):
                await websocket.send_json({"ok": True, "detected": False})
                continue
            await websocket.send_json(
                {
                    "ok": True,
                    "detected": True,
                    "wake_word": str(result.get("wake_word") or wake_word_hint or "openwakeword"),
                    "score": float(result.get("score") or 0.0),
                    "engine": "openwakeword",
                    "model_source": str(result.get("model_source") or settings.model_source),
                    "model_label": str(result.get("model_label") or ""),
                }
            )
    except WebSocketDisconnect:
        pass
    finally:
        logger.info("openWakeWord stream stopped selector=%s frames=%s", selector, frame_count)


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
