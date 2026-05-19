from __future__ import annotations

import asyncio
import contextlib
import logging
import time
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
STREAM_QUEUE_MAX = 12
DETECT_LOG_EVERY = 120
DETECT_SLOW_LOG_S = 1.0


def _query_int(websocket: WebSocket, name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = str(websocket.query_params.get(name) or "").strip()
    try:
        parsed = int(raw) if raw else int(default)
    except Exception:
        parsed = int(default)
    return max(int(minimum), min(int(maximum), parsed))


def _detector_selector(selector: str, client_host: str) -> str:
    selector_token = str(selector or "").strip() or "remote"
    client_token = str(client_host or "").strip()
    if client_token and client_token != selector_token:
        return f"{selector_token}@{client_token}"
    return selector_token


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
    detector_selector = _detector_selector(selector, client_host)
    wake_word_hint = str(websocket.query_params.get("wake_word") or "").strip()
    audio_bits = _query_int(websocket, "bits", 16, minimum=8, maximum=32)
    audio_format = {
        "rate": _query_int(websocket, "rate", 16000, minimum=8000, maximum=48000),
        "width": max(1, audio_bits // 8),
        "channels": _query_int(websocket, "channels", 1, minimum=1, maximum=2),
    }
    await websocket.accept()
    frame_count = 0
    processed_count = 0
    dropped_count = 0
    stream_queue_max = max(1, min(120, int(settings.stream_queue_max or STREAM_QUEUE_MAX)))
    audio_queue: asyncio.Queue[tuple[float, bytes]] = asyncio.Queue(maxsize=stream_queue_max)
    receiver_done = asyncio.Event()
    receiver_task: asyncio.Task[None] | None = None
    logger.info(
        "openWakeWord stream started selector=%s detector=%s client=%s queue_max=%s drop_queued_frames=%s",
        selector,
        detector_selector,
        client_host or "-",
        stream_queue_max,
        settings.drop_queued_frames,
    )

    def drop_queued_frame(*, count_drop: bool = True) -> None:
        nonlocal dropped_count
        try:
            audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        if count_drop:
            dropped_count += 1

    async def receive_audio_frames() -> None:
        nonlocal frame_count
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
                    continue
                if len(audio_bytes) > settings.max_chunk_bytes:
                    await websocket.send_json({"ok": False, "error": "openWakeWord audio chunk is too large"})
                    await websocket.close(code=1009)
                    break
                if settings.drop_queued_frames:
                    while audio_queue.full():
                        drop_queued_frame()
                    audio_queue.put_nowait((time.time(), bytes(audio_bytes)))
                else:
                    await audio_queue.put((time.time(), bytes(audio_bytes)))
        except WebSocketDisconnect:
            pass
        finally:
            receiver_done.set()

    try:
        receiver_task = asyncio.create_task(receive_audio_frames())
        while True:
            if receiver_done.is_set() and audio_queue.empty():
                break
            try:
                received_ts, audio_bytes = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            started_ts = time.time()
            audio_bytes_len = len(audio_bytes or b"")
            processed_count += 1
            queue_delay_ms = max(0.0, (started_ts - received_ts) * 1000.0)

            try:
                result = await asyncio.to_thread(
                    engine.process_audio,
                    selector=detector_selector,
                    audio_bytes=bytes(audio_bytes),
                    audio_format=audio_format,
                    wake_word_hint=wake_word_hint,
                )
            except Exception as exc:
                detail = str(exc) or "openWakeWord detection is unavailable"
                logger.warning("openWakeWord stream detection failed selector=%s error=%s", detector_selector, detail)
                await websocket.send_json({"ok": False, "error": detail})
                continue

            if not bool(result.get("detected")):
                elapsed_s = time.time() - started_ts
                try:
                    score = float(result.get("score") or 0.0)
                except Exception:
                    score = 0.0
                try:
                    threshold = float(result.get("threshold") or settings.threshold)
                except Exception:
                    threshold = float(settings.threshold)
                try:
                    hit_count = int(result.get("hit_count") or 0)
                except Exception:
                    hit_count = 0
                try:
                    patience = int(result.get("patience") or settings.patience)
                except Exception:
                    patience = int(settings.patience)
                force_log = bool(settings.diagnostic_logging) and (
                    (threshold > 0.0 and score >= max(0.2, threshold - 0.25)) or hit_count > 0
                )
                should_log = (
                    bool(settings.diagnostic_logging) and (
                        force_log
                        or processed_count == 1
                        or processed_count % DETECT_LOG_EVERY == 0
                    )
                    or elapsed_s >= DETECT_SLOW_LOG_S
                )
                if should_log:
                    if settings.diagnostic_logging:
                        logger.info(
                            (
                                "openWakeWord detect selector=%s detected=False elapsed_ms=%.1f "
                                "bytes=%s count=%s dropped=%s queue_ms=%.1f best_label=%s "
                                "score=%.3f hits=%s/%s threshold=%.3f model=%s"
                            ),
                            detector_selector,
                            elapsed_s * 1000.0,
                            audio_bytes_len,
                            processed_count,
                            dropped_count,
                            queue_delay_ms,
                            str(result.get("best_label") or "-"),
                            score,
                            hit_count,
                            patience,
                            threshold,
                            str(result.get("model_source") or settings.model_source or "-"),
                        )
                    else:
                        logger.info(
                            "openWakeWord detect selector=%s detected=False elapsed_ms=%.1f bytes=%s count=%s dropped=%s queue_ms=%.1f",
                            detector_selector,
                            elapsed_s * 1000.0,
                            audio_bytes_len,
                            processed_count,
                            dropped_count,
                            queue_delay_ms,
                        )
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
            while not audio_queue.empty():
                drop_queued_frame(count_drop=False)
    except WebSocketDisconnect:
        pass
    finally:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            if receiver_task is not None:
                receiver_task.cancel()
                await receiver_task
        logger.info(
            "openWakeWord stream stopped selector=%s frames=%s processed=%s dropped=%s",
            detector_selector,
            frame_count,
            processed_count,
            dropped_count,
        )


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
