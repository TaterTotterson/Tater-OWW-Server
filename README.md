# Tater OWW Server

Standalone HTTP openWakeWord server for Tater satellite firmware.

This service implements the same endpoint that the Tater firmware expects:

```text
POST /api/openwakeword/detect
```

Satellites post raw PCM chunks to this endpoint. When a wake word is detected,
the server returns JSON that tells the device to start its normal ESPHome voice
assistant flow.

## Quick Start

CPU:

```bash
docker compose up --build
```

NVIDIA GPU:

```bash
docker compose -f docker-compose.nvidia.yml up --build
```

Then set the satellite firmware value:

```text
openwakeword_server_url: http://YOUR_SERVER_IP:8502
```

The firmware automatically posts to `/api/openwakeword/detect`.

## Firmware Contract

Request:

```text
POST /api/openwakeword/detect
Content-Type: application/octet-stream
X-Audio-Format: pcm_s16le
X-Audio-Rate: 16000
X-Audio-Bits: 16
X-Audio-Channels: 1
X-Source-Device: kitchen-satellite
X-Wake-Word: hey_tater
```

Response when no wake word is detected:

```json
{"ok": true, "detected": false}
```

Response when detected:

```json
{
  "ok": true,
  "detected": true,
  "wake_word": "hey_tater",
  "score": 0.72,
  "engine": "openwakeword",
  "model_source": "hey_jarvis"
}
```

## Configuration

Environment variables:

| Name | Default | Description |
| --- | --- | --- |
| `TATER_OWW_HOST` | `0.0.0.0` | HTTP bind host. |
| `TATER_OWW_PORT` | `8502` | HTTP bind port. |
| `TATER_OWW_MODEL` | `hey_jarvis` | Pretrained model name, local path, or HTTP(S) model URL. |
| `TATER_OWW_FRAMEWORK` | `onnx` | `onnx` or `tflite`. |
| `TATER_OWW_DEVICE` | `auto` | `auto`, `cpu`, or `gpu`. |
| `TATER_OWW_THRESHOLD` | `0.50` | Detection threshold. |
| `TATER_OWW_PATIENCE` | `2` | Consecutive threshold hits required. |
| `TATER_OWW_DEBOUNCE_S` | `2.0` | Minimum seconds between detections per satellite. |
| `TATER_OWW_VAD_THRESHOLD` | `0.0` | Optional openWakeWord internal VAD threshold. |
| `TATER_OWW_MODEL_DIR` | `/models` | Directory for downloaded/custom model files. |
| `TATER_OWW_IDLE_TTL_S` | `3600` | Seconds before unused device detectors are unloaded. |
| `TATER_OWW_MAX_CHUNK_BYTES` | `524288` | Max request body size. |
| `TATER_OWW_PREFER_HINT` | `true` | Return `X-Wake-Word` as the response wake word when provided. |
| `TATER_OWW_WARMUP` | `true` | Load one detector at startup. |
| `TATER_OWW_LOG_LEVEL` | `info` | Uvicorn/app log level. |

## Endpoints

```text
GET  /healthz
GET  /api/openwakeword/status
POST /api/openwakeword/detect
POST /api/openwakeword/reset
```

`reset` clears detector state and is useful after changing model env vars and
restarting the container.

## Notes

This is an HTTP adapter, not a Wyoming server. It is meant for the Tater
satellite firmware `remote_wake_word` component. The firmware can still fall
back to microWakeWord if this server becomes unavailable.
