# Pulse Point Architecture

This document summarizes the core end-to-end flow for both the web and mobile prototypes.

## High-Level Data Flow

```text
User voice/text target
        |
        v
Target extraction (voice.js)
        |
        v
COCO label resolution (coco.js)
        |
        v
Web: YOLO11n inference loop (yolo.js)
Mobile: simulated detection stage (spatialEngine.js)
        |
        v
Tracker + smoothing (tracker.js)
        |
        v
Distance + direction computation
(distance.js + guidance/compute.js)
        |
        v
Guidance outputs
  - Haptics pattern (guidance/haptics.js)
  - Speech prompts (guidance/speech.js, lib/voice.js)
```

## Web App (`pulse-point/`)

- React + Vite frontend handles camera access and render loop.
- YOLO11n TensorFlow.js model runs locally in browser for real-time detection.
- `tracker.js` stabilizes noisy frame-to-frame detections.
- `compute.js` determines directional guidance (`left`, `right`, `up`, `down`, `locked`, `closer`, `reach`).
- Service worker caches model shards in `public/yolo11n_web_model/` for faster/offline repeat loads.

## Mobile App (`pulse-point-mobile/`)

- Expo app provides the user flow, haptics, and orientation guidance UX.
- Current detection layer is simulated (`createDetectionResult`) for demo reliability in Expo Go.
- App structure keeps a clear seam for future native detection integration (ML Kit/CoreML/ARKit/ARCore).

## AI Proxy (`api/`)

- `/api/ai` is a server-side proxy to OpenRouter/Gemini.
- API key remains server-only (`OPENROUTER_API_KEY`), never exposed in browser bundle.
- Includes model allow-list, max token caps, strict origin check, and per-IP in-memory rate limiting.

## Error Reporting

- `ErrorBoundary` catches frontend render crashes and posts a short payload to `/api/error`.
- `/api/error` writes structured crash data to Vercel function logs for live-demo debugging.
