# Pulse Point

**Live demo:** https://pulse-point-steel.vercel.app

[![Tests](https://github.com/ketchup235/Pulse-Point/actions/workflows/test.yml/badge.svg)](https://github.com/ketchup235/Pulse-Point/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Pulse Point is a haptic-first assistive interface for blind and low-vision users. You say what you're looking for, the app finds it with real-time object detection, and guides you toward it through distinct vibration patterns — no spoken directions that compete with hearing.

---

## How It Works

The web app runs YOLO11n object detection entirely in-browser using TensorFlow.js (no server round-trip for inference). Voice input names a target; the COCO label resolver maps natural language ("phone", "TV") to the model's vocabulary. A bounding-box tracker with EMA-smoothed velocity keeps the lock stable across frames. A pinhole-camera distance model converts bbox size to meters. The guidance engine divides the frame into a center sweet spot and emits one of seven directional haptic signals until the target is centered and within reach. A Vercel serverless proxy handles out-of-vocabulary objects by querying a vision-language model (Gemini via OpenRouter), keeping the API key off the client.

```
Voice input ──► COCO resolver ──► YOLO loop ──► Box tracker
                                                     │
                                              Distance model
                                                     │
                                           Guidance compute
                                            ┌──────────────┐
                                         Haptics        Speech
```

---

## Built With

![React](https://img.shields.io/badge/React-19-61dafb?logo=react&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-7-646cff?logo=vite&logoColor=white)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.22-ff6f00?logo=tensorflow&logoColor=white)
![Expo](https://img.shields.io/badge/Expo-53-000020?logo=expo&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-serverless-000000?logo=vercel&logoColor=white)
![YOLO11n](https://img.shields.io/badge/YOLO11n-COCO-00bfff)

---

## Projects

### Web Prototype (`pulse-point`)

The public Vercel app deploys from `pulse-point` and runs in the browser with live camera object detection.

```bash
cd pulse-point
npm install
npm run dev        # development server
npm test           # run unit tests
npm run build      # production build
```

### Mobile App (`pulse-point-mobile`)

The real mobile app lives in `pulse-point-mobile`.

```bash
cd pulse-point-mobile
npm install --cache .npm-cache
npm start
```

Scan the Expo QR code with Expo Go on your phone. The app uses the phone camera, haptics, and motion sensors for the object-finding flow.

---

## Prototype Goals

- Keep guidance tactile so it doesn't compete with hearing
- Run browser-based object detection for requests like "find my mouse"
- Show what LiDAR/camera spatial awareness could look like through a 3D room-map concept
- Walk through the full flow: request, scan, target lock, orientation, walking, and close-range handoff
- Visualize a 3×3 haptic matrix ring vocabulary for direction, proximity, and confirmation signals
- Provide a clean foundation for a TSA concept demo, pitch deck, or future hardware prototype

---

## Current Status

### Web

The Vercel app works in the browser. It requests camera permission, loads an object detection model, draws boxes around detected objects, locks onto the requested target, estimates direction and distance from the camera frame, and triggers phone vibration where supported.

iPhone browsers don't expose reliable vibration APIs and websites can't access iPhone LiDAR room meshes directly, so true haptic guidance and LiDAR mapping belong in the native app.

### Mobile

The Expo app opens the camera, reads compass heading, triggers haptics, and runs a target-finding state machine with spatial guidance on screen. Object recognition is simulated for now so the flow works on any phone through Expo Go.

Real LiDAR mesh capture and live object recognition require a native build with iOS ARKit/CoreML or Android ARCore/ML Kit. The app is structured so those pieces can be swapped in later.

---

## Deploying to Vercel

Import this repo into Vercel. The included `vercel.json` builds the `pulse-point` site and publishes `pulse-point/dist`.

Keep the repository root as the root directory. Vercel will run:

```bash
cd pulse-point && npm install
cd pulse-point && npm run build
```

---

## Environment Variables

Set these in Vercel Project Settings → Environment Variables.

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | Server-side key used by `api/ai.js`. **Do not** prefix with `VITE_` — Vite inlines those into the client bundle and exposes them publicly. |
| `ALLOWED_ORIGIN` | Optional. Restricts CORS to a specific origin (defaults to `https://pulse-point-steel.vercel.app`). |

Old prototype builds used `VITE_GEMINI_API_KEY` shipped to the browser. Remove that variable from production. The proxy at `/api/ai` is the supported path.

For local development, `vercel dev` is recommended and runs the serverless function alongside Vite. Alternatively, set `VITE_GEMINI_API_KEY` in `pulse-point/.env.local` — the client falls back to direct OpenRouter calls only when the proxy returns 404. This is a dev-only convenience and must never be set in production.

---

## Acknowledgments

Pulse Point uses [Ultralytics YOLO11n](https://github.com/ultralytics/ultralytics) pretrained weights (Apache 2.0 license) for 80-class COCO object detection. The model was converted from `.pt` → ONNX → TensorFlow.js GraphModel format by our team using the included `convert_tfjs.py` script. All Pulse Point application code is original work by our team. Licensed under the MIT License — see [LICENSE](LICENSE).
