# Pulse Point

Pulse Point is an early mobile-app prototype for SSE, Synchronous Spatial Echo: a haptic-first assistive spatial interface for blind and low-vision users.

Instead of adding more spoken directions, Pulse Point explores how a phone and optional haptic ring can translate nearby space into touch. The core flow is simple: the user says what they need to find, the app scans the room with camera/LiDAR-style sensing, recognizes the object, builds a 3D spatial map, orients the user, and guides them with haptics until they are close enough to reach safely.

## Projects

The public Vercel app deploys from [`pulse-point`](./pulse-point). It runs in the browser with live camera object detection.

```bash
npm run build --prefix pulse-point
```

The real mobile app lives in [`pulse-point-mobile`](./pulse-point-mobile).

```bash
cd pulse-point-mobile
npm install --cache .npm-cache
npm start
```

Scan the Expo QR code with Expo Go on your phone. The app uses the phone camera, haptics, and motion sensors for the object-finding guidance flow.

The visual web prototype lives in [`pulse-point`](./pulse-point).

```bash
cd pulse-point
npm install
npm run dev
```

## Prototype Goals

- Preserve hearing by keeping guidance tactile instead of audio-heavy.
- Run browser camera object detection for requests like “find my mouse.”
- Show LiDAR/camera spatial awareness through a 3D room-map concept.
- Guide the user through request, scan, target lock, orientation, walking, and close-range handoff.
- Visualize a 3x3 haptic matrix ring vocabulary for direction, proximity, and confirmation signals.
- Provide a clean foundation for a TSA concept demo, pitch deck, or future hardware/software prototype.

## Current Web Status

The Vercel app is functional in the browser: it requests camera permission, loads an object detection model, draws boxes around detected objects, locks onto the requested target class, estimates direction/distance from the camera frame, and triggers phone vibration when supported.

Because websites cannot access iPhone LiDAR room meshes directly, full LiDAR mapping still belongs in the native mobile app path.

## Current Mobile Status

The mobile app is functional as an Expo app: it opens the camera, requests camera permission, reads compass heading, triggers haptics, tracks a target-finding state machine, and displays spatial guidance. The object-recognition result is currently simulated so the flow can run on any phone through Expo Go.

True LiDAR mesh capture and live object recognition require a native build path with iOS ARKit/CoreML or Android ARCore/ML Kit. The mobile app is structured so that detection and spatial mapping can be replaced with native services next.

## Deploy To Vercel

Import this GitHub repo into Vercel. The included `vercel.json` builds the `pulse-point` website and publishes `pulse-point/dist`.

For the Vercel project settings, keep the repository root as the root directory. Vercel will run:

```bash
cd pulse-point && npm install
cd pulse-point && npm run build
```
