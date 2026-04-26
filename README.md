# Pulse Point

Pulse Point is an early mobile-app prototype for SSE, Synchronous Spatial Echo: a haptic-first assistive spatial interface for blind and low-vision users.

Instead of adding more spoken directions, Pulse Point explores how a phone and optional haptic ring can translate nearby space into touch. The core flow is simple: the user says what they need to find, the app scans the room with camera/LiDAR-style sensing, recognizes the object, builds a 3D spatial map, orients the user, and guides them with haptics until they are close enough to reach safely.

## Project

The starter app lives in [`pulse-point`](./pulse-point).

```bash
cd pulse-point
npm install
npm run dev
```

## Prototype Goals

- Preserve hearing by keeping guidance tactile instead of audio-heavy.
- Demonstrate object detection and recognition for requests like “find my mouse.”
- Show LiDAR/camera spatial awareness through a 3D room-map concept.
- Guide the user through request, scan, target lock, orientation, walking, and close-range handoff.
- Visualize a 3x3 haptic matrix ring vocabulary for direction, proximity, and confirmation signals.
- Provide a clean foundation for a TSA concept demo, pitch deck, or future hardware/software prototype.
