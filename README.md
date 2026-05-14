# Pulse Point
 
Pulse Point is an early prototype for SSE (Synchronous Spatial Echo), a haptic-first assistive interface for blind and low-vision users.
 
The core idea: instead of adding more spoken directions, the phone translates nearby space into touch. You say what you're looking for, the app scans the room, finds the object, builds a rough spatial picture, and guides you toward it with haptics until you're close enough to reach it.
 
---
 
## Projects
 
### Web Prototype (`pulse-point`)
 
The public Vercel app deploys from `pulse-point` and runs in the browser with live camera object detection.
 
```bash
npm run build --prefix pulse-point
```
 
For local development:
 
```bash
cd pulse-point
npm install
npm run dev
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
- Visualize a 3x3 haptic matrix ring vocabulary for direction, proximity, and confirmation signals
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
| `OPENROUTER_API_KEY` | Server-side key used by `pulse-point/api/ai.js`. Don't prefix with `VITE_` or Vite will inline it into the client bundle and expose it publicly. |
 
Old prototype builds used `VITE_GEMINI_API_KEY` shipped to the browser. Remove that variable from production. The proxy at `/api/ai` is the supported path.
 
For local development, `vercel dev` is recommended and runs the serverless function alongside Vite. Alternatively, set `VITE_GEMINI_API_KEY` in `pulse-point/.env.local` and the client will fall back to direct OpenRouter calls only when the proxy returns 404. This is a dev-only convenience and should never be set in production.
