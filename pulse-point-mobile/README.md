# Pulse Point Mobile

Expo mobile app for the Pulse Point / SSE object-finding workflow.

## Run On Phone

```bash
npm install --cache .npm-cache
npm start
```

Then scan the QR code with Expo Go.

## What Works Now

- Camera permission and live camera view
- Object request input, for example `mouse`
- Scan-to-target guidance state machine
- Compass heading from device sensors
- Phone haptic feedback for scan, target lock, orientation, walking, reach, and complete states
- 3x3 haptic ring visualization
- Mini spatial map with user position, obstacles, route, and target

## Native Sensor Roadmap

Expo Go cannot expose full LiDAR mesh capture or high-performance frame-by-frame object recognition. The next native step is:

- iOS: ARKit scene reconstruction + CoreML/Vision object recognition
- Android: ARCore depth + ML Kit or TensorFlow Lite object recognition
- Shared: replace `src/services/spatialEngine.js` with native detection, room-mesh, and route services
