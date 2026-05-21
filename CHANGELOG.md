# Changelog

All notable changes to Pulse Point are documented here.

## [0.2.0] - 2026-05-20

### Added
- Unit test suite (vitest) covering compute, distance, coco, tracker, voice, and settings modules
- Service worker for offline YOLO model caching (`/yolo11n_web_model/` cached on first load)
- Server-side per-IP rate limiting in `/api/ai` (10 req/min sliding window)
- Strict same-origin CORS enforcement in `/api/ai`
- Client-side error reporter (`/api/error`) wired into ErrorBoundary for live-demo debuggability
- GitHub Actions CI workflow — runs `npm test` on every push and pull request
- MIT license
- `CONTRIBUTING.md` — branching and commit conventions
- `ARCHITECTURE.md` — full data-flow diagram with ASCII art

### Changed
- Restructured README: architectural overview and Built With section moved to top; Acknowledgments added; Environment Variables moved to end
- Service worker registration added to `main.jsx`

## [0.1.0] - 2026-05-01

### Added
- Web prototype: React 19 + Vite + TensorFlow.js, real-time YOLO11n inference in the browser
- Mobile prototype: React Native + Expo, haptic guidance, compass heading, simulated detection
- Vercel serverless AI proxy (`/api/ai`) with model allow-list and token cap
- 12-pattern haptic vocabulary (`haptics.js`)
- Adaptive frame-rate control (4–15 Hz based on measured inference latency)
- Box tracker with EMA-smoothed velocity + IoU snap (`tracker.js`)
- Distance estimation via pinhole camera model (`distance.js`)
- 2D guidance computation with dominant-axis selection (`compute.js`)
- Voice target extraction via Web Speech API (`voice.js`)
- Screen-reader Announcer component and ErrorBoundary
- Settings sheet with independent haptic/speech toggles and localStorage persistence
