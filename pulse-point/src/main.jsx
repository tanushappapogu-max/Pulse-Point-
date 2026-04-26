import React, { useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Bluetooth,
  Camera,
  CheckCircle2,
  Compass,
  Cuboid,
  Hand,
  Map,
  Mic,
  Mouse,
  Radar,
  Route,
  ScanLine,
  Smartphone,
  Sparkles,
  Vibrate,
  Waves
} from 'lucide-react';
import './styles.css';

const stages = [
  {
    id: 'request',
    label: 'Ask',
    title: 'Say what you need',
    instruction: 'Listening for: “I need to find my mouse.”',
    guidance: 'Pulse Point turns the request into a target object class and starts a room scan.',
    haptic: 'single confirmation pulse',
    progress: 18,
    distance: 'unknown',
    bearing: 'calibrating',
    pattern: 'confirm'
  },
  {
    id: 'scan',
    label: 'Scan',
    title: 'Build a spatial map',
    instruction: 'Slowly turn your phone across the room.',
    guidance: 'LiDAR depth, rear camera, front camera, motion sensors, and object recognition create a live 3D room model.',
    haptic: 'wide sweeping pulse',
    progress: 44,
    distance: '3.2 m',
    bearing: 'object candidate at 42° right',
    pattern: 'sweep'
  },
  {
    id: 'found',
    label: 'Found',
    title: 'Mouse recognized',
    instruction: 'Target locked near the desk edge.',
    guidance: 'The app alerts the user, estimates the safest route, and computes the first body orientation correction.',
    haptic: 'three sharp lock pulses',
    progress: 68,
    distance: '2.4 m',
    bearing: 'turn right 38°',
    pattern: 'lock'
  },
  {
    id: 'guide',
    label: 'Guide',
    title: 'Orient and walk',
    instruction: 'Turn right until the ring centers, then move forward.',
    guidance: 'Directional haptics guide orientation first, then walking direction, avoiding detected chairs and table legs.',
    haptic: 'right-side correction pulses',
    progress: 86,
    distance: '0.8 m',
    bearing: 'forward, slight left',
    pattern: 'right'
  },
  {
    id: 'handoff',
    label: 'Reach',
    title: 'Close enough',
    instruction: 'Stop walking. Reach forward and slightly down.',
    guidance: 'Once the user is close, the app switches from navigation to hand-level precision and then ends assistance.',
    haptic: 'center dot steady pulse',
    progress: 100,
    distance: '0.18 m',
    bearing: 'hand guidance active',
    pattern: 'center'
  }
];

const sensors = [
  { icon: Camera, label: 'Rear camera', text: 'Object recognition and scene understanding' },
  { icon: Cuboid, label: 'LiDAR depth', text: '3D room mesh, surfaces, distance, and occlusion' },
  { icon: Smartphone, label: 'Motion sensors', text: 'Phone pose, turns, steps, and orientation drift' },
  { icon: Radar, label: 'Front camera', text: 'User posture and near-body safety checks' },
  { icon: Vibrate, label: 'Haptics', text: 'Phone vibration plus optional ring patterns' },
  { icon: Bluetooth, label: 'Haptic ring', text: 'Nine-point direction, proximity, and urgency signals' }
];

const roomObjects = [
  { name: 'desk', x: 67, y: 25, w: 24, h: 17, type: 'surface' },
  { name: 'chair', x: 41, y: 55, w: 16, h: 16, type: 'obstacle' },
  { name: 'table', x: 15, y: 29, w: 25, h: 22, type: 'surface' },
  { name: 'mouse', x: 75, y: 31, w: 8, h: 5, type: 'target' }
];

function App() {
  const [stageIndex, setStageIndex] = useState(2);
  const [query, setQuery] = useState('my mouse');
  const stage = stages[stageIndex];
  const dots = useMemo(() => buildRingPattern(stage.pattern), [stage.pattern]);

  return (
    <main className="app-shell">
      <section className="phone-frame" aria-label="Pulse Point mobile app prototype">
        <header className="mobile-topbar">
          <div className="brand">
            <span className="brand-mark">
              <Waves size={21} />
            </span>
            <div>
              <strong>Pulse Point</strong>
              <small>SSE mobile guidance</small>
            </div>
          </div>
          <span className="live-chip">Live scan</span>
        </header>

        <section className="request-card">
          <label htmlFor="target-search">Find object</label>
          <div className="search-row">
            <Mic size={19} />
            <input
              id="target-search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              aria-label="Object to find"
            />
            <button type="button" onClick={() => setStageIndex(1)}>
              <ScanLine size={18} />
            </button>
          </div>
        </section>

        <section className="stage-card" aria-live="polite">
          <div className="stage-header">
            <span>{stage.label}</span>
            <strong>{stage.distance}</strong>
          </div>
          <h1>{stage.title}</h1>
          <p>{stage.instruction}</p>
          <div className="progress-track">
            <span style={{ width: `${stage.progress}%` }} />
          </div>
        </section>

        <section className="room-card" aria-label="3D room map concept">
          <div className="map-toolbar">
            <div>
              <Map size={18} />
              <span>3D room map</span>
            </div>
            <strong>{query || 'target'}</strong>
          </div>

          <div className={`room-map stage-${stage.id}`}>
            <div className="scan-cone" />
            <div className="route-line" />
            <div className="user-position">
              <Compass size={20} />
            </div>
            {roomObjects.map((object) => (
              <span
                key={object.name}
                className={`room-object ${object.type}`}
                style={{
                  left: `${object.x}%`,
                  top: `${object.y}%`,
                  width: `${object.w}%`,
                  height: `${object.h}%`
                }}
              >
                {object.type === 'target' ? <Mouse size={18} /> : object.name}
              </span>
            ))}
            <div className="depth-grid" />
          </div>
        </section>

        <section className="guidance-grid">
          <article>
            <Compass size={19} />
            <span>Orientation</span>
            <strong>{stage.bearing}</strong>
          </article>
          <article>
            <Vibrate size={19} />
            <span>Haptic cue</span>
            <strong>{stage.haptic}</strong>
          </article>
        </section>

        <section className="ring-card">
          <div className="ring-copy">
            <Sparkles size={18} />
            <span>Haptic ring output</span>
          </div>
          <div className="ring-grid">
            {dots.map((dot, index) => (
              <span
                key={index}
                className="ring-dot"
                style={{
                  '--scale': dot.scale,
                  '--alpha': dot.alpha,
                  '--delay': `${dot.delay}ms`
                }}
              />
            ))}
          </div>
        </section>

        <nav className="stepper" aria-label="Prototype flow stages">
          {stages.map((item, index) => (
            <button
              key={item.id}
              type="button"
              className={index === stageIndex ? 'active' : ''}
              onClick={() => setStageIndex(index)}
            >
              <span>{index + 1}</span>
              {item.label}
            </button>
          ))}
        </nav>
      </section>

      <aside className="concept-panel">
        <p className="eyebrow">Object detection + spatial awareness + haptic guidance</p>
        <h2>Find “{query || 'object'}” without relying on speech.</h2>
        <p>
          The mobile app scans the room, recognizes the target, builds a 3D map, then guides the
          user through orientation, walking, and close-range reaching. Speech can start the task,
          but the actual guidance is haptic so hearing stays free for the environment.
        </p>

        <div className="sensor-list">
          {sensors.map((sensor) => {
            const Icon = sensor.icon;
            return (
              <article key={sensor.label}>
                <Icon size={20} />
                <div>
                  <strong>{sensor.label}</strong>
                  <span>{sensor.text}</span>
                </div>
              </article>
            );
          })}
        </div>

        <div className="handoff-note">
          <CheckCircle2 size={20} />
          <span>
            Assistance ends when the app estimates the user is close enough to safely reach the
            object without navigation support.
          </span>
        </div>
      </aside>
    </main>
  );
}

function buildRingPattern(pattern) {
  const patterns = {
    confirm: [0.35, 0.35, 0.35, 0.35, 1, 0.35, 0.35, 0.35, 0.35],
    sweep: [0.85, 0.55, 0.25, 0.9, 0.6, 0.3, 0.85, 0.55, 0.25],
    lock: [0.75, 0.75, 0.75, 0.75, 1.1, 0.75, 0.75, 0.75, 0.75],
    right: [0.25, 0.45, 0.95, 0.25, 0.5, 1.05, 0.25, 0.45, 0.95],
    center: [0.35, 0.55, 0.35, 0.55, 1.12, 0.55, 0.35, 0.55, 0.35]
  };

  return (patterns[pattern] ?? patterns.center).map((scale, index) => ({
    scale,
    alpha: Math.min(1, 0.22 + scale * 0.72),
    delay: index * 48
  }));
}

createRoot(document.getElementById('root')).render(<App />);
