import React, { useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity,
  BellRing,
  Droplets,
  Eye,
  Hand,
  Move3D,
  Radar,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  VolumeX,
  Waves
} from 'lucide-react';
import './styles.css';

const modes = [
  {
    id: 'pinpoint',
    name: 'Pinpoint Search',
    icon: ScanSearch,
    signal: 'center-pulse',
    metric: '4 sec guided reach',
    headline: 'Guide the hand to small objects without spoken directions.',
    detail:
      'The phone builds a spatial reference, then pulses faster and stronger as the hand aligns with a pill, tool, key, or measuring spoon.',
    haptic: 'Center dot confirms alignment. Side dots indicate left, right, closer, or farther corrections.',
    accent: '#13a89e'
  },
  {
    id: 'social',
    name: 'Social Gaze Map',
    icon: Eye,
    signal: 'left-rise',
    metric: 'Peripheral social cue',
    headline: 'Turn group attention shifts into subtle directional touch.',
    detail:
      'Instead of narrating who looked where, Pulse Point gives a felt cue when attention moves across the room.',
    haptic: 'A rising stroke on the left side means the group attention shifted from that direction.',
    accent: '#f05d5e'
  },
  {
    id: 'ambient',
    name: 'Ambient Filter',
    icon: Radar,
    signal: 'warning',
    metric: 'Moving threats only',
    headline: 'Stay aware of unexpected movement while preserving hearing.',
    detail:
      'Static objects stay silent. Fast-moving or unexpected obstacles trigger urgent micro-burst patterns.',
    haptic: 'All dots contract inward for collision-course urgency.',
    accent: '#f7b32b'
  },
  {
    id: 'liquid',
    name: 'Liquid Level',
    icon: Droplets,
    signal: 'rising',
    metric: 'Target fill feedback',
    headline: 'Feel liquid depth rise in real time.',
    detail:
      'A tactile gradient tracks fill height so pouring, measuring, and hot-liquid tasks do not need audio indicators.',
    haptic: 'Active dots rise up the finger as the container fills.',
    accent: '#4f8cff'
  }
];

const trainingSteps = [
  'Calibrate hand position and phone distance',
  'Learn three base patterns: align, correct, urgent',
  'Practice on a mapped tabletop scene',
  'Save preferred signal strength and rhythm'
];

function App() {
  const [activeId, setActiveId] = useState('pinpoint');
  const [intensity, setIntensity] = useState(68);
  const active = modes.find((mode) => mode.id === activeId) ?? modes[0];

  const ringDots = useMemo(() => buildRingPattern(active.signal, intensity), [active.signal, intensity]);

  return (
    <main className="shell" style={{ '--accent': active.accent }}>
      <aside className="sidebar" aria-label="Pulse Point prototype controls">
        <div className="brand">
          <div className="brand-mark">
            <Waves size={24} />
          </div>
          <div>
            <p>Pulse Point</p>
            <span>Synchronous Spatial Echo</span>
          </div>
        </div>

        <nav className="mode-list" aria-label="SSE modes">
          {modes.map((mode) => {
            const Icon = mode.icon;
            const selected = mode.id === activeId;
            return (
              <button
                className={selected ? 'mode-button active' : 'mode-button'}
                key={mode.id}
                onClick={() => setActiveId(mode.id)}
                type="button"
                aria-pressed={selected}
              >
                <Icon size={19} />
                <span>{mode.name}</span>
              </button>
            );
          })}
        </nav>

        <section className="panel compact" aria-labelledby="signal-heading">
          <div className="panel-title">
            <Activity size={17} />
            <h2 id="signal-heading">Signal Strength</h2>
          </div>
          <input
            aria-label="Haptic intensity"
            type="range"
            min="20"
            max="100"
            value={intensity}
            onChange={(event) => setIntensity(Number(event.target.value))}
          />
          <div className="range-labels">
            <span>Subtle</span>
            <strong>{intensity}%</strong>
            <span>Strong</span>
          </div>
        </section>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Haptic-first assistive spatial prototype</p>
            <h1>{active.name}</h1>
          </div>
          <div className="status-pill">
            <VolumeX size={17} />
            <span>Audio preserved</span>
          </div>
        </header>

        <section className="hero-grid">
          <article className="spatial-stage" aria-label="Spatial haptic simulation">
            <div className="scan-field">
              <div className="phone">
                <Move3D size={30} />
                <span>Phone sensors</span>
              </div>
              <div className="target target-a" />
              <div className="target target-b" />
              <div className="target target-c" />
              <div className="pulse-ring ring-one" />
              <div className="pulse-ring ring-two" />
              <div className="tether" />
              <div className="hand-node">
                <Hand size={28} />
              </div>
            </div>
          </article>

          <article className="panel feature">
            <div className="metric">{active.metric}</div>
            <h2>{active.headline}</h2>
            <p>{active.detail}</p>
            <div className="quiet-note">
              <ShieldCheck size={18} />
              <span>{active.haptic}</span>
            </div>
          </article>
        </section>

        <section className="lower-grid">
          <article className="panel ring-panel" aria-labelledby="ring-heading">
            <div className="panel-title">
              <Sparkles size={18} />
              <h2 id="ring-heading">Haptic Matrix Ring</h2>
            </div>
            <div className={`ring-grid ${active.signal}`}>
              {ringDots.map((dot, index) => (
                <span
                  key={index}
                  className="dot"
                  style={{
                    '--scale': dot.scale,
                    '--delay': `${dot.delay}ms`,
                    '--alpha': dot.alpha
                  }}
                />
              ))}
            </div>
          </article>

          <article className="panel">
            <div className="panel-title">
              <BellRing size={18} />
              <h2>Training Flow</h2>
            </div>
            <ol className="training-list">
              {trainingSteps.map((step) => (
                <li key={step}>{step}</li>
              ))}
            </ol>
          </article>

          <article className="panel">
            <div className="panel-title">
              <ScanSearch size={18} />
              <h2>Build Roadmap</h2>
            </div>
            <div className="roadmap">
              <span>Prototype UI</span>
              <span>Sensor mock data</span>
              <span>Haptic vocabulary</span>
              <span>Phone camera proof</span>
            </div>
          </article>
        </section>
      </section>
    </main>
  );
}

function buildRingPattern(signal, intensity) {
  const base = Math.max(0.35, intensity / 100);
  const patternMap = {
    'center-pulse': [0.45, 0.65, 0.45, 0.65, 1, 0.65, 0.45, 0.65, 0.45],
    'left-rise': [0.85, 0.35, 0.25, 1, 0.45, 0.3, 0.85, 0.35, 0.25],
    warning: [0.9, 1, 0.9, 1, 1.15, 1, 0.9, 1, 0.9],
    rising: [0.25, 0.25, 0.25, 0.65, 0.65, 0.65, 1, 1, 1]
  };

  return (patternMap[signal] ?? patternMap['center-pulse']).map((scale, index) => ({
    scale: (scale * base).toFixed(2),
    alpha: Math.min(1, 0.28 + scale * base).toFixed(2),
    delay: index * 55
  }));
}

createRoot(document.getElementById('root')).render(<App />);
