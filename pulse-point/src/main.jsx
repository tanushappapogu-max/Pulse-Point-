import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs';
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  Compass,
  Crosshair,
  Loader2,
  Map,
  MousePointer2,
  ScanLine,
  Smartphone,
  Vibrate,
  Waves
} from 'lucide-react';
import './styles.css';

const guidanceSteps = ['Ask', 'Camera', 'Detect', 'Orient', 'Reach'];

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const modelRef = useRef(null);
  const streamRef = useRef(null);
  const loopRef = useRef(null);
  const lastBuzzRef = useRef(0);

  const [target, setTarget] = useState('mouse');
  const [cameraStatus, setCameraStatus] = useState('idle');
  const [modelStatus, setModelStatus] = useState('idle');
  const [detections, setDetections] = useState([]);
  const [targetLock, setTargetLock] = useState(null);
  const [frameSize, setFrameSize] = useState({ width: 640, height: 480 });
  const [heading, setHeading] = useState(null);
  const [error, setError] = useState('');

  const guidance = useMemo(() => getGuidance(targetLock, frameSize), [targetLock, frameSize]);
  const targetClass = normalizeTarget(target);

  useEffect(() => {
    const onOrientation = (event) => {
      if (typeof event.alpha === 'number') setHeading(Math.round(event.alpha));
    };
    window.addEventListener('deviceorientation', onOrientation, true);
    return () => window.removeEventListener('deviceorientation', onOrientation, true);
  }, []);

  useEffect(() => {
    return () => {
      stopScanner();
    };
  }, []);

  async function startScanner() {
    setError('');
    setTargetLock(null);
    setDetections([]);

    try {
      setCameraStatus('starting');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });

      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setFrameSize({
        width: videoRef.current.videoWidth || 640,
        height: videoRef.current.videoHeight || 480
      });
      setCameraStatus('running');

      if (!modelRef.current) {
        setModelStatus('loading');
        modelRef.current = await cocoSsd.load();
      }

      setModelStatus('ready');
      runDetectionLoop();
    } catch (err) {
      setCameraStatus('blocked');
      setError(err?.message || 'Camera could not start. Check browser permission and HTTPS.');
    }
  }

  function stopScanner() {
    if (loopRef.current) cancelAnimationFrame(loopRef.current);
    loopRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraStatus('idle');
  }

  async function runDetectionLoop() {
    const video = videoRef.current;
    const model = modelRef.current;
    if (!video || !model || video.readyState < 2) {
      loopRef.current = requestAnimationFrame(runDetectionLoop);
      return;
    }

    const predictions = await model.detect(video);
    setDetections(predictions);
    drawDetections(predictions);

    const lock = findTarget(predictions, targetClass);
    setTargetLock(lock);
    if (lock) buzzForGuidance(lock, video.videoWidth || 640, lastBuzzRef);

    loopRef.current = requestAnimationFrame(runDetectionLoop);
  }

  function drawDetections(predictions) {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    predictions.forEach((prediction) => {
      const [x, y, width, height] = prediction.bbox;
      const matched = normalizeTarget(prediction.class) === targetClass;
      ctx.strokeStyle = matched ? '#06d6a0' : '#ffd166';
      ctx.lineWidth = matched ? 5 : 3;
      ctx.strokeRect(x, y, width, height);
      ctx.fillStyle = matched ? '#06d6a0' : '#ffd166';
      ctx.font = '700 22px system-ui';
      ctx.fillText(`${prediction.class} ${Math.round(prediction.score * 100)}%`, x + 8, Math.max(28, y - 10));
    });
  }

  const activeStep = targetLock ? 4 : cameraStatus === 'running' ? 3 : target.trim() ? 1 : 0;

  return (
    <main className="app-shell">
      <section className="phone-app" aria-label="Pulse Point object detector">
        <header className="topbar">
          <div className="brand">
            <span className="brand-mark">
              <Waves size={21} />
            </span>
            <div>
              <strong>Pulse Point</strong>
              <small>Live object finder</small>
            </div>
          </div>
          <span className={targetLock ? 'status live' : 'status'}>{targetLock ? 'Locked' : modelStatus}</span>
        </header>

        <section className="search-panel">
          <label htmlFor="target">What do you need to find?</label>
          <div className="input-row">
            <MousePointer2 size={20} />
            <input id="target" value={target} onChange={(event) => setTarget(event.target.value)} />
            <button type="button" onClick={startScanner} aria-label="Start scanner">
              <ScanLine size={19} />
            </button>
          </div>
        </section>

        <section className="camera-panel">
          <video ref={videoRef} playsInline muted />
          <canvas ref={canvasRef} />
          {cameraStatus !== 'running' ? (
            <div className="camera-empty">
              <Camera size={46} />
              <h1>Start camera scan</h1>
              <p>Pulse Point will use your browser camera to search for “{target || 'object'}”.</p>
            </div>
          ) : null}
          {modelStatus === 'loading' ? (
            <div className="model-loading">
              <Loader2 size={20} />
              Loading object detector
            </div>
          ) : null}
        </section>

        {error ? (
          <div className="error-box">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        ) : null}

        <section className={targetLock ? 'guidance-card found' : 'guidance-card'}>
          <div className="guidance-header">
            <span>{targetLock ? 'Target found' : 'Scanning'}</span>
            <strong>{guidance.distance}</strong>
          </div>
          <h2>{guidance.title}</h2>
          <p>{guidance.text}</p>
          <div className="guidance-grid">
            <Info icon={<Compass size={19} />} label="Direction" value={guidance.direction} />
            <Info icon={<Vibrate size={19} />} label="Haptics" value={guidance.haptic} />
            <Info icon={<Smartphone size={19} />} label="Heading" value={heading === null ? '--' : `${heading}°`} />
          </div>
        </section>

        <section className="map-card">
          <div className="map-title">
            <Map size={18} />
            <span>Spatial estimate</span>
          </div>
          <div className="mini-map">
            <span className="user-dot">
              <Smartphone size={16} />
            </span>
            <span
              className={targetLock ? 'target-dot locked' : 'target-dot'}
              style={{ left: `${guidance.mapX}%`, top: `${guidance.mapY}%` }}
            >
              <Crosshair size={17} />
            </span>
            <span className="route-line" />
          </div>
        </section>

        <nav className="stepper" aria-label="Flow">
          {guidanceSteps.map((step, index) => (
            <span key={step} className={index <= activeStep ? 'active' : ''}>
              {index + 1}
              <small>{step}</small>
            </span>
          ))}
        </nav>
      </section>

      <aside className="detector-panel">
        <p className="eyebrow">Real browser object detection</p>
        <h1>Open the site, start the camera, and search for an object.</h1>
        <p>
          This Vercel version now runs object recognition in the browser. It draws boxes around
          detected objects, locks onto the target name, estimates direction and distance from the
          camera frame, and uses phone vibration when available.
        </p>
        <div className="detected-list">
          <h2>Detected now</h2>
          {detections.length ? (
            detections.slice(0, 6).map((item) => (
              <div key={`${item.class}-${item.score}-${item.bbox[0]}`}>
                <span>{item.class}</span>
                <strong>{Math.round(item.score * 100)}%</strong>
              </div>
            ))
          ) : (
            <div>
              <span>No objects yet</span>
              <strong>--</strong>
            </div>
          )}
        </div>
        <div className="truth-box">
          <CheckCircle2 size={20} />
          <span>
            This is real camera detection. LiDAR room meshes still need native ARKit/ARCore, but
            object recognition and haptic guidance can run from the website.
          </span>
        </div>
      </aside>
    </main>
  );
}

function Info({ icon, label, value }) {
  return (
    <article>
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function normalizeTarget(value) {
  const normalized = value.trim().toLowerCase();
  if (normalized === 'computer mouse' || normalized === 'my mouse') return 'mouse';
  if (normalized === 'cell phone' || normalized === 'phone') return 'cell phone';
  return normalized;
}

function findTarget(predictions, targetClass) {
  if (!targetClass) return null;
  return predictions
    .filter((prediction) => normalizeTarget(prediction.class) === targetClass && prediction.score > 0.45)
    .sort((a, b) => b.score - a.score)[0];
}

function getGuidance(lock, frameSize) {
  if (!lock) {
    return {
      title: 'Move the camera slowly',
      text: 'Scan the room left to right. When the target is recognized, Pulse Point will switch to orientation guidance.',
      direction: 'searching',
      distance: '--',
      haptic: 'slow scan pulse',
      mapX: 72,
      mapY: 34
    };
  }

  const [x, y, width, height] = lock.bbox;
  const centerX = x + width / 2;
  const centerY = y + height / 2;
  const frameWidth = frameSize.width || 640;
  const frameHeight = frameSize.height || 480;
  const horizontal = centerX / frameWidth;
  const distanceScore = Math.max(0.2, Math.min(1, 1 - width / frameWidth));
  const distanceMeters = Math.max(0.2, distanceScore * 3.4);

  let direction = 'centered';
  let haptic = 'center steady pulse';
  if (horizontal < 0.42) {
    direction = 'turn left';
    haptic = 'left-side pulses';
  } else if (horizontal > 0.58) {
    direction = 'turn right';
    haptic = 'right-side pulses';
  }

  return {
    title: distanceMeters < 0.7 ? 'Reach forward carefully' : 'Target locked',
    text:
      distanceMeters < 0.7
        ? 'You are close enough for hand-level guidance. Reach toward the highlighted object.'
        : 'Adjust your direction until the object is centered, then move forward slowly.',
    direction,
    distance: `${distanceMeters.toFixed(1)} m est.`,
    haptic,
    mapX: Math.max(16, Math.min(86, horizontal * 100)),
    mapY: Math.max(16, Math.min(78, (centerY / frameHeight) * 100))
  };
}

function buzzForGuidance(lock, frameWidth, lastBuzzRef) {
  if (!navigator.vibrate) return;
  const now = Date.now();
  if (now - lastBuzzRef.current < 1100) return;
  lastBuzzRef.current = now;

  const [x, , width] = lock.bbox;
  const horizontal = (x + width / 2) / frameWidth;
  if (horizontal < 0.42) navigator.vibrate([70, 60, 70]);
  else if (horizontal > 0.58) navigator.vibrate([70, 40, 70, 40, 70]);
  else navigator.vibrate(120);
}

createRoot(document.getElementById('root')).render(<App />);
