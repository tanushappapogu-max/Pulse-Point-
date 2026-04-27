import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs';
import { Camera, Loader2, ScanLine, Square } from 'lucide-react';
import './styles.css';

const targetAliases = {
  'my mouse': ['mouse'],
  mouse: ['mouse'],
  'computer mouse': ['mouse'],
  trackpad: ['mouse'],
  cursor: ['mouse'],
  phone: ['cell phone'],
  iphone: ['cell phone'],
  'cell phone': ['cell phone'],
  laptop: ['laptop'],
  keyboard: ['keyboard'],
  remote: ['remote'],
  cup: ['cup'],
  bottle: ['bottle'],
  book: ['book'],
  backpack: ['backpack'],
  chair: ['chair']
};

const hapticPatterns = {
  looking: [24, 260],
  found: [170, 80, 170, 80, 260],
  left: [80, 45, 80],
  right: [80, 45, 80, 45, 80],
  closer: [45, 38, 45, 38, 45],
  locked: [260],
  reach: [360, 80, 360],
  lost: [35, 120, 35]
};

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const modelRef = useRef(null);
  const streamRef = useRef(null);
  const loopRef = useRef(null);
  const lastHapticRef = useRef(0);
  const lastSignalRef = useRef('');
  const previousAreaRef = useRef(0);
  const foundOnceRef = useRef(false);

  const [target, setTarget] = useState('mouse');
  const [status, setStatus] = useState('ready');
  const [signal, setSignal] = useState('looking');
  const [match, setMatch] = useState(null);
  const [error, setError] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [hapticsSupported, setHapticsSupported] = useState(true);

  useEffect(() => {
    setHapticsSupported('vibrate' in navigator);
    return () => stopScanner();
  }, []);

  async function startScanner() {
    setError('');
    setStatus('camera');
    setMatch(null);
    setSignal('looking');
    foundOnceRef.current = false;
    previousAreaRef.current = 0;

    try {
      const stream = await getWideCameraStream();
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      await setWidestZoom(stream);

      setIsRunning(true);
      setStatus('loading');
      fireHaptic('looking', true);

      if (!modelRef.current) {
        modelRef.current = await cocoSsd.load({ base: 'mobilenet_v2' });
      }

      setStatus('looking');
      detect();
    } catch (err) {
      setStatus('blocked');
      setError(err?.message || 'Camera blocked. Open the site over HTTPS and allow camera access.');
    }
  }

  function stopScanner() {
    if (loopRef.current) cancelAnimationFrame(loopRef.current);
    loopRef.current = null;
    navigator.vibrate?.(0);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setIsRunning(false);
    setStatus('ready');
  }

  async function detect() {
    const video = videoRef.current;
    const model = modelRef.current;

    if (!video || !model || video.readyState < 2) {
      loopRef.current = requestAnimationFrame(detect);
      return;
    }

    const predictions = await model.detect(video, 60, 0.18);
    const frame = {
      width: video.videoWidth || 640,
      height: video.videoHeight || 480
    };
    const targetMatch = findTarget(predictions, target, frame);
    draw(predictions, targetMatch);

    if (!targetMatch) {
      setMatch(null);
      setStatus('looking');
      setSignal('looking');
      fireHaptic('looking');
      previousAreaRef.current = 0;
      loopRef.current = requestAnimationFrame(detect);
      return;
    }

    const guidance = getGuidance(targetMatch, frame, previousAreaRef.current);
    previousAreaRef.current = guidance.area;

    setMatch({
      name: targetMatch.isCandidate ? target : targetMatch.class,
      score: targetMatch.score,
      direction: guidance.direction,
      distance: guidance.distance
    });
    setStatus(guidance.status);
    setSignal(guidance.signal);

    if (!foundOnceRef.current) {
      foundOnceRef.current = true;
      fireHaptic('found', true);
    } else {
      fireHaptic(guidance.signal);
    }

    loopRef.current = requestAnimationFrame(detect);
  }

  function draw(predictions, targetMatch) {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);

    predictions.forEach((prediction) => {
      const isTarget = prediction === targetMatch;
      const [x, y, boxWidth, boxHeight] = prediction.bbox;
      ctx.strokeStyle = isTarget ? '#00ff9d' : 'rgba(255,255,255,0.55)';
      ctx.lineWidth = isTarget ? 8 : 2;
      ctx.strokeRect(x, y, boxWidth, boxHeight);

      if (isTarget) {
        ctx.fillStyle = '#00ff9d';
        ctx.fillRect(x, Math.max(0, y - 38), Math.min(220, boxWidth), 38);
        ctx.fillStyle = '#05100d';
        ctx.font = '800 22px system-ui';
        ctx.fillText(`${prediction.class} ${Math.round(prediction.score * 100)}%`, x + 10, Math.max(26, y - 12));
      }
    });
  }

  function fireHaptic(nextSignal, immediate = false) {
    if (!navigator.vibrate) return;

    const now = Date.now();
    const minimumGap = nextSignal === 'looking' ? 1150 : 520;
    if (!immediate && now - lastHapticRef.current < minimumGap) return;
    if (!immediate && nextSignal === lastSignalRef.current && nextSignal === 'found') return;

    lastHapticRef.current = now;
    lastSignalRef.current = nextSignal;
    navigator.vibrate(hapticPatterns[nextSignal] || hapticPatterns.looking);
  }

  return (
    <main className={`scanner signal-${signal}`}>
      <video ref={videoRef} playsInline muted />
      <canvas ref={canvasRef} />

      {!isRunning ? (
        <button className="start-button" type="button" onClick={startScanner}>
          <Camera size={30} />
          Start
        </button>
      ) : null}

      {status === 'loading' ? (
        <div className="loading-pill">
          <Loader2 size={18} />
          Loading
        </div>
      ) : null}

      {error ? <div className="error-pill">{error}</div> : null}

      <div className="reticle" aria-hidden="true">
        <span />
      </div>

      <form className="target-bar" onSubmit={(event) => event.preventDefault()}>
        <input
          aria-label="Object to find"
          value={target}
          onChange={(event) => setTarget(event.target.value)}
          placeholder="object"
        />
        <button type="button" onClick={isRunning ? stopScanner : startScanner} aria-label={isRunning ? 'Stop' : 'Scan'}>
          {isRunning ? <Square size={18} /> : <ScanLine size={20} />}
        </button>
      </form>

      <div className="signal-strip" aria-live="polite">
        <strong>{status}</strong>
        <span>
          {match
            ? `${match.direction} · ${match.distance}`
            : hapticsSupported
              ? 'scan slowly'
              : 'iPhone web haptics blocked'}
        </span>
      </div>
    </main>
  );
}

async function getWideCameraStream() {
  const baseConstraints = {
    video: {
      facingMode: { ideal: 'environment' },
      width: { ideal: 1920 },
      height: { ideal: 1080 },
      aspectRatio: { ideal: 16 / 9 }
    },
    audio: false
  };

  const firstStream = await navigator.mediaDevices.getUserMedia(baseConstraints);
  const devices = await navigator.mediaDevices.enumerateDevices();
  const videoInputs = devices.filter((device) => device.kind === 'videoinput');
  const wideCamera = videoInputs.find((device) => /ultra|wide|back|rear|environment/i.test(device.label));

  if (!wideCamera) return firstStream;

  firstStream.getTracks().forEach((track) => track.stop());
  return navigator.mediaDevices.getUserMedia({
    video: {
      deviceId: { exact: wideCamera.deviceId },
      width: { ideal: 1920 },
      height: { ideal: 1080 },
      aspectRatio: { ideal: 16 / 9 }
    },
    audio: false
  });
}

async function setWidestZoom(stream) {
  const track = stream.getVideoTracks()[0];
  const capabilities = track?.getCapabilities?.();
  if (!track || !capabilities?.zoom) return;

  try {
    await track.applyConstraints({ advanced: [{ zoom: capabilities.zoom.min }] });
  } catch {
    // Some browsers expose zoom capabilities but reject runtime constraints.
  }
}

function findTarget(predictions, target, frame) {
  const aliases = getAliases(target);
  const exactMatch = predictions
    .filter((prediction) => aliases.includes(prediction.class.toLowerCase()))
    .sort((a, b) => b.score - a.score)[0];

  if (exactMatch) return exactMatch;

  return findCenterCandidate(predictions, frame, target);
}

function getAliases(target) {
  const normalized = target.trim().toLowerCase();
  if (targetAliases[normalized]) return targetAliases[normalized];
  return [normalized];
}

function getGuidance(prediction, frame, previousArea) {
  const [x, , width, height] = prediction.bbox;
  const centerX = x + width / 2;
  const horizontal = centerX / frame.width;
  const area = (width * height) / (frame.width * frame.height);
  const distance = estimateDistance(area);
  const centered = horizontal > 0.42 && horizontal < 0.58;
  const close = area > 0.2;
  const gettingCloser = previousArea > 0 && area > previousArea * 1.08;

  if (centered && close) {
    return { signal: 'reach', status: 'reach', direction: 'center', distance, area };
  }

  if (centered) {
    return {
      signal: gettingCloser ? 'closer' : 'locked',
      status: gettingCloser ? 'closer' : 'locked',
      direction: 'center',
      distance,
      area
    };
  }

  if (horizontal < 0.42) {
    return { signal: 'left', status: 'left', direction: 'turn left', distance, area };
  }

  return { signal: 'right', status: 'right', direction: 'turn right', distance, area };
}

function findCenterCandidate(predictions, frame, target) {
  const normalizedTarget = target.trim().toLowerCase();
  const allowFallback = normalizedTarget.length > 0;
  if (!allowFallback) return null;

  const centerX = frame.width / 2;
  const centerY = frame.height / 2;

  return predictions
    .map((prediction) => {
      const [x, y, width, height] = prediction.bbox;
      const boxCenterX = x + width / 2;
      const boxCenterY = y + height / 2;
      const normalizedDistance = Math.hypot((boxCenterX - centerX) / frame.width, (boxCenterY - centerY) / frame.height);
      const area = (width * height) / (frame.width * frame.height);
      const centerScore = prediction.score + area * 1.6 - normalizedDistance * 1.4;
      return { ...prediction, score: prediction.score * 0.72, centerScore, isCandidate: true };
    })
    .filter((prediction) => prediction.centerScore > 0.12 && prediction.score > 0.14)
    .sort((a, b) => b.centerScore - a.centerScore)[0];
}

function estimateDistance(area) {
  if (area > 0.24) return 'very close';
  if (area > 0.14) return 'close';
  if (area > 0.07) return 'medium';
  return 'far';
}

createRoot(document.getElementById('root')).render(<App />);
