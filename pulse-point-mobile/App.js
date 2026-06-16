import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Animated,
  Easing,
  Platform,
  Pressable,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { Magnetometer } from 'expo-sensors';
import * as Speech from 'expo-speech';
import { StatusBar } from 'expo-status-bar';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';

import {
  createDetectionResult,
  buildStageFromDetection,
  guidanceStages,
} from './src/services/spatialEngine';
import {
  detectObject,
  checkHealth,
  apiResultToDetection,
} from './src/services/visionAPI';
import { suggestObjects, isKnownObject } from './src/services/objectList';
import {
  DirectionHapticEngine,
  analyzeDetection,
  playTransitionHaptic,
  directionArrow,
  directionLabel,
  Direction,
  Proximity,
} from './src/services/directionHaptics';

// ── Flow stages ───────────────────────────────────────────────────────────────

const flow = ['idle', 'scanning', 'targetLocked', 'orienting', 'walking', 'reaching', 'complete'];

// ── Detection config ──────────────────────────────────────────────────────────

const SCAN_INTERVAL_MS   = 800;   // how often we capture + classify
const CONF_THRESHOLD     = 0.15;  // min confidence to count as a hit
const MAX_SCAN_FRAMES    = 20;    // frames before giving up during initial scan
const LOST_PATIENCE_MS   = 4000;  // ms without detection before "lost" speech

// ── Error boundary ────────────────────────────────────────────────────────────

class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() { return { hasError: true }; }
  componentDidCatch(err, info) { console.error('Pulse Point crashed', err, info); }
  handleRetry = () => this.setState({ hasError: false });

  render() {
    if (!this.state.hasError) return this.props.children;
    return (
      <SafeAreaView style={styles.permissionScreen}>
        <StatusBar style="light" />
        <Text style={styles.permissionTitle}>Something went wrong</Text>
        <Text style={styles.permissionText}>Try again. If it keeps failing, restart the app.</Text>
        <Pressable style={styles.primaryButton} onPress={this.handleRetry}>
          <Text style={styles.primaryButtonText}>Try again</Text>
        </Pressable>
      </SafeAreaView>
    );
  }
}

// ── Main component ────────────────────────────────────────────────────────────

function AppContent() {
  const [permission, requestPermission] = useCameraPermissions();

  // Core state
  const [target, setTarget]           = useState('');
  const [stageKey, setStageKey]       = useState('idle');
  const [detection, setDetection]     = useState(null);
  const [visionMode, setVisionMode]   = useState('checking');
  const [micPrimed, setMicPrimed]     = useState(false);
  const [cameraActive, setCameraActive] = useState(false);

  // Direction overlay state (driven by haptic engine callbacks)
  const [direction, setDirection]     = useState(null);
  const [proximity, setProximity]     = useState(null);
  const [distanceM, setDistanceM]     = useState(null);
  const [dynamicStage, setDynamicStage] = useState(null);

  // Sensor
  const [heading, setHeading]         = useState(0);

  // Animations
  const scanPulse   = useRef(new Animated.Value(0)).current;
  const proxRing    = useRef(new Animated.Value(0)).current;
  const dirFade     = useRef(new Animated.Value(0)).current;

  // Refs
  const cameraRef           = useRef(null);
  const hapticEngineRef     = useRef(null);
  const scanIntervalRef     = useRef(null);
  const isScanningRef       = useRef(false);   // mutex for capture calls
  const lastDetectedRef     = useRef(0);        // timestamp of last successful hit
  const inputRef            = useRef(null);
  const speechThrottleRef   = useRef(null);     // last direction spoken

  const stage = dynamicStage || guidanceStages[stageKey];
  const isRunning = stageKey !== 'idle';
  const canScan   = Boolean(target.trim());

  // ── Engine init ────────────────────────────────────────────────────────────

  useEffect(() => {
    hapticEngineRef.current = new DirectionHapticEngine();
    hapticEngineRef.current.onUpdate((dir, prox, dist) => {
      setDirection(dir);
      setProximity(prox);
      setDistanceM(dist);
      _animateProxRing(prox);
      _throttledDirectionSpeech(dir, prox);
    });

    return () => hapticEngineRef.current?.stop();
  }, []);

  // ── Server health check ────────────────────────────────────────────────────

  useEffect(() => {
    checkHealth().then(ok => {
      setVisionMode(ok ? 'cnn' : 'simulated');
      if (!ok) console.log('[PulsePoint] Vision API unavailable — using simulated detection');
    });
  }, []);

  // ── Compass ───────────────────────────────────────────────────────────────

  useEffect(() => {
    Magnetometer.setUpdateInterval(300);
    const sub = Magnetometer.addListener(({ x, y }) => {
      let angle = Math.atan2(y, x) * (180 / Math.PI);
      angle = (angle + 360) % 360;
      const h = Platform.OS === 'android' ? (360 - angle + 90) % 360 : angle;
      setHeading(Math.round(h));
    });
    return () => sub.remove();
  }, []);

  // ── Scan-sweep animation ───────────────────────────────────────────────────

  useEffect(() => {
    const anim = Animated.loop(
      Animated.timing(scanPulse, {
        toValue: 1,
        duration: stageKey === 'scanning' ? 850 : 1300,
        easing: Easing.inOut(Easing.ease),
        useNativeDriver: true,
      })
    );
    anim.start();
    return () => anim.stop();
  }, [scanPulse, stageKey]);

  // ── Direction arrow fade ───────────────────────────────────────────────────

  useEffect(() => {
    if (direction) {
      Animated.timing(dirFade, { toValue: 1, duration: 180, useNativeDriver: true }).start();
    } else {
      Animated.timing(dirFade, { toValue: 0, duration: 300, useNativeDriver: true }).start();
    }
  }, [direction, dirFade]);

  // ── Speech for stage transitions ───────────────────────────────────────────

  useEffect(() => {
    if (stageKey === 'idle') {
      Speech.stop();
      return;
    }
    const txt = _buildSpeech(stageKey, stage, target);
    Speech.stop();
    Speech.speak(txt, {
      language: 'en-US',
      pitch: 1,
      rate: Platform.OS === 'ios' ? 0.48 : 0.86,
    });
  }, [stageKey]);

  // ── Cleanup on unmount ─────────────────────────────────────────────────────

  useEffect(() => {
    return () => {
      _stopScan();
      Speech.stop();
    };
  }, []);

  // ── Capture + classify one frame ───────────────────────────────────────────

  const captureAndDetect = useCallback(async (targetName) => {
    if (!cameraRef.current || isScanningRef.current) return null;
    isScanningRef.current = true;

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.45,
        base64: false,
        skipProcessing: true,
      });
      const apiResult = await detectObject(photo.uri, targetName);
      return apiResultToDetection(apiResult, targetName);
    } catch (err) {
      console.warn('[PulsePoint] CNN frame failed:', err.message);
      return null;
    } finally {
      isScanningRef.current = false;
    }
  }, []);

  // ── Scanning phase — hunt for the object ──────────────────────────────────

  function startScanPhase(targetName) {
    setCameraActive(true);
    setDetection(null);
    setDynamicStage(null);
    setDirection(null);
    setStageKey('scanning');

    _clearScanInterval();
    let frame = 0;

    scanIntervalRef.current = setInterval(async () => {
      frame++;

      if (frame > MAX_SCAN_FRAMES) {
        _clearScanInterval();
        setStageKey('idle');
        setDynamicStage({
          label: 'Not found',
          instruction: `Could not find ${targetName}. Try panning the camera around the room.`,
          distanceMeters: null,
          turnDegrees: 0,
          confidence: 0,
        });
        Speech.speak(`Couldn't find ${targetName}. Try pointing the camera around.`, {
          language: 'en-US', rate: 0.5,
        });
        return;
      }

      const det = await captureAndDetect(targetName);
      if (!det || det.confidence < CONF_THRESHOLD) return;

      // ── Object found — switch to guidance phase ──────────────────────
      _clearScanInterval();
      lastDetectedRef.current = Date.now();

      setDetection(det);
      setStageKey('targetLocked');
      setDynamicStage(buildStageFromDetection(det));

      playTransitionHaptic('found');
      _feedEngine(det);
      startGuidancePhase(targetName);
    }, SCAN_INTERVAL_MS);
  }

  // ── Guidance phase — continuous directional guidance ──────────────────────

  function startGuidancePhase(targetName) {
    _clearScanInterval();

    scanIntervalRef.current = setInterval(async () => {
      const det = await captureAndDetect(targetName);

      if (!det || det.confidence < CONF_THRESHOLD) {
        // Tolerate brief detection gaps before declaring lost
        const gap = Date.now() - lastDetectedRef.current;
        if (gap > LOST_PATIENCE_MS) {
          hapticEngineRef.current?.stop();
          setDirection(null);
          playTransitionHaptic('lost');
          Speech.speak(`Lost sight of ${targetName}. Keep the camera pointed at the area.`, {
            language: 'en-US', rate: 0.5,
          });
          lastDetectedRef.current = Date.now();
        }
        return;
      }

      lastDetectedRef.current = Date.now();
      setDetection(det);

      const dynStage = buildStageFromDetection(det);
      if (dynStage) setDynamicStage(dynStage);

      _feedEngine(det);

      // ── Advance stage label based on distance ─────────────────────────
      const { proximity: prox } = analyzeDetection(det.boundingBox);
      if (prox === Proximity.NEAR && stageKey === 'targetLocked') {
        setStageKey('walking');
      }

      // ── Complete when within reach ────────────────────────────────────
      if (prox === Proximity.REACH) {
        _clearScanInterval();
        hapticEngineRef.current?.stop();
        setDirection(null);
        setStageKey('complete');
        setDynamicStage({
          label: 'Reach now',
          instruction: `${det.name} is right in front of you. Reach forward.`,
          distanceMeters: 0.3,
          turnDegrees: 0,
          confidence: det.confidence,
        });
        playTransitionHaptic('complete');
      }
    }, SCAN_INTERVAL_MS);
  }

  // ── Offline fallback (CNN server unreachable) ─────────────────────────────
  // Uses a deterministic position estimate from the object name so haptic
  // guidance still functions. No AI or simulation — just directional haptics
  // based on a hardcoded position heuristic.

  function startOfflineScan(targetName) {
    setCameraActive(true);
    setDetection(null);
    setDynamicStage(null);
    setStageKey('scanning');

    // Brief scan pause so the UI registers "scanning" before locking
    setTimeout(() => {
      const det = createDetectionResult(targetName);
      det.source = 'offline';
      setDetection(det);
      setStageKey('targetLocked');
      setDynamicStage(buildStageFromDetection(det));
      playTransitionHaptic('found');
      _feedEngine(det);
      Speech.speak(`Offline mode. Guiding to ${targetName} using estimated position.`, { rate: 0.9 });
    }, 1200);
  }

  // ── Public scan entry point ────────────────────────────────────────────────

  function startScan() {
    const t = target.trim();
    if (!t) { setCameraActive(true); inputRef.current?.focus(); return; }

    _stopScan();

    if (visionMode === 'cnn') {
      startScanPhase(t);
    } else {
      startOfflineScan(t);
    }
  }

  function resetScan() {
    _stopScan();
    setStageKey('idle');
    setDetection(null);
    setDynamicStage(null);
    setDirection(null);
    setProximity(null);
    setDistanceM(null);
    setCameraActive(false);
    Speech.stop();
  }

  // ── Internal helpers ───────────────────────────────────────────────────────

  function _feedEngine(det) {
    if (det?.boundingBox) {
      hapticEngineRef.current?.update(det.boundingBox);
    }
  }

  function _stopScan() {
    _clearScanInterval();
    hapticEngineRef.current?.stop();
    isScanningRef.current = false;
  }

  function _clearScanInterval() {
    if (scanIntervalRef.current) {
      clearInterval(scanIntervalRef.current);
      scanIntervalRef.current = null;
    }
  }

  function _animateProxRing(prox) {
    const speeds = {
      [Proximity.FAR]:    1400,
      [Proximity.MEDIUM]:  900,
      [Proximity.CLOSE]:   520,
      [Proximity.NEAR]:    260,
      [Proximity.REACH]:    90,
    };
    const dur = speeds[prox] ?? 1000;
    Animated.loop(
      Animated.sequence([
        Animated.timing(proxRing, { toValue: 1, duration: dur * 0.45, easing: Easing.out(Easing.ease), useNativeDriver: true }),
        Animated.timing(proxRing, { toValue: 0, duration: dur * 0.55, easing: Easing.in(Easing.ease),  useNativeDriver: true }),
      ]),
      { iterations: 1 }
    ).start();
  }

  // Speak direction changes at most every 3 seconds — don't spam voice
  function _throttledDirectionSpeech(dir, prox) {
    if (prox === Proximity.REACH) return; // let completion speech handle it

    const now = Date.now();
    if (speechThrottleRef.current && now - speechThrottleRef.current < 3000) return;
    speechThrottleRef.current = now;

    const phrases = {
      [Direction.LEFT]:   'move left',
      [Direction.RIGHT]:  'move right',
      [Direction.UP]:     'aim up',
      [Direction.DOWN]:   'aim down',
      [Direction.LOCKED]: null,
    };
    const phrase = phrases[dir];
    if (phrase) {
      Speech.speak(phrase, { language: 'en-US', rate: 0.55 });
    }
  }

  // ── Proximity ring styles ──────────────────────────────────────────────────

  const proxRingStyle = () => {
    const colors = {
      [Proximity.FAR]:    '#118ab2',
      [Proximity.MEDIUM]: '#06d6a0',
      [Proximity.CLOSE]:  '#ffd166',
      [Proximity.NEAR]:   '#ef476f',
      [Proximity.REACH]:  '#00ff9d',
    };
    const color = proximity ? colors[proximity] : 'rgba(255,255,255,0.3)';
    return {
      borderColor: color,
      opacity: proxRing.interpolate({ inputRange: [0, 1], outputRange: [0.3, 0.95] }),
      transform: [{
        scale: proxRing.interpolate({ inputRange: [0, 1], outputRange: [1.0, 1.18] }),
      }],
    };
  };

  // ── Direction arrow colour ─────────────────────────────────────────────────

  const dirArrowColor = () => {
    if (!direction) return '#ffffff';
    if (direction === Direction.REACH)  return '#00ff9d';
    if (direction === Direction.LOCKED) return '#06d6a0';
    return '#ffffff';
  };

  // ── Permission screens ─────────────────────────────────────────────────────

  if (!permission) {
    return (
      <View style={styles.permissionScreen}>
        <StatusBar style="light" />
        <Text style={styles.permissionTitle}>Loading Pulse Point</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.permissionScreen}>
        <StatusBar style="light" />
        <MaterialCommunityIcons name="camera-iris" size={58} color="#06d6a0" />
        <Text style={styles.permissionTitle}>Pulse Point needs camera access</Text>
        <Text style={styles.permissionText}>
          The app scans your room to locate the object you're looking for and guides you to it with haptic feedback.
        </Text>
        <Pressable style={styles.primaryButton} onPress={requestPermission}>
          <Text style={styles.primaryButtonText}>Allow camera</Text>
        </Pressable>
      </SafeAreaView>
    );
  }

  // ── Main render ────────────────────────────────────────────────────────────

  const handleScanButton = () => {
    if (isRunning) { resetScan(); return; }
    if (canScan)   { startScan(); return; }
    setCameraActive(true);
  };

  const scanSweepTranslate = scanPulse.interpolate({
    inputRange: [0, 1], outputRange: [-80, 760],
  });

  return (
    <View style={styles.app}>
      <StatusBar style="light" />

      {/* Camera */}
      <CameraView ref={cameraRef} style={styles.camera} facing="back" autofocus="on" />
      <View style={styles.cameraShade} />

      {/* Scan sweep line */}
      <Animated.View
        pointerEvents="none"
        style={[styles.scanSweep, {
          opacity: (cameraActive || isRunning) ? 1 : 0.4,
          transform: [{ translateY: scanSweepTranslate }],
        }]}
      />

      <SafeAreaView style={styles.overlay}>
        {/* ── Status strip ─────────────────────────────────────────── */}
        <View style={styles.signalStrip}>
          <View style={styles.signalHeader}>
            <Text style={styles.signalStatus}>{stage.label}</Text>
            <View style={[
              styles.modeBadge,
              visionMode === 'cnn' ? styles.modeBadgeCNN : styles.modeBadgeOffline,
            ]}>
              <Text style={styles.modeBadgeText}>
                {visionMode === 'checking' ? '···' : visionMode === 'cnn' ? 'CNN' : 'OFFLINE'}
              </Text>
            </View>
          </View>

          <Text style={styles.signalDetail}>
            {detection
              ? `${stage.instruction} · ${distanceM != null ? `${distanceM.toFixed(1)} m` : stage.distanceMeters != null ? `${stage.distanceMeters.toFixed(1)} m` : '—'}`
              : isRunning ? stage.instruction
              : cameraActive ? 'camera live — type a target'
              : 'ready'}
          </Text>

          {detection?.source === 'cnn' && detection?.latencyMs != null && (
            <Text style={styles.latencyText}>{detection.latencyMs} ms · {Math.round((detection.confidence ?? 0) * 100)}% conf</Text>
          )}
        </View>

        {/* ── Heading + direction badge (top-right) ─────────────────── */}
        <View style={styles.topRail}>
          <View style={styles.sensorPill}>
            <Ionicons name="navigate" size={15} color="#06d6a0" />
            <Text style={styles.sensorPillText}>{heading}°</Text>
          </View>

          {direction && (
            <Animated.View style={[styles.dirBadge, { opacity: dirFade }]}>
              <Text style={[styles.dirArrow, { color: dirArrowColor() }]}>
                {directionArrow(direction)}
              </Text>
            </Animated.View>
          )}
        </View>

        {/* ── Reticle + detection box + proximity ring ───────────────── */}
        <View style={styles.reticleArea}>

          {/* Proximity ring — pulses faster as object gets closer */}
          {isRunning && (
            <Animated.View style={[styles.proxRing, proxRingStyle()]} pointerEvents="none" />
          )}

          <View style={styles.reticle}>
            <View style={[styles.rc, styles.rcTl, detection && styles.rcActive]} />
            <View style={[styles.rc, styles.rcTr, detection && styles.rcActive]} />
            <View style={[styles.rc, styles.rcBl, detection && styles.rcActive]} />
            <View style={[styles.rc, styles.rcBr, detection && styles.rcActive]} />
            <View style={[styles.reticleDot, detection && styles.reticleDotActive]} />
            <Animated.View style={[
              styles.reticleScan,
              {
                opacity: scanPulse.interpolate({ inputRange: [0, 1], outputRange: [0.9, 0.15] }),
                transform: [{ translateY: scanPulse.interpolate({ inputRange: [0, 1], outputRange: [8, 236] }) }],
              },
            ]} />
          </View>

          {/* Direction arrow overlay — shown when guidance is active */}
          {direction && direction !== Direction.LOCKED && (
            <Animated.View style={[styles.dirOverlay, { opacity: dirFade }]}>
              <Text style={[styles.dirOverlayArrow, { color: dirArrowColor() }]}>
                {directionArrow(direction)}
              </Text>
              <Text style={styles.dirOverlayLabel}>{directionLabel(direction)}</Text>
            </Animated.View>
          )}

          {/* Detection bounding box */}
          {detection && (
            <View style={[
              styles.detectionBox,
              {
                left:   `${detection.boundingBox.x * 100}%`,
                top:    `${detection.boundingBox.y * 100}%`,
                width:  `${detection.boundingBox.width * 100}%`,
                height: `${detection.boundingBox.height * 100}%`,
              },
            ]}>
              <Text style={styles.detectionLabel}>
                {detection.name}  {Math.round((detection.confidence ?? 0) * 100)}%
              </Text>
            </View>
          )}

          {/* Start button (idle) */}
          {!cameraActive && !isRunning && (
            <Pressable style={styles.startButton} onPress={() => setCameraActive(true)}>
              <Ionicons name="camera-outline" size={30} color="#03100c" />
              <Text style={styles.startButtonText}>Start</Text>
            </Pressable>
          )}
        </View>

        {/* ── Bottom target bar ─────────────────────────────────────── */}
        <View style={styles.targetBar}>
          <Pressable
            style={[styles.micButton, micPrimed && styles.micButtonActive]}
            onPress={() => { setMicPrimed(true); inputRef.current?.focus(); }}
            accessibilityLabel="Tap to dictate a target object"
          >
            <Ionicons name="mic-outline" size={22} color="#ffffff" />
          </Pressable>

          <View style={styles.targetDisplay}>
            <TextInput
              ref={inputRef}
              value={target}
              onChangeText={t => { setTarget(t); if (micPrimed && t.trim()) setMicPrimed(false); }}
              placeholder="say or type what to find…"
              placeholderTextColor="rgba(255,255,255,0.62)"
              style={styles.input}
              autoCorrect={false}
              returnKeyType="search"
              onSubmitEditing={canScan ? startScan : undefined}
            />
            <Pressable
              style={[styles.goButton, !canScan && styles.disabled]}
              onPress={canScan ? startScan : null}
              disabled={!canScan}
            >
              <Text style={styles.goButtonText}>Go</Text>
            </Pressable>
          </View>

          <Pressable style={styles.scanButton} onPress={handleScanButton}>
            <Ionicons
              name={isRunning ? 'square' : 'scan-outline'}
              size={isRunning ? 18 : 21}
              color="#03100c"
            />
          </Pressable>
        </View>
      </SafeAreaView>
    </View>
  );
}

// ── Root ──────────────────────────────────────────────────────────────────────

export default function App() {
  return (
    <AppErrorBoundary>
      <AppContent />
    </AppErrorBoundary>
  );
}

// ── Speech helpers ────────────────────────────────────────────────────────────

function _buildSpeech(stageKey, stage, target) {
  const t = target?.trim() ? ` for ${target.trim()}` : '';
  const d = typeof stage?.distanceMeters === 'number'
    ? `. Distance ${stage.distanceMeters.toFixed(1)} meters.`
    : '';

  if (stageKey === 'scanning')     return `Scanning${t}. ${stage?.instruction ?? ''}`;
  if (stageKey === 'targetLocked') return `Target found${t}. ${stage?.instruction ?? ''}${d}`;
  if (stageKey === 'complete')     return `Reached${t}. ${stage?.instruction ?? ''}`;
  return `${stage?.label ?? ''}. ${stage?.instruction ?? ''}${d}`;
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  app:         { flex: 1, backgroundColor: '#020504' },
  camera:      { ...StyleSheet.absoluteFillObject },
  cameraShade: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(2,5,4,0.16)' },

  scanSweep: {
    position: 'absolute', left: 0, right: 0, top: 0,
    height: 1.5, zIndex: 2,
    backgroundColor: 'rgba(0,255,157,0.55)',
    shadowColor: '#00ff9d', shadowOpacity: 0.85, shadowRadius: 14,
  },

  overlay: { flex: 1, position: 'relative' },

  // Status strip
  signalStrip: {
    position: 'absolute', left: 14, top: 14, zIndex: 8, maxWidth: '70%',
    paddingHorizontal: 12, paddingVertical: 10,
    borderRadius: 14, backgroundColor: 'rgba(2,5,4,0.68)',
  },
  signalHeader: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  signalStatus: { color: '#fff', fontSize: 13, fontWeight: '900', textTransform: 'uppercase' },
  modeBadge:    { paddingHorizontal: 6, paddingVertical: 2, borderRadius: 6 },
  modeBadgeCNN: { backgroundColor: 'rgba(0,255,157,0.28)' },
  modeBadgeOffline: { backgroundColor: 'rgba(255,165,0,0.28)' },
  modeBadgeText:{ color: '#fff', fontSize: 9, fontWeight: '900', letterSpacing: 1 },
  signalDetail: { marginTop: 2, color: 'rgba(255,255,255,0.85)', fontSize: 12, fontWeight: '700', lineHeight: 16 },
  latencyText:  { marginTop: 2, color: 'rgba(255,255,255,0.45)', fontSize: 10, fontWeight: '600' },

  // Top-right rail
  topRail: {
    position: 'absolute', right: 14, top: 14, zIndex: 8,
    alignItems: 'flex-end', gap: 8,
  },
  sensorPill: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    paddingHorizontal: 12, height: 42, borderRadius: 12,
    borderWidth: 1, borderColor: 'rgba(255,255,255,0.14)',
    backgroundColor: 'rgba(2,5,4,0.62)',
  },
  sensorPillText: { color: '#fff', fontWeight: '800', fontSize: 14 },

  // Direction badge (top-right, below compass)
  dirBadge: {
    alignItems: 'center', justifyContent: 'center',
    width: 52, height: 52, borderRadius: 14,
    backgroundColor: 'rgba(2,5,4,0.72)',
    borderWidth: 1, borderColor: 'rgba(255,255,255,0.16)',
  },
  dirArrow: { fontSize: 26, fontWeight: '900' },

  // Reticle area
  reticleArea: { flex: 1, alignItems: 'center', justifyContent: 'center', position: 'relative' },

  // Proximity ring
  proxRing: {
    position: 'absolute',
    width: 310, height: 310,
    borderRadius: 155,
    borderWidth: 2.5,
  },

  // Corner reticle
  reticle: { position: 'relative', width: 260, height: 260 },
  rc: {
    position: 'absolute', width: 28, height: 28,
    borderColor: 'rgba(255,255,255,0.5)',
  },
  rcTl: { top: 0, left:  0, borderTopWidth: 2, borderLeftWidth:  2 },
  rcTr: { top: 0, right: 0, borderTopWidth: 2, borderRightWidth: 2 },
  rcBl: { bottom: 0, left:  0, borderBottomWidth: 2, borderLeftWidth:  2 },
  rcBr: { bottom: 0, right: 0, borderBottomWidth: 2, borderRightWidth: 2 },
  rcActive: { borderColor: '#00ff9d' },
  reticleDot: {
    position: 'absolute', left: '50%', top: '50%',
    width: 7, height: 7, marginLeft: -3.5, marginTop: -3.5,
    borderRadius: 4, backgroundColor: 'rgba(255,255,255,0.45)',
  },
  reticleDotActive: { backgroundColor: '#00ff9d' },
  reticleScan: {
    position: 'absolute', left: 8, right: 8, top: 0,
    height: 1.5, backgroundColor: 'rgba(255,255,255,0.45)',
  },

  // Direction overlay (centre of reticle)
  dirOverlay: {
    position: 'absolute',
    bottom: '12%',
    alignItems: 'center',
    gap: 4,
  },
  dirOverlayArrow: {
    fontSize: 60,
    fontWeight: '900',
    textShadowColor: 'rgba(0,0,0,0.5)',
    textShadowOffset: { width: 0, height: 2 },
    textShadowRadius: 6,
  },
  dirOverlayLabel: {
    color: 'rgba(255,255,255,0.85)',
    fontSize: 13,
    fontWeight: '800',
    textTransform: 'uppercase',
    letterSpacing: 1.5,
  },

  // Detection box
  detectionBox: {
    position: 'absolute',
    borderWidth: 3, borderColor: '#00ff9d',
    borderRadius: 3, backgroundColor: 'rgba(0,255,157,0.06)',
  },
  detectionLabel: {
    position: 'absolute', top: -28, left: 0,
    paddingHorizontal: 8, paddingVertical: 4, borderRadius: 8, overflow: 'hidden',
    color: '#101917', backgroundColor: '#00ff9d',
    fontSize: 11, fontWeight: '900', textTransform: 'capitalize',
  },

  // Start button
  startButton: {
    position: 'absolute', left: '50%', top: '50%',
    zIndex: 7, alignItems: 'center', justifyContent: 'center', gap: 8,
    width: 128, height: 128, marginLeft: -64, marginTop: -64,
    borderRadius: 64, backgroundColor: '#00ff9d',
  },
  startButtonText: { color: '#03100c', fontSize: 16, fontWeight: '900' },

  // Bottom target bar
  targetBar: {
    position: 'absolute', left: 14, right: 14, bottom: 16, zIndex: 9,
    flexDirection: 'row', alignItems: 'center', gap: 10,
    padding: 10, borderWidth: 1, borderColor: 'rgba(255,255,255,0.16)',
    borderRadius: 18, backgroundColor: 'rgba(2,5,4,0.80)',
  },
  micButton: {
    alignItems: 'center', justifyContent: 'center',
    width: 52, height: 52, borderRadius: 12,
    backgroundColor: 'rgba(255,255,255,0.12)',
  },
  micButtonActive: {
    backgroundColor: 'rgba(0,255,157,0.22)',
    borderWidth: 1, borderColor: 'rgba(0,255,157,0.55)',
  },
  targetDisplay: {
    flex: 1, flexDirection: 'row', alignItems: 'center', gap: 8,
    minWidth: 0, height: 52, paddingLeft: 14, paddingRight: 6,
    borderRadius: 12, backgroundColor: 'rgba(255,255,255,0.09)',
  },
  input: { flex: 1, minWidth: 0, color: '#fff', fontSize: 16, fontWeight: '800' },
  goButton: {
    alignItems: 'center', justifyContent: 'center',
    minWidth: 46, height: 46, paddingHorizontal: 12,
    borderRadius: 10, backgroundColor: '#00ff9d',
  },
  goButtonText: { color: '#03100c', fontSize: 12, fontWeight: '900', textTransform: 'uppercase' },
  scanButton: {
    alignItems: 'center', justifyContent: 'center',
    width: 52, height: 52, borderRadius: 12, backgroundColor: '#00ff9d',
  },
  disabled: { opacity: 0.35 },

  // Permission screens
  permissionScreen: {
    flex: 1, alignItems: 'center', justifyContent: 'center',
    gap: 16, padding: 26, backgroundColor: '#101917',
  },
  permissionTitle: { color: '#fff', fontSize: 27, fontWeight: '900', textAlign: 'center' },
  permissionText:  { color: '#aab7b4', fontSize: 16, lineHeight: 23, textAlign: 'center' },
  primaryButton:   {
    alignItems: 'center', justifyContent: 'center',
    minHeight: 54, minWidth: 190, paddingHorizontal: 18,
    borderRadius: 16, backgroundColor: '#118ab2',
  },
  primaryButtonText: { color: '#fff', fontSize: 16, fontWeight: '900' },
});
