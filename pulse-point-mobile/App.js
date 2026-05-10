import React, { useEffect, useRef, useState } from 'react';
import {
  Animated,
  Easing,
  Platform,
  Pressable,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  View
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Haptics from 'expo-haptics';
import { Magnetometer } from 'expo-sensors';
import { StatusBar } from 'expo-status-bar';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import Svg, { Circle, Line, Rect } from 'react-native-svg';
import { createDetectionResult, getHapticPattern, guidanceStages } from './src/services/spatialEngine';

const flow = ['idle', 'scanning', 'targetLocked', 'orienting', 'walking', 'reaching', 'complete'];

class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, info) {
    console.error('Pulse Point crashed', error, info);
  }

  handleRetry = () => {
    this.setState({ hasError: false });
  };

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

function AppContent() {
  const [permission, requestPermission] = useCameraPermissions();
  const [target, setTarget] = useState('');
  const [stageKey, setStageKey] = useState('idle');
  const [detection, setDetection] = useState(null);
  const [heading, setHeading] = useState(0);
  const pulse = useRef(new Animated.Value(0)).current;
  const scanTimerRef = useRef(null);
  const stage = guidanceStages[stageKey];
  const stageIndex = flow.indexOf(stageKey);

  useEffect(() => {
    Magnetometer.setUpdateInterval(300);
    const subscription = Magnetometer.addListener((data) => {
      // ANDROID axes are messy, normalize to 0-360
      let angle = Math.atan2(data.y, data.x) * (180 / Math.PI);
      angle = (angle + 360) % 360;
      const headingValue = Platform.OS === 'android'
        ? (360 - angle + 90) % 360
        : angle;
      setHeading(Math.round(headingValue));
    });

    return () => subscription.remove();
  }, []);

  useEffect(() => {
    const animation = Animated.loop(
      Animated.timing(pulse, {
        toValue: 1,
        duration: stageKey === 'scanning' ? 900 : 1250,
        easing: Easing.inOut(Easing.ease),
        useNativeDriver: true
      })
    );

    animation.start();
    return () => animation.stop();
  }, [pulse, stageKey]);

  useEffect(() => {
    runStageHaptics(stageKey);
  }, [stageKey]);

  useEffect(() => {
    return () => {
      if (scanTimerRef.current) clearTimeout(scanTimerRef.current);
    };
  }, []);

  function startScan() {
    if (scanTimerRef.current) clearTimeout(scanTimerRef.current);
    setDetection(null);
    setStageKey('scanning');
    const targetAtScanStart = target.trim();
    scanTimerRef.current = setTimeout(() => {
      setDetection(createDetectionResult(targetAtScanStart));
      setStageKey('targetLocked');
      scanTimerRef.current = null;
    }, 2600);
  }

  function resetScan() {
    if (scanTimerRef.current) {
      clearTimeout(scanTimerRef.current);
      scanTimerRef.current = null;
    }
    setStageKey('idle');
    setDetection(null);
  }

  function nextStage() {
    if (stageKey === 'complete') {
      resetScan();
      return;
    }
    const next = flow[Math.min(stageIndex + 1, flow.length - 1)];
    setStageKey(next);
  }

  const pulseScale = pulse.interpolate({
    inputRange: [0, 1],
    outputRange: [0.85, 1.25]
  });

  const pulseOpacity = pulse.interpolate({
    inputRange: [0, 1],
    outputRange: [0.95, 0.25]
  });

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
          The app scans your room to recognize the object you are trying to find and estimate where it is.
        </Text>
        <Pressable style={styles.primaryButton} onPress={requestPermission}>
          <Text style={styles.primaryButtonText}>Allow camera</Text>
        </Pressable>
      </SafeAreaView>
    );
  }

  return (
    <View style={styles.app}>
      <StatusBar style="light" />
      {/* autofocus: lowercase is correct for expo-camera v16 CameraView
          in the current CameraView API. */}
      <CameraView style={styles.camera} facing="back" autofocus="on" />
      <View style={styles.cameraShade} />

      <SafeAreaView style={styles.overlay}>
        <View style={styles.header}>
          <View>
            <Text style={styles.kicker}>Pulse Point</Text>
            <Text style={styles.heading}>Find object</Text>
          </View>
          <View style={styles.sensorPill}>
            <Ionicons name="navigate" size={16} color="#06d6a0" />
            <Text style={styles.sensorPillText}>{heading}°</Text>
          </View>
        </View>

        <View style={styles.targetInput}>
          <Ionicons name="mic" size={20} color="#101917" />
          <TextInput
            value={target}
            onChangeText={setTarget}
            placeholder="What do you need to find?"
            placeholderTextColor="#687674"
            style={styles.input}
            autoCorrect={false}
            returnKeyType="search"
            onSubmitEditing={startScan}
          />
          <Pressable
            style={[
              styles.scanButton,
              !target.trim() && { opacity: 0.38 }
            ]}
            onPress={target.trim() ? startScan : null}
            disabled={!target.trim()}
          >
            <Ionicons name="scan" size={19} color="#ffffff" />
          </Pressable>
        </View>

        <View style={styles.reticleArea}>
          <Animated.View
            style={[
              styles.reticlePulse,
              {
                opacity: pulseOpacity,
                transform: [{ scale: pulseScale }]
              }
            ]}
          />
          <View style={styles.reticle}>
            <Ionicons name={stageKey === 'complete' ? 'checkmark' : 'locate'} size={36} color="#06d6a0" />
          </View>

          {detection ? (
            <View
              style={[
                styles.detectionBox,
                {
                  left: `${detection.boundingBox.x * 100}%`,
                  top: `${detection.boundingBox.y * 100}%`,
                  width: `${detection.boundingBox.width * 100}%`,
                  height: `${detection.boundingBox.height * 100}%`
                }
              ]}
            >
              <Text style={styles.detectionLabel}>
                {detection.name} {Math.round(detection.confidence * 100)}%
              </Text>
            </View>
          ) : null}
        </View>

        <View style={styles.bottomSheet}>
          <View style={styles.stageRow}>
            <View>
              <Text style={styles.stageLabel}>{stage.label}</Text>
              <Text style={styles.stageInstruction}>{stage.instruction}</Text>
            </View>
            <View style={styles.distanceBadge}>
              <Text style={styles.distanceValue}>
                {stage.distanceMeters === null ? '--' : `${stage.distanceMeters.toFixed(1)}m`}
              </Text>
              <Text style={styles.distanceLabel}>distance</Text>
            </View>
          </View>

          <View style={styles.progressTrack}>
            <View style={[styles.progressFill, { width: `${Math.max(8, (stageIndex / (flow.length - 1)) * 100)}%` }]} />
          </View>

          <View style={styles.guidanceRow}>
            <InfoBlock icon="compass-outline" label="Turn" value={`${stage.turnDegrees}°`} />
            <InfoBlock icon="analytics-outline" label="Confidence" value={`${Math.round(stage.confidence * 100)}%`} />
            <InfoBlock icon="phone-portrait-outline" label="Mode" value={Platform.OS === 'ios' ? 'iOS' : 'Android'} />
          </View>

          <View style={styles.mapAndRing}>
            <MiniMap stageKey={stageKey} detection={detection} />
            <HapticGrid pattern={getHapticPattern(stageKey)} />
          </View>

          <View style={styles.actions}>
            <Pressable style={styles.secondaryButton} onPress={resetScan}>
              <Text style={styles.secondaryButtonText}>Reset</Text>
            </Pressable>
            <Pressable
              style={[
                styles.primaryButtonSmall,
                stageKey === 'idle' && !target.trim() && { opacity: 0.38 }
              ]}
              onPress={stageKey === 'idle' ? (target.trim() ? startScan : null) : nextStage}
              disabled={stageKey === 'idle' && !target.trim()}
            >
              <Text style={styles.primaryButtonText}>
                {stageKey === 'idle' ? 'Start scan' : stageKey === 'complete' ? 'Done' : 'Continue'}
              </Text>
            </Pressable>
          </View>
        </View>
      </SafeAreaView>
    </View>
  );
}

export default function App() {
  return (
    <AppErrorBoundary>
      <AppContent />
    </AppErrorBoundary>
  );
}

function InfoBlock({ icon, label, value }) {
  return (
    <View style={styles.infoBlock}>
      <Ionicons name={icon} size={18} color="#06d6a0" />
      <Text style={styles.infoLabel}>{label}</Text>
      <Text style={styles.infoValue}>{value}</Text>
    </View>
  );
}

function MiniMap({ stageKey, detection }) {
  const targetX = detection ? 245 : 205;
  const targetY = detection ? 74 : 108;

  return (
    <View style={styles.miniMap}>
      <Text style={styles.panelTitle}>Spatial map</Text>
      <Svg width="100%" height="132" viewBox="0 0 300 132">
        <Rect x="10" y="10" width="280" height="112" rx="12" fill="#e8f1ef" />
        <Rect x="52" y="34" width="62" height="34" rx="6" fill="#ffffff" opacity="0.95" />
        <Rect x="140" y="74" width="44" height="30" rx="6" fill="#ef476f" opacity="0.9" />
        <Rect x="214" y="42" width="54" height="28" rx="6" fill="#ffffff" opacity="0.95" />
        <Line x1="68" y1="106" x2={targetX} y2={targetY} stroke="#118ab2" strokeWidth="5" strokeLinecap="round" />
        <Circle cx="68" cy="106" r="12" fill="#118ab2" />
        <Circle cx={targetX} cy={targetY} r={stageKey === 'scanning' ? 8 : 12} fill="#06d6a0" />
      </Svg>
    </View>
  );
}

function HapticGrid({ pattern }) {
  return (
    <View style={styles.hapticPanel}>
      <Text style={styles.panelTitle}>Haptics</Text>
      <View style={styles.hapticGrid}>
        {pattern.map((strength, index) => (
          <View
            key={index}
            style={[
              styles.hapticDot,
              {
                opacity: 0.22 + strength * 0.72,
                transform: [{ scale: 0.72 + strength * 0.58 }]
              }
            ]}
          />
        ))}
      </View>
    </View>
  );
}

async function runStageHaptics(stageKey) {
  if (stageKey === 'idle') return;

  if (stageKey === 'targetLocked') {
    await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    return;
  }

  if (stageKey === 'complete') {
    await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    return;
  }

  const impact =
    stageKey === 'orienting' || stageKey === 'walking'
      ? Haptics.ImpactFeedbackStyle.Medium
      : Haptics.ImpactFeedbackStyle.Light;

  await Haptics.impactAsync(impact);
}

const styles = StyleSheet.create({
  app: {
    flex: 1,
    backgroundColor: '#101917'
  },
  camera: {
    ...StyleSheet.absoluteFillObject
  },
  cameraShade: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(5, 11, 10, 0.24)'
  },
  overlay: {
    flex: 1,
    paddingHorizontal: 18
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: 10
  },
  kicker: {
    color: '#06d6a0',
    fontSize: 13,
    fontWeight: '800',
    textTransform: 'uppercase'
  },
  heading: {
    color: '#ffffff',
    fontSize: 34,
    fontWeight: '900',
    letterSpacing: -0.3
  },
  sensorPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    height: 38,
    borderRadius: 19,
    backgroundColor: 'rgba(255,255,255,0.14)'
  },
  sensorPillText: {
    color: '#ffffff',
    fontWeight: '800'
  },
  targetInput: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    minHeight: 54,
    marginTop: 14,
    paddingHorizontal: 12,
    borderRadius: 16,
    backgroundColor: 'rgba(255,255,255,0.92)'
  },
  input: {
    flex: 1,
    color: '#101917',
    fontSize: 17,
    fontWeight: '800'
  },
  scanButton: {
    alignItems: 'center',
    justifyContent: 'center',
    width: 38,
    height: 38,
    borderRadius: 13,
    backgroundColor: '#101917'
  },
  reticleArea: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center'
  },
  reticlePulse: {
    position: 'absolute',
    width: 154,
    height: 154,
    borderRadius: 77,
    borderWidth: 3,
    borderColor: '#06d6a0'
  },
  reticle: {
    alignItems: 'center',
    justifyContent: 'center',
    width: 82,
    height: 82,
    borderRadius: 41,
    borderWidth: 2,
    borderColor: '#06d6a0',
    backgroundColor: 'rgba(16,25,23,0.78)'
  },
  detectionBox: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#06d6a0',
    borderRadius: 12,
    backgroundColor: 'rgba(6,214,160,0.12)'
  },
  detectionLabel: {
    position: 'absolute',
    top: -30,
    left: 0,
    paddingHorizontal: 9,
    paddingVertical: 5,
    borderRadius: 8,
    overflow: 'hidden',
    color: '#101917',
    backgroundColor: '#06d6a0',
    fontSize: 12,
    fontWeight: '900',
    textTransform: 'capitalize'
  },
  bottomSheet: {
    marginBottom: 14,
    padding: 16,
    borderRadius: 24,
    backgroundColor: 'rgba(255,255,255,0.95)'
  },
  stageRow: {
    flexDirection: 'row',
    gap: 14,
    justifyContent: 'space-between'
  },
  stageLabel: {
    color: '#101917',
    fontSize: 23,
    fontWeight: '900'
  },
  stageInstruction: {
    maxWidth: 235,
    marginTop: 4,
    color: '#52615e',
    fontSize: 14,
    lineHeight: 19
  },
  distanceBadge: {
    alignItems: 'center',
    justifyContent: 'center',
    width: 86,
    height: 74,
    borderRadius: 18,
    backgroundColor: '#e8f7f3'
  },
  distanceValue: {
    color: '#101917',
    fontSize: 20,
    fontWeight: '900'
  },
  distanceLabel: {
    color: '#65726f',
    fontSize: 11,
    fontWeight: '800',
    textTransform: 'uppercase'
  },
  progressTrack: {
    overflow: 'hidden',
    height: 8,
    marginTop: 14,
    borderRadius: 8,
    backgroundColor: '#e3eae8'
  },
  progressFill: {
    height: '100%',
    borderRadius: 8,
    backgroundColor: '#06d6a0'
  },
  guidanceRow: {
    flexDirection: 'row',
    gap: 9,
    marginTop: 12
  },
  infoBlock: {
    flex: 1,
    minHeight: 78,
    padding: 10,
    borderRadius: 14,
    backgroundColor: '#101917'
  },
  infoLabel: {
    marginTop: 6,
    color: '#a9b7b4',
    fontSize: 11,
    fontWeight: '800',
    textTransform: 'uppercase'
  },
  infoValue: {
    marginTop: 2,
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '900'
  },
  mapAndRing: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 12
  },
  miniMap: {
    flex: 1.4,
    padding: 10,
    borderRadius: 16,
    backgroundColor: '#f2f6f5'
  },
  hapticPanel: {
    width: 116,
    padding: 10,
    borderRadius: 16,
    backgroundColor: '#f2f6f5'
  },
  panelTitle: {
    marginBottom: 5,
    color: '#53615e',
    fontSize: 11,
    fontWeight: '900',
    textTransform: 'uppercase'
  },
  hapticGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 7
  },
  hapticDot: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#06d6a0'
  },
  actions: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 13
  },
  secondaryButton: {
    alignItems: 'center',
    justifyContent: 'center',
    flex: 0.8,
    minHeight: 50,
    borderRadius: 15,
    backgroundColor: '#e7eeec'
  },
  secondaryButtonText: {
    color: '#101917',
    fontWeight: '900'
  },
  primaryButtonSmall: {
    alignItems: 'center',
    justifyContent: 'center',
    flex: 1.2,
    minHeight: 50,
    borderRadius: 15,
    backgroundColor: '#118ab2'
  },
  primaryButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '900'
  },
  permissionScreen: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
    padding: 26,
    backgroundColor: '#101917'
  },
  permissionTitle: {
    color: '#ffffff',
    fontSize: 27,
    fontWeight: '900',
    textAlign: 'center'
  },
  permissionText: {
    color: '#aab7b4',
    fontSize: 16,
    lineHeight: 23,
    textAlign: 'center'
  },
  primaryButton: {
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 54,
    minWidth: 190,
    paddingHorizontal: 18,
    borderRadius: 16,
    backgroundColor: '#118ab2'
  }
});
