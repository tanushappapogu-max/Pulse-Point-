export const guidanceStages = {
  idle: {
    label: 'Ready',
    instruction: 'Enter an object and start scanning.',
    distanceMeters: null,
    turnDegrees: 0,
    confidence: 0
  },
  scanning: {
    label: 'Scanning room',
    instruction: 'Pan slowly left to right so the camera can build context.',
    distanceMeters: null,
    turnDegrees: 0,
    confidence: 0.22
  },
  targetLocked: {
    label: 'Target found',
    instruction: 'Object recognized. Hold the phone steady.',
    distanceMeters: 2.6,
    turnDegrees: 42,
    confidence: 0.78
  },
  orienting: {
    label: 'Turn right',
    instruction: 'Rotate your body until the center haptic pulse triggers.',
    distanceMeters: 2.3,
    turnDegrees: 22,
    confidence: 0.84
  },
  walking: {
    label: 'Move forward',
    instruction: 'Walk forward slowly. Haptics will pull left or right if you drift.',
    distanceMeters: 1.1,
    turnDegrees: -8,
    confidence: 0.88
  },
  reaching: {
    label: 'Reach zone',
    instruction: 'Stop walking. Reach forward and slightly down.',
    distanceMeters: 0.24,
    turnDegrees: 0,
    confidence: 0.93
  },
  complete: {
    label: 'Assistance complete',
    instruction: 'You are close enough to continue without navigation support.',
    distanceMeters: 0.12,
    turnDegrees: 0,
    confidence: 0.96
  }
};

export function createDetectionResult(targetName) {
  return {
    id: `${targetName.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`,
    name: targetName.trim() || 'target object',
    confidence: 0.82,
    position: {
      x: 0.72,
      y: 0.38,
      zMeters: 2.6
    },
    boundingBox: {
      x: 0.62,
      y: 0.28,
      width: 0.24,
      height: 0.18
    }
  };
}

export function getHapticPattern(stageKey) {
  const patterns = {
    idle: [0, 0, 0, 0, 0.2, 0, 0, 0, 0],
    scanning: [0.8, 0.5, 0.2, 0.8, 0.5, 0.2, 0.8, 0.5, 0.2],
    targetLocked: [0.7, 0.7, 0.7, 0.7, 1, 0.7, 0.7, 0.7, 0.7],
    orienting: [0.2, 0.4, 1, 0.2, 0.4, 1, 0.2, 0.4, 1],
    walking: [0.3, 0.5, 0.3, 0.5, 1, 0.5, 0.3, 0.5, 0.3],
    reaching: [0.2, 0.5, 0.2, 0.5, 1, 0.5, 0.2, 0.5, 0.2],
    complete: [0, 0.4, 0, 0.4, 0.9, 0.4, 0, 0.4, 0]
  };

  return patterns[stageKey] ?? patterns.idle;
}
