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
  const cleanName = (targetName || '').trim();
  const seedText = cleanName || 'target';
  let seed = 2166136261;
  for (let i = 0; i < seedText.length; i++) {
    seed ^= seedText.charCodeAt(i);
    seed = Math.imul(seed, 16777619);
  }

  let salt = 0;
  const nextUnit = () => {
    let value = (seed >>> 0) + Math.imul(++salt, 0x9e3779b1);
    value ^= value >>> 16;
    value = Math.imul(value, 0x85ebca6b);
    value ^= value >>> 13;
    value = Math.imul(value, 0xc2b2ae35);
    value ^= value >>> 16;
    return (value >>> 0) / 0xffffffff;
  };

  const width = 0.18 + nextUnit() * 0.16;
  const height = 0.14 + nextUnit() * 0.18;
  const x = 0.06 + nextUnit() * (0.94 - width);
  const y = 0.08 + nextUnit() * (0.88 - height);
  const centerX = x + width / 2;
  const centerY = y + height / 2;

  return {
    id: `${(cleanName || 'target').toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`,
    name: cleanName || 'target object',
    confidence: 0.72 + nextUnit() * 0.22,
    position: {
      x: centerX,
      y: centerY,
      zMeters: 0.8 + nextUnit() * 2.2
    },
    boundingBox: {
      x,
      y,
      width,
      height
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
