// 2D guidance computation: takes a target box + frame size and returns the
// haptic signal, status string, and a full-sentence description for speech /
// screen readers.
//
// The frame is divided into a center sweet spot (centered horizontally AND
// vertically). Outside the spot, we pick the dominant correction axis: if the
// horizontal error is bigger than the vertical, we say "turn left/right",
// otherwise "tilt up/down". Picking the dominant axis (instead of always
// emitting both) keeps the haptic vocabulary discriminable.

import { estimateDistanceMeters, describeDistanceMeters, describeDistanceArea, formatMeters } from '../detection/distance.js';

// Center sweet spot
const H_LEFT = 0.40, H_RIGHT = 0.60;
const V_TOP  = 0.38, V_BOTTOM = 0.62;

// "very close" thresholds (either real meters or fallback area)
const REACH_METERS = 0.55;
const REACH_AREA = 0.20;

// growth threshold for "getting closer" between consecutive frames
const NEAR_GROWTH = 1.05;

/**
 * @param {{bbox:[number,number,number,number], class:string, label?:string, fromAi?:boolean}} match
 * @param {{width:number,height:number}} frame
 * @param {number} prevArea
 * @returns {{
 *   signal: 'reach'|'closer'|'locked'|'left'|'right'|'up'|'down'|'looking',
 *   status: string,
 *   direction: string,
 *   distance: string|null,
 *   distanceMeters: number|null,
 *   area: number,
 *   cx: number,
 *   cy: number,
 *   sentence: string
 * }}
 */
export function computeGuidance(match, frame, prevArea) {
  const [x, y, w, h] = match.bbox;
  const cx = (x + w / 2) / frame.width;
  const cy = (y + h / 2) / frame.height;
  const area = (w * h) / (frame.width * frame.height);

  const labelForDistance = (match.class || match.label || '').toLowerCase();
  const meters = estimateDistanceMeters(labelForDistance, w, frame.width);
  const distanceText = meters != null
    ? describeDistanceMeters(meters)
    : describeDistanceArea(match.bbox, frame);

  const inH = cx >= H_LEFT && cx <= H_RIGHT;
  const inV = cy >= V_TOP && cy <= V_BOTTOM;
  const veryClose = (meters != null && meters < REACH_METERS) || area > REACH_AREA;
  const gettingNear = prevArea > 0 && area > prevArea * NEAR_GROWTH;

  let signal, status, direction;

  if (inH && inV && veryClose) {
    signal = 'reach';
    status = 'reach';
    direction = 'right in front of you';
  } else if (inH && inV) {
    signal = gettingNear ? 'closer' : 'locked';
    status = gettingNear ? 'closer' : 'locked';
    direction = 'centered';
  } else {
    // Outside the sweet spot — pick the dominant correction axis.
    const horizErr = inH ? 0 : Math.abs(cx - 0.5);
    const vertErr  = inV ? 0 : Math.abs(cy - 0.5);
    if (horizErr >= vertErr) {
      signal = cx < 0.5 ? 'left' : 'right';
      status = signal;
      direction = signal === 'left' ? 'turn left' : 'turn right';
    } else {
      signal = cy < 0.5 ? 'up' : 'down';
      status = signal;
      direction = signal === 'up' ? 'tilt up' : 'tilt down';
    }
  }

  return {
    signal,
    status,
    direction,
    distance: distanceText,
    distanceMeters: meters,
    distanceFormatted: formatMeters(meters),
    area,
    cx,
    cy,
    sentence: buildSentence(match, signal, distanceText, meters),
    speechPhrase: buildSpeechPhrase(signal, meters),
  };
}

function buildSentence(match, signal, distanceText, meters) {
  const name = (match.displayClass || match.class || match.label || 'target').toString();
  const dist = meters != null ? `${meters < 1 ? Math.round(meters * 100) + ' centimeters' : meters.toFixed(1) + ' meters'}` : distanceText;

  switch (signal) {
    case 'reach':   return `Reach now. ${capitalize(name)} is right in front of you.`;
    case 'closer':  return `${capitalize(name)} is centered, getting closer, ${dist}.`;
    case 'locked':  return `${capitalize(name)} is centered, ${dist}.`;
    case 'left':    return `Turn left. ${capitalize(name)} ${dist}.`;
    case 'right':   return `Turn right. ${capitalize(name)} ${dist}.`;
    case 'up':      return `Tilt up. ${capitalize(name)} ${dist}.`;
    case 'down':    return `Tilt down. ${capitalize(name)} ${dist}.`;
    default:        return `${capitalize(name)} ${dist}.`;
  }
}

function buildSpeechPhrase(signal, meters) {
  // Generate very short phrases for speech synthesis to avoid overlaps.
  // Phrases are directional commands, not location descriptions.
  const distCommand = !meters ? '' : meters < 0.5 ? 'hold' : 'move closer';
  
  switch (signal) {
    case 'reach':   return 'Reach';
    case 'closer':  return 'Closer';
    case 'locked':  return distCommand || 'Set';
    case 'left':    return 'Left';
    case 'right':   return 'Right';
    case 'up':      return 'Up';
    case 'down':    return 'Down';
    default:        return 'Found';
  }
}

function capitalize(s) {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}
