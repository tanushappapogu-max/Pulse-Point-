import React, { useEffect, useRef, useState } from 'react';

// Screen-reader-only assertive live region. The visible status strip uses
// aria-live="polite" which queues behind anything else the screen reader is
// saying. Direction changes (left → right → reach) are time-critical for a
// blind user navigating space, so they need ASSERTIVE delivery in their own
// region with full sentences ("Turn left, one step away") instead of "left".
//
// Why a separate component: assertive regions interrupt. We want exactly one
// at a time, sized to one short sentence, throttled by content-equality so
// repeated identical phrases don't re-interrupt the SR every render.

const MIN_REPEAT_MS = 1000;

export default function Announcer({ message, urgent }) {
  const [text, setText] = useState('');
  const lastTextRef = useRef('');
  const lastTimeRef = useRef(0);

  useEffect(() => {
    if (!message) return;
    const now = Date.now();
    if (
      message === lastTextRef.current &&
      now - lastTimeRef.current < MIN_REPEAT_MS
    ) return;
    lastTextRef.current = message;
    lastTimeRef.current = now;
    // Re-emit even when text is identical by clearing first, so screen readers
    // notice the change even on repeats after the throttle.
    setText('');
    const id = requestAnimationFrame(() => setText(message));
    return () => cancelAnimationFrame(id);
  }, [message]);

  return (
    <div
      className="sr-only"
      role={urgent ? 'alert' : 'status'}
      aria-live={urgent ? 'assertive' : 'polite'}
      aria-atomic="true"
    >
      {text}
    </div>
  );
}
