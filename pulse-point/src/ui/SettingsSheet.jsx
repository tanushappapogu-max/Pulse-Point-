import React, { useEffect, useRef } from 'react';
import { X } from 'lucide-react';

// Settings sheet — bottom-sheet modal for accessibility-relevant prefs.
// Keyboard-trapped while open (Esc to dismiss, Tab cycles within), focuses the
// close button on open, restores focus to the opener on close.

export default function SettingsSheet({
  open,
  onClose,
  settings,
  onChange,
  hapticsAvailable,
  speechAvailable,
}) {
  const closeBtnRef = useRef(null);
  const sheetRef = useRef(null);
  const lastFocusRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    lastFocusRef.current = document.activeElement;
    closeBtnRef.current?.focus();

    const onKey = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key === 'Tab' && sheetRef.current) {
        // basic focus trap
        const focusable = sheetRef.current.querySelectorAll(
          'button, input, [tabindex]:not([tabindex="-1"])'
        );
        if (!focusable.length) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault(); last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault(); first.focus();
        }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
      lastFocusRef.current?.focus?.();
    };
  }, [open, onClose]);

  if (!open) return null;

  const set = (k, v) => onChange({ ...settings, [k]: v });

  return (
    <>
      <div
        className="settings-overlay"
        onClick={onClose}
        aria-hidden="true"
      />
      <div
        className="settings-sheet"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        ref={sheetRef}
      >
        <header className="settings-header">
          <h2 id="settings-title">Settings</h2>
          <button
            ref={closeBtnRef}
            type="button"
            className="settings-close"
            onClick={onClose}
            aria-label="Close settings"
          >
            <X size={22} />
          </button>
        </header>

        <div className="settings-body">
          <Toggle
            id="haptics-toggle"
            label="Haptic feedback"
            hint={hapticsAvailable ? 'Phone vibration patterns for direction and distance.' : 'Not supported on this device.'}
            checked={settings.haptics && hapticsAvailable}
            disabled={!hapticsAvailable}
            onChange={v => set('haptics', v)}
          />

          <Toggle
            id="speech-toggle"
            label="Spoken guidance"
            hint={speechAvailable ? 'Reads turn directions and distance aloud.' : 'Not supported on this device.'}
            checked={settings.speech && speechAvailable}
            disabled={!speechAvailable}
            onChange={v => set('speech', v)}
          />

          {settings.speech && speechAvailable && (
            <div className="settings-row">
              <label htmlFor="speech-rate">Speech rate</label>
              <input
                id="speech-rate"
                type="range"
                min="0.7"
                max="1.8"
                step="0.05"
                value={settings.speechRate}
                onChange={e => set('speechRate', parseFloat(e.target.value))}
              />
              <span className="settings-row-value" aria-live="polite">
                {settings.speechRate.toFixed(2)}×
              </span>
            </div>
          )}

          <div className="settings-row">
            <span className="settings-row-label">Sensitivity</span>
            <div className="settings-segmented" role="radiogroup" aria-label="Guidance sensitivity">
              {['gentle', 'medium', 'sharp'].map(level => (
                <button
                  key={level}
                  type="button"
                  role="radio"
                  aria-checked={settings.sensitivity === level}
                  className={`settings-seg${settings.sensitivity === level ? ' active' : ''}`}
                  onClick={() => set('sensitivity', level)}
                >
                  {level}
                </button>
              ))}
            </div>
          </div>

          <Toggle
            id="boxes-toggle"
            label="Show all detections"
            hint="Faintly draw every detected object, not just the target."
            checked={settings.showAllBoxes}
            onChange={v => set('showAllBoxes', v)}
          />
        </div>

        <footer className="settings-footer">
          <p className="settings-help">
            Pulse Point runs a CNN on-device to detect and locate objects in real time.
            Audio and vibration guide you toward them.
          </p>
        </footer>
      </div>
    </>
  );
}

function Toggle({ id, label, hint, checked, disabled, onChange }) {
  return (
    <div className={`settings-row${disabled ? ' disabled' : ''}`}>
      <label htmlFor={id} className="settings-row-text">
        <span className="settings-row-label">{label}</span>
        {hint && <span className="settings-row-hint">{hint}</span>}
      </label>
      <input
        id={id}
        type="checkbox"
        role="switch"
        aria-checked={checked}
        checked={checked}
        disabled={disabled}
        onChange={e => onChange(e.target.checked)}
        className="settings-switch"
      />
    </div>
  );
}
