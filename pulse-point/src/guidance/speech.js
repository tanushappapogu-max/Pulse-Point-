// Spoken guidance via the browser's SpeechSynthesis API.
//
// For users who can't reliably feel haptics (iPhone Safari has weak Vibration
// API support, certain Android phones suppress short vibrations) speech is the
// crucial alternative output channel. Even with haptics, speech adds the
// "what" + "how far" context that vibration can't carry.
//
// Behavior:
//   - Queues utterances to prevent overlapping by waiting for current speech to finish.
//   - Only interrupts for truly urgent signals (reach, closer).
//   - Tracks speaking state and plays pending speech when current finishes.
//   - Throttles repetition of the same phrase to MIN_GAP_MS unless `urgent`.
//   - Prefers a quality voice if one is available.

const MIN_GAP_MS = 2000;
const URGENT_GAP_MS = 600;

export class Speaker {
  constructor() {
    this.lastSaid = '';
    this.lastTime = 0;
    this.enabled = false;
    this.rate = 1.15;
    this.voice = null;
    this.isSpeaking = false;
    this.pendingText = null;
    this.pendingUrgent = false;
    this._tryPickVoice();
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      // voice list loads async on first call in most browsers
      window.speechSynthesis.addEventListener?.('voiceschanged', () => this._tryPickVoice());
    }
  }

  isAvailable() {
    return typeof window !== 'undefined' && 'speechSynthesis' in window;
  }

  setEnabled(on) {
    this.enabled = on && this.isAvailable();
    if (!on) this.cancel();
  }

  setRate(r) {
    this.rate = Math.max(0.5, Math.min(2.5, r || 1));
  }

  cancel() {
    if (this.isAvailable()) window.speechSynthesis.cancel();
    this.isSpeaking = false;
    this.pendingText = null;
  }

  /**
   * Speak a short phrase. Queues if currently speaking.
   * @param {string} text
   * @param {{urgent?: boolean, force?: boolean}} opts
   */
  say(text, { urgent = false, force = false } = {}) {
    if (!this.enabled || !this.isAvailable() || !text) return;
    const now = Date.now();
    const gap = urgent ? URGENT_GAP_MS : MIN_GAP_MS;
    
    // Don't repeat the same phrase too soon
    if (!force && text === this.lastSaid && now - this.lastTime < gap) return;
    
    // If currently speaking, queue instead of interrupting (except for forced urgent)
    if (this.isSpeaking && !(force && urgent)) {
      this.pendingText = text;
      this.pendingUrgent = urgent;
      return;
    }
    
    // For forced urgent, cancel immediately
    if (force && urgent) {
      window.speechSynthesis.cancel();
      this.isSpeaking = false;
      this.pendingText = null;
    }

    const u = new SpeechSynthesisUtterance(text);
    u.rate = this.rate;
    u.pitch = 1;
    u.volume = 1;
    if (this.voice) u.voice = this.voice;
    
    u.onstart = () => { this.isSpeaking = true; };
    u.onend = () => { 
      this.isSpeaking = false;
      this._playPending();
    };
    u.onerror = () => { 
      this.isSpeaking = false;
      this._playPending();
    };
    
    window.speechSynthesis.speak(u);
    this.lastSaid = text;
    this.lastTime = now;
  }

  _playPending() {
    if (this.pendingText) {
      const text = this.pendingText;
      const urgent = this.pendingUrgent;
      this.pendingText = null;
      this.pendingUrgent = false;
      this.say(text, { urgent });
    }
  }

  _tryPickVoice() {
    if (!this.isAvailable()) return;
    const voices = window.speechSynthesis.getVoices() || [];
    if (!voices.length) return;
    // Prefer en-US, then en-GB, then any en-*
    const preferred =
      voices.find(v => /en[-_]US/i.test(v.lang) && /Google|Microsoft|Samantha|Alex/i.test(v.name))
      || voices.find(v => /en[-_]US/i.test(v.lang))
      || voices.find(v => /en[-_]GB/i.test(v.lang))
      || voices.find(v => /^en/i.test(v.lang))
      || null;
    this.voice = preferred;
  }
}
