import React from 'react';

// Last line of defense — if a render throws, we don't want a blank screen.
// The user (likely blind/low-vision) needs to know the app crashed and how
// to recover. The reload button is keyboard-focusable and has clear copy.

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // eslint-disable-next-line no-console
    console.error('Pulse Point crashed:', error, info);
  }

  handleReload = () => {
    if (typeof window !== 'undefined') window.location.reload();
  };

  render() {
    if (!this.state.error) return this.props.children;

    return (
      <main className="crash-screen" role="alert">
        <h1>Pulse Point hit an error</h1>
        <p>
          The app crashed. Your camera and data are safe. Reload to try again.
        </p>
        <p className="crash-detail">
          {this.state.error?.message || 'Unknown error'}
        </p>
        <button type="button" className="crash-reload" onClick={this.handleReload}>
          Reload
        </button>
      </main>
    );
  }
}
