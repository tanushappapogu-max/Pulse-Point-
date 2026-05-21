// Lightweight client-side error reporter.
// ErrorBoundary POSTs here when the React tree crashes so live-demo failures
// are visible in Vercel's log viewer without needing browser DevTools open.

export const config = {
  runtime: 'nodejs',
  maxDuration: 5,
};

export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');

  if (req.method === 'OPTIONS') {
    res.setHeader('Allow', 'POST, OPTIONS');
    return res.status(204).end();
  }
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { message, stack, componentStack } = req.body || {};

  // Vercel surfaces console.error lines in the Functions log tab.
  console.error('[client-crash]', {
    message: String(message || '').slice(0, 300),
    stack: String(stack || '').slice(0, 500),
    componentStack: String(componentStack || '').slice(0, 500),
    userAgent: req.headers['user-agent'],
    timestamp: new Date().toISOString(),
  });

  return res.status(200).json({ ok: true });
}
