import { useEffect, useRef } from 'react';

const CELL_SIZE = 26;
const COLS = 4;
const ROWS = 2;
const NUM_CELLS = COLS * ROWS;

function renderCell(ctx, W, H, cellIdx, active, conf, time) {
  const img = ctx.createImageData(W, H);
  const seed = cellIdx * 1.618 + 0.3;

  for (let py = 0; py < H; py++) {
    for (let px = 0; px < W; px++) {
      const i = (py * W + px) * 4;

      // Gabor-like noise: different orientation per cell
      const angle = (cellIdx / NUM_CELLS) * Math.PI;
      const u = px * Math.cos(angle) + py * Math.sin(angle);
      const v = -px * Math.sin(angle) + py * Math.cos(angle);
      const spatial = Math.sin(u * (0.6 + seed * 0.3) + time * 0.4 * (seed + 0.5))
                    * Math.cos(v * (0.5 + seed * 0.2) + time * 0.25 * seed);
      const n = (spatial + 1) / 2; // 0..1

      if (active) {
        const intensity = n * conf * 1.2;
        img.data[i]   = Math.min(255, intensity * 18);
        img.data[i+1] = Math.min(255, intensity * 255);
        img.data[i+2] = Math.min(255, intensity * 80);
        img.data[i+3] = Math.min(255, 60 + intensity * 200);
      } else {
        const v2 = n * 0.25;
        img.data[i] = img.data[i+1] = img.data[i+2] = Math.round(v2 * 50);
        img.data[i+3] = 140;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}

export default function FeatureGrid({ active, confidence = 0 }) {
  const refs  = useRef([]);
  const rafId = useRef(null);
  const conf  = Math.max(0.05, confidence / 100);

  useEffect(() => {
    const canvases = refs.current.filter(Boolean);
    let t = 0;

    function frame() {
      t += 0.025;
      canvases.forEach((c, i) => {
        const ctx = c.getContext('2d');
        renderCell(ctx, CELL_SIZE, CELL_SIZE, i, active, conf, t);
      });
      rafId.current = requestAnimationFrame(frame);
    }

    rafId.current = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(rafId.current);
  }, [active, conf]);

  return (
    <div className="feat-grid" aria-hidden="true">
      <div className="feat-grid-header">
        <span className="feat-grid-title">FEAT MAPS</span>
        <span className="feat-grid-dim">L3 · 128ch</span>
      </div>
      <div className="feat-grid-cells">
        {Array.from({ length: NUM_CELLS }, (_, i) => (
          <canvas
            key={i}
            ref={el => refs.current[i] = el}
            width={CELL_SIZE}
            height={CELL_SIZE}
            className={`feat-cell${active ? ' feat-cell-active' : ''}`}
          />
        ))}
      </div>
    </div>
  );
}
