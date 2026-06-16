import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // GitHub Pages serves from /Pulse-Point-/ — set base so assets resolve correctly
  base: process.env.NODE_ENV === 'production' ? '/Pulse-Point-/' : '/',
});
