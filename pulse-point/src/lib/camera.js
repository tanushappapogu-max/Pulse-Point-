// Camera helpers: pick the widest back camera, set min zoom, toggle torch,
// capture a JPEG of the current frame.

export async function getWideCameraStream() {
  const base = {
    video: {
      facingMode: { ideal: 'environment' },
      width: { ideal: 1920 },
      height: { ideal: 1080 },
      aspectRatio: { ideal: 16 / 9 },
    },
    audio: false,
  };
  const first = await navigator.mediaDevices.getUserMedia(base);
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter(d => d.kind === 'videoinput');
  const backCams = cameras.filter(d => /back|rear|environment/i.test(d.label));
  const wide = backCams.find(d => /ultra|wide/i.test(d.label))
    || backCams[0]
    || cameras.find(d => /ultra|wide/i.test(d.label));
  if (!wide) return first;
  first.getTracks().forEach(t => t.stop());
  return navigator.mediaDevices.getUserMedia({
    video: {
      deviceId: { exact: wide.deviceId },
      width: { ideal: 1920 },
      height: { ideal: 1080 },
    },
    audio: false,
  });
}

export async function setWidestZoom(stream) {
  const track = stream?.getVideoTracks()[0];
  const caps = track?.getCapabilities?.();
  if (!caps?.zoom) return;
  try {
    await track.applyConstraints({ advanced: [{ zoom: caps.zoom.min }] });
  } catch {
    // some platforms reject mid-stream zoom — fine to ignore
  }
}

export function hasTorchSupport(stream) {
  const track = stream?.getVideoTracks()[0];
  return Boolean(track?.getCapabilities?.()?.torch);
}

export async function setTorch(stream, on) {
  const track = stream?.getVideoTracks()[0];
  if (!track) return false;
  try {
    await track.applyConstraints({ advanced: [{ torch: !!on }] });
    return true;
  } catch {
    return false;
  }
}

export function captureJpeg(video, quality = 0.82) {
  if (!video) return null;
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', quality);
}

export function stopStream(stream) {
  stream?.getTracks().forEach(t => t.stop());
}
