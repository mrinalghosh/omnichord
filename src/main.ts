import {
  HandLandmarker,
  FilesetResolver,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { GRID, ROWS, COLS, chordAt, type Chord } from "./chords";
import {
  initAudio,
  pluck,
  setHeldChord,
  cycleWaveform,
  getWaveform,
  setVibrato,
} from "./audio";

const video = document.getElementById("video") as HTMLVideoElement;
const canvas = document.getElementById("overlay") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const startOverlay = document.getElementById("start-overlay")!;
const startBtn = document.getElementById("start-btn") as HTMLButtonElement;

let landmarker: HandLandmarker | null = null;
let lastVideoTime = -1;

const STRUM_BANDS = 16;
let lastStrumBand: number | null = null;
let lastStrumX: number | null = null;
let lastStrumT = 0;

// Heavy smoothing — feels like a damped follow rather than 1:1 tracking.
const SMOOTH = 0.28;
const HYSTERESIS = 0.22;

// Layout: chord grid takes the top GRID_FRACTION; strum strip is the bottom.
// Note: strum *triggers* don't require the hand to be in the strum strip — the
// strip is only a visual reference. The strum hand's x-coordinate is tracked
// wherever it is on screen.
const GRID_FRACTION = 0.6;

type Vec2 = { x: number; y: number };

// Per-role state. Identity is matched across frames by nearest-centroid so
// MediaPipe re-ordering can't swap chord/strum mid-play. Each role has a
// 150ms grace window so a single dropped detection frame doesn't release the
// pad or re-fire pinch.
type RoleState = {
  smoothed: Vec2 | null;
  lastCentroid: Vec2 | null;
  lastSeenAt: number;
};
const chord: RoleState = { smoothed: null, lastCentroid: null, lastSeenAt: 0 };
const strum: RoleState = { smoothed: null, lastCentroid: null, lastSeenAt: 0 };
const GRACE_MS = 150;

let currentChord: Chord = chordAt(0, 1);
let currentCell: { row: number; col: number } | null = { row: 0, col: 1 };
let heldCellKey: string | null = null;
let lastStrumNoteIdx: number | null = null;

function smooth(prev: Vec2 | null, next: Vec2): Vec2 {
  if (!prev) return next;
  return {
    x: prev.x + (next.x - prev.x) * SMOOTH,
    y: prev.y + (next.y - prev.y) * SMOOTH,
  };
}

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

async function setupCamera() {
  // Mobile browsers downscale anyway, and lower res keeps the GPU delegate
  // hitting frame rate. Desktop still gets the higher resolution.
  const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
  const stream = await navigator.mediaDevices.getUserMedia({
    video: isMobile
      ? { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
      : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise<void>((res) => {
    video.onloadedmetadata = () => {
      video.play();
      res();
    };
  });
}

async function setupLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  const modelAssetPath =
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
  try {
    landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
    });
  } catch {
    // Some mobile browsers (esp. iOS Safari) reject the GPU delegate.
    landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath, delegate: "CPU" },
      runningMode: "VIDEO",
      numHands: 2,
    });
  }
}

async function requestWakeLock() {
  try {
    await (navigator as any).wakeLock?.request("screen");
  } catch {
    // Non-fatal: phone may dim during play, but everything still works.
  }
}

function mirrorX(x: number): number {
  return 1 - x;
}

type Hand = { landmarks: Vec2[] };

function extractHands(result: HandLandmarkerResult): Hand[] {
  return result.landmarks.map((lm) => ({
    landmarks: lm.map((p) => ({ x: p.x, y: p.y })),
  }));
}

function handCentroid(hand: Hand): Vec2 {
  const w = hand.landmarks[0];
  const m = hand.landmarks[9];
  return { x: (w.x + m.x) / 2, y: (w.y + m.y) / 2 };
}

function indexTip(hand: Hand): Vec2 {
  return hand.landmarks[8];
}

// Pinch = thumb tip (4) close to index tip (8), normalized by the hand's
// bounding-box diagonal (rotation-robust — wrist→MCP collapses when the palm
// is edge-on to the camera).
function pinchAmount(hand: Hand): number {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of hand.landmarks) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const handSize = Math.hypot(maxX - minX, maxY - minY) || 0.0001;
  const thumb = hand.landmarks[4];
  const index = hand.landmarks[8];
  const d = Math.hypot(thumb.x - index.x, thumb.y - index.y);
  return d / handSize;
}

// Thresholds tuned for bbox-diagonal normalization (open hand ≈ 0.5–0.8,
// pinched ≈ 0.05–0.15).
const PINCH_ON = 0.18;
const PINCH_OFF = 0.32;
let chordPinched = false;

function chordCellFromDisplay(
  dx: number,
  dy: number,
  current: { row: number; col: number } | null
): { row: number; col: number } | null {
  if (dy > GRID_FRACTION) return null;
  const gx = dx * COLS;
  const gy = (dy / GRID_FRACTION) * ROWS;
  let col = Math.min(COLS - 1, Math.max(0, Math.floor(gx)));
  let row = Math.min(ROWS - 1, Math.max(0, Math.floor(gy)));
  if (current) {
    const fracX = gx - current.col;
    const fracY = gy - current.row;
    if (col !== current.col) {
      if (col < current.col && fracX > -HYSTERESIS) col = current.col;
      else if (col > current.col && fracX < 1 + HYSTERESIS) col = current.col;
    }
    if (row !== current.row) {
      if (row < current.row && fracY > -HYSTERESIS) row = current.row;
      else if (row > current.row && fracY < 1 + HYSTERESIS) row = current.row;
    }
  }
  return { row, col };
}

function strumBandFromX(dx: number, current: number | null): number {
  const gx = dx * STRUM_BANDS;
  let band = Math.min(STRUM_BANDS - 1, Math.max(0, Math.floor(gx)));
  if (current !== null && band !== current) {
    const frac = gx - current;
    if (band < current && frac > -HYSTERESIS) band = current;
    else if (band > current && frac < 1 + HYSTERESIS) band = current;
  }
  return band;
}

function drawGrid() {
  const w = canvas.width;
  const h = canvas.height;
  const gridH = h * GRID_FRACTION;
  const cellW = w / COLS;
  const cellH = gridH / ROWS;

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const x = c * cellW;
      const y = r * cellH;
      const isActive =
        currentCell && currentCell.row === r && currentCell.col === c;
      ctx.fillStyle = isActive
        ? "rgba(108, 92, 231, 0.45)"
        : "rgba(255,255,255,0.04)";
      ctx.fillRect(x + 4, y + 4, cellW - 8, cellH - 8);
      ctx.strokeStyle = isActive ? "#a29bfe" : "rgba(255,255,255,0.18)";
      ctx.lineWidth = isActive ? 3 : 1;
      ctx.strokeRect(x + 4, y + 4, cellW - 8, cellH - 8);

      ctx.fillStyle = isActive ? "#fff" : "rgba(255,255,255,0.7)";
      ctx.font = "600 36px -apple-system, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(GRID[r][c].label, x + cellW / 2, y + cellH / 2);
    }
  }

  const stripY = gridH;
  const stripH = h - gridH;
  ctx.fillStyle = "rgba(255, 234, 167, 0.04)";
  ctx.fillRect(0, stripY, w, stripH);
  for (let i = 1; i < STRUM_BANDS; i++) {
    const x = (i / STRUM_BANDS) * w;
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, stripY);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(255,255,255,0.25)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, stripY);
  ctx.lineTo(w, stripY);
  ctx.stroke();

  // Highlight the current strum band column for feedback.
  if (lastStrumBand !== null) {
    const x = (lastStrumBand / STRUM_BANDS) * w;
    ctx.fillStyle = "rgba(255, 234, 167, 0.12)";
    ctx.fillRect(x, stripY, w / STRUM_BANDS, stripH);
  }
}

function drawStrumDot(p: Vec2, vibrato: number) {
  const x = p.x * canvas.width;
  const y = p.y * canvas.height;
  // Pulsing halo whose size grows with vibrato depth.
  if (vibrato > 0.02) {
    const phase = (performance.now() / 1000) * (4.5 + vibrato * 3.5) * Math.PI * 2;
    const wobble = Math.sin(phase);
    const r = 18 + vibrato * (14 + wobble * 6);
    ctx.strokeStyle = `rgba(255, 234, 167, ${0.25 + vibrato * 0.4})`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  drawHandDot(p, "#ffeaa7", "strum");
}

function drawHandDot(p: Vec2, color: string, label?: string) {
  const x = p.x * canvas.width;
  const y = p.y * canvas.height;
  ctx.shadowColor = color;
  ctx.shadowBlur = 18;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, 14, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;
  if (label) {
    ctx.fillStyle = "#fff";
    ctx.font = "12px -apple-system, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(label, x, y - 22);
  }
}

// Draw one period of the given waveform as a glyph centered on p. Used as the
// chord-hand cursor so the user always sees which oscillator the strum will use.
function drawWaveformGlyph(
  p: Vec2,
  wave: "triangle" | "sine" | "sawtooth" | "square",
  pinched: boolean
) {
  const cx = p.x * canvas.width;
  const cy = p.y * canvas.height;
  const halfW = 22; // glyph half-width
  const halfH = 10; // glyph half-height
  const color = pinched ? "#fd79a8" : "#a29bfe";

  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = 16;
  ctx.strokeStyle = color;
  ctx.lineWidth = 3.5;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();

  const x0 = cx - halfW;
  const x1 = cx + halfW;

  if (wave === "sine") {
    const steps = 48;
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const x = x0 + t * (x1 - x0);
      const y = cy - Math.sin(t * Math.PI * 2) * halfH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
  } else if (wave === "triangle") {
    // Up to peak at 1/4, down through 3/4, back to baseline.
    ctx.moveTo(x0, cy);
    ctx.lineTo(x0 + halfW * 0.5, cy - halfH);
    ctx.lineTo(cx + halfW * 0.5, cy + halfH);
    ctx.lineTo(x1, cy);
  } else if (wave === "sawtooth") {
    // Ramp up over the full period, vertical drop at the end.
    ctx.moveTo(x0, cy + halfH);
    ctx.lineTo(x1 - 0.001, cy - halfH);
    ctx.lineTo(x1, cy + halfH);
  } else {
    // square: half-period high, half-period low.
    ctx.moveTo(x0, cy + halfH);
    ctx.lineTo(x0, cy - halfH);
    ctx.lineTo(cx, cy - halfH);
    ctx.lineTo(cx, cy + halfH);
    ctx.lineTo(x1, cy + halfH);
    ctx.lineTo(x1, cy - halfH);
  }
  ctx.stroke();
  ctx.restore();
}

function handleStrum(dx: number) {
  const band = strumBandFromX(dx, lastStrumBand);
  const now = performance.now();
  if (lastStrumBand === null) {
    lastStrumBand = band;
    lastStrumX = dx;
    lastStrumT = now;
    return;
  }
  if (band !== lastStrumBand) {
    const dir = band > lastStrumBand ? 1 : -1;
    const dt = Math.max(1, now - lastStrumT);
    const dxAbs = Math.abs(dx - (lastStrumX ?? dx));
    const speed = dxAbs / dt;
    // Cap velocity below 1 to leave headroom for the chorus + delay + reverb
    // chain on aggressive strums.
    const velocity = Math.min(0.85, 0.4 + speed * 50);

    const notes = currentChord.strum;
    let cursor = lastStrumBand;
    while (cursor !== band) {
      cursor += dir;
      const noteIdx = Math.min(
        notes.length - 1,
        Math.max(0, Math.floor((cursor / STRUM_BANDS) * notes.length))
      );
      // Skip duplicate-pitch plucks: with 16 bands across ~9 chord tones,
      // adjacent bands often resolve to the same note index.
      if (noteIdx !== lastStrumNoteIdx) {
        pluck(notes[noteIdx], velocity);
        lastStrumNoteIdx = noteIdx;
      }
    }
    lastStrumBand = band;
    lastStrumX = dx;
    lastStrumT = now;
  }
}

function frame() {
  if (!landmarker || video.readyState < 2) {
    requestAnimationFrame(frame);
    return;
  }
  let result: HandLandmarkerResult | null = null;
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    result = landmarker.detectForVideo(video, performance.now());
  }

  const now = performance.now();
  const chordFresh = chord.lastSeenAt > 0 && now - chord.lastSeenAt < GRACE_MS;
  const strumFresh = strum.lastSeenAt > 0 && now - strum.lastSeenAt < GRACE_MS;

  // Update target positions only on fresh detections; smoothed positions are
  // continuous and rendered every animation frame.
  if (result) {
    const hands = extractHands(result);
    const centroids = hands.map(handCentroid);

    // Match detected hands to roles by nearest previous centroid. This keeps
    // chord/strum identities stable across MediaPipe's frame-to-frame
    // re-ordering and small position swaps.
    let chordIdx = -1;
    let strumIdx = -1;
    const dist = (a: Vec2, b: Vec2) => Math.hypot(a.x - b.x, a.y - b.y);

    if (hands.length === 1) {
      const c = centroids[0];
      const dC = chordFresh && chord.lastCentroid ? dist(c, chord.lastCentroid) : Infinity;
      const dS = strumFresh && strum.lastCentroid ? dist(c, strum.lastCentroid) : Infinity;
      if (dC === Infinity && dS === Infinity) {
        // Cold start: assign by y-zone.
        if (c.y < GRID_FRACTION) chordIdx = 0;
        else strumIdx = 0;
      } else if (dC <= dS) chordIdx = 0;
      else strumIdx = 0;
    } else if (hands.length >= 2) {
      if (chordFresh && strumFresh && chord.lastCentroid && strum.lastCentroid) {
        // 2-way assignment: pick the pairing with the smaller total distance.
        const d00 =
          dist(centroids[0], chord.lastCentroid) +
          dist(centroids[1], strum.lastCentroid);
        const d01 =
          dist(centroids[0], strum.lastCentroid) +
          dist(centroids[1], chord.lastCentroid);
        if (d00 <= d01) { chordIdx = 0; strumIdx = 1; }
        else { chordIdx = 1; strumIdx = 0; }
      } else {
        // Cold start: top hand → chord, lower hand → strum.
        const order = [0, 1].sort(
          (a, b) => centroids[a].y - centroids[b].y
        );
        chordIdx = order[0];
        strumIdx = order[1];
      }
    }

    if (chordIdx >= 0) {
      const h = hands[chordIdx];
      const c = centroids[chordIdx];
      chord.lastCentroid = c;
      chord.lastSeenAt = now;
      chord.smoothed = smooth(chord.smoothed, { x: mirrorX(c.x), y: c.y });

      const p = pinchAmount(h);
      if (!chordPinched && p < PINCH_ON) {
        chordPinched = true;
        cycleWaveform();
      } else if (chordPinched && p > PINCH_OFF) {
        chordPinched = false;
      }
    }

    if (strumIdx >= 0) {
      const h = hands[strumIdx];
      const c = centroids[strumIdx];
      const tip = indexTip(h);
      strum.lastCentroid = c;
      strum.lastSeenAt = now;
      strum.smoothed = smooth(strum.smoothed, { x: mirrorX(tip.x), y: tip.y });
    }
  }

  // Apply grace period: only clear role state once it's been missing > GRACE_MS.
  if (chord.lastSeenAt > 0 && now - chord.lastSeenAt >= GRACE_MS) {
    chord.smoothed = null;
    chord.lastCentroid = null;
    chordPinched = false;
  }
  if (strum.lastSeenAt > 0 && now - strum.lastSeenAt >= GRACE_MS) {
    strum.smoothed = null;
    strum.lastCentroid = null;
    lastStrumBand = null;
    lastStrumX = null;
    lastStrumNoteIdx = null;
  }

  // Drive logic and rendering from the smoothed positions every frame.
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid();

  if (chord.smoothed) {
    const cell = chordCellFromDisplay(
      chord.smoothed.x,
      chord.smoothed.y,
      currentCell
    );
    if (cell) {
      currentCell = cell;
      currentChord = chordAt(cell.row, cell.col);
      const key = `${cell.row},${cell.col}`;
      if (key !== heldCellKey) {
        heldCellKey = key;
        setHeldChord(currentChord.pad);
      }
    } else if (heldCellKey !== null) {
      heldCellKey = null;
      setHeldChord(null);
    }
    drawWaveformGlyph(chord.smoothed, getWaveform(), chordPinched);
  } else if (heldCellKey !== null) {
    heldCellKey = null;
    setHeldChord(null);
  }

  if (strum.smoothed) {
    handleStrum(strum.smoothed.x);
    const vibAmount = Math.max(0, Math.min(1, (strum.smoothed.y - 0.4) / 0.55));
    setVibrato(vibAmount);
    drawStrumDot(strum.smoothed, vibAmount);
  } else {
    setVibrato(0);
  }

  requestAnimationFrame(frame);
}

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  startBtn.textContent = "Loading…";
  try {
    await initAudio();
    await setupCamera();
    await setupLandmarker();
    requestWakeLock();
    startOverlay.classList.add("hidden");
    requestAnimationFrame(frame);
  } catch (err) {
    startBtn.textContent = `Error: ${(err as Error).message} — Retry`;
    startBtn.disabled = false;
  }
});
