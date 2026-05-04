import {
  HandLandmarker,
  FilesetResolver,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { GRID, ROWS, COLS, chordAt, type Chord } from "./chords";
import { initAudio, pluck, setHeldChord, cycleWaveform, getWaveform } from "./audio";

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
let chordSmoothed: Vec2 | null = null;
let strumSmoothed: Vec2 | null = null;

let currentChord: Chord = chordAt(0, 1);
let currentCell: { row: number; col: number } | null = { row: 0, col: 1 };
let heldCellKey: string | null = null;

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
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720, facingMode: "user" },
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
  landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
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

// Pinch = thumb tip (4) close to index tip (8), measured relative to hand size
// (wrist 0 → middle MCP 9) so it's distance-invariant.
function pinchAmount(hand: Hand): number {
  const thumb = hand.landmarks[4];
  const index = hand.landmarks[8];
  const wrist = hand.landmarks[0];
  const mcp = hand.landmarks[9];
  const handSize = Math.hypot(mcp.x - wrist.x, mcp.y - wrist.y) || 0.0001;
  const d = Math.hypot(thumb.x - index.x, thumb.y - index.y);
  return d / handSize; // ~0.2 when pinched, ~1+ when open
}

const PINCH_ON = 0.45;
const PINCH_OFF = 0.7;
let chordPinched = false;
let waveLabel: string = "triangle";
let waveLabelUntil = 0;

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
    const velocity = Math.min(1, 0.4 + speed * 50);

    const notes = currentChord.strum;
    let cursor = lastStrumBand;
    while (cursor !== band) {
      cursor += dir;
      // Map band index → note index, ascending strum walks up the chord
      // tones across all octaves.
      const noteIdx = Math.min(
        notes.length - 1,
        Math.max(0, Math.floor((cursor / STRUM_BANDS) * notes.length))
      );
      pluck(notes[noteIdx], velocity);
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
  const t = performance.now();
  let result: HandLandmarkerResult | null = null;
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    result = landmarker.detectForVideo(video, t);
  }

  // Update target positions only on fresh detections; smoothed positions are
  // continuous and rendered every animation frame, which kills visual flicker.
  if (result) {
    const hands = extractHands(result);

    let chordHand: Hand | null = null;
    let strumHand: Hand | null = null;
    if (hands.length === 1) {
      const palm = handCentroid(hands[0]);
      if (palm.y < GRID_FRACTION) chordHand = hands[0];
      else strumHand = hands[0];
    } else if (hands.length >= 2) {
      // Hand whose palm is higher on the screen (smaller y) drives the chord;
      // the other hand strums regardless of its y position.
      const sorted = [...hands].sort(
        (a, b) => handCentroid(a).y - handCentroid(b).y
      );
      chordHand = sorted[0];
      strumHand = sorted[1];
    }

    if (chordHand) {
      const c = handCentroid(chordHand);
      chordSmoothed = smooth(chordSmoothed, { x: mirrorX(c.x), y: c.y });

      // Pinch gesture (with hysteresis) toggles the strum oscillator waveform
      // on the rising edge.
      const p = pinchAmount(chordHand);
      if (!chordPinched && p < PINCH_ON) {
        chordPinched = true;
        waveLabel = cycleWaveform();
        waveLabelUntil = performance.now() + 1200;
      } else if (chordPinched && p > PINCH_OFF) {
        chordPinched = false;
      }
    } else {
      chordSmoothed = null;
      chordPinched = false;
    }

    if (strumHand) {
      const tip = indexTip(strumHand);
      strumSmoothed = smooth(strumSmoothed, { x: mirrorX(tip.x), y: tip.y });
    } else {
      strumSmoothed = null;
      lastStrumBand = null;
      lastStrumX = null;
    }
  }

  // Drive logic and rendering from the smoothed positions every frame.
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid();

  if (chordSmoothed) {
    const cell = chordCellFromDisplay(
      chordSmoothed.x,
      chordSmoothed.y,
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
    } else {
      // Hand drifted out of the grid area — release the pad.
      if (heldCellKey !== null) {
        heldCellKey = null;
        setHeldChord(null);
      }
    }
    drawHandDot(chordSmoothed, chordPinched ? "#fd79a8" : "#a29bfe", "chord");
  } else {
    if (heldCellKey !== null) {
      heldCellKey = null;
      setHeldChord(null);
    }
  }

  if (strumSmoothed) {
    handleStrum(strumSmoothed.x);
    drawHandDot(strumSmoothed, "#ffeaa7", "strum");
  }

  // Brief on-screen confirmation when the waveform changes.
  if (performance.now() < waveLabelUntil) {
    ctx.fillStyle = "rgba(253, 121, 168, 0.95)";
    ctx.font = "600 28px -apple-system, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`～ ${waveLabel}`, canvas.width / 2, canvas.height - 40);
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
    waveLabel = getWaveform();
    startOverlay.classList.add("hidden");
    requestAnimationFrame(frame);
  } catch (err) {
    startBtn.textContent = `Error: ${(err as Error).message} — Retry`;
    startBtn.disabled = false;
  }
});
