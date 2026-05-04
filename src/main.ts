import {
  HandLandmarker,
  FilesetResolver,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { GRID, ROWS, COLS, chordAt, type Chord } from "./chords";
import { initAudio, pluck, previewChord } from "./audio";

const video = document.getElementById("video") as HTMLVideoElement;
const canvas = document.getElementById("overlay") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const chordNameEl = document.getElementById("chord-name")!;
const statusEl = document.getElementById("status")!;
const startOverlay = document.getElementById("start-overlay")!;
const startBtn = document.getElementById("start-btn") as HTMLButtonElement;

let landmarker: HandLandmarker | null = null;
let lastVideoTime = -1;

const STRUM_BANDS = 8;
let lastStrumBand: number | null = null;
let lastStrumX: number | null = null;
let lastStrumT = 0;

// Smoothed positions (EMA) so cell/band selection doesn't flicker on jitter.
const SMOOTH = 0.35; // 0=no smoothing, 1=frozen
let chordSmoothed: { x: number; y: number } | null = null;
let strumSmoothed: { x: number; y: number } | null = null;

// Hysteresis: require the smoothed position to be at least this fraction of a
// cell/band past the boundary before switching.
const HYSTERESIS = 0.18;

function smooth(
  prev: { x: number; y: number } | null,
  next: { x: number; y: number }
): { x: number; y: number } {
  if (!prev) return next;
  return {
    x: prev.x + (next.x - prev.x) * SMOOTH,
    y: prev.y + (next.y - prev.y) * SMOOTH,
  };
}

// Currently selected chord (sticky once chosen). Default to C major.
let currentChord: Chord = chordAt(1, 1); // Am middle — but we'll update immediately
currentChord = chordAt(0, 1); // C
let currentCell: { row: number; col: number } | null = { row: 0, col: 1 };

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

// In MediaPipe normalized coords, x grows left→right of the *raw* camera frame.
// Because we mirror the display (scaleX(-1)), what the user sees on the left side
// of the screen is x > 0.5 in raw coords. We mirror x for all logic so it matches
// what the user sees.
function mirrorX(x: number): number {
  return 1 - x;
}

type Hand = {
  landmarks: { x: number; y: number; z: number }[];
  handedness: "Left" | "Right"; // as labeled by MediaPipe (the user's actual hand)
};

function extractHands(result: HandLandmarkerResult): Hand[] {
  const hands: Hand[] = [];
  for (let i = 0; i < result.landmarks.length; i++) {
    const lm = result.landmarks[i];
    // MediaPipe handedness is from the camera's perspective. Since we mirror the
    // display, MediaPipe's "Left" appears on the user's left side of the screen,
    // which is actually their right hand (selfie view). Flip the label.
    const rawLabel = result.handednesses[i]?.[0]?.categoryName ?? "Right";
    const flipped: "Left" | "Right" = rawLabel === "Left" ? "Right" : "Left";
    hands.push({
      landmarks: lm.map((p) => ({ x: p.x, y: p.y, z: p.z })),
      handedness: flipped,
    });
  }
  return hands;
}

function handCentroid(hand: Hand): { x: number; y: number } {
  // Use wrist (0) + middle MCP (9) midpoint as a stable palm center.
  const w = hand.landmarks[0];
  const m = hand.landmarks[9];
  return { x: (w.x + m.x) / 2, y: (w.y + m.y) / 2 };
}

function indexTip(hand: Hand): { x: number; y: number } {
  return hand.landmarks[8];
}

// Layout: grid occupies the top GRID_FRACTION of the screen, strum bands occupy
// the bottom (1 - GRID_FRACTION).
const GRID_FRACTION = 0.6;

function chordCellFromDisplay(
  dx: number,
  dy: number,
  current: { row: number; col: number } | null
): { row: number; col: number } | null {
  if (dy > GRID_FRACTION) return null;
  const gx = dx * COLS; // continuous column index
  const gy = (dy / GRID_FRACTION) * ROWS;
  let col = Math.min(COLS - 1, Math.max(0, Math.floor(gx)));
  let row = Math.min(ROWS - 1, Math.max(0, Math.floor(gy)));

  // Hysteresis: if we have a current cell, require crossing the boundary by
  // HYSTERESIS before switching to the new cell.
  if (current) {
    const fracX = gx - current.col; // distance into current col, 0..1 if same col
    const fracY = gy - current.row;
    if (col !== current.col) {
      // Want to switch column. Demand the new cell is entered by HYSTERESIS.
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

function strumBandFromDisplay(
  dx: number,
  dy: number,
  current: number | null
): number | null {
  if (dy < GRID_FRACTION) return null;
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

  // chord grid (top)
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

  // strum bands (bottom)
  const stripY = gridH;
  const stripH = h - gridH;
  ctx.fillStyle = "rgba(255, 234, 167, 0.04)";
  ctx.fillRect(0, stripY, w, stripH);
  for (let i = 1; i < STRUM_BANDS; i++) {
    const x = (i / STRUM_BANDS) * w;
    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, stripY);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  // divider
  ctx.strokeStyle = "rgba(255,255,255,0.25)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, stripY);
  ctx.lineTo(w, stripY);
  ctx.stroke();

  ctx.fillStyle = "rgba(255,255,255,0.35)";
  ctx.font = "12px -apple-system, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("strum ←→", w / 2, stripY + stripH / 2);
}

function drawHandDot(dx: number, dy: number, color: string, label?: string) {
  const x = dx * canvas.width;
  const y = dy * canvas.height;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, 14, 0, Math.PI * 2);
  ctx.fill();
  if (label) {
    ctx.fillStyle = "#fff";
    ctx.font = "12px -apple-system, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(label, x, y - 22);
  }
}

function handleStrum(dx: number, dy: number) {
  const band = strumBandFromDisplay(dx, dy, lastStrumBand);
  const now = performance.now();
  if (band === null) {
    lastStrumBand = null;
    lastStrumX = null;
    return;
  }
  if (lastStrumBand === null) {
    lastStrumBand = band;
    lastStrumX = dx;
    lastStrumT = now;
    return;
  }
  if (band !== lastStrumBand) {
    // Crossed at least one band boundary. Trigger one note per crossing.
    const dir = band > lastStrumBand ? 1 : -1;
    const dt = Math.max(1, now - lastStrumT);
    const dxAbs = Math.abs((dx - (lastStrumX ?? dx)));
    const speed = dxAbs / dt; // display units / ms
    const velocity = Math.min(1, 0.35 + speed * 60);

    const notes = currentChord.notes;
    let cursor = lastStrumBand;
    while (cursor !== band) {
      cursor += dir;
      // Pick a note from the chord, cycling. Ascending strum walks up.
      const idx = ((cursor % notes.length) + notes.length) % notes.length;
      const noteMidi =
        dir > 0 ? notes[idx] : notes[notes.length - 1 - idx];
      pluck(noteMidi, velocity);
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

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid();

  if (result) {
    const hands = extractHands(result);

    // Position-based assignment (ignore MediaPipe handedness, which flips
    // unreliably under mirrored selfie video). Whichever hand's palm sits on
    // the display-left half picks the chord; whichever index tip sits on the
    // display-right half drives the strum.
    let chordHand: Hand | null = null;
    let strumHand: Hand | null = null;
    for (const h of hands) {
      const palm = handCentroid(h);
      if (palm.y < GRID_FRACTION) {
        if (!chordHand) chordHand = h;
      } else {
        if (!strumHand) strumHand = h;
      }
    }

    if (chordHand) {
      const c = handCentroid(chordHand);
      const raw = { x: mirrorX(c.x), y: c.y };
      chordSmoothed = smooth(chordSmoothed, raw);
      const cell = chordCellFromDisplay(
        chordSmoothed.x,
        chordSmoothed.y,
        currentCell
      );
      if (cell) {
        const changed =
          !currentCell ||
          currentCell.row !== cell.row ||
          currentCell.col !== cell.col;
        currentCell = cell;
        currentChord = chordAt(cell.row, cell.col);
        if (changed) previewChord(currentChord.notes);
      }
      drawHandDot(chordSmoothed.x, chordSmoothed.y, "#a29bfe", "chord");
    } else {
      chordSmoothed = null;
    }

    if (strumHand) {
      const tip = indexTip(strumHand);
      const raw = { x: mirrorX(tip.x), y: tip.y };
      strumSmoothed = smooth(strumSmoothed, raw);
      handleStrum(strumSmoothed.x, strumSmoothed.y);
      drawHandDot(strumSmoothed.x, strumSmoothed.y, "#ffeaa7", "strum");
    } else {
      lastStrumBand = null;
      lastStrumX = null;
      strumSmoothed = null;
    }
  }

  chordNameEl.textContent = currentChord.label;
  requestAnimationFrame(frame);
}

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  startBtn.textContent = "Loading…";
  try {
    await initAudio();
    statusEl.textContent = "audio ready · loading camera…";
    await setupCamera();
    statusEl.textContent = "camera ready · loading hand model…";
    await setupLandmarker();
    statusEl.textContent = "ready";
    startOverlay.classList.add("hidden");
    requestAnimationFrame(frame);
  } catch (err) {
    statusEl.textContent = `error: ${(err as Error).message}`;
    startBtn.disabled = false;
    startBtn.textContent = "Retry";
  }
});
