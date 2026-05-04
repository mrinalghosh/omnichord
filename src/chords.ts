export type Chord = {
  label: string;
  // Pad voicing: low-register held chord (root+3rd+5th near middle C).
  pad: number[];
  // Strum voicing: chord tones repeated across multiple octaves, low → high.
  strum: number[];
};

// Build a chord from a root MIDI note + chord-tone intervals (within one octave).
// Strum array tiles those tones across `octaves` octaves starting at the root.
const c = (
  label: string,
  root: number,
  intervals: number[],
  octaves = 3
): Chord => {
  const pad = [root, ...intervals.slice(1).map((i) => root + i)];
  const strum: number[] = [];
  for (let o = 0; o < octaves; o++) {
    for (const i of intervals) strum.push(root + i + o * 12);
  }
  return { label, pad, strum };
};

const MAJ = [0, 4, 7];
const MIN = [0, 3, 7];
const DIM = [0, 3, 6];
const DOM7 = [0, 4, 7, 10];
const MAJ7 = [0, 4, 7, 11];

// MIDI: C4 = 60. Pad voicings stay near middle C; strums span 3 octaves above.
// All roots placed in MIDI 48–59 (C3–B3) so pad voicings and strum ranges
// stay consistent across the grid — otherwise low roots disappear into the
// speaker rolloff.
export const GRID: Chord[][] = [
  [c("F",  53, MAJ),  c("C",  48, MAJ),  c("G",  55, MAJ)],
  [c("Dm", 50, MIN),  c("Am", 57, MIN),  c("Em", 52, MIN)],
  [c("G7", 55, DOM7), c("Cmaj7", 48, MAJ7), c("Bdim", 59, DIM)],
];

export const ROWS = GRID.length;
export const COLS = GRID[0].length;

export function chordAt(row: number, col: number): Chord {
  return GRID[row][col];
}
