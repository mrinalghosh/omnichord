// 3x3 chord grid in C major. Row 0 = top of screen.
// Layout chosen so the most common chords (I, IV, V, vi) sit in the easy
// middle/top zones, with color chords on the bottom row.
export type Chord = {
  label: string;
  // MIDI note numbers, root + chord tones, ordered low → high for strumming.
  notes: number[];
};

// Helper: build a chord from a root midi + intervals.
const c = (label: string, root: number, intervals: number[]): Chord => ({
  label,
  notes: intervals.map((i) => root + i),
});

const MAJ = [0, 4, 7, 12];
const MIN = [0, 3, 7, 12];
const DIM = [0, 3, 6, 12];
const DOM7 = [0, 4, 7, 10, 12];
const MAJ7 = [0, 4, 7, 11, 12];

// MIDI: C4 = 60
export const GRID: Chord[][] = [
  // top row
  [c("F",  65, MAJ),  c("C",  60, MAJ),  c("G",  67, MAJ)],
  // middle row
  [c("Dm", 62, MIN),  c("Am", 57, MIN),  c("Em", 64, MIN)],
  // bottom row
  [c("G7", 67, DOM7), c("Cmaj7", 60, MAJ7), c("Bdim", 59, DIM)],
];

export const ROWS = GRID.length;
export const COLS = GRID[0].length;

export function chordAt(row: number, col: number): Chord {
  return GRID[row][col];
}
