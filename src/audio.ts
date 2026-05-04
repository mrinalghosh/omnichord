import * as Tone from "tone";

export type Waveform = "triangle" | "sine" | "sawtooth" | "square";
export const WAVEFORMS: Waveform[] = ["triangle", "sine", "sawtooth", "square"];

let synth: Tone.PolySynth | null = null; // strum notes ("dreamy" pluck)
let pad: Tone.PolySynth | null = null; // sustained chord pad
let reverb: Tone.Reverb | null = null;
let chorus: Tone.Chorus | null = null;
let delay: Tone.FeedbackDelay | null = null;
let heldPad: number[] | null = null;
let currentWave: Waveform = "triangle";

export async function initAudio() {
  await Tone.start();
  reverb = new Tone.Reverb({ decay: 6, wet: 0.55 }).toDestination();
  delay = new Tone.FeedbackDelay({
    delayTime: "8n.",
    feedback: 0.35,
    wet: 0.25,
  }).connect(reverb);
  chorus = new Tone.Chorus({ frequency: 0.6, delayTime: 4, depth: 0.7, wet: 0.5 })
    .connect(delay)
    .start();

  synth = new Tone.PolySynth(Tone.Synth, {
    oscillator: { type: currentWave },
    envelope: { attack: 0.06, decay: 0.4, sustain: 0.4, release: 2.4 },
  }).connect(chorus);
  synth.volume.value = -12;

  pad = new Tone.PolySynth(Tone.Synth, {
    oscillator: { type: "sine" },
    envelope: { attack: 0.5, decay: 0.3, sustain: 0.7, release: 1.4 },
  }).connect(reverb);
  pad.volume.value = -22;
}

export function setWaveform(w: Waveform) {
  currentWave = w;
  if (synth) synth.set({ oscillator: { type: w } });
}

export function cycleWaveform(): Waveform {
  const i = WAVEFORMS.indexOf(currentWave);
  const next = WAVEFORMS[(i + 1) % WAVEFORMS.length];
  setWaveform(next);
  return next;
}

export function getWaveform(): Waveform {
  return currentWave;
}

export function midiToFreq(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

export function pluck(midi: number, velocity = 0.7) {
  if (!synth) return;
  synth.triggerAttackRelease(midiToFreq(midi), "8n", undefined, velocity);
}

// Hold a sustained chord on the pad. Calling again releases the previous chord
// and attacks the new one. Pass null to release.
export function setHeldChord(midis: number[] | null) {
  if (!pad) return;
  if (heldPad) {
    pad.triggerRelease(heldPad.map(midiToFreq));
  }
  heldPad = midis;
  if (midis) {
    pad.triggerAttack(midis.map(midiToFreq), undefined, 0.6);
  }
}

export function isReady(): boolean {
  return synth !== null;
}
