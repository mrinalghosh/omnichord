import * as Tone from "tone";

let synth: Tone.PolySynth | null = null;
let reverb: Tone.Reverb | null = null;

export async function initAudio() {
  await Tone.start();
  reverb = new Tone.Reverb({ decay: 3.5, wet: 0.35 }).toDestination();
  synth = new Tone.PolySynth(Tone.Synth, {
    oscillator: { type: "triangle" },
    envelope: { attack: 0.005, decay: 0.2, sustain: 0.4, release: 1.6 },
  }).connect(reverb);
  synth.volume.value = -8;
}

export function midiToFreq(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

export function pluck(midi: number, velocity = 0.7) {
  if (!synth) return;
  synth.triggerAttackRelease(midiToFreq(midi), "8n", undefined, velocity);
}

export function previewChord(midis: number[], velocity = 0.35) {
  if (!synth) return;
  const freqs = midis.map(midiToFreq);
  synth.triggerAttackRelease(freqs, "16n", undefined, velocity);
}

export function isReady(): boolean {
  return synth !== null;
}
