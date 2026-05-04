import * as Tone from "tone";

let synth: Tone.PolySynth | null = null; // bright pluck for strum notes
let pad: Tone.PolySynth | null = null; // sustained chord pad
let reverb: Tone.Reverb | null = null;
let heldPad: number[] | null = null;

export async function initAudio() {
  await Tone.start();
  reverb = new Tone.Reverb({ decay: 4, wet: 0.35 }).toDestination();
  synth = new Tone.PolySynth(Tone.Synth, {
    oscillator: { type: "triangle" },
    envelope: { attack: 0.003, decay: 0.25, sustain: 0.0, release: 1.2 },
  }).connect(reverb);
  synth.volume.value = -10;

  pad = new Tone.PolySynth(Tone.Synth, {
    oscillator: { type: "sine" },
    envelope: { attack: 0.4, decay: 0.3, sustain: 0.7, release: 1.2 },
  }).connect(reverb);
  pad.volume.value = -22;
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
