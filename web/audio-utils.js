// Shared mono 16-bit PCM <-> WAV helpers for the 變聲 page. Kept separate from
// app.js so the voice controller can reuse the recording/encoding plumbing
// without dragging in the transcribe page's state.

export const VOICE_SAMPLE_RATE = 16000;

function writeAscii(view, offset, text) {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index));
  }
}

// Build a 16-bit mono WAV blob from raw little-endian Int16 PCM bytes.
export function pcm16BytesToWavBlob(byteChunks, sampleRate) {
  const chunks = byteChunks.filter((chunk) => chunk && chunk.byteLength > 0);
  const dataSize = chunks.reduce((total, chunk) => total + chunk.byteLength, 0);
  const header = new ArrayBuffer(44);
  const view = new DataView(header);
  const blockAlign = 2; // mono * 16-bit
  const byteRate = sampleRate * blockAlign;

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataSize, true);

  return new Blob([header, ...chunks], { type: "audio/wav" });
}

export function int16ToWavBlob(int16, sampleRate) {
  return pcm16BytesToWavBlob([int16.buffer.slice(0)], sampleRate);
}

export async function blobToBase64(blob) {
  const buffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

function downmixToMono(audioBuffer) {
  const channels = audioBuffer.numberOfChannels;
  if (channels === 1) {
    return audioBuffer.getChannelData(0).slice();
  }
  const length = audioBuffer.length;
  const mono = new Float32Array(length);
  for (let channel = 0; channel < channels; channel += 1) {
    const data = audioBuffer.getChannelData(channel);
    for (let index = 0; index < length; index += 1) {
      mono[index] += data[index];
    }
  }
  for (let index = 0; index < length; index += 1) {
    mono[index] /= channels;
  }
  return mono;
}

async function resampleMono(channelData, inputRate, targetRate) {
  if (inputRate === targetRate || channelData.length === 0) {
    return channelData;
  }
  const length = Math.max(1, Math.ceil((channelData.length * targetRate) / inputRate));
  const offline = new OfflineAudioContext(1, length, targetRate);
  const buffer = offline.createBuffer(1, channelData.length, inputRate);
  buffer.copyToChannel(channelData, 0);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}

function floatToInt16(floatData) {
  const out = new Int16Array(floatData.length);
  for (let index = 0; index < floatData.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, floatData[index]));
    out[index] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
  }
  return out;
}

// Decode any browser-supported audio file into mono 16-bit PCM at targetRate.
export async function decodeFileToInt16(file, targetRate = VOICE_SAMPLE_RATE) {
  const arrayBuffer = await file.arrayBuffer();
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const context = new AudioContextClass();
  let audioBuffer;
  try {
    audioBuffer = await context.decodeAudioData(arrayBuffer.slice(0));
  } finally {
    void context.close();
  }
  const mono = downmixToMono(audioBuffer);
  const resampled = await resampleMono(mono, audioBuffer.sampleRate, targetRate);
  return { pcm: floatToInt16(resampled), sampleRate: targetRate };
}
