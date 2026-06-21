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

export function base64ToBytes(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

// CRC-32 (IEEE 802.3) — required by the ZIP local + central headers below.
const _CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let crc = 0xffffffff;
  for (let index = 0; index < bytes.length; index += 1) {
    crc = _CRC32_TABLE[(crc ^ bytes[index]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

// Pack a few files into a single ZIP (store / no compression). WAV audio barely
// compresses, so storing keeps the helper tiny and dependency-free while still
// producing a real, standard .zip the OS can open. ``files`` is a list of
// ``{ name, bytes }`` where ``name`` is ASCII and ``bytes`` is a Uint8Array.
export function filesToZipBlob(files) {
  const encoder = new TextEncoder();
  const locals = [];
  const centrals = [];
  let offset = 0;

  for (const file of files) {
    const nameBytes = encoder.encode(file.name);
    const data = file.bytes;
    const crc = crc32(data);

    const local = new Uint8Array(30 + nameBytes.length + data.length);
    const lview = new DataView(local.buffer);
    lview.setUint32(0, 0x04034b50, true); // local file header signature
    lview.setUint16(4, 20, true); // version needed
    lview.setUint16(6, 0, true); // flags
    lview.setUint16(8, 0, true); // method: store
    lview.setUint16(10, 0, true); // mod time
    lview.setUint16(12, 0, true); // mod date
    lview.setUint32(14, crc, true);
    lview.setUint32(18, data.length, true); // compressed size
    lview.setUint32(22, data.length, true); // uncompressed size
    lview.setUint16(26, nameBytes.length, true);
    lview.setUint16(28, 0, true); // extra length
    local.set(nameBytes, 30);
    local.set(data, 30 + nameBytes.length);
    locals.push(local);

    const central = new Uint8Array(46 + nameBytes.length);
    const cview = new DataView(central.buffer);
    cview.setUint32(0, 0x02014b50, true); // central dir signature
    cview.setUint16(4, 20, true); // version made by
    cview.setUint16(6, 20, true); // version needed
    cview.setUint16(8, 0, true); // flags
    cview.setUint16(10, 0, true); // method: store
    cview.setUint16(12, 0, true); // mod time
    cview.setUint16(14, 0, true); // mod date
    cview.setUint32(16, crc, true);
    cview.setUint32(20, data.length, true);
    cview.setUint32(24, data.length, true);
    cview.setUint16(28, nameBytes.length, true);
    cview.setUint16(30, 0, true); // extra length
    cview.setUint16(32, 0, true); // comment length
    cview.setUint16(34, 0, true); // disk number start
    cview.setUint16(36, 0, true); // internal attrs
    cview.setUint32(38, 0, true); // external attrs
    cview.setUint32(42, offset, true); // local header offset
    central.set(nameBytes, 46);
    centrals.push(central);

    offset += local.length;
  }

  const centralSize = centrals.reduce((total, part) => total + part.length, 0);
  const end = new Uint8Array(22);
  const eview = new DataView(end.buffer);
  eview.setUint32(0, 0x06054b50, true); // end of central dir signature
  eview.setUint16(8, files.length, true); // entries this disk
  eview.setUint16(10, files.length, true); // total entries
  eview.setUint32(12, centralSize, true);
  eview.setUint32(16, offset, true); // central dir offset
  eview.setUint16(20, 0, true); // comment length

  return new Blob([...locals, ...centrals, end], { type: "application/zip" });
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
