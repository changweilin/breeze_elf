const els = {
  start: document.querySelector("#start"),
  stop: document.querySelector("#stop"),
  clear: document.querySelector("#clear"),
  copy: document.querySelector("#copy"),
  download: document.querySelector("#download"),
  status: document.querySelector("#status"),
  stats: document.querySelector("#stats"),
  lines: document.querySelector("#lines"),
  partial: document.querySelector("#partial"),
  backend: document.querySelector("#backend"),
  clock: document.querySelector("#clock"),
  level: document.querySelector("#level"),
};

const AUDIO_CHUNK_MS = 250;
const MAX_WS_BUFFERED_BYTES = 256 * 1024;

const state = {
  ws: null,
  audioContext: null,
  stream: null,
  source: null,
  worklet: null,
  silence: null,
  startedAt: 0,
  clockTimer: 0,
  transcript: "",
  droppedClientChunks: 0,
  statsTimer: 0,
};

function websocketUrl() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}/ws/audio`;
}

function setStatus(text, mode = "") {
  els.status.textContent = text;
  els.status.className = `status ${mode}`.trim();
}

function setRunning(isRunning) {
  els.start.disabled = isRunning;
  els.stop.disabled = !isRunning;
}

function renderTranscript(text) {
  state.transcript = text;
  els.lines.textContent = text;
  els.lines.scrollTop = els.lines.scrollHeight;
  setTranscriptActions(Boolean(text.trim()));
}

function appendTranscript(text) {
  const next = state.transcript ? `${state.transcript}${text}` : text;
  renderTranscript(next);
}

function startClock() {
  state.startedAt = Date.now();
  window.clearInterval(state.clockTimer);
  state.clockTimer = window.setInterval(() => {
    const seconds = Math.floor((Date.now() - state.startedAt) / 1000);
    const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
    const ss = String(seconds % 60).padStart(2, "0");
    els.clock.textContent = `${mm}:${ss}`;
  }, 500);
}

function renderStats(data = {}) {
  const parts = [];
  if (typeof data.asrMs === "number") {
    parts.push(`${data.asrMs} ms`);
  } else if (data.speech === false) {
    parts.push("靜音");
  } else if (data.backpressure) {
    parts.push("延遲");
  }

  if (data.segmentKind === "utterance") {
    parts.push("段落");
  }
  if (typeof data.asrQueueWaitMs === "number" && data.asrQueueWaitMs > 0) {
    parts.push(`等候 ${data.asrQueueWaitMs} ms`);
  }
  if (typeof data.droppedWindows === "number" && data.droppedWindows > 0) {
    parts.push(`後端丟 ${data.droppedWindows}`);
  }
  if (state.droppedClientChunks > 0) {
    parts.push(`前端丟 ${state.droppedClientChunks}`);
  }
  if (typeof data.queueDepth === "number" && data.queueDepth > 0) {
    parts.push(`佇列 ${data.queueDepth}`);
  }
  if (typeof data.asrQueueDepth === "number" && data.asrQueueDepth > 0) {
    parts.push(`ASR ${data.asrQueueDepth}`);
  }

  if (parts.length) {
    els.stats.textContent = parts.join(" · ");
  }
}

function flashStats(text) {
  window.clearTimeout(state.statsTimer);
  const previous = els.stats.textContent;
  els.stats.textContent = text;
  state.statsTimer = window.setTimeout(() => {
    els.stats.textContent = previous;
  }, 1200);
}

function renderLevel(rms = 0) {
  const scaled = Math.min(1, Math.max(0, rms / 0.12));
  els.level.style.transform = `scaleX(${scaled.toFixed(3)})`;
}

function setTranscriptActions(hasTranscript) {
  els.copy.disabled = !hasTranscript;
  els.download.disabled = !hasTranscript;
}

function stopClock() {
  window.clearInterval(state.clockTimer);
  state.clockTimer = 0;
}

function waitForOpen(ws) {
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => reject(new Error("連線逾時")), 6000);
    ws.addEventListener(
      "open",
      () => {
        window.clearTimeout(timeout);
        resolve();
      },
      { once: true },
    );
    ws.addEventListener(
      "error",
      () => {
        window.clearTimeout(timeout);
        reject(new Error("連線失敗"));
      },
      { once: true },
    );
  });
}

async function start() {
  if (!window.isSecureContext && !["localhost", "127.0.0.1"].includes(location.hostname)) {
    setStatus("需要 HTTPS", "error");
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("無麥克風", "error");
    return;
  }

  setRunning(true);
  setStatus("連線中");
  state.droppedClientChunks = 0;
  els.stats.textContent = "0 ms";
  renderLevel(0);

  try {
    const ws = new WebSocket(websocketUrl());
    ws.binaryType = "arraybuffer";
    ws.addEventListener("message", handleServerMessage);
    ws.addEventListener("close", () => {
      cleanupAudio();
      setRunning(false);
      setStatus("待命");
      stopClock();
    });
    state.ws = ws;
    await waitForOpen(ws);

    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });

    state.audioContext = new AudioContext({ latencyHint: "interactive" });
    await state.audioContext.audioWorklet.addModule("/static/audio-worklet.js");

    state.source = state.audioContext.createMediaStreamSource(state.stream);
    state.worklet = new AudioWorkletNode(state.audioContext, "breeze-mic-processor", {
      processorOptions: { targetSampleRate: 16000, chunkMs: AUDIO_CHUNK_MS },
    });
    state.silence = state.audioContext.createGain();
    state.silence.gain.value = 0;

    state.worklet.port.onmessage = (event) => {
      if (event.data?.type !== "audio") {
        return;
      }
      renderLevel(event.data.rms || 0);
      if (state.ws?.readyState === WebSocket.OPEN) {
        if (state.ws.bufferedAmount > MAX_WS_BUFFERED_BYTES) {
          state.droppedClientChunks += 1;
          renderStats({ backpressure: true });
          return;
        }
        state.ws.send(event.data.buffer);
      }
    };

    state.source.connect(state.worklet);
    state.worklet.connect(state.silence).connect(state.audioContext.destination);

    ws.send(JSON.stringify({ type: "start", sampleRate: 16000, language: "zh", chunkMs: AUDIO_CHUNK_MS }));
    setStatus("收音中", "live");
    startClock();
  } catch (error) {
    setStatus(error.message || "啟動失敗", "error");
    stop();
  }
}

function stop() {
  if (state.ws?.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: "stop" }));
    window.setTimeout(() => state.ws?.close(), 100);
  } else {
    state.ws?.close();
  }
  cleanupAudio();
  setRunning(false);
  stopClock();
}

function cleanupAudio() {
  state.worklet?.disconnect();
  state.source?.disconnect();
  state.silence?.disconnect();
  state.stream?.getTracks().forEach((track) => track.stop());
  state.audioContext?.close();
  state.worklet = null;
  state.source = null;
  state.silence = null;
  state.stream = null;
  state.audioContext = null;
  renderLevel(0);
}

function handleServerMessage(event) {
  let data;
  try {
    data = JSON.parse(event.data);
  } catch {
    return;
  }

  if (data.type === "ready") {
    els.backend.textContent = `${data.backend} · ${data.device} · ${data.segmenter || "audio"}`;
    setStatus("收音中", "live");
    return;
  }

  if (data.type === "partial") {
    els.partial.textContent = data.text || "";
    return;
  }

  if (data.type === "final") {
    els.partial.textContent = "";
    if (data.transcript) {
      renderTranscript(data.transcript);
    } else if (data.text) {
      appendTranscript(data.text);
    }
    return;
  }

  if (data.type === "stats") {
    renderStats(data);
    return;
  }

  if (data.type === "error") {
    setStatus(data.message || "錯誤", "error");
  }
}

els.start.addEventListener("click", start);
els.stop.addEventListener("click", stop);
els.clear.addEventListener("click", () => {
  renderTranscript("");
  els.partial.textContent = "";
});
els.copy.addEventListener("click", async () => {
  if (!state.transcript.trim()) {
    return;
  }
  try {
    await navigator.clipboard.writeText(state.transcript);
    flashStats("已複製");
  } catch {
    flashStats("複製失敗");
  }
});
els.download.addEventListener("click", () => {
  if (!state.transcript.trim()) {
    return;
  }
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const blob = new Blob([state.transcript], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `breeze-elf-${stamp}.txt`;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  flashStats("已下載");
});

if ("serviceWorker" in navigator && window.isSecureContext) {
  navigator.serviceWorker.register("/service-worker.js").catch(() => {});
}
