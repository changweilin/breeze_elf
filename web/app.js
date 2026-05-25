const els = {
  start: document.querySelector("#start"),
  stop: document.querySelector("#stop"),
  clear: document.querySelector("#clear"),
  status: document.querySelector("#status"),
  stats: document.querySelector("#stats"),
  lines: document.querySelector("#lines"),
  partial: document.querySelector("#partial"),
  backend: document.querySelector("#backend"),
  clock: document.querySelector("#clock"),
};

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
      processorOptions: { targetSampleRate: 16000, chunkMs: 250 },
    });
    state.silence = state.audioContext.createGain();
    state.silence.gain.value = 0;

    state.worklet.port.onmessage = (event) => {
      if (event.data?.type !== "audio") {
        return;
      }
      if (state.ws?.readyState === WebSocket.OPEN) {
        state.ws.send(event.data.buffer);
      }
    };

    state.source.connect(state.worklet);
    state.worklet.connect(state.silence).connect(state.audioContext.destination);

    ws.send(JSON.stringify({ type: "start", sampleRate: 16000, language: "zh", chunkMs: 1000 }));
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
}

function handleServerMessage(event) {
  let data;
  try {
    data = JSON.parse(event.data);
  } catch {
    return;
  }

  if (data.type === "ready") {
    els.backend.textContent = `${data.backend} · ${data.device}`;
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
    if (typeof data.asrMs === "number") {
      els.stats.textContent = `${data.asrMs} ms`;
    } else if (data.speech === false) {
      els.stats.textContent = "靜音";
    }
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

if ("serviceWorker" in navigator && window.isSecureContext) {
  navigator.serviceWorker.register("/service-worker.js").catch(() => {});
}

