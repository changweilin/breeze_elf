const els = {
  start: document.querySelector("#start"),
  stop: document.querySelector("#stop"),
  clear: document.querySelector("#clear"),
  copy: document.querySelector("#copy"),
  download: document.querySelector("#download"),
  save: document.querySelector("#save"),
  theme: document.querySelector("#theme"),
  themeColor: document.querySelector("meta[name='theme-color']"),
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
const THEME_STORAGE_KEY = "breeze-elf-theme";
const SYSTEM_DARK_QUERY = window.matchMedia("(prefers-color-scheme: dark)");
const SEARCH_PARAMS = new URLSearchParams(location.search);
const DEMO_MODE =
  SEARCH_PARAMS.has("demo") ||
  SEARCH_PARAMS.get("mode") === "demo" ||
  location.protocol === "file:" ||
  location.hostname.endsWith(".github.io");
const THEME_COLORS = {
  light: "#f6f7f8",
  dark: "#111614",
};
const DEMO_EVENTS = [
  {
    delay: 350,
    partial: "今天先確認 GitHub Actions 的靜態展示。",
    rms: 0.045,
    stats: { asrMs: 18, segmentKind: "utterance" },
  },
  {
    delay: 1100,
    final: "今天先確認 GitHub Actions 的靜態展示。",
    rms: 0.02,
    stats: { asrMs: 18, segmentKind: "utterance" },
  },
  {
    delay: 1850,
    partial: "麥克風、WebSocket 與遠端儲存都維持凍結。",
    rms: 0.05,
    stats: { asrMs: 22, segmentKind: "utterance" },
  },
  {
    delay: 2750,
    final: "\n麥克風、WebSocket 與遠端儲存都維持凍結，只呈現操作流程。",
    rms: 0.01,
    stats: { asrMs: 22, segmentKind: "utterance" },
  },
];

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
  savingRemote: false,
  demoRunning: false,
  demoTimers: [],
};

function storedTheme() {
  try {
    const value = localStorage.getItem(THEME_STORAGE_KEY);
    return value === "dark" || value === "light" ? value : null;
  } catch {
    return null;
  }
}

function preferredTheme() {
  return storedTheme() || (SYSTEM_DARK_QUERY.matches ? "dark" : "light");
}

function applyTheme(theme, { persist = false } = {}) {
  const nextTheme = theme === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = nextTheme;
  els.themeColor.content = THEME_COLORS[nextTheme];

  const switchLabel = nextTheme === "dark" ? "切換淺色模式" : "切換深色模式";
  els.theme.textContent = nextTheme === "dark" ? "☀" : "☾";
  els.theme.setAttribute("aria-label", switchLabel);
  els.theme.setAttribute("title", switchLabel);
  els.theme.setAttribute("aria-pressed", String(nextTheme === "dark"));

  if (persist) {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
    } catch {
      flashStats("無法記住主題");
    }
  }
}

function selectedTheme() {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

function toggleTheme() {
  applyTheme(selectedTheme() === "dark" ? "light" : "dark", { persist: true });
}

function syncSystemTheme() {
  if (!storedTheme()) {
    applyTheme(preferredTheme());
  }
}

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
  if (data.filtered) {
    parts.push("靜音");
  } else if (typeof data.asrMs === "number") {
    parts.push(`${data.asrMs} ms`);
  } else if (data.speech === false) {
    parts.push("靜音");
  } else if (data.backpressure) {
    parts.push("延遲");
  } else if (data.stopped) {
    parts.push("完成");
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

function flashStats(text, restoreText = els.stats.textContent) {
  window.clearTimeout(state.statsTimer);
  els.stats.textContent = text;
  state.statsTimer = window.setTimeout(() => {
    els.stats.textContent = restoreText;
  }, 1200);
}

function renderLevel(rms = 0) {
  const scaled = Math.min(1, Math.max(0, rms / 0.12));
  els.level.style.transform = `scaleX(${scaled.toFixed(3)})`;
}

function setTranscriptActions(hasTranscript) {
  els.copy.disabled = !hasTranscript;
  els.download.disabled = !hasTranscript;
  els.save.disabled = DEMO_MODE || !hasTranscript || state.savingRemote;
}

function transcriptTitle(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find(Boolean)
    ?.slice(0, 80);
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

function clearDemoTimers() {
  state.demoTimers.forEach((timer) => window.clearTimeout(timer));
  state.demoTimers = [];
}

function startDemo() {
  clearDemoTimers();
  state.demoRunning = true;
  state.droppedClientChunks = 0;
  renderTranscript("");
  els.partial.textContent = "";
  els.stats.textContent = "示意模式";
  els.backend.textContent = "GitHub Actions 示意 · 隱私功能凍結";
  setRunning(true);
  setStatus("示意中", "live");
  startClock();
  renderLevel(0.02);

  DEMO_EVENTS.forEach((entry) => {
    const timer = window.setTimeout(() => {
      if (!state.demoRunning) {
        return;
      }
      if (entry.partial) {
        els.partial.textContent = entry.partial;
      }
      if (entry.final) {
        els.partial.textContent = "";
        appendTranscript(entry.final);
      }
      renderLevel(entry.rms || 0);
      renderStats(entry.stats);
    }, entry.delay);
    state.demoTimers.push(timer);
  });

  state.demoTimers.push(
    window.setTimeout(() => {
      if (!state.demoRunning) {
        return;
      }
      state.demoRunning = false;
      clearDemoTimers();
      stopClock();
      renderLevel(0);
      renderStats({ stopped: true });
      setRunning(false);
      setStatus("示意完成");
    }, DEMO_EVENTS.at(-1).delay + 650),
  );
}

function stopDemo(status = "示意待命") {
  clearDemoTimers();
  state.demoRunning = false;
  els.partial.textContent = "";
  stopClock();
  renderLevel(0);
  setRunning(false);
  setStatus(status);
}

function applyRuntimeMode() {
  if (!DEMO_MODE) {
    return;
  }

  els.backend.textContent = "GitHub Actions 示意 · 隱私功能凍結";
  els.save.textContent = "儲存凍結";
  els.save.setAttribute("title", "示意模式不會寫入遠端主機");
  els.save.setAttribute("aria-label", "遠端儲存已凍結");
  els.stats.textContent = "示意模式";
  setStatus("示意");
  setTranscriptActions(false);
}

async function start() {
  if (DEMO_MODE) {
    startDemo();
    return;
  }

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
      state.ws = null;
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
    await state.audioContext.audioWorklet.addModule(
      new URL("audio-worklet.js", import.meta.url).href,
    );

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
  if (DEMO_MODE) {
    stopDemo();
    return;
  }

  cleanupAudio();
  stopClock();

  if (state.ws?.readyState === WebSocket.OPEN) {
    els.start.disabled = true;
    els.stop.disabled = true;
    setStatus("收尾中");
    state.ws.send(JSON.stringify({ type: "stop" }));
    return;
  } else {
    state.ws?.close();
  }
  setRunning(false);
  setStatus("待命");
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
    if (data.stopped) {
      state.ws?.close();
    }
    return;
  }

  if (data.type === "error") {
    setStatus(data.message || "錯誤", "error");
  }
}

applyTheme(preferredTheme());
applyRuntimeMode();

els.theme.addEventListener("click", toggleTheme);
SYSTEM_DARK_QUERY.addEventListener("change", syncSystemTheme);
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
els.save.addEventListener("click", async () => {
  if (DEMO_MODE) {
    flashStats("示意模式不會遠端儲存");
    return;
  }

  const text = state.transcript.trim();
  if (!text || state.savingRemote) {
    return;
  }

  state.savingRemote = true;
  setTranscriptActions(true);
  window.clearTimeout(state.statsTimer);
  const previousStats = els.stats.textContent;
  els.stats.textContent = "遠端儲存中";

  try {
    const response = await fetch("/api/transcripts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        title: transcriptTitle(text),
      }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      throw new Error(data.detail || "遠端儲存失敗");
    }
    flashStats(
      data.filename ? `已遠端儲存 ${data.filename}` : "已遠端儲存",
      previousStats,
    );
  } catch (error) {
    flashStats(error.message || "遠端儲存失敗", previousStats);
  } finally {
    state.savingRemote = false;
    setTranscriptActions(Boolean(state.transcript.trim()));
  }
});

if ("serviceWorker" in navigator && window.isSecureContext) {
  navigator.serviceWorker.register(new URL("service-worker.js", location.href).href).catch(() => {});
}
