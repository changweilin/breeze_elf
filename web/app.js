const els = {
  start: document.querySelector("#start"),
  stop: document.querySelector("#stop"),
  clear: document.querySelector("#clear"),
  pitch: document.querySelector("#pitch"),
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
const PITCH_MIN_HZ = 70;
const PITCH_MAX_HZ = 500;
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
    startSeconds: 0.35,
    endSeconds: 1.1,
    pitch: demoPitch(188, [176, 184, 194, 202, 190, 181]),
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
    startSeconds: 1.85,
    endSeconds: 2.75,
    pitch: demoPitch(162, [154, 158, 166, 172, 160, 150]),
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
  transcriptBlocks: [],
  pitchMode: SEARCH_PARAMS.get("pitch") === "1",
  droppedClientChunks: 0,
  statsTimer: 0,
  savingRemote: false,
  demoRunning: false,
  demoTimers: [],
};

function demoPitch(medianHz, values) {
  return {
    medianHz,
    minHz: Math.min(...values),
    maxHz: Math.max(...values),
    voicedRatio: 0.92,
    points: values.map((hz, index) => ({
      offsetSeconds: index * 0.14,
      hz,
      confidence: 0.82,
    })),
  };
}

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
  state.transcriptBlocks = [];
  renderTranscriptView();
  setTranscriptActions(Boolean(text.trim()));
}

function appendTranscript(text) {
  appendTranscriptBlock({ text });
}

function appendTranscriptBlock(data) {
  const text = typeof data?.text === "string" ? data.text : "";
  if (!text) {
    return;
  }

  state.transcript = typeof data.transcript === "string"
    ? data.transcript
    : state.transcript
      ? `${state.transcript}${text}`
      : text;
  state.transcriptBlocks.push({
    text,
    startSeconds: finiteNumber(data.startSeconds),
    endSeconds: finiteNumber(data.endSeconds),
    segmentKind: data.segmentKind || "",
    windowIndex: Number.isInteger(data.windowIndex) ? data.windowIndex : null,
    pitch: normalizePitch(data.pitch),
  });
  renderTranscriptView();
  setTranscriptActions(Boolean(state.transcript.trim()));
}

function renderTranscriptView() {
  els.lines.classList.toggle("pitch-mode", state.pitchMode && state.transcriptBlocks.length > 0);
  els.lines.replaceChildren();

  if (state.pitchMode && state.transcriptBlocks.length > 0) {
    const fragment = document.createDocumentFragment();
    state.transcriptBlocks.forEach((block) => {
      fragment.append(renderTranscriptBlock(block));
    });
    els.lines.append(fragment);
  } else {
    els.lines.textContent = state.transcript;
  }

  els.lines.scrollTop = els.lines.scrollHeight;
}

function renderTranscriptBlock(block) {
  const row = document.createElement("div");
  row.className = "transcript-block";

  const text = document.createElement("span");
  text.className = "transcript-text";
  text.textContent = block.text.trimStart();

  const meta = document.createElement("span");
  meta.className = "pitch-meta";

  const range = document.createElement("span");
  range.textContent = formatTimeRange(block.startSeconds, block.endSeconds);

  const pitch = document.createElement("span");
  pitch.className = "pitch-value";
  pitch.textContent = formatPitch(block.pitch);

  meta.append(range, pitch);
  row.append(text, meta, renderPitchSpark(block.pitch));
  return row;
}

function renderPitchSpark(pitch) {
  const spark = document.createElement("span");
  const points = Array.isArray(pitch?.points) ? pitch.points.filter((point) => point.hz) : [];
  if (!points.length) {
    spark.className = "pitch-spark empty";
    return spark;
  }

  spark.className = "pitch-spark";
  const stride = Math.max(1, Math.ceil(points.length / 36));
  points.filter((_, index) => index % stride === 0).forEach((point) => {
    const bar = document.createElement("span");
    const normalized = (point.hz - PITCH_MIN_HZ) / (PITCH_MAX_HZ - PITCH_MIN_HZ);
    const height = 18 + Math.min(1, Math.max(0, normalized)) * 82;
    bar.style.setProperty("--pitch-height", `${height.toFixed(1)}%`);
    spark.append(bar);
  });
  return spark;
}

function renderPartial(data) {
  const text = data.text || "";
  const pitch = normalizePitch(data.pitch);
  if (state.pitchMode && pitch) {
    els.partial.textContent = `${text}\n${formatPitch(pitch)}`;
    return;
  }
  els.partial.textContent = text;
}

function finiteNumber(value) {
  return Number.isFinite(value) ? Number(value) : null;
}

function normalizePitch(pitch) {
  if (!pitch || typeof pitch !== "object") {
    return null;
  }

  return {
    medianHz: finiteNumber(pitch.medianHz),
    minHz: finiteNumber(pitch.minHz),
    maxHz: finiteNumber(pitch.maxHz),
    voicedRatio: finiteNumber(pitch.voicedRatio),
    points: Array.isArray(pitch.points)
      ? pitch.points
          .map((point) => ({
            offsetSeconds: finiteNumber(point.offsetSeconds),
            hz: finiteNumber(point.hz),
            confidence: finiteNumber(point.confidence),
          }))
          .filter((point) => point.hz)
      : [],
  };
}

function formatPitch(pitch) {
  if (!Number.isFinite(pitch?.medianHz)) {
    return "音高未偵測";
  }
  return `音高 ${Math.round(pitch.medianHz)} Hz`;
}

function formatTimeRange(startSeconds, endSeconds) {
  if (!Number.isFinite(startSeconds) || !Number.isFinite(endSeconds)) {
    return "段落";
  }
  return `${formatClockTime(startSeconds)}-${formatClockTime(endSeconds)}`;
}

function formatClockTime(totalSeconds) {
  const safeSeconds = Math.max(0, totalSeconds);
  const minutes = Math.floor(safeSeconds / 60);
  const seconds = Math.floor(safeSeconds % 60);
  const fraction = Math.floor((safeSeconds % 1) * 10);
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${fraction}`;
}

function setPitchMode(enabled) {
  state.pitchMode = Boolean(enabled);
  els.pitch.classList.toggle("active", state.pitchMode);
  els.pitch.setAttribute("aria-pressed", String(state.pitchMode));
  els.pitch.setAttribute("aria-label", state.pitchMode ? "隱藏音高模式" : "顯示音高模式");
  els.pitch.setAttribute("title", state.pitchMode ? "隱藏每段文字的音高" : "顯示每段文字的音高");
  renderTranscriptView();
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
        appendTranscriptBlock({ ...entry, text: entry.final });
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
    renderPartial(data);
    return;
  }

  if (data.type === "final") {
    els.partial.textContent = "";
    if (data.text) {
      appendTranscriptBlock(data);
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
setPitchMode(state.pitchMode);

els.theme.addEventListener("click", toggleTheme);
els.pitch.addEventListener("click", () => setPitchMode(!state.pitchMode));
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
