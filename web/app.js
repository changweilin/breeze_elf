const els = {
  start: document.querySelector("#start"),
  stop: document.querySelector("#stop"),
  clear: document.querySelector("#clear"),
  pitch: document.querySelector("#pitch"),
  copy: document.querySelector("#copy"),
  download: document.querySelector("#download"),
  save: document.querySelector("#save"),
  audioPanel: document.querySelector("#audio-panel"),
  audioPlayer: document.querySelector("#recording"),
  audioDownload: document.querySelector("#audio-download"),
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
const SETTINGS_STORAGE_KEY = "breeze-elf-settings-v1";
const SESSION_STORAGE_KEY = "breeze-elf-session-v1";
const AUDIO_DB_NAME = "breeze-elf-audio-v1";
const AUDIO_STORE_NAME = "sessions";
const AUDIO_RECORD_ID = "current";
const AUDIO_SAMPLE_RATE = 16000;
const AUDIO_CHANNEL_COUNT = 1;
const AUDIO_BYTES_PER_SAMPLE = 2;
const AUDIO_PERSIST_DELAY_MS = 1200;
const AUDIO_PREVIEW_DELAY_MS = 900;
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
const INITIAL_SETTINGS = readStoredSettings();
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
  pitchMode: SEARCH_PARAMS.has("pitch")
    ? SEARCH_PARAMS.get("pitch") === "1"
    : INITIAL_SETTINGS.pitchMode === true,
  droppedClientChunks: 0,
  statsTimer: 0,
  sessionPersistTimer: 0,
  audioChunks: [],
  audioBytes: 0,
  audioSampleRate: AUDIO_SAMPLE_RATE,
  audioObjectUrl: "",
  audioPersistTimer: 0,
  audioPreviewTimer: 0,
  audioDirty: false,
  audioStorageFailed: false,
  savingRemote: false,
  demoRunning: false,
  demoTimers: [],
};

let lastPitchTouchToggleAt = 0;
let audioDatabasePromise = null;

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

function readStoredSettings() {
  try {
    const data = JSON.parse(localStorage.getItem(SETTINGS_STORAGE_KEY) || "{}");
    return data && typeof data === "object" ? data : {};
  } catch {
    return {};
  }
}

function persistSettings() {
  if (DEMO_MODE) {
    return;
  }

  try {
    localStorage.setItem(
      SETTINGS_STORAGE_KEY,
      JSON.stringify({
        pitchMode: state.pitchMode,
      }),
    );
  } catch {
    flashStats("設定儲存失敗");
  }
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

function renderTranscript(text, { persist = true } = {}) {
  state.transcript = text;
  state.transcriptBlocks = [];
  renderTranscriptView();
  setTranscriptActions(Boolean(text.trim()));
  if (persist) {
    scheduleSessionPersist();
  }
}

function appendTranscript(text) {
  appendTranscriptBlock({ text });
}

function appendTranscriptBlock(data, { persist = true } = {}) {
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
  if (persist) {
    scheduleSessionPersist();
  }
}

function renderTranscriptView() {
  els.lines.classList.toggle("pitch-mode", state.pitchMode);
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

function setPitchMode(enabled, { persist = true } = {}) {
  state.pitchMode = Boolean(enabled);
  els.pitch.classList.toggle("active", state.pitchMode);
  els.pitch.textContent = "音高";
  els.pitch.setAttribute("aria-pressed", String(state.pitchMode));
  els.pitch.setAttribute("aria-label", state.pitchMode ? "隱藏音高模式" : "顯示音高模式");
  els.pitch.setAttribute("title", state.pitchMode ? "隱藏每段文字的音高" : "顯示每段文字的音高");
  renderTranscriptView();
  if (persist) {
    persistSettings();
  }
}

function togglePitchModeFromEvent(event) {
  const now = Date.now();
  const isPointer = event?.type === "pointerup";
  const isPointerTouch = isPointer && event.pointerType !== "mouse";
  const isTouch = event?.type === "touchend" || isPointerTouch;
  if (isPointer && !isPointerTouch) {
    return;
  }
  if (event?.type === "click" && now - lastPitchTouchToggleAt < 700) {
    return;
  }
  if (isTouch) {
    event.preventDefault();
    lastPitchTouchToggleAt = now;
  }
  setPitchMode(!state.pitchMode);
}

function bindPitchToggle() {
  if (window.PointerEvent) {
    els.pitch.addEventListener("pointerup", togglePitchModeFromEvent);
  } else {
    els.pitch.addEventListener("touchend", togglePitchModeFromEvent, { passive: false });
  }
  els.pitch.addEventListener("click", togglePitchModeFromEvent);
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

function readStoredSession() {
  try {
    const data = JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY) || "null");
    return data && typeof data === "object" ? data : null;
  } catch {
    return null;
  }
}

function normalizeTranscriptBlockForRestore(block) {
  const text = typeof block?.text === "string" ? block.text : "";
  if (!text) {
    return null;
  }

  return {
    text,
    startSeconds: finiteNumber(block.startSeconds),
    endSeconds: finiteNumber(block.endSeconds),
    segmentKind: block.segmentKind || "",
    windowIndex: Number.isInteger(block.windowIndex) ? block.windowIndex : null,
    pitch: normalizePitch(block.pitch),
  };
}

function restoreTranscriptSession() {
  if (DEMO_MODE) {
    return;
  }

  const session = readStoredSession();
  if (!session) {
    return;
  }

  const transcript = typeof session.transcript === "string" ? session.transcript : "";
  const blocks = Array.isArray(session.transcriptBlocks)
    ? session.transcriptBlocks.map(normalizeTranscriptBlockForRestore).filter(Boolean)
    : [];
  if (!transcript.trim() && blocks.length === 0) {
    return;
  }

  state.transcript = transcript || blocks.map((block) => block.text).join("");
  state.transcriptBlocks = blocks;
  renderTranscriptView();
  setTranscriptActions(Boolean(state.transcript.trim()));
  flashStats("已恢復本機記憶");
}

function scheduleSessionPersist() {
  if (DEMO_MODE) {
    return;
  }

  window.clearTimeout(state.sessionPersistTimer);
  state.sessionPersistTimer = window.setTimeout(persistSessionNow, 250);
}

function persistSessionNow() {
  if (DEMO_MODE) {
    return;
  }

  window.clearTimeout(state.sessionPersistTimer);
  state.sessionPersistTimer = 0;

  try {
    if (!state.transcript.trim() && state.transcriptBlocks.length === 0) {
      localStorage.removeItem(SESSION_STORAGE_KEY);
      return;
    }

    localStorage.setItem(
      SESSION_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        transcript: state.transcript,
        transcriptBlocks: state.transcriptBlocks,
        updatedAt: Date.now(),
      }),
    );
  } catch {
    flashStats("本機文字儲存失敗");
  }
}

function writeAscii(view, offset, text) {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index));
  }
}

function recordedAudioBlob() {
  const dataSize = state.audioChunks.reduce((total, chunk) => total + chunk.byteLength, 0);
  const header = new ArrayBuffer(44);
  const view = new DataView(header);
  const sampleRate = state.audioSampleRate || AUDIO_SAMPLE_RATE;
  const blockAlign = AUDIO_CHANNEL_COUNT * AUDIO_BYTES_PER_SAMPLE;
  const byteRate = sampleRate * blockAlign;

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, AUDIO_CHANNEL_COUNT, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, AUDIO_BYTES_PER_SAMPLE * 8, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataSize, true);

  return new Blob([header, ...state.audioChunks], { type: "audio/wav" });
}

function refreshAudioPreview() {
  window.clearTimeout(state.audioPreviewTimer);
  state.audioPreviewTimer = 0;

  if (state.audioObjectUrl) {
    URL.revokeObjectURL(state.audioObjectUrl);
    state.audioObjectUrl = "";
  }

  if (!state.audioBytes) {
    els.audioPanel.hidden = true;
    els.audioPlayer.removeAttribute("src");
    els.audioPlayer.load();
    els.audioDownload.disabled = true;
    return;
  }

  state.audioObjectUrl = URL.createObjectURL(recordedAudioBlob());
  els.audioPlayer.src = state.audioObjectUrl;
  els.audioPanel.hidden = false;
  els.audioDownload.disabled = false;
}

function scheduleAudioPreviewRefresh() {
  if (state.audioPreviewTimer) {
    return;
  }
  state.audioPreviewTimer = window.setTimeout(refreshAudioPreview, AUDIO_PREVIEW_DELAY_MS);
}

function appendRecordedAudioChunk(buffer) {
  if (DEMO_MODE || !(buffer instanceof ArrayBuffer) || buffer.byteLength === 0) {
    return;
  }

  const chunk = buffer.slice(0);
  state.audioChunks.push(chunk);
  state.audioBytes += chunk.byteLength;
  state.audioDirty = true;
  els.audioPanel.hidden = false;
  els.audioDownload.disabled = false;
  scheduleAudioPreviewRefresh();
  scheduleAudioPersist();
}

function openAudioDatabase() {
  if (!("indexedDB" in window)) {
    return Promise.resolve(null);
  }

  if (!audioDatabasePromise) {
    audioDatabasePromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(AUDIO_DB_NAME, 1);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(AUDIO_STORE_NAME)) {
          db.createObjectStore(AUDIO_STORE_NAME, { keyPath: "id" });
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error("IndexedDB failed"));
      request.onblocked = () => reject(new Error("IndexedDB blocked"));
    });
  }

  return audioDatabasePromise;
}

async function runAudioStore(operation, mode = "readonly") {
  const db = await openAudioDatabase();
  if (!db) {
    return null;
  }

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(AUDIO_STORE_NAME, mode);
    const store = transaction.objectStore(AUDIO_STORE_NAME);
    const request = operation(store);
    let result = null;

    request.onsuccess = () => {
      result = request.result;
    };
    request.onerror = () => reject(request.error || new Error("IndexedDB request failed"));
    transaction.oncomplete = () => resolve(result);
    transaction.onerror = () => reject(transaction.error || new Error("IndexedDB transaction failed"));
    transaction.onabort = () => reject(transaction.error || new Error("IndexedDB transaction aborted"));
  });
}

function scheduleAudioPersist() {
  if (DEMO_MODE) {
    return;
  }

  if (state.audioPersistTimer) {
    return;
  }
  state.audioPersistTimer = window.setTimeout(() => {
    void persistAudioSession();
  }, AUDIO_PERSIST_DELAY_MS);
}

async function persistAudioSession({ force = false } = {}) {
  if (DEMO_MODE || (!force && !state.audioDirty)) {
    return;
  }

  window.clearTimeout(state.audioPersistTimer);
  state.audioPersistTimer = 0;

  try {
    const persistedBytes = state.audioBytes;

    if (!state.audioBytes) {
      await runAudioStore((store) => store.delete(AUDIO_RECORD_ID), "readwrite");
      state.audioDirty = state.audioBytes !== persistedBytes;
      if (state.audioDirty) {
        scheduleAudioPersist();
      }
      return;
    }

    await runAudioStore(
      (store) =>
        store.put({
          id: AUDIO_RECORD_ID,
          pcm: new Blob(state.audioChunks, { type: "application/octet-stream" }),
          sampleRate: state.audioSampleRate,
          bytes: state.audioBytes,
          updatedAt: Date.now(),
        }),
      "readwrite",
    );
    state.audioDirty = state.audioBytes !== persistedBytes;
    if (state.audioDirty) {
      scheduleAudioPersist();
    }
    state.audioStorageFailed = false;
  } catch {
    if (!state.audioStorageFailed) {
      flashStats("本機錄音儲存失敗");
    }
    state.audioStorageFailed = true;
  }
}

async function restoreAudioSession() {
  if (DEMO_MODE) {
    return;
  }

  try {
    const record = await runAudioStore((store) => store.get(AUDIO_RECORD_ID));
    if (!record?.pcm || !record.bytes) {
      refreshAudioPreview();
      return;
    }

    const buffer = await record.pcm.arrayBuffer();
    state.audioChunks = [buffer];
    state.audioBytes = buffer.byteLength;
    state.audioSampleRate = Number.isFinite(record.sampleRate) ? record.sampleRate : AUDIO_SAMPLE_RATE;
    state.audioDirty = false;
    refreshAudioPreview();
  } catch {
    flashStats("本機錄音讀取失敗");
  }
}

async function clearRecordedAudio() {
  window.clearTimeout(state.audioPersistTimer);
  state.audioPersistTimer = 0;
  state.audioChunks = [];
  state.audioBytes = 0;
  state.audioDirty = false;
  state.audioStorageFailed = false;
  refreshAudioPreview();

  try {
    await runAudioStore((store) => store.delete(AUDIO_RECORD_ID), "readwrite");
  } catch {
    flashStats("本機錄音清除失敗");
  }
}

function downloadRecordedAudio() {
  if (!state.audioBytes) {
    return;
  }

  void persistAudioSession({ force: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const url = URL.createObjectURL(recordedAudioBlob());
  const link = document.createElement("a");
  link.href = url;
  link.download = `breeze-elf-audio-${stamp}.wav`;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  flashStats("音檔已下載");
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
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
      video: false,
    });

    state.audioContext = new AudioContext({ latencyHint: "interactive" });
    await state.audioContext.audioWorklet.addModule(
      new URL("audio-worklet.js", import.meta.url).href,
    );

    state.source = state.audioContext.createMediaStreamSource(state.stream);
    state.worklet = new AudioWorkletNode(state.audioContext, "breeze-mic-processor", {
      processorOptions: { targetSampleRate: AUDIO_SAMPLE_RATE, chunkMs: AUDIO_CHUNK_MS },
    });
    state.silence = state.audioContext.createGain();
    state.silence.gain.value = 0;

    state.worklet.port.onmessage = (event) => {
      if (event.data?.type !== "audio") {
        return;
      }
      renderLevel(event.data.rms || 0);
      appendRecordedAudioChunk(event.data.buffer);
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

    ws.send(JSON.stringify({ type: "start", sampleRate: AUDIO_SAMPLE_RATE, language: "zh", chunkMs: AUDIO_CHUNK_MS }));
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
setPitchMode(state.pitchMode, { persist: false });
restoreTranscriptSession();
void restoreAudioSession();

els.theme.addEventListener("click", toggleTheme);
bindPitchToggle();
SYSTEM_DARK_QUERY.addEventListener("change", syncSystemTheme);
els.start.addEventListener("click", start);
els.stop.addEventListener("click", stop);
els.clear.addEventListener("click", () => {
  renderTranscript("", { persist: false });
  els.partial.textContent = "";
  persistSessionNow();
  void clearRecordedAudio();
});
els.audioDownload.addEventListener("click", downloadRecordedAudio);
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

window.addEventListener("pagehide", () => {
  persistSessionNow();
  void persistAudioSession({ force: true });
});
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") {
    persistSessionNow();
    void persistAudioSession({ force: true });
  }
});

if ("serviceWorker" in navigator && window.isSecureContext) {
  navigator.serviceWorker.register(new URL("service-worker.js", location.href).href).catch(() => {});
}
