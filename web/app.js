const els = {
  toggle: document.querySelector("#toggle"),
  load: document.querySelector("#load"),
  fileInput: document.querySelector("#file"),
  clear: document.querySelector("#clear"),
  pitch: document.querySelector("#pitch"),
  copy: document.querySelector("#copy"),
  download: document.querySelector("#download"),
  save: document.querySelector("#save"),
  audioPanel: document.querySelector("#audio-panel"),
  audioPlayer: document.querySelector("#recording"),
  theme: document.querySelector("#theme"),
  themeColor: document.querySelector("meta[name='theme-color']"),
  status: document.querySelector("#status"),
  stats: document.querySelector("#stats"),
  lines: document.querySelector("#lines"),
  partial: document.querySelector("#partial"),
  backend: document.querySelector("#backend"),
  about: document.querySelector("#about"),
  aboutBackend: document.querySelector("#about-backend"),
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
    characters: demoCharacters("今天先確認 GitHub Actions 的靜態展示。", 0.35, 1.1, [
      "1",
      "2",
      "3",
      "5",
      "6̇",
      "5",
      "3",
      "2",
    ]),
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
    characters: demoCharacters(
      "麥克風、WebSocket 與遠端儲存都維持凍結，只呈現操作流程。",
      1.85,
      2.75,
      ["5̣", "6̣", "1", "2", "3", "2", "1", "6̣"],
    ),
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
  stopping: false,
  running: false,
  analyzing: false,
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
  fileLanguageShown: false,
  openBlocks: new Set(),
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

function demoCharacters(text, startSeconds, endSeconds, jianpu) {
  const chars = [...text].filter((char) => char.trim());
  if (chars.length === 0) {
    return [];
  }
  const step = (endSeconds - startSeconds) / chars.length;
  return chars.map((char, index) => {
    const charStart = startSeconds + index * step;
    const baseHz = 180 + ((index * 17) % 90);
    const isGlide = index % 4 === 3;
    const endHz = isGlide ? baseHz + 28 : baseHz + ((index % 3) - 1) * 4;
    const degree = jianpu[index % jianpu.length];
    const glideDegree = jianpu[(index + 2) % jianpu.length];
    const intensityStart = 0.05 + ((index * 7) % 30) / 1000;
    const intensityEnd = intensityStart * (isGlide ? 1.4 : index % 2 ? 0.72 : 1.05);
    return {
      char,
      startSeconds: Number(charStart.toFixed(3)),
      endSeconds: Number((charStart + step).toFixed(3)),
      durationSeconds: Number(step.toFixed(3)),
      hz: Number(baseHz.toFixed(1)),
      minHz: Number(Math.min(baseHz, endHz).toFixed(1)),
      maxHz: Number(Math.max(baseHz, endHz).toFixed(1)),
      startHz: Number(baseHz.toFixed(1)),
      endHz: Number(endHz.toFixed(1)),
      jianpu: isGlide ? `${degree}↗${glideDegree}` : degree,
      jianpuStart: degree,
      jianpuEnd: isGlide ? glideDegree : degree,
      isGlide,
      centsOff: ((index * 13) % 70) - 35,
      intensity: Number(((intensityStart + intensityEnd) / 2).toFixed(4)),
      intensityStart: Number(intensityStart.toFixed(4)),
      intensityEnd: Number(intensityEnd.toFixed(4)),
    };
  });
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

function openAbout() {
  if (!els.about) {
    return;
  }
  if (els.aboutBackend) {
    const label = els.backend.textContent?.trim();
    els.aboutBackend.textContent = label && label !== "ASR" ? label : "尚未連線";
  }
  if (typeof els.about.showModal === "function") {
    if (!els.about.open) {
      els.about.showModal();
    }
  } else {
    els.about.setAttribute("open", "");
  }
}

function setRunning(isRunning) {
  state.running = isRunning;
  els.toggle.disabled = false;
  els.toggle.classList.toggle("primary", !isRunning);
  els.toggle.classList.toggle("recording", isRunning);
  els.toggle.textContent = isRunning ? "■ 停止" : "▶ 開始";
  els.toggle.setAttribute("aria-pressed", String(isRunning));
  els.load.disabled = DEMO_MODE || isRunning;
}

function handleToggle() {
  if (state.running) {
    stop();
  } else {
    void start();
  }
}

function renderTranscript(text, { persist = true } = {}) {
  state.transcript = text;
  state.transcriptBlocks = [];
  state.openBlocks.clear();
  renderTranscriptView({ scrollToEnd: true });
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
    characters: normalizeCharacters(data.characters),
  });
  renderTranscriptView({ scrollToEnd: true });
  setTranscriptActions(Boolean(state.transcript.trim()));
  if (persist) {
    scheduleSessionPersist();
  }
}

function renderTranscriptView({ scrollToEnd = false } = {}) {
  const hasBlocks = state.transcriptBlocks.length > 0;
  const previousScroll = els.lines.scrollTop;
  const wasAtBottom = els.lines.scrollHeight - previousScroll - els.lines.clientHeight < 48;

  els.lines.classList.toggle("pitch-mode", state.pitchMode);
  els.lines.classList.toggle("block-mode", hasBlocks);
  els.lines.replaceChildren();

  if (hasBlocks) {
    const fragment = document.createDocumentFragment();
    state.transcriptBlocks.forEach((block, index) => {
      fragment.append(renderTranscriptEntry(block, index));
    });
    els.lines.append(fragment);
  } else {
    els.lines.textContent = state.transcript;
  }

  if (scrollToEnd || wasAtBottom) {
    els.lines.scrollTop = els.lines.scrollHeight;
  } else {
    els.lines.scrollTop = previousScroll;
  }
}

function renderTranscriptEntry(block, index) {
  const entry = document.createElement("div");
  entry.className = "transcript-entry";

  const characters = Array.isArray(block.characters) ? block.characters : [];
  const jianpuMode = state.pitchMode && characters.length > 0;
  const hasDetails = characters.length > 0 || Boolean(block.pitch);
  const open = hasDetails && state.openBlocks.has(index);

  const main = document.createElement("div");
  main.className = "entry-main";

  const meta = document.createElement("span");
  meta.className = "entry-meta";
  const range = document.createElement("span");
  range.textContent = formatTimeRange(block.startSeconds, block.endSeconds);
  meta.append(range);
  if (state.pitchMode && block.pitch) {
    const pitch = document.createElement("span");
    pitch.className = "pitch-value";
    pitch.textContent = formatPitch(block.pitch);
    meta.append(pitch);
  }
  if (hasDetails) {
    const chevron = document.createElement("span");
    chevron.className = "entry-chevron";
    chevron.setAttribute("aria-hidden", "true");
    chevron.textContent = "▸";
    meta.append(chevron);
  }

  if (jianpuMode) {
    entry.classList.add("jianpu");
    main.append(meta, renderJianpuLine(characters));
  } else {
    const text = document.createElement("span");
    text.className = "transcript-text";
    text.textContent = block.text.trimStart();
    const body = document.createElement("span");
    body.className = "entry-body";
    body.append(text);
    main.append(body, meta);
  }

  if (hasDetails) {
    entry.classList.add("expandable");
    main.setAttribute("role", "button");
    main.setAttribute("tabindex", "0");
    main.setAttribute("aria-expanded", String(open));
    main.dataset.index = String(index);
    main.setAttribute("aria-label", `${block.text.trim().slice(0, 24) || "段落"} 詳細資訊`);
  }
  entry.append(main);

  if (hasDetails) {
    const details = renderEntryDetails(block);
    details.hidden = !open;
    entry.classList.toggle("open", open);
    entry.append(details);
  }
  return entry;
}

function toggleEntryDetails(index) {
  if (!Number.isInteger(index)) {
    return;
  }
  if (state.openBlocks.has(index)) {
    state.openBlocks.delete(index);
  } else {
    state.openBlocks.add(index);
  }
  renderTranscriptView();
}

function renderEntryDetails(block) {
  const details = document.createElement("div");
  details.className = "entry-details";
  const characters = Array.isArray(block.characters) ? block.characters : [];
  if (characters.length > 0) {
    details.append(renderCharacterTable(characters));
  } else if (block.pitch) {
    details.append(renderPitchSummaryDetail(block.pitch));
  }
  return details;
}

function renderCharacterTable(characters) {
  const table = document.createElement("table");
  table.className = "char-table";

  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["字", "時間", "頻率", "音階", "強度"].forEach((label) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = label;
    headRow.append(th);
  });
  head.append(headRow);

  const tbody = document.createElement("tbody");
  characters.forEach((character) => tbody.append(renderCharacterRow(character)));

  table.append(head, tbody);
  return table;
}

function renderCharacterRow(character) {
  const row = document.createElement("tr");

  const charCell = document.createElement("td");
  charCell.className = "char-cell";
  const glyph = document.createElement("span");
  glyph.className = "char-glyph";
  glyph.textContent = character.char;
  charCell.append(glyph);
  if (character.jianpu) {
    const jp = document.createElement("span");
    jp.className = "char-jianpu";
    jp.textContent = character.jianpu;
    charCell.append(jp);
  }

  const timeCell = document.createElement("td");
  timeCell.textContent = formatCharTime(character);

  const freqCell = document.createElement("td");
  freqCell.textContent = formatCharFrequency(character);

  const scaleCell = document.createElement("td");
  renderScaleCell(character).forEach((node) => scaleCell.append(node));

  const intensityCell = document.createElement("td");
  intensityCell.append(renderIntensityCell(character));

  row.append(charCell, timeCell, freqCell, scaleCell, intensityCell);
  return row;
}

function formatCharTime(character) {
  if (!Number.isFinite(character.startSeconds) || !Number.isFinite(character.endSeconds)) {
    return "—";
  }
  const ms = Number.isFinite(character.durationSeconds)
    ? Math.round(character.durationSeconds * 1000)
    : Math.round((character.endSeconds - character.startSeconds) * 1000);
  return `${formatClockTime(character.startSeconds)}–${formatClockTime(character.endSeconds)} · ${ms}ms`;
}

function formatCharFrequency(character) {
  if (character.isGlide && Number.isFinite(character.startHz) && Number.isFinite(character.endHz)) {
    return `${Math.round(character.startHz)}→${Math.round(character.endHz)} Hz`;
  }
  if (Number.isFinite(character.hz)) {
    return `${Math.round(character.hz)} Hz`;
  }
  return "—";
}

function renderScaleCell(character) {
  const nodes = [];
  const degree = document.createElement("span");
  degree.className = "scale-degree";
  degree.textContent = character.jianpu || "—";
  nodes.push(degree);
  if (Number.isFinite(character.centsOff) && Number.isFinite(character.hz)) {
    const cents = document.createElement("span");
    cents.className = "scale-cents";
    const sign = character.centsOff > 0 ? "+" : "";
    cents.textContent = `${sign}${Math.round(character.centsOff)}¢`;
    if (Math.abs(character.centsOff) > 35) {
      cents.classList.add("off");
    }
    nodes.push(cents);
  }
  return nodes;
}

function renderIntensityCell(character) {
  const wrap = document.createElement("span");
  wrap.className = "intensity-cell";
  const trend = intensityTrend(character.intensityStart, character.intensityEnd);
  const arrow = document.createElement("span");
  arrow.className = `intensity-arrow ${trend.tone}`.trim();
  arrow.textContent = trend.arrow;
  const label = document.createElement("span");
  label.textContent =
    trend.db === null ? trend.label : `${trend.label} ${trend.db > 0 ? "+" : ""}${trend.db.toFixed(1)}dB`;
  wrap.append(arrow, label);
  return wrap;
}

function intensityTrend(start, end) {
  if (!(start > 0) || !(end > 0)) {
    return { label: "—", arrow: "·", db: null, tone: "" };
  }
  const db = 20 * Math.log10(end / start);
  if (db >= 1.5) {
    return { label: "漸強", arrow: "↗", db, tone: "up" };
  }
  if (db <= -1.5) {
    return { label: "漸弱", arrow: "↘", db, tone: "down" };
  }
  return { label: "持平", arrow: "→", db, tone: "flat" };
}

function renderPitchSummaryDetail(pitch) {
  const wrap = document.createElement("div");
  wrap.className = "pitch-summary-detail";
  const rows = [
    ["中位音高", Number.isFinite(pitch.medianHz) ? `${Math.round(pitch.medianHz)} Hz` : "未偵測"],
    [
      "音高範圍",
      Number.isFinite(pitch.minHz) && Number.isFinite(pitch.maxHz)
        ? `${Math.round(pitch.minHz)}–${Math.round(pitch.maxHz)} Hz`
        : "—",
    ],
    ["濁音比例", Number.isFinite(pitch.voicedRatio) ? `${Math.round(pitch.voicedRatio * 100)}%` : "—"],
  ];
  rows.forEach(([label, value]) => {
    const line = document.createElement("div");
    line.className = "summary-line";
    const key = document.createElement("span");
    key.className = "summary-key";
    key.textContent = label;
    const val = document.createElement("span");
    val.textContent = value;
    line.append(key, val);
    wrap.append(line);
  });
  wrap.append(renderPitchSpark(pitch));
  return wrap;
}

function renderJianpuLine(characters) {
  const line = document.createElement("div");
  line.className = "jianpu-line";
  characters.forEach((character) => {
    const cell = document.createElement("span");
    cell.className = character.jianpu ? "jianpu-char" : "jianpu-char rest";

    const jp = document.createElement("span");
    jp.className = "jp";
    jp.textContent = character.jianpu || "·";

    const ch = document.createElement("span");
    ch.className = "ch";
    ch.textContent = character.char;

    cell.append(jp, ch);
    if (Number.isFinite(character.hz)) {
      cell.title = `${character.char} · ${Math.round(character.hz)} Hz`;
    }
    line.append(cell);
  });
  return line;
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

function normalizeCharacters(characters) {
  if (!Array.isArray(characters)) {
    return [];
  }

  return characters
    .map((character) => ({
      char: typeof character?.char === "string" ? character.char : "",
      startSeconds: finiteNumber(character?.startSeconds),
      endSeconds: finiteNumber(character?.endSeconds),
      durationSeconds: finiteNumber(character?.durationSeconds),
      hz: finiteNumber(character?.hz),
      minHz: finiteNumber(character?.minHz),
      maxHz: finiteNumber(character?.maxHz),
      startHz: finiteNumber(character?.startHz),
      endHz: finiteNumber(character?.endHz),
      jianpu: typeof character?.jianpu === "string" ? character.jianpu : "",
      jianpuStart: typeof character?.jianpuStart === "string" ? character.jianpuStart : "",
      jianpuEnd: typeof character?.jianpuEnd === "string" ? character.jianpuEnd : "",
      isGlide: Boolean(character?.isGlide),
      centsOff: finiteNumber(character?.centsOff),
      intensity: finiteNumber(character?.intensity),
      intensityStart: finiteNumber(character?.intensityStart),
      intensityEnd: finiteNumber(character?.intensityEnd),
    }))
    .filter((character) => character.char);
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
  renderTranscriptView({ scrollToEnd: true });
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

function entryIndexFromEvent(event) {
  const main = event.target.closest?.(".entry-main[role='button']");
  if (!main || !els.lines.contains(main)) {
    return null;
  }
  const index = Number(main.dataset.index);
  return Number.isInteger(index) ? index : null;
}

function bindEntryToggle() {
  els.lines.addEventListener("click", (event) => {
    const index = entryIndexFromEvent(event);
    if (index !== null) {
      toggleEntryDetails(index);
    }
  });
  els.lines.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " " && event.key !== "Spacebar") {
      return;
    }
    const index = entryIndexFromEvent(event);
    if (index !== null) {
      event.preventDefault();
      toggleEntryDetails(index);
    }
  });
}

function startClock() {
  if (!els.clock) {
    return;
  }
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
  if (!els.level) {
    return;
  }
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
    characters: normalizeCharacters(block.characters),
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
  state.openBlocks.clear();
  renderTranscriptView({ scrollToEnd: true });
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
    return;
  }

  state.audioObjectUrl = URL.createObjectURL(recordedAudioBlob());
  els.audioPlayer.src = state.audioObjectUrl;
  els.audioPanel.hidden = false;
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

function connectSocket() {
  const ws = new WebSocket(websocketUrl());
  ws.binaryType = "arraybuffer";
  ws.addEventListener("message", handleServerMessage);
  ws.addEventListener("close", handleSocketClose);
  state.ws = ws;
  return ws;
}

function handleSocketClose() {
  cleanupAudio();
  state.analyzing = false;
  setRunning(false);
  if (els.status.classList.contains("error")) {
    // 保留啟動失敗的錯誤訊息,別被「待命」蓋掉
  } else if (state.stopping) {
    setStatus("待命");
  } else {
    setStatus("連線中斷", "error");
  }
  state.stopping = false;
  stopClock();
  state.ws = null;
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
  state.stopping = false;
  state.droppedClientChunks = 0;
  els.stats.textContent = "0 ms";
  renderLevel(0);

  try {
    const ws = connectSocket();
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
    console.error("startStreaming failed", error);
    state.stopping = true;
    cleanupAudio();
    stopClock();
    setRunning(false);
    state.ws?.close();
    const name = error?.name ? `${error.name}: ` : "";
    setStatus(`${name}${error?.message || "啟動失敗"}`, "error");
  }
}

async function analyzeFile(file) {
  if (DEMO_MODE) {
    flashStats("示意模式無法分析音檔");
    return;
  }
  if (state.running || !file) {
    return;
  }

  setStatus("解碼音檔");
  let decoded;
  try {
    decoded = await decodeAudioFile(file);
  } catch (error) {
    console.error("decodeAudioFile failed", error);
    setStatus("音檔解碼失敗", "error");
    flashStats(error?.message || "音檔解碼失敗");
    return;
  }

  if (!decoded.pcm.length) {
    setStatus("音檔無聲音", "error");
    return;
  }

  renderTranscript("", { persist: false });
  els.partial.textContent = "";
  await clearRecordedAudio();
  ingestLoadedAudio(decoded);

  state.analyzing = true;
  state.fileLanguageShown = false;
  state.stopping = false;
  state.droppedClientChunks = 0;
  setRunning(true);
  setStatus("分析音檔", "live");
  els.stats.textContent = "0 ms";
  renderLevel(0);

  try {
    const ws = connectSocket();
    await waitForOpen(ws);
    ws.send(
      JSON.stringify({
        type: "start",
        sampleRate: AUDIO_SAMPLE_RATE,
        // Loaded files may be music or non-Chinese audio; let the model detect
        // the language instead of forcing zh, which garbles melodies.
        language: "auto",
        chunkMs: AUDIO_CHUNK_MS,
        mode: "file",
      }),
    );
    startClock();
    await streamPcmToSocket(ws, decoded.pcm);
    if (!state.stopping && ws.readyState === WebSocket.OPEN) {
      state.stopping = true;
      setStatus("辨識中", "live");
      ws.send(JSON.stringify({ type: "stop", reason: "file" }));
    }
  } catch (error) {
    console.error("analyzeFile failed", error);
    state.stopping = true;
    stopClock();
    setRunning(false);
    state.ws?.close();
    setStatus(error?.message || "分析失敗", "error");
  }
}

async function streamPcmToSocket(ws, pcm) {
  const chunkSamples = Math.max(1, Math.round((AUDIO_SAMPLE_RATE * AUDIO_CHUNK_MS) / 1000));
  for (let offset = 0; offset < pcm.length; offset += chunkSamples) {
    if (state.stopping || ws.readyState !== WebSocket.OPEN) {
      return;
    }
    const slice = pcm.subarray(offset, offset + chunkSamples);
    ws.send(slice.slice().buffer);
    renderLevel(chunkRms(slice));
    while (ws.bufferedAmount > MAX_WS_BUFFERED_BYTES && ws.readyState === WebSocket.OPEN) {
      await delay(15);
    }
    await delay(0);
  }
}

function chunkRms(int16) {
  if (!int16.length) {
    return 0;
  }
  let sum = 0;
  for (let index = 0; index < int16.length; index += 1) {
    const value = int16[index] / 0x8000;
    sum += value * value;
  }
  return Math.sqrt(sum / int16.length);
}

function delay(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function decodeAudioFile(file) {
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
  const resampled = await resampleMono(mono, audioBuffer.sampleRate, AUDIO_SAMPLE_RATE);
  return { pcm: floatToInt16(resampled), sampleRate: AUDIO_SAMPLE_RATE };
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

function ingestLoadedAudio(decoded) {
  const buffer = decoded.pcm.buffer.slice(0);
  state.audioChunks = [buffer];
  state.audioBytes = buffer.byteLength;
  state.audioSampleRate = decoded.sampleRate;
  state.audioDirty = true;
  refreshAudioPreview();
  scheduleAudioPersist();
}

function stop() {
  if (DEMO_MODE) {
    stopDemo();
    return;
  }

  state.stopping = true;
  cleanupAudio();
  stopClock();

  if (state.ws?.readyState === WebSocket.OPEN) {
    els.toggle.disabled = true;
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
    const deviceLabel = data.computeType ? `${data.device}/${data.computeType}` : data.device;
    const modelLabel = data.model ? ` · ${data.model}` : "";
    els.backend.textContent = `${data.backend}${modelLabel} · ${deviceLabel} · ${data.segmenter || "audio"}`;
    if (!state.analyzing) {
      setStatus("收音中", "live");
    }
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
    if (state.analyzing && data.language && !state.fileLanguageShown) {
      state.fileLanguageShown = true;
      flashStats(`偵測語言 ${data.language}`);
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
setRunning(false);
setPitchMode(state.pitchMode, { persist: false });
restoreTranscriptSession();
void restoreAudioSession();

els.theme.addEventListener("click", toggleTheme);
els.backend.addEventListener("click", openAbout);
els.about?.addEventListener("click", (event) => {
  if (event.target === els.about) {
    els.about.close();
  }
});
bindPitchToggle();
bindEntryToggle();
SYSTEM_DARK_QUERY.addEventListener("change", syncSystemTheme);
els.toggle.addEventListener("click", handleToggle);
els.load.addEventListener("click", () => {
  if (!els.load.disabled) {
    els.fileInput.click();
  }
});
els.fileInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  event.target.value = "";
  if (file) {
    void analyzeFile(file);
  }
});
els.clear.addEventListener("click", () => {
  renderTranscript("", { persist: false });
  els.partial.textContent = "";
  persistSessionNow();
  void clearRecordedAudio();
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
    const payload = {
      text,
      title: transcriptTitle(text),
      sampleRate: state.audioSampleRate,
      blocks: serializeBlocksForSave(),
    };
    const audioBase64 = await encodeRecordedAudioBase64();
    if (audioBase64) {
      payload.audioBase64 = audioBase64;
    }

    const response = await fetch("/api/transcripts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      throw new Error(data.detail || "遠端儲存失敗");
    }
    const extras = [data.audioFilename ? "音檔" : "", data.jsonFilename ? "簡譜" : ""]
      .filter(Boolean)
      .join("+");
    const suffix = extras ? `(含${extras})` : "";
    flashStats(
      data.filename ? `已遠端儲存 ${data.filename}${suffix}` : "已遠端儲存",
      previousStats,
    );
  } catch (error) {
    flashStats(error.message || "遠端儲存失敗", previousStats);
  } finally {
    state.savingRemote = false;
    setTranscriptActions(Boolean(state.transcript.trim()));
  }
});

function serializeBlocksForSave() {
  return state.transcriptBlocks.map((block) => ({
    text: block.text,
    startSeconds: block.startSeconds,
    endSeconds: block.endSeconds,
    segmentKind: block.segmentKind || "",
    pitch: block.pitch || null,
    characters: Array.isArray(block.characters) ? block.characters : [],
  }));
}

async function encodeRecordedAudioBase64() {
  if (!state.audioBytes) {
    return "";
  }
  const buffer = await recordedAudioBlob().arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

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
