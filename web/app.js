const els = {
  toggle: document.querySelector("#toggle"),
  load: document.querySelector("#load"),
  fileInput: document.querySelector("#file"),
  scenario: document.querySelector("#scenario"),
  clear: document.querySelector("#clear"),
  lang: document.querySelector("#lang"),
  glossary: document.querySelector("#glossary"),
  translate: document.querySelector("#translate"),
  search: document.querySelector("#search"),
  searchDialog: document.querySelector("#search-dialog"),
  searchInput: document.querySelector("#search-input"),
  searchResults: document.querySelector("#search-results"),
  summarize: document.querySelector("#summarize"),
  summaryDialog: document.querySelector("#summary-dialog"),
  summaryBody: document.querySelector("#summary-body"),
  viewTabs: Array.from(document.querySelectorAll(".view-tab")),
  analyze: document.querySelector("#analyze"),
  analyzeProgress: document.querySelector("#analyze-progress"),
  analyzeProgressLabel: document.querySelector("#analyze-progress-label"),
  analyzeProgressPct: document.querySelector("#analyze-progress-pct"),
  analyzeProgressBar: document.querySelector("#analyze-progress-bar"),
  copy: document.querySelector("#copy"),
  download: document.querySelector("#download"),
  save: document.querySelector("#save"),
  audioPanel: document.querySelector("#audio-panel"),
  audioPlayer: document.querySelector("#recording"),
  audioDownload: document.querySelector("#audio-download"),
  audioSave: document.querySelector("#audio-save"),
  theme: document.querySelector("#theme"),
  settings: document.querySelector("#settings"),
  settingsPanel: document.querySelector("#settings-panel"),
  themeColor: document.querySelector("meta[name='theme-color']"),
  status: document.querySelector("#status"),
  stats: document.querySelector("#stats"),
  lines: document.querySelector("#lines"),
  partial: document.querySelector("#partial"),
  backend: document.querySelector("#backend"),
  about: document.querySelector("#about"),
  aboutBackend: document.querySelector("#about-backend"),
};

// Buttons are icon-only; swap the sprite glyph instead of writing text (which would wipe the icon).
function iconSvg(id) {
  return `<svg class="ic" aria-hidden="true"><use href="#${id}"></use></svg>`;
}

const AUDIO_CHUNK_MS = 250;
const MAX_WS_BUFFERED_BYTES = 256 * 1024;
const THEME_STORAGE_KEY = "breeze-elf-theme";
const SETTINGS_STORAGE_KEY = "breeze-elf-settings-v1";
const SESSION_STORAGE_KEY = "breeze-elf-session-v1";
const GLOSSARY_STORAGE_KEY = "breeze-elf-glossary-v1";
// 辨識語言設定:預設繁中＋英文,最多可多選 4 種,或切到「自由偵測」不限制。
const MAX_LANGUAGES = 4;
const MAX_GLOSSARY_ENTRIES = 200;
const MAX_GLOSSARY_TERM_CHARS = 80;
const DEFAULT_LANGUAGES = ["zh", "en"];
const LANGUAGE_OPTIONS = [
  { code: "zh", label: "繁體中文" },
  { code: "en", label: "English 英文" },
  { code: "ja", label: "日本語 日文" },
  { code: "ko", label: "한국어 韓文" },
  { code: "es", label: "Español 西班牙文" },
  { code: "fr", label: "Français 法文" },
  { code: "de", label: "Deutsch 德文" },
  { code: "vi", label: "Tiếng Việt 越南文" },
  { code: "th", label: "ไทย 泰文" },
  { code: "id", label: "Indonesia 印尼文" },
  { code: "ru", label: "Русский 俄文" },
  { code: "pt", label: "Português 葡萄牙文" },
];
const LANGUAGE_LABELS = new Map(LANGUAGE_OPTIONS.map((item) => [item.code, item.label]));
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
// VAD 範圍 overlay colour — the green band on the 基頻分析 spectrogram marking the
// time spans the system treats as 字詞 (a character sounding, after the 後處理 VAD
// extension covers sub-threshold onsets/tails).
const VAD_RANGE_COLOR = "rgba(122, 232, 160, 0.92)";
// Transcript display tabs: 文字 / 文字+簡譜 / 基頻分析. Declared up here because
// initialViewMode() (used in the state initializer below) reads it.
const VIEW_MODES = ["text", "jianpu", "spectrum"];
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
    partial: "麥克風、WebSocket 與雲端儲存都維持凍結。",
    rms: 0.05,
    stats: { asrMs: 22, segmentKind: "utterance" },
  },
  {
    delay: 2750,
    final: "\n麥克風、WebSocket 與雲端儲存都維持凍結，只呈現操作流程。",
    startSeconds: 1.85,
    endSeconds: 2.75,
    pitch: demoPitch(162, [154, 158, 166, 172, 160, 150]),
    characters: demoCharacters(
      "麥克風、WebSocket 與雲端儲存都維持凍結，只呈現操作流程。",
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
  // Per-block 基頻分析 (spectrogram + f0), produced by 後處理, parallel to
  // transcriptBlocks by index. Kept out of the persisted/saved transcript.
  analysisSpectra: [],
  viewMode: initialViewMode(),
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
  // 段落播放: a dedicated player seeks the recorded WAV to a block's time range.
  segmentPlayer: null,
  segmentPlayIndex: null,
  segmentPlayEnd: Infinity,
  segmentAudioUrl: "",
  segmentAudioUrlBytes: -1,
  savingRemote: false,
  savingAudio: false,
  analyzingPitch: false,
  fileLanguageShown: false,
  // Whether the server offers whole-file batched transcription (BREEZE_ASR_FILE_BATCH_SIZE>0);
  // resolved once from /health. null = not yet checked → fall back to WS streaming.
  fileBatchAvailable: null,
  openBlocks: new Set(),
  demoRunning: false,
  demoTimers: [],
  // 辨識語言限制 + 自由偵測 + 慣用詞庫,從 localStorage 還原(下方函式宣告已提升)。
  languages: sanitizeLanguages(INITIAL_SETTINGS.languages) || DEFAULT_LANGUAGES.slice(),
  autoDetectLang: Boolean(INITIAL_SETTINGS.autoDetectLang),
  // 載入音檔的後製情境:"general" 直接辨識,"music" 先 Demucs 分離人聲再辨識歌詞。
  scenario: INITIAL_SETTINGS.scenario === "music" ? "music" : "general",
  // 翻譯:辨識後把每段翻成繁體中文(伺服器需啟用 NLLB;預設關)。雙語堆疊顯示。
  translateEnabled: Boolean(INITIAL_SETTINGS.translateEnabled),
  glossary: loadStoredGlossary(),
  editingIndex: null,
};

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
        viewMode: state.viewMode,
        languages: state.languages,
        autoDetectLang: state.autoDetectLang,
        scenario: state.scenario,
        translateEnabled: state.translateEnabled,
      }),
    );
  } catch {
    flashStats("設定儲存失敗");
  }
}

// ── 辨識語言限制 + 慣用詞庫 ──────────────────────────────────────────────────
// Languages restrict recognition to the chosen set (default 繁中＋英文, 最多 4
// 種) unless 自由偵測 is on; the 慣用詞庫 holds 原詞→改成 corrections learned from
// transcript edits. Both ride along in the WebSocket start message so the server
// can bias Whisper's prompt and auto-replace the recognised text.

function sanitizeLanguages(value) {
  if (!Array.isArray(value)) {
    return null;
  }
  const seen = [];
  for (const item of value) {
    const code = String(item || "").trim().toLowerCase();
    if (code && code !== "auto" && !seen.includes(code)) {
      seen.push(code);
    }
    if (seen.length >= MAX_LANGUAGES) {
      break;
    }
  }
  return seen.length ? seen : null;
}

function loadStoredGlossary() {
  try {
    const data = JSON.parse(localStorage.getItem(GLOSSARY_STORAGE_KEY) || "[]");
    if (!Array.isArray(data)) {
      return [];
    }
    return data
      .map((entry) => ({
        from: String(entry?.from || "").trim(),
        to: String(entry?.to || "").trim(),
      }))
      .filter((entry) => entry.from && entry.to && entry.from !== entry.to)
      .slice(0, MAX_GLOSSARY_ENTRIES);
  } catch {
    return [];
  }
}

function persistGlossary() {
  if (DEMO_MODE) {
    return;
  }
  try {
    localStorage.setItem(GLOSSARY_STORAGE_KEY, JSON.stringify(state.glossary));
  } catch {
    flashStats("詞庫儲存失敗");
  }
}

// The languages payload sent to the server: ["auto"] for 自由偵測, otherwise the
// chosen set (never empty — falls back to the default).
function startLanguages() {
  if (state.autoDetectLang || !state.languages.length) {
    return ["auto"];
  }
  return state.languages.slice(0, MAX_LANGUAGES);
}

function languageLabel(code) {
  return LANGUAGE_LABELS.get(code) || code;
}

// A short label for the 語言 button: 自動 for free-detect, else the codes' 中文 names.
function languageButtonText() {
  if (state.autoDetectLang || !state.languages.length) {
    return "🌐 語言:自動";
  }
  const names = state.languages.map((code) => {
    const label = languageLabel(code);
    return label.split(/\s+/)[0];
  });
  const shown = names.slice(0, 2).join("・");
  const extra = names.length > 2 ? ` +${names.length - 2}` : "";
  return `🌐 語言:${shown}${extra}`;
}

function updateLangButton() {
  if (els.lang) {
    const text = languageButtonText().replace(/^🌐\s*/u, "");
    els.lang.title = text;
    els.lang.setAttribute("aria-label", text);
  }
}

function updateGlossaryButton() {
  if (els.glossary) {
    const count = state.glossary.length;
    const label = count ? `慣用詞庫(${count} 筆)` : "慣用詞庫";
    els.glossary.title = label;
    els.glossary.setAttribute("aria-label", label);
    els.glossary.dataset.count = count ? String(count) : "";
    els.glossary.classList.toggle("has-count", count > 0);
  }
}

function openLanguageDialog() {
  const dialog = document.querySelector("#lang-dialog");
  if (!dialog) {
    return;
  }
  const grid = dialog.querySelector("#lang-grid");
  const autoToggle = dialog.querySelector("#lang-auto");
  const confirmBtn = dialog.querySelector(".lang-confirm");
  const cancelBtn = dialog.querySelector(".lang-cancel");

  const working = state.languages.length ? state.languages.slice() : DEFAULT_LANGUAGES.slice();
  let auto = state.autoDetectLang;
  const inputs = new Map();

  grid.replaceChildren();
  LANGUAGE_OPTIONS.forEach((option) => {
    const label = document.createElement("label");
    label.className = "pick-item";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = option.code;
    checkbox.checked = working.includes(option.code);
    const span = document.createElement("span");
    span.textContent = option.label;
    label.append(checkbox, span);
    grid.append(label);
    inputs.set(option.code, checkbox);
    checkbox.addEventListener("change", () => {
      const checkedCount = [...inputs.values()].filter((box) => box.checked).length;
      if (checkedCount > MAX_LANGUAGES) {
        checkbox.checked = false;
        flashStats(`最多選 ${MAX_LANGUAGES} 種語言`);
      }
    });
  });

  const applyAutoState = () => {
    grid.querySelectorAll(".pick-item").forEach((item) => item.classList.toggle("disabled", auto));
  };
  autoToggle.checked = auto;
  applyAutoState();
  autoToggle.onchange = () => {
    auto = autoToggle.checked;
    applyAutoState();
  };

  let settled = false;
  const onClose = () => finish(false);
  const finish = (commit) => {
    if (settled) {
      return;
    }
    settled = true;
    confirmBtn.onclick = null;
    cancelBtn.onclick = null;
    autoToggle.onchange = null;
    dialog.removeEventListener("close", onClose);
    if (dialog.open) {
      dialog.close();
    }
    if (!commit) {
      return;
    }
    const chosen = LANGUAGE_OPTIONS.filter((option) => inputs.get(option.code).checked)
      .map((option) => option.code)
      .slice(0, MAX_LANGUAGES);
    state.autoDetectLang = auto;
    state.languages = chosen.length ? chosen : DEFAULT_LANGUAGES.slice();
    persistSettings();
    updateLangButton();
    flashStats(auto ? "語言:自由偵測" : `語言:${languageButtonText().replace("🌐 語言:", "")}`);
  };
  confirmBtn.onclick = () => finish(true);
  cancelBtn.onclick = () => finish(false);
  dialog.addEventListener("close", onClose);

  if (typeof dialog.showModal === "function") {
    dialog.showModal();
  } else {
    dialog.setAttribute("open", "");
  }
}

function openGlossaryDialog() {
  const dialog = document.querySelector("#glossary-dialog");
  if (!dialog) {
    return;
  }
  renderGlossaryList(dialog);
  const clearBtn = dialog.querySelector(".gloss-clear");
  const closeBtn = dialog.querySelector(".gloss-close");
  closeBtn.onclick = () => {
    if (dialog.open) {
      dialog.close();
    }
  };
  clearBtn.onclick = () => {
    if (!state.glossary.length) {
      return;
    }
    state.glossary = [];
    persistGlossary();
    updateGlossaryButton();
    renderGlossaryList(dialog);
    flashStats("已清空詞庫");
  };
  if (typeof dialog.showModal === "function") {
    dialog.showModal();
  } else {
    dialog.setAttribute("open", "");
  }
}

function renderGlossaryList(dialog) {
  const list = dialog.querySelector("#gloss-list");
  list.replaceChildren();
  if (!state.glossary.length) {
    const empty = document.createElement("div");
    empty.className = "gloss-empty";
    empty.textContent = "還沒有任何慣用詞。到逐字稿點 ✎ 改字就會自動學習。";
    list.append(empty);
    return;
  }
  state.glossary.forEach((entry, index) => {
    const item = document.createElement("div");
    item.className = "gloss-item";
    const pair = document.createElement("div");
    pair.className = "gloss-pair";
    const from = document.createElement("span");
    from.className = "gloss-from";
    from.textContent = entry.from;
    const arrow = document.createElement("span");
    arrow.className = "gloss-arrow";
    arrow.textContent = "→";
    const to = document.createElement("span");
    to.className = "gloss-to";
    to.textContent = entry.to;
    pair.append(from, arrow, to);
    const del = document.createElement("button");
    del.type = "button";
    del.className = "gloss-del";
    del.textContent = "✕";
    del.title = "刪除這個詞";
    del.addEventListener("click", () => {
      state.glossary.splice(index, 1);
      persistGlossary();
      updateGlossaryButton();
      renderGlossaryList(dialog);
    });
    item.append(pair, del);
    list.append(item);
  });
}

// Extract the single changed span between the old and new text (common prefix /
// suffix stripped) so an edit like 「機器學習很好」→「Mike 學習很好」 yields the
// minimal pair 機器→Mike rather than a whole-line replacement.
function diffTerms(oldText, newText) {
  let start = 0;
  const max = Math.min(oldText.length, newText.length);
  while (start < max && oldText[start] === newText[start]) {
    start += 1;
  }
  let oldEnd = oldText.length;
  let newEnd = newText.length;
  while (oldEnd > start && newEnd > start && oldText[oldEnd - 1] === newText[newEnd - 1]) {
    oldEnd -= 1;
    newEnd -= 1;
  }
  return { from: oldText.slice(start, oldEnd).trim(), to: newText.slice(start, newEnd).trim() };
}

// Record a 原詞→改成 correction (upserting by 原詞). Returns true when the 詞庫
// actually changed, so the caller can confirm the learning to the user.
function learnGlossary(from, to) {
  from = (from || "").trim();
  to = (to || "").trim();
  if (!from || !to || from === to) {
    return false;
  }
  if (from.length > MAX_GLOSSARY_TERM_CHARS || to.length > MAX_GLOSSARY_TERM_CHARS) {
    return false;
  }
  const existing = state.glossary.find((entry) => entry.from === from);
  if (existing) {
    if (existing.to === to) {
      return false;
    }
    existing.to = to;
  } else {
    state.glossary.push({ from, to });
    if (state.glossary.length > MAX_GLOSSARY_ENTRIES) {
      state.glossary.shift();
    }
  }
  persistGlossary();
  updateGlossaryButton();
  return true;
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
  document.querySelectorAll(".theme-toggle").forEach((button) => {
    button.textContent = nextTheme === "dark" ? "☀" : "☾";
    button.setAttribute("aria-label", switchLabel);
    button.setAttribute("title", switchLabel);
    button.setAttribute("aria-pressed", String(nextTheme === "dark"));
  });

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
  els.toggle.innerHTML = iconSvg(isRunning ? "i-stop" : "i-play");
  els.toggle.title = isRunning ? "停止辨識" : "開始辨識";
  els.toggle.setAttribute("aria-label", isRunning ? "停止辨識" : "開始辨識");
  els.toggle.setAttribute("aria-pressed", String(isRunning));
  els.load.disabled = DEMO_MODE || isRunning;
  setAudioActions();
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
  state.analysisSpectra = [];
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
    translation: typeof data.translation === "string" ? data.translation : null,
    speaker: Number.isInteger(data.speaker) ? data.speaker : null,
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

  els.lines.classList.toggle("block-mode", hasBlocks);
  els.lines.classList.toggle("view-jianpu", state.viewMode === "jianpu");
  els.lines.classList.toggle("view-spectrum", state.viewMode === "spectrum");
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
  const mode = state.viewMode;
  const jianpuMode = mode === "jianpu" && characters.length > 0;
  const spectrumMode = mode === "spectrum";
  // In 基頻分析 the spectrogram is shown inline (not behind the expand chevron).
  const hasDetails = !spectrumMode && (characters.length > 0 || Boolean(block.pitch));
  const open = hasDetails && state.openBlocks.has(index);

  const main = document.createElement("div");
  main.className = "entry-main";

  const meta = document.createElement("span");
  meta.className = "entry-meta";
  // 語者分離:匿名的 session 內說話者標籤(說話者 1／2…),顏色依索引區分。
  if (Number.isInteger(block.speaker)) {
    meta.append(createSpeakerChip(block.speaker));
  }
  // 段落播放 of the original recording — available in 文字/簡譜/基頻 views alike.
  const playButton = createSegmentPlayButton(block, index);
  if (playButton) {
    meta.append(playButton);
  }
  const range = document.createElement("span");
  range.textContent = formatTimeRange(block.startSeconds, block.endSeconds);
  meta.append(range);
  if (mode !== "text" && block.pitch) {
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
    // ✎ lets the user correct this block; corrections feed the 慣用詞庫. Hidden
    // in 示意模式 (no real recognition to improve) and while streaming.
    if (!DEMO_MODE) {
      meta.append(createBlockEditButton(index, text));
    }
    const body = document.createElement("span");
    body.className = "entry-body";
    body.append(text);
    // 雙語堆疊行:翻譯掛在整段之下(只在 final 產生,restore/儲存也帶著)。
    if (block.translation) {
      body.append(createTranslationLine(block.translation));
    }
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

  if (spectrumMode) {
    entry.classList.add("spectrum");
    entry.append(renderSpectrogram(state.analysisSpectra[index]));
  }

  if (hasDetails) {
    const details = renderEntryDetails(block);
    details.hidden = !open;
    entry.classList.toggle("open", open);
    entry.append(details);
  }
  return entry;
}

// Distinct chip colours per anonymous speaker (cycled; white text sits legibly on
// all of them). The server label is 0-based, so the UI shows `speaker + 1`.
const SPEAKER_COLORS = ["#2f80ed", "#27ae60", "#eb5757", "#9b51e0", "#f2994a", "#0aa2c0"];

function createSpeakerChip(speaker) {
  const chip = document.createElement("span");
  chip.className = "speaker-chip";
  chip.style.setProperty("--speaker-color", SPEAKER_COLORS[speaker % SPEAKER_COLORS.length]);
  chip.textContent = `說話者 ${speaker + 1}`;
  return chip;
}

function createTranslationLine(translation) {
  const line = document.createElement("span");
  line.className = "entry-translation";
  line.textContent = translation;
  return line;
}

function setTranslateEnabled(on) {
  state.translateEnabled = Boolean(on);
  persistSettings();
  updateTranslateButton();
  flashStats(state.translateEnabled ? "翻譯已開啟(下次開始生效)" : "翻譯已關閉");
}

function updateTranslateButton() {
  if (!els.translate) {
    return;
  }
  els.translate.classList.toggle("on", state.translateEnabled);
  els.translate.setAttribute("aria-pressed", String(state.translateEnabled));
}

// ── 逐字稿編輯 → 慣用詞庫學習 ─────────────────────────────────────────────────
// Each block's text can be edited in place; the diff between the original and the
// edit is learned as a 原詞→改成 correction, and the whole transcript is rebuilt
// from the blocks so 複製/下載/儲存 reflect the change.

function createBlockEditButton(index, textEl) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "entry-edit";
  button.textContent = "✎";
  button.title = "編輯這段文字(系統會記住你的更正,加入慣用詞庫)";
  button.setAttribute("aria-label", "編輯這段文字");
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    beginEditBlock(index, textEl);
  });
  return button;
}

function beginEditBlock(index, textEl) {
  if (state.editingIndex !== null) {
    return;
  }
  state.editingIndex = index;
  textEl.contentEditable = "true";
  textEl.spellcheck = false;
  textEl.focus();

  const selection = window.getSelection();
  if (selection) {
    const range = document.createRange();
    range.selectNodeContents(textEl);
    selection.removeAllRanges();
    selection.addRange(range);
  }

  const commit = () => {
    textEl.removeEventListener("blur", commit);
    textEl.removeEventListener("keydown", onKeydown);
    commitEditBlock(index, textEl);
  };
  const onKeydown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      textEl.blur();
    } else if (event.key === "Escape") {
      event.preventDefault();
      // Cancel: restore the original text, then commit (a no-op diff).
      const block = state.transcriptBlocks[index];
      if (block) {
        textEl.textContent = block.text.trimStart();
      }
      textEl.blur();
    }
  };
  textEl.addEventListener("blur", commit);
  textEl.addEventListener("keydown", onKeydown);
}

function commitEditBlock(index, textEl) {
  if (state.editingIndex !== index) {
    return;
  }
  state.editingIndex = null;
  textEl.contentEditable = "false";

  const block = state.transcriptBlocks[index];
  if (!block) {
    renderTranscriptView();
    return;
  }

  const oldDisplayed = block.text.trimStart();
  const leading = block.text.slice(0, block.text.length - oldDisplayed.length);
  const newDisplayed = textEl.textContent.replace(/\s+$/u, "");

  if (!newDisplayed || newDisplayed === oldDisplayed) {
    renderTranscriptView();
    return;
  }

  const { from, to } = diffTerms(oldDisplayed, newDisplayed);
  const learned = learnGlossary(from, to);

  block.text = leading + newDisplayed;
  state.transcript = state.transcriptBlocks.map((entry) => entry.text).join("").replace(/^\s+/u, "");
  renderTranscriptView();
  setTranscriptActions(Boolean(state.transcript.trim()));
  scheduleSessionPersist();
  flashStats(learned ? `已學習:${from} → ${to}` : "已更新文字");
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

// ── 段落播放 ────────────────────────────────────────────────────────────────
// Each transcript block carries absolute start/end seconds, so playback just
// seeks the recorded WAV and stops at the block's end. One shared player and a
// single "currently playing" index keep the ▶/⏸ buttons in sync across views.

function createSegmentPlayButton(block, index) {
  if (!Number.isFinite(block.startSeconds)) {
    return null;
  }
  const button = document.createElement("button");
  button.type = "button";
  button.className = "seg-play";
  button.dataset.segIndex = String(index);
  paintSegmentButton(button, state.segmentPlayIndex === index);
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    toggleSegmentPlayback(block, index);
  });
  return button;
}

function paintSegmentButton(button, playing) {
  button.textContent = playing ? "⏸" : "▶";
  button.classList.toggle("playing", playing);
  const title = playing ? "暫停這個段落" : "播放這個段落";
  button.title = title;
  button.setAttribute("aria-label", title);
}

function paintSegmentButtons() {
  els.lines.querySelectorAll(".seg-play").forEach((button) => {
    paintSegmentButton(button, Number(button.dataset.segIndex) === state.segmentPlayIndex);
  });
}

function ensureSegmentPlayer() {
  if (!state.segmentPlayer) {
    const player = new Audio();
    player.addEventListener("timeupdate", () => {
      if (Number.isFinite(state.segmentPlayEnd) && player.currentTime >= state.segmentPlayEnd) {
        stopSegmentPlayback();
      }
    });
    player.addEventListener("ended", stopSegmentPlayback);
    player.addEventListener("error", stopSegmentPlayback);
    state.segmentPlayer = player;
  }
  return state.segmentPlayer;
}

// Reuse one object URL for the whole recording, rebuilt only when the audio grew.
function segmentAudioUrl() {
  if (state.segmentAudioUrl && state.segmentAudioUrlBytes === state.audioBytes) {
    return state.segmentAudioUrl;
  }
  if (state.segmentAudioUrl) {
    URL.revokeObjectURL(state.segmentAudioUrl);
  }
  state.segmentAudioUrl = URL.createObjectURL(recordedAudioBlob());
  state.segmentAudioUrlBytes = state.audioBytes;
  return state.segmentAudioUrl;
}

function stopSegmentPlayback() {
  if (state.segmentPlayer) {
    state.segmentPlayer.pause();
  }
  if (state.segmentPlayIndex !== null) {
    state.segmentPlayIndex = null;
    state.segmentPlayEnd = Infinity;
    paintSegmentButtons();
  }
}

function toggleSegmentPlayback(block, index) {
  if (state.segmentPlayIndex === index) {
    stopSegmentPlayback();
    return;
  }
  stopSegmentPlayback();
  if (!state.audioBytes) {
    flashStats("沒有可播放的錄音");
    return;
  }
  const start = Number(block.startSeconds);
  if (!Number.isFinite(start)) {
    flashStats("這個段落沒有時間資訊");
    return;
  }
  const end = Number(block.endSeconds);
  const player = ensureSegmentPlayer();
  const url = segmentAudioUrl();
  if (player.src !== url) {
    player.src = url;
  }
  state.segmentPlayIndex = index;
  state.segmentPlayEnd = Number.isFinite(end) && end > start ? end : Infinity;
  paintSegmentButtons();

  const begin = () => {
    try {
      player.currentTime = Math.max(0, start);
    } catch {
      /* seek attempted before metadata is ready — ignore and play from 0 */
    }
    player.play().catch(() => {
      flashStats("無法播放段落");
      stopSegmentPlayback();
    });
  };
  if (player.readyState >= 1 /* HAVE_METADATA */) {
    begin();
  } else {
    player.addEventListener("loadedmetadata", begin, { once: true });
  }
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
  if (state.viewMode !== "text" && pitch) {
    els.partial.textContent = `${text}\n${formatPitch(pitch)}`;
    return;
  }
  els.partial.textContent = text;
}

// 基頻分析:render a block's STFT spectrogram (uint8 magnitudes from 後處理) with
// the continuous time→基頻 curve overlaid; non-voiced frames (null f0) leave gaps.
function renderSpectrogram(spectro) {
  const wrap = document.createElement("div");
  wrap.className = "spectro-wrap";
  if (!spectro || !spectro.magnitudes || !spectro.timeBins || !spectro.freqBins) {
    const empty = document.createElement("div");
    empty.className = "spectro-empty";
    empty.textContent = "尚無基頻分析資料,請先點上方的「後處理」。";
    wrap.append(empty);
    return wrap;
  }

  const timeBins = spectro.timeBins;
  const freqBins = spectro.freqBins;
  const maxHz = spectro.maxHz || 2000;
  const magnitudes = base64ToUint8(spectro.magnitudes);

  // Paint the magnitudes onto a low-res offscreen canvas (one pixel per bin),
  // then scale it up smoothly onto the display canvas.
  const source = document.createElement("canvas");
  source.width = timeBins;
  source.height = freqBins;
  const sctx = source.getContext("2d");
  const image = sctx.createImageData(timeBins, freqBins);
  for (let t = 0; t < timeBins; t += 1) {
    for (let f = 0; f < freqBins; f += 1) {
      const value = magnitudes[t * freqBins + f] || 0;
      const [r, g, b] = spectroColor(value);
      const y = freqBins - 1 - f; // low freq at the bottom
      const offset = (y * timeBins + t) * 4;
      image.data[offset] = r;
      image.data[offset + 1] = g;
      image.data[offset + 2] = b;
      image.data[offset + 3] = 255;
    }
  }
  sctx.putImageData(image, 0, 0);

  const width = 480;
  const height = 132;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(source, 0, 0, timeBins, freqBins, 0, 0, width, height);

  // Overlay the f0 curve on the same freq axis (0–maxHz), breaking at nulls.
  const f0 = Array.isArray(spectro.f0) ? spectro.f0 : [];
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#7df9ff";
  ctx.lineJoin = "round";
  ctx.beginPath();
  let drawing = false;
  f0.forEach((hz, index) => {
    if (hz == null || !Number.isFinite(hz)) {
      drawing = false;
      return;
    }
    const x = f0.length > 1 ? (index / (f0.length - 1)) * width : width / 2;
    const y = height - Math.min(1, hz / maxHz) * height;
    if (drawing) {
      ctx.lineTo(x, y);
    } else {
      ctx.moveTo(x, y);
      drawing = true;
    }
  });
  ctx.stroke();

  // VAD 範圍:a green band along the top edge over the time spans where a 字 is
  // sounding (text-present bins, already grown to cover the 字's onset/tail by the
  // 後處理 VAD extension); the gaps between bands are the rests / silences.
  const texts = Array.isArray(spectro.text) ? spectro.text : [];
  const bins = Math.max(f0.length, texts.length);
  if (texts.length && bins > 1) {
    const binW = width / (bins - 1);
    ctx.fillStyle = VAD_RANGE_COLOR;
    let runStart = -1;
    const flushVadBand = (end) => {
      if (runStart < 0) {
        return;
      }
      const x0 = Math.max(0, (runStart / (bins - 1)) * width - binW / 2);
      const x1 = Math.min(width, (end / (bins - 1)) * width + binW / 2);
      ctx.fillRect(x0, 1, Math.max(1, x1 - x0), 4);
      runStart = -1;
    };
    for (let i = 0; i < bins; i += 1) {
      if ((texts[i] || "").toString().trim()) {
        if (runStart < 0) {
          runStart = i;
        }
      } else {
        flushVadBand(i - 1);
      }
    }
    flushVadBand(bins - 1);
  }

  wrap.append(canvas);
  const legend = document.createElement("div");
  legend.className = "spectro-legend";
  const left = document.createElement("span");
  left.textContent = `0–${Math.round(maxHz)} Hz`;
  const mid = document.createElement("span");
  mid.className = "spectro-legend-vad";
  mid.textContent = "▬ VAD範圍";
  const right = document.createElement("span");
  right.textContent = "基頻曲線";
  legend.append(left, mid, right);
  wrap.append(legend);
  return wrap;
}

// Inferno-ish ramp (dark → purple → orange → pale) readable on the dark canvas.
function spectroColor(value) {
  const stops = [
    [0.0, 12, 14, 32],
    [0.25, 64, 22, 96],
    [0.5, 146, 42, 94],
    [0.7, 222, 92, 52],
    [0.85, 250, 172, 56],
    [1.0, 255, 252, 206],
  ];
  const t = value / 255;
  for (let i = 1; i < stops.length; i += 1) {
    if (t <= stops[i][0]) {
      const [t0, r0, g0, b0] = stops[i - 1];
      const [t1, r1, g1, b1] = stops[i];
      const k = (t - t0) / (t1 - t0 || 1);
      return [
        Math.round(r0 + (r1 - r0) * k),
        Math.round(g0 + (g1 - g0) * k),
        Math.round(b0 + (b1 - b0) * k),
      ];
    }
  }
  return [255, 252, 206];
}

function base64ToUint8(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
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
    return "音階未偵測";
  }
  return `音階 ${Math.round(pitch.medianHz)} Hz`;
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

function initialViewMode() {
  const requested = SEARCH_PARAMS.get("view");
  if (VIEW_MODES.includes(requested)) {
    return requested;
  }
  if (SEARCH_PARAMS.has("pitch")) {
    return SEARCH_PARAMS.get("pitch") === "1" ? "jianpu" : "text";
  }
  if (VIEW_MODES.includes(INITIAL_SETTINGS.viewMode)) {
    return INITIAL_SETTINGS.viewMode;
  }
  // Migrate the old boolean 音高 setting to the matching tab.
  return INITIAL_SETTINGS.pitchMode === true ? "jianpu" : "text";
}

function setViewMode(mode, { persist = true } = {}) {
  state.viewMode = VIEW_MODES.includes(mode) ? mode : "text";
  els.viewTabs.forEach((tab) => {
    tab.setAttribute("aria-selected", String(tab.dataset.view === state.viewMode));
  });
  renderTranscriptView({ scrollToEnd: true });
  reapplyCaptureConstraints();
  if (persist) {
    persistSettings();
  }
}

function bindViewTabs() {
  els.viewTabs.forEach((tab) => {
    tab.addEventListener("click", () => setViewMode(tab.dataset.view));
  });
}

function entryIndexFromEvent(event) {
  // The 段落播放 / 編輯 buttons live inside entry-main but drive their own action,
  // and an in-progress inline edit must not toggle the details panel either.
  if (
    event.target.closest?.(".seg-play") ||
    event.target.closest?.(".entry-edit") ||
    event.target.closest?.("[contenteditable='true']") ||
    state.editingIndex !== null
  ) {
    return null;
  }
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
  if (els.summarize) {
    els.summarize.disabled = DEMO_MODE || !hasTranscript;
  }
  updateAnalyzeButton();
}

// 校準音準 needs both the recorded/loaded audio and a structured (per-character)
// transcript to recompute against; it is pointless (and disabled) otherwise.
function updateAnalyzeButton() {
  if (!els.analyze) {
    return;
  }
  els.analyze.disabled =
    DEMO_MODE ||
    state.analyzingPitch ||
    state.running ||
    !state.audioBytes ||
    !hasStructuredTranscript();
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
    translation: typeof block.translation === "string" ? block.translation : null,
    speaker: Number.isInteger(block.speaker) ? block.speaker : null,
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

// --- 跨稿搜尋 ---------------------------------------------------------------

function openSearchDialog() {
  if (DEMO_MODE) {
    flashStats("示意模式無法搜尋");
    return;
  }
  const dialog = els.searchDialog;
  if (!dialog || typeof dialog.showModal !== "function") {
    return;
  }
  els.searchInput.value = "";
  dialog.showModal();
  void runSearch(""); // browse newest-first on open
  els.searchInput.focus();
}

function scheduleSearch() {
  window.clearTimeout(state.searchTimer);
  state.searchTimer = window.setTimeout(() => void runSearch(els.searchInput.value), 200);
}

async function runSearch(query) {
  const list = els.searchResults;
  if (!list) {
    return;
  }
  // Guard against out-of-order responses (a slow earlier query resolving after a
  // faster later one) clobbering the results for what's currently in the box.
  const seq = (state.searchSeq = (state.searchSeq || 0) + 1);
  try {
    const params = new URLSearchParams({ q: query.trim() });
    const response = await fetch(`/api/transcripts/search?${params}`);
    if (seq !== state.searchSeq) {
      return;
    }
    if (!response.ok) {
      list.textContent = response.status === 503 ? "搜尋功能未啟用" : "搜尋失敗";
      return;
    }
    const data = await response.json();
    if (seq !== state.searchSeq) {
      return;
    }
    renderSearchResults(Array.isArray(data.results) ? data.results : []);
  } catch (error) {
    if (seq !== state.searchSeq) {
      return;
    }
    console.error("search failed", error);
    list.textContent = "搜尋失敗";
  }
}

function renderSearchResults(results) {
  const list = els.searchResults;
  list.replaceChildren();
  if (!results.length) {
    const empty = document.createElement("p");
    empty.className = "lang-hint";
    empty.textContent = "沒有符合的逐字稿。";
    list.append(empty);
    return;
  }
  for (const item of results) {
    list.append(buildSearchResult(item));
  }
}

function buildSearchResult(item) {
  const entry = document.createElement("button");
  entry.type = "button";
  entry.className = "search-result";

  const title = document.createElement("span");
  title.className = "search-result-title";
  title.textContent = item.title || item.filename || item.id;

  const meta = document.createElement("span");
  meta.className = "search-result-meta";
  const badges = [item.hasAudio ? "🔊" : "", item.hasJianpu ? "🎵" : ""].filter(Boolean).join(" ");
  meta.textContent = badges ? `${formatSearchDate(item.createdAt)} · ${badges}` : formatSearchDate(item.createdAt);

  entry.append(title, meta);
  if (item.snippet) {
    const snippet = document.createElement("span");
    snippet.className = "search-result-snippet";
    snippet.textContent = item.snippet;
    entry.append(snippet);
  }
  entry.addEventListener("click", () => void openSearchResult(item.id));
  return entry;
}

function formatSearchDate(value) {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

async function openSearchResult(id) {
  // Never silently clobber an unsaved live/edited transcript with a stored one.
  if (state.running) {
    flashStats("請先停止收音再開啟");
    return;
  }
  if (
    hasUnsavedTranscript() &&
    !window.confirm("開啟這份逐字稿會覆蓋目前畫面的內容,是否繼續?")
  ) {
    return;
  }
  try {
    const response = await fetch(`/api/transcripts/${encodeURIComponent(id)}`);
    if (!response.ok) {
      flashStats("讀取逐字稿失敗");
      return;
    }
    loadTranscriptRecord(await response.json());
    els.searchDialog?.close();
  } catch (error) {
    console.error("open transcript failed", error);
    flashStats("讀取逐字稿失敗");
  }
}

function hasUnsavedTranscript() {
  return Boolean(state.transcript.trim()) || state.transcriptBlocks.length > 0;
}

function loadTranscriptRecord(data) {
  const blocks = Array.isArray(data.blocks)
    ? data.blocks.map(normalizeTranscriptBlockForRestore).filter(Boolean)
    : [];
  const text = typeof data.text === "string" ? data.text : "";
  state.transcript = text || blocks.map((block) => block.text).join("");
  state.transcriptBlocks = blocks;
  state.openBlocks.clear();
  // Drop any prior 基頻分析 spectra — they are index-parallel to the OLD blocks; left
  // stale they'd render under the new text and poison hasAnalysis()/the CSV export.
  state.analysisSpectra = [];
  renderTranscriptView({ scrollToEnd: false });
  setTranscriptActions(Boolean(state.transcript.trim()));
  persistSessionNow();
  flashStats(`已開啟:${data.title || data.id}`);
}

// --- 會後摘要 ---------------------------------------------------------------

async function openSummary() {
  if (DEMO_MODE) {
    flashStats("示意模式無法摘要");
    return;
  }
  const text = state.transcript.trim();
  if (!text) {
    flashStats("沒有可摘要的逐字稿");
    return;
  }
  const dialog = els.summaryDialog;
  if (!dialog || typeof dialog.showModal !== "function") {
    return;
  }
  const body = els.summaryBody;
  const copyButton = dialog.querySelector(".summary-copy");
  body.textContent = "摘要中…";
  body.classList.add("loading");
  if (copyButton) {
    copyButton.disabled = true;
  }
  dialog.showModal();

  const seq = (state.summarySeq = (state.summarySeq || 0) + 1);
  try {
    const response = await fetch("/api/summary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (seq !== state.summarySeq || !dialog.open) {
      return;
    }
    if (!response.ok) {
      body.textContent = response.status === 503 ? "摘要功能未啟用" : "摘要失敗";
      body.classList.remove("loading");
      return;
    }
    const data = await response.json();
    if (seq !== state.summarySeq || !dialog.open) {
      return;
    }
    const summary = (data.summary || "").trim();
    body.textContent = summary || "(無摘要)";
    body.classList.remove("loading");
    if (copyButton && summary) {
      copyButton.disabled = false;
    }
  } catch (error) {
    if (seq !== state.summarySeq) {
      return;
    }
    console.error("summary failed", error);
    body.textContent = "摘要失敗";
    body.classList.remove("loading");
  }
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

function setAudioActions() {
  const hasAudio = state.audioBytes > 0;
  els.audioDownload.disabled = !hasAudio;
  els.audioSave.disabled = DEMO_MODE || !hasAudio || state.savingAudio || state.running;
  // Audio availability gates 校準音準 too, so keep that button in sync.
  updateAnalyzeButton();
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
    setAudioActions();
    return;
  }

  state.audioObjectUrl = URL.createObjectURL(recordedAudioBlob());
  els.audioPlayer.src = state.audioObjectUrl;
  els.audioPanel.hidden = false;
  setAudioActions();
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
  stopSegmentPlayback();
  if (state.segmentAudioUrl) {
    URL.revokeObjectURL(state.segmentAudioUrl);
    state.segmentAudioUrl = "";
    state.segmentAudioUrlBytes = -1;
  }
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
  els.save.setAttribute("title", "示意模式不會寫入遠端主機");
  els.save.setAttribute("aria-label", "雲端儲存已凍結");
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

// 收音設定檔:純逐字稿(text 檢視)時開啟瀏覽器端回音消除與降噪,讓 Whisper 拿到乾淨音訊;
// 一旦進入音高/簡譜/基頻(jianpu/spectrum)或音樂情境,就退回 raw(全關)保住基頻分析準度。
function wantsRawCapture() {
  return state.viewMode !== "text" || state.scenario === "music";
}

function captureConstraints() {
  const raw = wantsRawCapture();
  return {
    channelCount: AUDIO_CHANNEL_COUNT,
    echoCancellation: !raw,
    noiseSuppression: !raw,
    // 自動增益(AGC)一律關閉,即使逐字稿模式:伺服器端 prepare_asr_audio 已做 RMS 正規化
    // (瀏覽器 AGC 因此多餘),而 AGC 會把靜段的 RMS 抬高,反而破壞 server 端以 RMS 校準的
    // VAD 起點門檻與靜音幻覺過濾(asr_hallucination_rms_threshold)。降噪則相反、會壓低噪聲,
    // 對兩者都有利,所以逐字稿模式只開 EC+NS。
    autoGainControl: false,
  };
}

// 錄音中切換檢視/情境時,盡量即時套用新設定檔;部分行動瀏覽器不支援對已開啟的 track
// applyConstraints,失敗就維持啟動時的設定(下次重錄才生效),不中斷收音。
function reapplyCaptureConstraints() {
  const track = state.stream?.getAudioTracks?.()?.[0];
  if (!track || track.readyState !== "live") {
    return;
  }
  track.applyConstraints(captureConstraints()).catch((error) => {
    console.warn("applyConstraints failed", error);
  });
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
      audio: captureConstraints(),
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

    ws.send(
      JSON.stringify({
        type: "start",
        sampleRate: AUDIO_SAMPLE_RATE,
        languages: startLanguages(),
        glossary: state.glossary,
        chunkMs: AUDIO_CHUNK_MS,
        translate: state.translateEnabled,
      }),
    );
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

  // 音樂情境:先用 Demucs 分離人聲,後續辨識/基頻/播放都跑在乾淨人聲上,避免
  // Whisper 在整段混音上幻覺。分離失敗就退回原始混音,至少還能辨識。
  if (state.scenario === "music") {
    try {
      setStatus("分離人聲中(Demucs)", "live");
      const vocals = await separateVocals(decoded.pcm, decoded.sampleRate);
      if (vocals && vocals.length) {
        decoded = { pcm: vocals, sampleRate: decoded.sampleRate };
      }
    } catch (error) {
      console.error("separateVocals failed", error);
      flashStats(error?.message || "人聲分離失敗,改用原始音檔");
    }
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

  // Long recordings: one batched pass (3-4x) when the server offers it, instead of
  // streaming utterance-by-utterance. No live partials; per-character 基頻/簡譜 comes
  // from 校準音準 (the offline analyze pass) afterwards.
  await ensureServerCapabilities();
  if (state.fileBatchAvailable) {
    startClock();
    try {
      await transcribeFileBatched(decoded);
      setStatus("待命");
    } catch (error) {
      console.error("batched analyzeFile failed", error);
      setStatus(error?.message || "分析失敗", "error");
    } finally {
      state.analyzing = false;
      state.stopping = false;
      setRunning(false);
      stopClock();
    }
    return;
  }

  try {
    const ws = connectSocket();
    await waitForOpen(ws);
    ws.send(
      JSON.stringify({
        type: "start",
        sampleRate: AUDIO_SAMPLE_RATE,
        // Loaded files may be music or non-Chinese audio; let the model detect
        // the language instead of forcing zh, which garbles melodies. The 慣用詞庫
        // corrections still apply to whatever is recognised.
        languages: ["auto"],
        glossary: state.glossary,
        chunkMs: AUDIO_CHUNK_MS,
        mode: "file",
        translate: state.translateEnabled,
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

let capabilitiesPromise = null;

// Resolve server capabilities once (cached). Only /health.fileBatchAvailable is needed
// today; a failed fetch leaves the flag false so file loads use the WS streaming path.
function ensureServerCapabilities() {
  if (!capabilitiesPromise) {
    capabilitiesPromise = fetch("/health")
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        state.fileBatchAvailable = Boolean(data?.fileBatchAvailable);
      })
      .catch(() => {
        state.fileBatchAvailable = false;
      });
  }
  return capabilitiesPromise;
}

async function transcribeFileBatched(decoded) {
  setStatus("辨識中(批次)", "live");
  const response = await fetch("/api/transcribe/file", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      pcmBase64: pcmToBase64(decoded.pcm),
      sampleRate: decoded.sampleRate,
      // Loaded files use free detection (see the streaming path); 慣用詞庫 + 翻譯 still apply.
      languages: ["auto"],
      glossary: state.glossary,
      translate: state.translateEnabled,
    }),
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      detail = (await response.json())?.detail || detail;
    } catch {
      // non-JSON error body; keep the status line
    }
    throw new Error(detail);
  }
  const data = await response.json();
  (data.blocks || []).forEach((block) => appendTranscriptBlock(block, { persist: false }));
  // The server returns the space-joined transcript; keep it for copy/save/search
  // (block display already covers the on-screen view).
  if (typeof data.transcript === "string" && data.transcript) {
    state.transcript = data.transcript;
  }
  setTranscriptActions(Boolean(state.transcript.trim()));
  scheduleSessionPersist();
  if (data.language) {
    flashStats(`偵測語言 ${data.language}`);
  }
}

async function separateVocals(pcm, sampleRate) {
  const response = await fetch("/api/enhance/separate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pcmBase64: pcmToBase64(pcm), sampleRate, scenario: "music" }),
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      detail = (await response.json())?.detail || detail;
    } catch {
      // non-JSON error body; keep the status line
    }
    throw new Error(detail);
  }
  const data = await response.json();
  return base64ToPcm(data.pcmBase64);
}

function pcmToBase64(int16) {
  const bytes = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);
  let binary = "";
  const CHUNK = 0x8000; // chunked so a big song does not blow the arg-count limit
  for (let offset = 0; offset < bytes.length; offset += CHUNK) {
    binary += String.fromCharCode.apply(null, bytes.subarray(offset, offset + CHUNK));
  }
  return btoa(binary);
}

function base64ToPcm(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Int16Array(bytes.buffer, 0, bytes.byteLength >> 1);
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
setViewMode(state.viewMode, { persist: false });
restoreTranscriptSession();
void restoreAudioSession();

document.querySelectorAll(".theme-toggle").forEach((button) => {
  button.addEventListener("click", toggleTheme);
});

// ── 齒輪設定選單 ──────────────────────────────────────────────────────────────
// 收納語言/翻譯/慣用詞庫/搜尋。內含按鈕沿用原本的 id 與處理器,這裡只管開合。
function closeSettingsMenu() {
  if (!els.settings || !els.settingsPanel || els.settingsPanel.hidden) {
    return;
  }
  els.settingsPanel.hidden = true;
  els.settings.setAttribute("aria-expanded", "false");
}

if (els.settings && els.settingsPanel) {
  els.settings.addEventListener("click", (event) => {
    event.stopPropagation();
    const willOpen = els.settingsPanel.hidden;
    els.settingsPanel.hidden = !willOpen;
    els.settings.setAttribute("aria-expanded", String(willOpen));
  });

  // 整列可點:點文字說明轉觸發該列按鈕;選了任一項就收合選單。
  els.settingsPanel.querySelectorAll(".settings-row").forEach((row) => {
    row.addEventListener("click", (event) => {
      const button = row.querySelector("button");
      if (!button) {
        return;
      }
      if (!event.target.closest("button")) {
        button.click();
      }
      closeSettingsMenu();
    });
  });

  // 點選單外 / 按 Esc 收合。
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".settings-menu")) {
      closeSettingsMenu();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeSettingsMenu();
    }
  });
}

els.backend.addEventListener("click", openAbout);
document.addEventListener("breeze:page", (event) => {
  if (event.detail?.page !== "transcribe" && (state.running || state.analyzing)) {
    stop();
  }
});
els.about?.addEventListener("click", (event) => {
  if (event.target === els.about) {
    els.about.close();
  }
});
bindViewTabs();
bindEntryToggle();
updateLangButton();
updateGlossaryButton();
els.lang?.addEventListener("click", openLanguageDialog);
els.glossary?.addEventListener("click", openGlossaryDialog);
els.translate?.addEventListener("click", () => setTranslateEnabled(!state.translateEnabled));
updateTranslateButton();
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
if (els.scenario) {
  els.scenario.value = state.scenario;
  els.scenario.addEventListener("change", () => {
    state.scenario = els.scenario.value === "music" ? "music" : "general";
    reapplyCaptureConstraints();
    persistSettings();
  });
}
if (els.search) {
  els.search.addEventListener("click", openSearchDialog);
}
if (els.searchInput) {
  els.searchInput.addEventListener("input", scheduleSearch);
}
if (els.searchDialog) {
  els.searchDialog
    .querySelector(".search-close")
    ?.addEventListener("click", () => els.searchDialog.close());
  // Enter in the sole text input would implicitly submit the method="dialog" form
  // and close the dialog; search is already live on input, so swallow the submit.
  els.searchDialog.querySelector("form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    void runSearch(els.searchInput.value);
  });
}
if (els.summarize) {
  els.summarize.addEventListener("click", () => void openSummary());
}
if (els.summaryDialog) {
  els.summaryDialog
    .querySelector(".summary-close")
    ?.addEventListener("click", () => els.summaryDialog.close());
  els.summaryDialog.querySelector(".summary-copy")?.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(els.summaryBody.textContent || "");
      flashStats("已複製摘要");
    } catch {
      flashStats("複製失敗");
    }
  });
}
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
els.download.addEventListener("click", async () => {
  const text = state.transcript.trim();
  if (!text) {
    return;
  }
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const downloadTxt = () =>
    downloadBlobAs(
      new Blob([state.transcript], { type: "text/plain;charset=utf-8" }),
      `breeze-elf-${stamp}.txt`,
    );
  // The structured .json (簡譜/timing) is what the 簡譜唱歌 page reads, so a
  // downloaded transcript can be uploaded there directly.
  // The structured .json (簡譜/timing) is what the 簡譜唱歌 page reads, so a
  // downloaded transcript can be uploaded there directly.
  const downloadJson = () =>
    downloadBlobAs(
      new Blob([JSON.stringify(buildTranscriptDocument(state.transcript), null, 2)], {
        type: "application/json;charset=utf-8",
      }),
      `breeze-elf-${stamp}.json`,
    );
  const downloadCsv = () =>
    downloadBlobAs(
      new Blob([buildPitchCsv()], { type: "text/csv;charset=utf-8" }),
      `breeze-elf-${stamp}-基頻分析.csv`,
    );

  // Plain text only → a single file, no need to ask.
  if (!hasStructuredTranscript()) {
    downloadTxt();
    flashStats("已下載");
    return;
  }

  const items = [
    { key: "txt", label: "逐字稿文字 (.txt)" },
    { key: "json", label: "簡譜 JSON (.json)" },
  ];
  if (hasAnalysis()) {
    items.push({ key: "csv", label: "基頻分析 (.csv)" });
  }
  const keys = await pickItems({ title: "下載逐字稿", storageKey: "transcript-download", items });
  if (!keys || keys.length === 0) {
    return;
  }
  if (keys.includes("txt")) {
    downloadTxt();
  }
  if (keys.includes("json")) {
    downloadJson();
  }
  if (keys.includes("csv")) {
    downloadCsv();
  }
  flashStats("已下載");
});
els.save.addEventListener("click", async () => {
  if (DEMO_MODE) {
    flashStats("示意模式不會雲端儲存");
    return;
  }

  const text = state.transcript.trim();
  if (!text || state.savingRemote) {
    return;
  }

  // When the transcript has 簡譜/timing, ask which parts to save (the text is
  // always written; the 簡譜 JSON and 基頻分析 CSV are optional). Remembered.
  let includeJianpu = true;
  let includeCsv = false;
  if (hasStructuredTranscript()) {
    const items = [
      { key: "text", label: "逐字稿文字", locked: true },
      { key: "jianpu", label: "簡譜 / 音階資料" },
    ];
    if (hasAnalysis()) {
      items.push({ key: "csv", label: "基頻分析 (.csv)" });
    }
    const keys = await pickItems({ title: "雲端儲存逐字稿", storageKey: "transcript-save", items });
    if (!keys) {
      return;
    }
    includeJianpu = keys.includes("jianpu");
    includeCsv = keys.includes("csv");
  }

  state.savingRemote = true;
  setTranscriptActions(true);
  window.clearTimeout(state.statsTimer);
  const previousStats = els.stats.textContent;
  els.stats.textContent = "雲端儲存中";

  try {
    // The transcript save carries the transcript text plus, when chosen, its
    // 簡譜/timing and 基頻分析 CSV. Omitting `blocks` entirely tells the server to
    // write txt only (sending [] would still emit an empty .json).
    const payload = {
      text,
      title: transcriptTitle(text),
      sampleRate: state.audioSampleRate,
    };
    if (includeJianpu) {
      payload.blocks = serializeBlocksForSave();
    }
    if (includeCsv) {
      payload.pitchCsv = buildPitchCsv();
    }

    const response = await fetch("/api/transcripts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      throw new Error(data.detail || "雲端儲存失敗");
    }
    const parts = [];
    if (data.jsonFilename) {
      parts.push("簡譜");
    }
    if (data.csvFilename) {
      parts.push("基頻分析");
    }
    const suffix = parts.length ? `(含${parts.join("、")})` : "";
    flashStats(
      data.filename ? `已雲端儲存逐字稿 ${data.filename}${suffix}` : "已雲端儲存逐字稿",
      previousStats,
    );
  } catch (error) {
    flashStats(error.message || "雲端儲存失敗", previousStats);
  } finally {
    state.savingRemote = false;
    setTranscriptActions(Boolean(state.transcript.trim()));
  }
});

els.analyze.addEventListener("click", () => {
  void analyzePitch();
});

els.audioDownload.addEventListener("click", () => {
  if (!state.audioBytes) {
    return;
  }
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const url = URL.createObjectURL(recordedAudioBlob());
  const link = document.createElement("a");
  link.href = url;
  link.download = `breeze-elf-${stamp}.wav`;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  flashStats("已下載音檔");
});

els.audioSave.addEventListener("click", async () => {
  if (DEMO_MODE) {
    flashStats("示意模式不會雲端儲存");
    return;
  }
  if (!state.audioBytes || state.savingAudio) {
    return;
  }

  state.savingAudio = true;
  setAudioActions();
  window.clearTimeout(state.statsTimer);
  const previousStats = els.stats.textContent;
  els.stats.textContent = "音檔雲端儲存中";

  try {
    const audioBase64 = await encodeRecordedAudioBase64();
    if (!audioBase64) {
      throw new Error("沒有可儲存的音檔");
    }
    const response = await fetch("/api/voice/outputs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: "recording", audioBase64 }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      throw new Error(data.detail || "雲端儲存失敗");
    }
    flashStats(
      data.filename ? `已雲端儲存音檔 ${data.filename}` : "已雲端儲存音檔",
      previousStats,
    );
  } catch (error) {
    flashStats(error.message || "雲端儲存失敗", previousStats);
  } finally {
    state.savingAudio = false;
    setAudioActions();
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
    translation: block.translation || null,
    speaker: Number.isInteger(block.speaker) ? block.speaker : null,
  }));
}

// True when the transcript carries per-character 簡譜/timing or block pitch — the
// data the 簡譜唱歌 page needs. Plain-text-only transcripts have none of it.
function hasStructuredTranscript() {
  return state.transcriptBlocks.some(
    (block) => (Array.isArray(block.characters) && block.characters.length > 0) || block.pitch,
  );
}

// True once 後處理 has produced 基頻分析 data (spectrogram + time series).
function hasAnalysis() {
  return state.analysisSpectra.some((spectro) => spectro && Array.isArray(spectro.times));
}

// Build the 基頻分析 CSV: one row per time point with 時間/基頻/強度/文字. Blocks
// (段落) are separated by a blank line so 基頻唱歌 can segment by paragraph and
// never mistakes a wide time bin in one block for a paragraph break.
function buildPitchCsv() {
  const rows = ["time_seconds,hz,intensity,text"];
  let emitted = false;
  state.transcriptBlocks.forEach((block, index) => {
    const spectro = state.analysisSpectra[index];
    if (!spectro || !Array.isArray(spectro.times) || !spectro.times.length) {
      return;
    }
    if (emitted) {
      rows.push(""); // blank line = paragraph break between blocks
    }
    emitted = true;
    const base = Number.isFinite(block.startSeconds) ? block.startSeconds : 0;
    const f0 = Array.isArray(spectro.f0) ? spectro.f0 : [];
    const intensity = Array.isArray(spectro.intensity) ? spectro.intensity : [];
    const text = Array.isArray(spectro.text) ? spectro.text : [];
    spectro.times.forEach((time, i) => {
      const hz = f0[i] == null ? "" : f0[i];
      const level = intensity[i] == null ? "" : intensity[i];
      rows.push(`${(base + time).toFixed(3)},${hz},${level},${csvCell(text[i] || "")}`);
    });
  });
  return rows.join("\n");
}

function csvCell(value) {
  return /[",\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

// Build the structured transcript document. Mirrors the remote-save payload (plus
// createdAt) so the downloaded .json and the server-saved .json are identical and
// both load fully in 簡譜唱歌.
function buildTranscriptDocument(text) {
  return {
    text,
    title: transcriptTitle(text),
    sampleRate: state.audioSampleRate,
    blocks: serializeBlocksForSave(),
    createdAt: new Date().toISOString(),
  };
}

function downloadBlobAs(blob, name) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// --------------------------------------------------------------------------- //
// multi-file picker dialog (remembers the last selection per operation)
// --------------------------------------------------------------------------- //

const PICK_STORAGE_PREFIX = "breeze-elf-pick-";

function readPickedKeys(storageKey) {
  try {
    const value = JSON.parse(localStorage.getItem(PICK_STORAGE_PREFIX + storageKey));
    return Array.isArray(value) ? value : null;
  } catch {
    return null;
  }
}

function writePickedKeys(storageKey, keys) {
  try {
    localStorage.setItem(PICK_STORAGE_PREFIX + storageKey, JSON.stringify(keys));
  } catch {
    /* storage full / unavailable — selection just won't be remembered */
  }
}

// Show the shared checkbox dialog. Resolves to the array of chosen keys (locked
// items are always included), or null when cancelled. The selection is restored
// from / saved to localStorage under `storageKey` so it is remembered.
function pickItems({ title, storageKey, items, confirmLabel = "確認" }) {
  return new Promise((resolve) => {
    const dialog = document.querySelector("#pick-dialog");
    if (!dialog) {
      resolve(items.map((item) => item.key));
      return;
    }
    dialog.querySelector(".pick-title").textContent = title;
    const list = dialog.querySelector(".pick-list");
    list.replaceChildren();

    const remembered = readPickedKeys(storageKey);
    items.forEach((item) => {
      const checked = item.locked
        ? true
        : remembered
          ? remembered.includes(item.key)
          : item.defaultChecked !== false;
      const label = document.createElement("label");
      label.className = "pick-item";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = item.key;
      checkbox.checked = checked;
      checkbox.disabled = Boolean(item.locked);
      const span = document.createElement("span");
      span.textContent = item.label;
      label.append(checkbox, span);
      list.append(label);
    });

    const confirmBtn = dialog.querySelector(".pick-confirm");
    const cancelBtn = dialog.querySelector(".pick-cancel");
    confirmBtn.textContent = confirmLabel;

    let settled = false;
    const onClose = () => finish(null);
    const finish = (result) => {
      if (settled) {
        return;
      }
      settled = true;
      confirmBtn.onclick = null;
      cancelBtn.onclick = null;
      dialog.removeEventListener("close", onClose);
      if (dialog.open) {
        dialog.close();
      }
      resolve(result);
    };
    confirmBtn.onclick = () => {
      const inputs = list.querySelectorAll("input");
      const keys = items
        .filter((item, index) => item.locked || inputs[index].checked)
        .map((item) => item.key);
      writePickedKeys(storageKey, keys);
      finish(keys);
    };
    cancelBtn.onclick = () => finish(null);
    dialog.addEventListener("close", onClose); // ESC / programmatic close → cancel

    if (typeof dialog.showModal === "function") {
      dialog.showModal();
    } else {
      dialog.setAttribute("open", "");
    }
  });
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

// 後處理 — re-measure every character's pitch from the recorded/loaded audio with
// the more accurate (offline) tracker and a single global 主音, and compute each
// block's 基頻分析 (STFT + f0). Everything post-processable is done in one pass,
// server-side, behind a real progress bar (it does not need to be real-time).
async function analyzePitch() {
  if (DEMO_MODE) {
    flashStats("示意模式不會執行後處理");
    return;
  }
  if (state.analyzingPitch || state.running) {
    return;
  }
  if (!hasStructuredTranscript()) {
    flashStats("沒有可後處理的逐字稿,請先錄製或載入");
    return;
  }
  if (!state.audioBytes) {
    flashStats("找不到對應的音檔,無法後處理");
    return;
  }

  state.analyzingPitch = true;
  setTranscriptActions(Boolean(state.transcript.trim()));
  showAnalyzeProgress("後處理中", 0, true);
  try {
    const audioBase64 = await encodeRecordedAudioBase64();
    if (!audioBase64) {
      throw new Error("找不到音檔");
    }
    const response = await fetch("/api/transcript/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        audioBase64,
        sampleRate: state.audioSampleRate,
        blocks: serializeBlocksForSave(),
      }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      throw new Error(data.detail || "後處理失敗");
    }

    const result = await pollAnalyzeStatus();
    applyAnalyzedBlocks(result.blocks);
    state.openBlocks.clear();
    renderTranscriptView({ scrollToEnd: false });
    scheduleSessionPersist();
    const tonic = Number.isFinite(result.tonicHz) ? `,主音 ${Math.round(result.tonicHz)} Hz` : "";
    flashStats(`後處理完成(${result.characterCount || 0} 字${tonic})`);
  } catch (error) {
    flashStats(error.message || "後處理失敗");
  } finally {
    state.analyzingPitch = false;
    hideAnalyzeProgress();
    setTranscriptActions(Boolean(state.transcript.trim()));
  }
}

// Poll the analysis job, updating the progress bar, until it finishes or fails.
function pollAnalyzeStatus() {
  return new Promise((resolve, reject) => {
    const timer = window.setInterval(async () => {
      let data;
      try {
        const response = await fetch("/api/transcript/analyze/status");
        data = await response.json().catch(() => ({}));
      } catch {
        return; // transient — keep polling
      }
      if (data.status === "running") {
        const fraction = Number.isFinite(data.progress) ? data.progress : 0;
        showAnalyzeProgress(data.stage || "分析中", fraction, fraction <= 0.001);
      } else if (data.status === "done") {
        window.clearInterval(timer);
        resolve(data.result || { blocks: [] });
      } else if (data.status === "error") {
        window.clearInterval(timer);
        reject(new Error(data.error || "後處理失敗"));
      } else {
        window.clearInterval(timer);
        reject(new Error("後處理未啟動"));
      }
    }, 300);
  });
}

// Fold refined pitch + per-character data back onto the existing blocks (matched
// by order); text and timings are left untouched.
function applyAnalyzedBlocks(blocks) {
  if (!Array.isArray(blocks)) {
    return;
  }
  state.analysisSpectra = [];
  blocks.forEach((refined, index) => {
    const target = state.transcriptBlocks[index];
    if (!target) {
      return;
    }
    target.pitch = normalizePitch(refined.pitch);
    target.characters = normalizeCharacters(refined.characters);
    // Spectrogram stays out of transcriptBlocks so it is never persisted/saved.
    state.analysisSpectra[index] = refined.spectrogram || null;
  });
}

function showAnalyzeProgress(label, fraction, indeterminate) {
  els.analyzeProgress.hidden = false;
  els.analyzeProgressLabel.textContent = label;
  els.analyzeProgressBar.classList.toggle("indeterminate", Boolean(indeterminate));
  if (indeterminate) {
    els.analyzeProgressPct.textContent = "";
  } else {
    const pct = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
    els.analyzeProgressPct.textContent = `${pct}%`;
    els.analyzeProgressBar.style.width = `${pct}%`;
  }
}

function hideAnalyzeProgress() {
  els.analyzeProgress.hidden = true;
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
