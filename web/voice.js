// 變聲工作室 page controller. A single 聲音庫 holds every saved voice (add /
// favorite / rename / delete / pick-as-target are all in one place). Recording
// or uploading runs the matching job immediately — building a voiceprint for a
// new library voice, or re-voicing your clip as the target — behind a busy
// overlay that locks the page and offers a 中斷 button. Finished audio lands in
// a player so you can listen before downloading or saving it remotely.

import {
  VOICE_SAMPLE_RATE,
  base64ToBytes,
  blobToBase64,
  decodeFileToInt16,
  filesToZipBlob,
  int16ToWavBlob,
  pcm16BytesToWavBlob,
  // Versioned so a cached older audio-utils.js (without base64ToBytes /
  // filesToZipBlob) can't break this whole module — a failed named import would
  // stop voice.js from running, which also owns the page tab navigation.
} from "./audio-utils.js?v=2";

const els = {
  page: document.querySelector("#page-voice"),
  status: document.querySelector("#voice-status"),
  engineTag: document.querySelector("#voice-engine-tag"),
  targetName: document.querySelector("#voice-target-name"),
  empty: document.querySelector("#voice-empty"),
  // model-load progress (in-card)
  progress: document.querySelector("#voice-progress"),
  progressLabel: document.querySelector("#voice-progress-label"),
  progressPct: document.querySelector("#voice-progress-pct"),
  progressBar: document.querySelector("#voice-progress-bar"),
  // busy overlay (locks the page during a job)
  busy: document.querySelector("#voice-busy"),
  busyLabel: document.querySelector("#voice-busy-label"),
  busyPct: document.querySelector("#voice-busy-pct"),
  busyBar: document.querySelector("#voice-busy-bar"),
  busyCancel: document.querySelector("#voice-busy-cancel"),
  // add to library
  vcRecord: document.querySelector("#vc-record"),
  vcUpload: document.querySelector("#vc-upload"),
  vcFile: document.querySelector("#vc-file"),
  // convert
  cvRecord: document.querySelector("#cv-record"),
  cvUpload: document.querySelector("#cv-upload"),
  cvFile: document.querySelector("#cv-file"),
  cvResultWrap: document.querySelector("#cv-result-wrap"),
  cvSource: document.querySelector("#cv-source"),
  cvResult: document.querySelector("#cv-result"),
  cvSrcDownload: document.querySelector("#cv-src-download"),
  cvSrcSave: document.querySelector("#cv-src-save"),
  cvDownload: document.querySelector("#cv-download"),
  cvSave: document.querySelector("#cv-save"),
  cvBundleDownload: document.querySelector("#cv-bundle-download"),
  cvBundleSave: document.querySelector("#cv-bundle-save"),
  // text -> target voice
  ttsUpload: document.querySelector("#tts-upload"),
  ttsFile: document.querySelector("#tts-file"),
  ttsText: document.querySelector("#tts-text"),
  ttsBase: document.querySelector("#tts-base"),
  ttsSpeed: document.querySelector("#tts-speed"),
  ttsRun: document.querySelector("#tts-run"),
  ttsResultWrap: document.querySelector("#tts-result-wrap"),
  ttsResult: document.querySelector("#tts-result"),
  ttsDownload: document.querySelector("#tts-download"),
  ttsSave: document.querySelector("#tts-save"),
  // 簡譜 singing
  singUpload: document.querySelector("#sing-upload"),
  singFile: document.querySelector("#sing-file"),
  singBlocks: document.querySelector("#sing-blocks"),
  singAdd: document.querySelector("#sing-add"),
  singTonic: document.querySelector("#sing-tonic"),
  singSpeed: document.querySelector("#sing-speed"),
  singRun: document.querySelector("#sing-run"),
  singResultWrap: document.querySelector("#sing-result-wrap"),
  singResult: document.querySelector("#sing-result"),
  singDownload: document.querySelector("#sing-download"),
  singSave: document.querySelector("#sing-save"),
  // 基頻唱歌 (sing by the measured pitch in a 基頻分析 CSV)
  pitchUpload: document.querySelector("#pitch-analysis-upload"),
  pitchFile: document.querySelector("#pitch-analysis-file"),
  pitchEmpty: document.querySelector("#pitch-empty"),
  pitchVis: document.querySelector("#pitch-vis"),
  pitchParas: document.querySelector("#pitch-paras"),
  pitchVisMeta: document.querySelector("#pitch-vis-meta"),
  pitchSpeed: document.querySelector("#pitch-speed"),
  pitchRun: document.querySelector("#pitch-run"),
  pitchResultWrap: document.querySelector("#pitch-result-wrap"),
  pitchResult: document.querySelector("#pitch-result"),
  pitchDownload: document.querySelector("#pitch-download"),
  pitchSave: document.querySelector("#pitch-save"),
  // library
  list: document.querySelector("#voice-list"),
};

const state = {
  initialized: false,
  modelReady: false,
  loading: false,
  pollTimer: 0,
  voices: [],
  selectedId: "",
  busy: false,
  abort: null,
  cvSourceB64: "",
  cvSourceUrl: "",
  cvResultB64: "",
  cvResultUrl: "",
  ttsResultB64: "",
  ttsResultUrl: "",
  singResultB64: "",
  singResultUrl: "",
  pitchResultB64: "",
  pitchResultUrl: "",
  // 基頻唱歌 state: the raw CSV rows (for the relationship chart) and the rich
  // notes (per-syllable pitch contour + 氣音) sung verbatim by measured pitch.
  pitchRows: null,
  pitchNotes: null,
  pitchTonic: 0,
  // 簡譜唱歌 state: one editable sentence per row, each with its 文字 + 簡譜 (and
  // per-character durations / glide positions carried from the loaded 逐字稿).
  singBlocks: [],
};

// --------------------------------------------------------------------------- //
// page navigation (owns the icon tab bar)
// --------------------------------------------------------------------------- //

const tabs = Array.from(document.querySelectorAll(".tabbar .tab"));

function showPage(page) {
  document.body.dataset.page = page;
  document.querySelectorAll(".page").forEach((element) => {
    element.hidden = element.id !== `page-${page}`;
  });
  tabs.forEach((tab) => {
    if (tab.dataset.page === page) {
      tab.setAttribute("aria-current", "page");
    } else {
      tab.removeAttribute("aria-current");
    }
  });
  // Keep the URL in sync so the voice page is deep-linkable (e.g. the PWA
  // "變聲" home-screen shortcut opens ?page=voice directly) and survives reload.
  try {
    const url = page === "voice" ? "?page=voice" : location.pathname;
    history.replaceState(null, "", url);
  } catch {
    /* history may be unavailable in some embedded contexts */
  }
  document.dispatchEvent(new CustomEvent("breeze:page", { detail: { page } }));
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    // A running job locks the page, including page switches.
    if (state.busy) {
      return;
    }
    showPage(tab.dataset.page);
  });
});

document.addEventListener("breeze:page", (event) => {
  if (event.detail?.page === "voice") {
    void initVoicePage();
  } else {
    stopSamplePlayback();
    closeAllPlayerMenus();
    void stopActiveRecording("切換頁面");
  }
});

// Sub-tabs inside the voice page: 聲音轉換 / 文字轉換 / 簡譜唱歌 / 基頻唱歌 share
// the one 聲音庫 above them, so only the active operation panel is shown at once.
const subtabs = Array.from(document.querySelectorAll("#page-voice .subtab"));

function showVoicePanel(panel) {
  closeAllPlayerMenus();
  document.querySelectorAll("#page-voice .vpanel").forEach((element) => {
    element.hidden = element.id !== `vpanel-${panel}`;
  });
  subtabs.forEach((tab) => {
    tab.setAttribute("aria-selected", String(tab.dataset.panel === panel));
  });
  // The chart can only measure its width once its panel is visible, so (re)draw
  // it when 基頻唱歌 is shown and there is data to render.
  if (panel === "pitch" && state.pitchRows) {
    renderPitchRelation(state.pitchRows);
  }
}

subtabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    if (state.busy) {
      return;
    }
    showVoicePanel(tab.dataset.panel);
  });
});

// --------------------------------------------------------------------------- //
// microphone recorder (mono 16k PCM via the shared AudioWorklet)
// --------------------------------------------------------------------------- //

class Recorder {
  constructor() {
    this.context = null;
    this.stream = null;
    this.worklet = null;
    this.source = null;
    this.silence = null;
    this.chunks = [];
    this.recording = false;
  }

  async start() {
    if (this.recording) {
      return;
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
      video: false,
    });
    this.context = new AudioContext({ latencyHint: "interactive" });
    await this.context.audioWorklet.addModule(new URL("audio-worklet.js", import.meta.url).href);
    this.source = this.context.createMediaStreamSource(this.stream);
    this.worklet = new AudioWorkletNode(this.context, "breeze-mic-processor", {
      processorOptions: { targetSampleRate: VOICE_SAMPLE_RATE, chunkMs: 200 },
    });
    this.silence = this.context.createGain();
    this.silence.gain.value = 0;
    this.chunks = [];
    this.worklet.port.onmessage = (event) => {
      if (event.data?.type === "audio" && event.data.buffer?.byteLength) {
        this.chunks.push(event.data.buffer);
      }
    };
    this.source.connect(this.worklet);
    this.worklet.connect(this.silence).connect(this.context.destination);
    this.recording = true;
  }

  async stop() {
    this.recording = false;
    this.worklet?.disconnect();
    this.source?.disconnect();
    this.silence?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    await this.context?.close().catch(() => {});
    this.worklet = null;
    this.source = null;
    this.silence = null;
    this.stream = null;
    this.context = null;
    return pcm16BytesToWavBlob(this.chunks, VOICE_SAMPLE_RATE);
  }
}

let activeRecorder = null;
let activeRecordButton = null;

function idleLabel(button) {
  return button?.dataset.idleLabel || "● 開始錄音";
}

// Record/stop toggle. On stop the captured blob is handed to onDone, which kicks
// off the matching job (build voiceprint / convert) right away.
async function toggleRecording(button, onDone) {
  if (state.busy) {
    return;
  }
  if (activeRecorder && activeRecordButton !== button) {
    await stopActiveRecording("先停止其他錄音");
  }

  if (activeRecorder && activeRecordButton === button) {
    const recorder = activeRecorder;
    activeRecorder = null;
    activeRecordButton = null;
    button.classList.remove("recording");
    button.textContent = idleLabel(button);
    let blob = null;
    try {
      blob = await recorder.stop();
    } catch (error) {
      setStatus(error?.message || "錄音失敗", "error");
      return;
    }
    if (!blob || blob.size <= 44) {
      setStatus("沒有錄到聲音", "error");
      return;
    }
    setStatus("錄音完成");
    await onDone(blob);
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("此裝置無法錄音", "error");
    return;
  }
  if (!window.isSecureContext && !["localhost", "127.0.0.1"].includes(location.hostname)) {
    setStatus("錄音需要 HTTPS", "error");
    return;
  }

  const recorder = new Recorder();
  try {
    await recorder.start();
  } catch (error) {
    setStatus(error?.message || "無法啟動麥克風", "error");
    return;
  }
  activeRecorder = recorder;
  activeRecordButton = button;
  button.classList.add("recording");
  button.textContent = "■ 停止錄音";
  setStatus("錄音中…", "live");
}

async function stopActiveRecording(reason) {
  if (!activeRecorder) {
    return;
  }
  const recorder = activeRecorder;
  const button = activeRecordButton;
  activeRecorder = null;
  activeRecordButton = null;
  if (button) {
    button.classList.remove("recording");
    button.textContent = idleLabel(button);
  }
  try {
    await recorder.stop();
  } catch {
    /* discarded — recording was cancelled */
  }
  if (reason) {
    setStatus(reason);
  }
}

// --------------------------------------------------------------------------- //
// model loading + progress bar
// --------------------------------------------------------------------------- //

async function initVoicePage() {
  if (!state.initialized) {
    state.initialized = true;
    await refreshVoices();
  }
  void ensureModelLoaded();
}

async function ensureModelLoaded() {
  if (state.modelReady || state.loading) {
    return;
  }
  state.loading = true;
  showProgress("載入變聲模型中", 0, true);
  try {
    const response = await fetch("/api/voice/load", { method: "POST" });
    const data = await response.json().catch(() => ({}));
    applyEngineMeta(data);
    applyStatus(data);
  } catch (error) {
    state.loading = false;
    showProgress(error?.message || "無法連線到伺服器", 0, false, true);
    setStatus("無法連線到伺服器", "error");
    return;
  }
  if (state.loading) {
    startPolling();
  }
}

// Resolve once the model is ready; reject if it ends in an error state.
function waitForModelReady() {
  if (state.modelReady) {
    return Promise.resolve();
  }
  void ensureModelLoaded();
  return new Promise((resolve, reject) => {
    const timer = window.setInterval(() => {
      if (state.modelReady) {
        window.clearInterval(timer);
        resolve();
      } else if (!state.loading) {
        window.clearInterval(timer);
        reject(new Error("模型尚未就緒,請稍後再試"));
      }
    }, 200);
  });
}

function startPolling() {
  stopPolling();
  state.pollTimer = window.setInterval(async () => {
    try {
      const response = await fetch("/api/voice/status");
      const data = await response.json().catch(() => ({}));
      applyEngineMeta(data);
      applyStatus(data);
    } catch {
      /* transient — keep polling */
    }
  }, 400);
}

function stopPolling() {
  if (state.pollTimer) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = 0;
  }
}

function applyEngineMeta(data) {
  if (data?.backend) {
    const device = data.device && data.device !== "unknown" ? ` · ${data.device}` : "";
    els.engineTag.textContent = `${data.provider || data.backend}${device}`;
  }
}

function applyStatus(data) {
  const status = data?.status;
  if (status === "ready") {
    state.modelReady = true;
    state.loading = false;
    stopPolling();
    showProgress("模型就緒", 1, false);
    window.setTimeout(() => {
      if (state.modelReady) {
        els.progress.hidden = true;
      }
    }, 700);
    if (!state.busy) {
      setStatus("待命");
    }
  } else if (status === "error") {
    state.modelReady = false;
    state.loading = false;
    stopPolling();
    showProgress(data.error || "模型載入失敗", 0, false, true);
    setStatus("模型載入失敗,點任一動作可重試", "error");
  } else if (status === "loading") {
    state.loading = true;
    const fraction = typeof data.progress === "number" ? data.progress : 0;
    showProgress(data.stage || "載入中", fraction, fraction <= 0.001);
  }
  refreshButtons();
}

function showProgress(label, fraction, indeterminate = false, isError = false) {
  els.progress.hidden = false;
  els.progressLabel.textContent = label;
  els.progressBar.classList.toggle("indeterminate", Boolean(indeterminate) && !isError);
  els.progressBar.classList.toggle("error", Boolean(isError));
  if (indeterminate && !isError) {
    els.progressPct.textContent = "";
  } else {
    const pct = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
    els.progressPct.textContent = `${pct}%`;
    els.progressBar.style.width = `${pct}%`;
  }
}

// --------------------------------------------------------------------------- //
// busy overlay — lock the page, show progress, allow 中斷
// --------------------------------------------------------------------------- //

function showBusy(label) {
  els.busyLabel.textContent = label;
  els.busyPct.textContent = "";
  els.busyBar.classList.add("indeterminate");
  els.busy.hidden = false;
}

function hideBusy() {
  els.busy.hidden = true;
}

// Run a server job exclusively: lock the page with the busy overlay, expose an
// AbortController to the 中斷 button, and always unlock afterwards.
async function runExclusive(label, task) {
  if (state.busy) {
    return;
  }
  state.busy = true;
  const controller = new AbortController();
  state.abort = controller;
  showBusy(label);
  refreshButtons();
  try {
    await task(controller.signal);
  } catch (error) {
    if (error?.name === "AbortError") {
      setStatus("已中斷");
    } else {
      setStatus(error?.message || "操作失敗", "error");
    }
  } finally {
    state.busy = false;
    state.abort = null;
    hideBusy();
    refreshButtons();
  }
}

els.busyCancel.addEventListener("click", () => {
  if (state.abort) {
    state.abort.abort();
  }
});

// --------------------------------------------------------------------------- //
// library: add / list / favorite / rename / delete / pick target
// --------------------------------------------------------------------------- //

function defaultVoiceName(fileName) {
  if (fileName) {
    const stem = fileName.replace(/\.[^.]+$/, "").trim();
    if (stem) {
      return stem.slice(0, 120);
    }
  }
  return `我的聲音 ${state.voices.length + 1}`;
}

async function addVoiceFromBlob(blob, fileName) {
  await runExclusive("建立聲紋中…", async (signal) => {
    await waitForModelReady();
    setStatus("擷取聲音特徵中…", "live");
    const audioBase64 = await blobToBase64(blob);
    const response = await fetch("/api/voices", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: defaultVoiceName(fileName), audioBase64 }),
      signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      throw new Error(data.detail || "建立聲紋失敗");
    }
    await refreshVoices(data.voice.id);
    setStatus(`已新增「${data.voice.name}」並設為目標聲音`);
  });
}

async function refreshVoices(selectId) {
  try {
    const response = await fetch("/api/voices");
    const data = await response.json().catch(() => ({}));
    state.voices = Array.isArray(data.voices) ? data.voices : [];
  } catch {
    state.voices = [];
  }
  if (selectId && state.voices.some((voice) => voice.id === selectId)) {
    state.selectedId = selectId;
  } else if (!state.voices.some((voice) => voice.id === state.selectedId)) {
    state.selectedId = state.voices[0]?.id || "";
  }
  renderVoiceList();
  renderTargetName();
  refreshButtons();
}

function renderTargetName() {
  const selected = state.voices.find((voice) => voice.id === state.selectedId);
  const label = selected ? selected.name : "尚未選擇";
  els.targetName.textContent = label;
  // Echo the target inside each operation panel so it is always clear which
  // voice a convert / TTS / sing will use, even when the 聲音庫 is scrolled away.
  document.querySelectorAll("#page-voice [data-role='panel-target']").forEach((node) => {
    node.textContent = label;
  });
}

function renderVoiceList() {
  // Re-rendering recreates every row, so any in-flight preview button reference
  // would dangle — stop playback first so the ⏸ state never gets stranded.
  stopSamplePlayback();
  els.list.innerHTML = "";
  state.voices.forEach((voice) => {
    els.list.appendChild(renderVoiceItem(voice));
  });
  els.empty.hidden = state.voices.length !== 0;
}

function renderVoiceItem(voice) {
  const item = document.createElement("li");
  item.className = "voice-item";
  item.dataset.id = voice.id;
  if (voice.id === state.selectedId) {
    item.classList.add("selected");
  }
  // The whole row is the target picker: tapping anywhere that is not an action
  // button selects this voice. This is the reliable way to choose the target —
  // the small radio alone was easy to miss, so conversions could run against the
  // wrong (previously selected) voice.
  item.addEventListener("click", () => selectVoice(voice.id));

  const pick = document.createElement("input");
  pick.type = "radio";
  pick.name = "voice-pick";
  pick.className = "voice-pick";
  pick.checked = voice.id === state.selectedId;
  pick.setAttribute("aria-label", `設為目標聲音:${voice.name}`);
  pick.addEventListener("click", (event) => event.stopPropagation());
  pick.addEventListener("change", () => selectVoice(voice.id));

  const body = document.createElement("div");
  body.style.minWidth = "0";
  const name = document.createElement("div");
  name.className = "voice-item-name";
  name.textContent = voice.name;
  const meta = document.createElement("div");
  meta.className = "voice-item-meta";
  meta.textContent = `${formatDuration(voice.durationSeconds)} · ${formatDate(voice.createdAt)}`;
  body.append(name, meta);

  const actions = document.createElement("div");
  actions.className = "voice-item-actions";

  const stopRowSelect = (handler) => (event) => {
    event.stopPropagation();
    handler();
  };

  const star = document.createElement("button");
  star.type = "button";
  star.className = `icon-btn star${voice.favorite ? " on" : ""}`;
  star.textContent = voice.favorite ? "★" : "☆";
  star.title = voice.favorite ? "取消我的最愛" : "加入我的最愛";
  star.setAttribute("aria-label", star.title);
  star.addEventListener("click", stopRowSelect(() => toggleFavorite(voice)));
  actions.appendChild(star);

  if (voice.hasSample) {
    const play = document.createElement("button");
    play.type = "button";
    play.className = "icon-btn play";
    play.textContent = "▶";
    play.title = "試聽這個聲音";
    play.setAttribute("aria-label", play.title);
    play.addEventListener("click", stopRowSelect(() => toggleSample(voice.id, play)));
    actions.appendChild(play);
  }

  const edit = document.createElement("button");
  edit.type = "button";
  edit.className = "icon-btn edit";
  edit.textContent = "✎";
  edit.title = "重新命名";
  edit.setAttribute("aria-label", edit.title);
  edit.addEventListener("click", stopRowSelect(() => renameVoice(voice)));
  actions.appendChild(edit);

  const del = document.createElement("button");
  del.type = "button";
  del.className = "icon-btn del";
  del.textContent = "🗑";
  del.title = "刪除這個聲音";
  del.setAttribute("aria-label", del.title);
  del.addEventListener("click", stopRowSelect(() => deleteVoice(voice)));
  actions.appendChild(del);

  item.append(pick, body, actions);
  return item;
}

function selectVoice(voiceId) {
  state.selectedId = voiceId;
  renderVoiceList();
  renderTargetName();
  refreshButtons();
}

async function toggleFavorite(voice) {
  await patchVoice(voice.id, { favorite: !voice.favorite });
}

async function renameVoice(voice) {
  const next = window.prompt("重新命名聲音", voice.name);
  if (next === null) {
    return;
  }
  const name = next.trim();
  if (!name || name === voice.name) {
    return;
  }
  await patchVoice(voice.id, { name });
}

async function patchVoice(voiceId, body) {
  try {
    const response = await fetch(`/api/voices/${encodeURIComponent(voiceId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      throw new Error(data.detail || "更新失敗");
    }
    await refreshVoices(state.selectedId);
  } catch (error) {
    setStatus(error?.message || "更新失敗", "error");
  }
}

async function deleteVoice(voice) {
  if (!window.confirm(`確定刪除「${voice.name}」?`)) {
    return;
  }
  try {
    const response = await fetch(`/api/voices/${encodeURIComponent(voice.id)}`, {
      method: "DELETE",
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      throw new Error(data.detail || "刪除失敗");
    }
    setStatus(`已刪除「${voice.name}」`);
    await refreshVoices();
  } catch (error) {
    setStatus(error?.message || "刪除失敗", "error");
  }
}

let samplePlayer = null;
let samplePlayButton = null;

// Reflect the player state on its button: ▶ when idle/paused, ⏸ while playing.
function paintSampleButton(playing) {
  if (!samplePlayButton) {
    return;
  }
  samplePlayButton.textContent = playing ? "⏸" : "▶";
  samplePlayButton.classList.toggle("playing", playing);
  samplePlayButton.title = playing ? "暫停試聽" : "試聽這個聲音";
  samplePlayButton.setAttribute("aria-label", samplePlayButton.title);
}

function stopSamplePlayback() {
  if (samplePlayer) {
    samplePlayer.pause();
    samplePlayer = null;
  }
  paintSampleButton(false);
  samplePlayButton = null;
}

// Toggle a 聲音庫 preview. Clicking the active clip's button pauses/resumes it;
// clicking a different voice switches to that clip. The button shows ⏸ while
// playing so it doubles as a pause control.
function toggleSample(voiceId, button) {
  if (samplePlayer && samplePlayButton === button) {
    if (samplePlayer.paused) {
      void samplePlayer.play().catch(() => setStatus("無法播放試聽", "error"));
    } else {
      samplePlayer.pause();
    }
    return;
  }

  stopSamplePlayback();
  samplePlayer = new Audio(`/api/voices/${encodeURIComponent(voiceId)}/sample`);
  samplePlayButton = button;
  samplePlayer.addEventListener("play", () => {
    if (samplePlayButton === button) {
      paintSampleButton(true);
    }
  });
  samplePlayer.addEventListener("pause", () => {
    if (samplePlayButton === button) {
      paintSampleButton(false);
    }
  });
  samplePlayer.addEventListener("ended", () => {
    if (samplePlayButton === button) {
      stopSamplePlayback();
    }
  });
  samplePlayer.play().catch(() => {
    setStatus("無法播放試聽", "error");
    stopSamplePlayback();
  });
}

// --------------------------------------------------------------------------- //
// convert (your voice -> target) and text -> target
// --------------------------------------------------------------------------- //

async function convertFromBlob(blob) {
  if (!state.selectedId) {
    setStatus("請先在聲音庫選擇或新增一個目標聲音", "error");
    return;
  }
  await runExclusive("轉換聲音中…", async (signal) => {
    await waitForModelReady();
    setStatus("轉換中…", "live");
    const audioBase64 = await blobToBase64(blob);
    const response = await fetch("/api/voice/convert", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ voiceId: state.selectedId, audioBase64 }),
      signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      throw new Error(data.detail || "轉換失敗");
    }
    showConvertResult(audioBase64, data.audioBase64);
    setStatus("轉換完成,可試聽後打包下載或打包雲端儲存");
  });
}

async function runTts() {
  const text = els.ttsText.value.trim();
  if (!text) {
    setStatus("請先輸入文字", "error");
    return;
  }
  if (!state.selectedId) {
    setStatus("請先在聲音庫選擇或新增一個目標聲音", "error");
    return;
  }
  await runExclusive("合成語音中…", async (signal) => {
    await waitForModelReady();
    setStatus("合成語音中…", "live");
    const body = { voiceId: state.selectedId, text };
    const baseHz = positiveNumber(els.ttsBase.value);
    if (baseHz) {
      body.baseHz = baseHz;
    }
    const speed = positiveNumber(els.ttsSpeed.value);
    if (speed) {
      body.speed = speed;
    }
    const response = await fetch("/api/voice/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      throw new Error(data.detail || "合成失敗");
    }
    showResult("tts", data.audioBase64);
    setStatus("合成完成,可試聽後下載或雲端儲存");
  });
}

// --------------------------------------------------------------------------- //
// 簡譜 singing
// --------------------------------------------------------------------------- //

async function loadTextFile(file) {
  if (!file) {
    return;
  }
  try {
    const text = await file.text();
    els.ttsText.value = text.slice(0, 2000);
    refreshButtons();
    setStatus(`已載入文字檔「${file.name}」`);
  } catch (error) {
    setStatus(error?.message || "讀取文字檔失敗", "error");
  }
}

// Turn a transcript JSON (the 逐字稿 bundle) into one editable sentence per row.
// Each row keeps a 簡譜 line (degrees, glides as 3↗5) and a 文字 line, separately
// editable, plus per-character durations + glide positions and the silence before
// it (so the sung clip keeps the original timing). 主音 defaults to the *original*
// recording's pitch so the melody stays in its own key.
async function loadTranscriptFile(file) {
  if (!file) {
    return;
  }
  let doc;
  try {
    doc = JSON.parse(await file.text());
  } catch {
    setStatus("不是有效的逐字稿 JSON", "error");
    return;
  }
  // Accept the downloaded / remote-saved transcript ({blocks}), the locally
  // persisted session shape ({transcriptBlocks}), or a bare array of blocks.
  const blocks = Array.isArray(doc?.blocks)
    ? doc.blocks
    : Array.isArray(doc?.transcriptBlocks)
      ? doc.transcriptBlocks
      : Array.isArray(doc)
        ? doc
        : [];
  const singBlocks = [];
  const medians = [];
  let sungCount = 0;
  let lastBlockEnd = null;
  for (const block of blocks) {
    if (block.pitch && Number.isFinite(Number(block.pitch.medianHz))) {
      medians.push(Number(block.pitch.medianHz));
    }
    const chars = Array.isArray(block.characters) ? block.characters : [];
    const texts = [];
    const tokens = [];
    const durations = [];
    const glides = [];
    let blockStart = null;
    let blockEnd = null;
    for (const entry of chars) {
      const char = (entry.char || "").trim();
      if (!char) continue;
      const start = Number(entry.startSeconds);
      const end = Number(entry.endSeconds);
      const rawDur = Number(entry.durationSeconds);
      const duration =
        Number.isFinite(rawDur) && rawDur > 0
          ? rawDur
          : Number.isFinite(start) && Number.isFinite(end) && end > start
            ? end - start
            : 0.4;
      if (blockStart == null && Number.isFinite(start)) {
        blockStart = start;
      }
      blockEnd = Number.isFinite(end) ? end : (Number.isFinite(start) ? start + duration : blockEnd);
      texts.push(char);
      tokens.push((entry.jianpu || "").trim() || "0");
      durations.push(Number(duration.toFixed(3)));
      const mid = Number(entry.glideMid);
      glides.push(entry.isGlide && Number.isFinite(mid) ? mid : null);
      sungCount += 1;
    }
    if (!texts.length) continue;
    // Silence before this sentence (kept as a rest so timing matches the original).
    const leadRest =
      singBlocks.length && lastBlockEnd != null && blockStart != null
        ? Math.max(0, blockStart - lastBlockEnd)
        : 0;
    singBlocks.push({
      text: texts.join(""),
      jianpu: tokens.join(" "),
      durations,
      glides,
      leadRest: leadRest > 0.08 ? Number(leadRest.toFixed(3)) : 0,
    });
    if (blockEnd != null) {
      lastBlockEnd = blockEnd;
    }
  }
  if (!sungCount) {
    setStatus("逐字稿裡找不到音高資料", "error");
    return;
  }
  state.singBlocks = singBlocks;
  renderSingBlocks();
  const tonic = medians.length ? Math.round(median(medians)) : 0;
  els.singTonic.value = tonic ? String(tonic) : "";
  els.singTonic.placeholder = "自動";
  refreshButtons();
  setStatus(
    `已載入逐字稿「${file.name}」,共 ${sungCount} 個音、${singBlocks.length} 句${
      tonic ? ` · 以原曲音高 ${tonic} Hz 演唱(可調整基音 Hz)` : ""
    }`,
  );
}

// Render one editable row per sentence: 簡譜 (top) + 文字 (bottom), each editable.
function renderSingBlocks() {
  const host = els.singBlocks;
  if (!host) {
    return;
  }
  host.replaceChildren();
  if (!state.singBlocks.length) {
    state.singBlocks.push({ text: "", jianpu: "", durations: [], glides: [], leadRest: 0 });
  }
  state.singBlocks.forEach((block, index) => {
    const row = document.createElement("div");
    row.className = "sing-block";

    const fields = document.createElement("div");
    fields.className = "sing-block-fields";

    const jianpu = document.createElement("input");
    jianpu.className = "sing-field jianpu";
    jianpu.type = "text";
    jianpu.value = block.jianpu;
    jianpu.placeholder = "1 1 5 5 6 6 5";
    jianpu.setAttribute("aria-label", `第 ${index + 1} 句的簡譜`);
    jianpu.addEventListener("input", () => {
      block.jianpu = jianpu.value;
      refreshButtons();
    });

    const lyric = document.createElement("input");
    lyric.className = "sing-field lyric";
    lyric.type = "text";
    lyric.value = block.text;
    lyric.placeholder = "小星星亮晶晶";
    lyric.setAttribute("aria-label", `第 ${index + 1} 句的文字`);
    lyric.addEventListener("input", () => {
      block.text = lyric.value;
      refreshButtons();
    });

    fields.append(jianpu, lyric);

    const del = document.createElement("button");
    del.type = "button";
    del.className = "sing-block-del";
    del.textContent = "🗑";
    del.title = "刪除這一句";
    del.setAttribute("aria-label", del.title);
    del.addEventListener("click", () => {
      state.singBlocks.splice(index, 1);
      renderSingBlocks();
      refreshButtons();
    });

    row.append(fields, del);
    host.append(row);
  });
}

// Rebuild singable notes from the editable rows. Characters and 簡譜 tokens are
// zipped per sentence; stored per-character durations / glide positions are used
// when the counts still line up, otherwise the sentence's time is shared evenly.
function buildSingNotes(blocks) {
  const notes = [];
  for (const block of blocks) {
    const chars = Array.from((block.text || "").trim()).filter((c) => c.trim());
    const tokens = (block.jianpu || "").trim().split(/\s+/).filter(Boolean);
    const count = Math.max(chars.length, tokens.length);
    if (!count) continue;
    if (notes.length && block.leadRest > 0.08) {
      notes.push({ char: "-", jianpu: "0", durationSeconds: block.leadRest });
    }
    const stored = Array.isArray(block.durations) ? block.durations : [];
    const aligned = stored.length === count;
    const total = stored.length ? stored.reduce((a, b) => a + b, 0) : count * 0.4;
    const glides = Array.isArray(block.glides) ? block.glides : [];
    for (let i = 0; i < count; i += 1) {
      const char = chars[i] != null ? chars[i] : "啦";
      const jianpu = tokens[i] != null ? tokens[i] : tokens[tokens.length - 1] || "1";
      const duration = aligned && stored[i] ? stored[i] : total / count;
      const note = { char, jianpu, durationSeconds: Number(duration.toFixed(3)) };
      if (aligned && glides[i] != null && /[↗↘]/.test(jianpu)) {
        note.glideMid = glides[i];
      }
      notes.push(note);
    }
  }
  return notes;
}

function median(values) {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

// Turn 基頻分析 CSV rows (time,hz,intensity,text) into rich singing notes.
//
// Segmentation is driven by the **text column**: each run of one character is one
// sung note, so *every* character is sung — none get dropped just because the
// pitch tracker missed a few of its frames (a character with no detected pitch
// carries the nearest known pitch so it still sounds). A run's voiced frames give
// its pitch *contour* (抑揚頓挫 / glides). The gaps between characters (empty text)
// are silence by default — only a clearly energetic, sustained gap becomes a 氣音
// (breath), which keeps the brief unvoiced transitions between words from turning
// into 奇怪短促的 noise. Every run keeps its real length, so the notes line up in
// time with the original (head/tail silence is trimmed server-side).
function buildNotesFromRows(rows) {
  const voiced = rows.filter((r) => Number.isFinite(r.hz) && r.hz > 0);
  const tonic = median(voiced.map((r) => r.hz));
  const voicedIntensity = median(voiced.map((r) => r.intensity));
  const breathThreshold = Math.max(0.02, 0.45 * voicedIntensity);
  const deltas = [];
  for (let i = 1; i < rows.length; i += 1) {
    const dt = rows[i].time - rows[i - 1].time;
    if (dt > 0 && dt < 1) {
      deltas.push(dt);
    }
  }
  const binStep = deltas.length ? median(deltas) : 0.02;
  const MIN_BREATH = 0.16; // shorter energetic gaps stay silent, not 氣音

  const notes = [];
  let lastHz = 0;
  // A voiced span with no text is the tail of the previous syllable — extend it
  // (legato) rather than re-articulating it as a separate note.
  const extendOrAddVoiced = (char, contour, span) => {
    const prev = notes[notes.length - 1];
    if (!char && prev && prev.kind === "voiced") {
      prev.contour = subsampleContour(prev.contour.concat(contour), 8);
      prev.durationSeconds += span;
      prev.hz = prev.contour[Math.floor(prev.contour.length / 2)];
      return;
    }
    const points = subsampleContour(contour, 8);
    notes.push({
      char: char || "啦",
      kind: "voiced",
      contour: points,
      hz: points[Math.floor(points.length / 2)],
      durationSeconds: Math.max(0.05, span),
    });
  };

  let i = 0;
  let prevRowTime = null;
  while (i < rows.length) {
    // The silence between paragraphs/blocks (a blank line in the CSV, or — for
    // older CSVs — a big jump in time) becomes a rest so the sung clip stays
    // aligned with the original. We deliberately do NOT split on ordinary bin
    // spacing: that varies per block (a long block has wide bins) and would
    // otherwise chop a character into fragments and drop its lyric.
    if (prevRowTime != null) {
      const gap = rows[i].time - prevRowTime;
      if ((rows[i].blockStart || gap > 0.35) && gap > binStep) {
        notes.push({ char: "-", kind: "rest", durationSeconds: gap - binStep });
      }
    }
    const char = (rows[i].text || "").trim();
    let j = i;
    const hzs = [];
    const intensities = [];
    while (
      j < rows.length &&
      (rows[j].text || "").trim() === char &&
      (j === i || !rows[j].blockStart)
    ) {
      const row = rows[j];
      if (Number.isFinite(row.hz) && row.hz > 0) {
        hzs.push(row.hz);
      }
      intensities.push(row.intensity || 0);
      j += 1;
    }
    const span = rows[j - 1].time - rows[i].time + binStep;
    prevRowTime = rows[j - 1].time;
    if (char) {
      // A character always becomes a sung note: use its measured contour, or the
      // nearest known pitch when its frames were all unvoiced.
      let contour;
      if (hzs.length) {
        contour = hzs;
        lastHz = median(hzs);
      } else {
        contour = [lastHz || tonic || 200];
      }
      extendOrAddVoiced(char, contour, span);
    } else if (hzs.length) {
      lastHz = median(hzs);
      extendOrAddVoiced("", hzs, span); // voiced tail with no text → legato
    } else if (median(intensities) >= breathThreshold && span >= MIN_BREATH) {
      notes.push({ char: "", kind: "breath", intensity: median(intensities), durationSeconds: span });
    } else {
      notes.push({ char: "-", kind: "rest", durationSeconds: span }); // silence keeps timing
    }
    i = j;
  }
  return { notes, tonic, voicedCount: notes.filter((n) => n.kind === "voiced").length };
}

// Load a 基頻分析 CSV for 基頻唱歌: build the rich notes, remember the raw rows for
// the relationship chart, and draw it. Singing then uses state.pitchNotes verbatim.
async function loadAnalysisFile(file) {
  if (!file) {
    return;
  }
  let rows;
  try {
    rows = parsePitchCsv(await file.text());
  } catch {
    setStatus("不是有效的基頻分析 CSV", "error");
    return;
  }
  if (!rows.length) {
    setStatus("基頻分析 CSV 沒有資料", "error");
    return;
  }
  const built = buildNotesFromRows(rows);
  if (!built.voicedCount) {
    setStatus("基頻分析裡找不到可唱的音", "error");
    return;
  }
  state.pitchRows = rows;
  state.pitchNotes = built.notes;
  state.pitchTonic = built.tonic;
  els.pitchEmpty.hidden = true;
  els.pitchVis.hidden = false;
  renderPitchRelation(rows);
  if (els.pitchVisMeta) {
    const seconds = rows[rows.length - 1].time - rows[0].time;
    els.pitchVisMeta.textContent = `${built.voicedCount} 音 · ${seconds.toFixed(1)} 秒`;
  }
  refreshButtons();
  setStatus(
    `已載入基頻分析「${file.name}」,共 ${built.voicedCount} 音(含滑音/氣音),將依實測音高演唱`,
  );
}

// Split rows into paragraphs at the block markers (blank lines in the CSV), or —
// for older CSVs without markers — at big time gaps, mirroring how the 逐字稿
// 基頻分析 view shows one panel per 段落.
function splitPitchParagraphs(rows) {
  const paras = [];
  let current = [rows[0]];
  for (let k = 1; k < rows.length; k += 1) {
    if (rows[k].blockStart || rows[k].time - rows[k - 1].time > 0.35) {
      paras.push(current);
      current = [];
    }
    current.push(rows[k]);
  }
  if (current.length) paras.push(current);
  return paras;
}

// Render the 基頻 / 強度 / 文字 relationship, one panel per 段落 (like the 逐字稿
// 基頻分析 view). All panels share one pitch axis so heights are comparable.
function renderPitchRelation(rows) {
  const host = els.pitchParas;
  if (!host || !rows || !rows.length) {
    return;
  }
  host.replaceChildren();
  let loHz = Infinity;
  let hiHz = -Infinity;
  for (const row of rows) {
    if (Number.isFinite(row.hz) && row.hz > 0) {
      loHz = Math.min(loHz, row.hz);
      hiHz = Math.max(hiHz, row.hz);
    }
  }
  if (!Number.isFinite(loHz) || !Number.isFinite(hiHz)) {
    loHz = 110;
    hiHz = 330;
  }
  if (hiHz - loHz < 1) {
    loHz *= 0.94;
    hiHz *= 1.06;
  }
  splitPitchParagraphs(rows).forEach((para, index) => {
    const figure = document.createElement("figure");
    figure.className = "pitch-para";
    const head = document.createElement("figcaption");
    head.className = "pitch-para-head";
    head.textContent = `段落 ${index + 1} · ${para[0].time.toFixed(1)}–${para[para.length - 1].time.toFixed(1)} 秒`;
    const canvas = document.createElement("canvas");
    canvas.className = "pitch-canvas";
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", `段落 ${index + 1} 的基頻、強度與文字對應關係`);
    figure.append(head, canvas);
    host.append(figure);
    drawPitchCanvas(canvas, para, loHz, hiHz);
  });
}

// Draw one paragraph: the pitch contour (cyan, log-Hz, broken at unvoiced gaps),
// the intensity envelope (orange fill, own scale) and the per-syllable text (top,
// with a faint time guide), on the paragraph's own time axis.
function drawPitchCanvas(canvas, rows, loHz, hiHz) {
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || 480;
  const cssHeight = 140;
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#0b1020";
  ctx.fillRect(0, 0, cssWidth, cssHeight);

  const padX = 10;
  const padTop = 20;
  const padBottom = 22;
  const plotW = Math.max(1, cssWidth - padX * 2);
  const plotH = Math.max(1, cssHeight - padTop - padBottom);
  const bottom = padTop + plotH;

  const t0 = rows[0].time;
  const tSpan = Math.max(1e-3, rows[rows.length - 1].time - t0);
  const xOf = (t) => padX + ((t - t0) / tSpan) * plotW;

  const logLo = Math.log(loHz);
  const logSpan = Math.log(hiHz) - logLo || 1;
  const yOfHz = (hz) => bottom - ((Math.log(hz) - logLo) / logSpan) * plotH;

  let maxI = 1e-6;
  for (const row of rows) maxI = Math.max(maxI, row.intensity || 0);

  // 強度:filled envelope along the bottom (its own scale, up to ~55% height).
  ctx.beginPath();
  ctx.moveTo(xOf(t0), bottom);
  for (const row of rows) {
    const h = Math.min(1, (row.intensity || 0) / maxI) * plotH * 0.55;
    ctx.lineTo(xOf(row.time), bottom - h);
  }
  ctx.lineTo(xOf(rows[rows.length - 1].time), bottom);
  ctx.closePath();
  ctx.fillStyle = "rgba(240, 168, 56, 0.22)";
  ctx.fill();

  // 基頻:the pitch contour, broken wherever the frame is unvoiced.
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#7df9ff";
  ctx.lineJoin = "round";
  ctx.beginPath();
  let drawing = false;
  for (const row of rows) {
    if (!(Number.isFinite(row.hz) && row.hz > 0)) {
      drawing = false;
      continue;
    }
    const x = xOf(row.time);
    const y = yOfHz(row.hz);
    if (drawing) {
      ctx.lineTo(x, y);
    } else {
      ctx.moveTo(x, y);
      drawing = true;
    }
  }
  ctx.stroke();

  // 文字:one label per run of the same character, at the run's centre, skipping
  // labels that would collide with the previous one.
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  let lastLabelX = -Infinity;
  let i = 0;
  while (i < rows.length) {
    const char = (rows[i].text || "").trim();
    let j = i;
    while (j < rows.length && (rows[j].text || "").trim() === char) {
      j += 1;
    }
    if (char) {
      const cx = (xOf(rows[i].time) + xOf(rows[j - 1].time)) / 2;
      ctx.strokeStyle = "rgba(230, 237, 243, 0.12)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, padTop);
      ctx.lineTo(cx, bottom);
      ctx.stroke();
      if (cx - lastLabelX >= 13) {
        ctx.fillStyle = "#e6edf3";
        ctx.font = "13px system-ui, sans-serif";
        ctx.fillText(char, Math.max(8, Math.min(cssWidth - 8, cx)), padTop / 2 + 1);
        lastLabelX = cx;
      }
    }
    i = j;
  }

  // Time axis end labels (absolute seconds, matching the paragraph header).
  ctx.fillStyle = "rgba(230, 237, 243, 0.55)";
  ctx.font = "11px system-ui, sans-serif";
  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left";
  ctx.fillText(`${t0.toFixed(1)} 秒`, padX, cssHeight - 6);
  ctx.textAlign = "right";
  ctx.fillText(`${(t0 + tSpan).toFixed(1)} 秒`, cssWidth - padX, cssHeight - 6);
}

// Pick up to ``maxPoints`` values evenly across the contour (keep the shape).
function subsampleContour(values, maxPoints) {
  if (values.length <= maxPoints) {
    return values.slice();
  }
  const out = [];
  for (let i = 0; i < maxPoints; i += 1) {
    out.push(values[Math.round((i * (values.length - 1)) / (maxPoints - 1))]);
  }
  return out;
}

// Parse the 基頻分析 CSV. A blank line marks a paragraph (block) break — the first
// data row after it is flagged ``blockStart`` so segmentation never has to guess
// boundaries from the (per-block-variable) time-bin spacing.
function parsePitchCsv(textData) {
  const rows = [];
  let seenData = false;
  let blockBreak = false;
  for (const raw of textData.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) {
      if (seenData) {
        blockBreak = true; // blank line = paragraph break
      }
      continue;
    }
    const cells = splitCsvLine(line);
    const time = Number.parseFloat(cells[0]);
    if (!Number.isFinite(time)) {
      continue; // header row or stray line
    }
    const hz = Number.parseFloat(cells[1]);
    const intensity = Number.parseFloat(cells[2]);
    rows.push({
      time,
      hz: Number.isFinite(hz) ? hz : null,
      intensity: Number.isFinite(intensity) ? intensity : 0,
      text: (cells[3] || "").trim(),
      blockStart: blockBreak,
    });
    seenData = true;
    blockBreak = false;
  }
  return rows;
}

function splitCsvLine(line) {
  const cells = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (quoted) {
      if (char === '"' && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        current += char;
      }
    } else if (char === ",") {
      cells.push(current);
      current = "";
    } else if (char === '"') {
      quoted = true;
    } else {
      current += char;
    }
  }
  cells.push(current);
  return cells;
}

// Shared singing job: POST the notes to /api/voice/sing and show the result in
// the named player. ``resultKind`` picks the panel ("sing" or "pitch").
async function performSing({ notes, useMeasured, tonicHz, speed, resultKind }) {
  await runExclusive("合成歌聲中…", async (signal) => {
    await waitForModelReady();
    setStatus("唱歌合成中…", "live");
    const body = { voiceId: state.selectedId, notes };
    if (Number.isFinite(tonicHz) && tonicHz > 0) {
      body.tonicHz = tonicHz;
    }
    if (useMeasured) {
      body.useMeasuredHz = true;
    }
    if (speed) {
      body.speed = speed;
    }
    const response = await fetch("/api/voice/sing", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      throw new Error(data.detail || "唱歌合成失敗");
    }
    showResult(resultKind, data.audioBase64);
    setStatus("歌聲完成,可試聽後下載或雲端儲存");
  });
}

// 簡譜唱歌: sing the editable per-sentence rows (relative degrees, transposed into
// the target voice's register; glides ride along as 3↗5 with their measured position).
async function runSing() {
  if (!state.selectedId) {
    setStatus("請先在聲音庫選擇或新增一個目標聲音", "error");
    return;
  }
  const notes = buildSingNotes(state.singBlocks);
  const singable = notes.some(
    (note) => note.jianpu && note.jianpu !== "0" && note.jianpu !== "-",
  );
  if (!singable) {
    setStatus("沒有可唱的內容,請先上傳逐字稿或輸入簡譜", "error");
    return;
  }
  await performSing({
    notes,
    useMeasured: false,
    tonicHz: Number.parseFloat(els.singTonic.value),
    speed: positiveNumber(els.singSpeed.value),
    resultKind: "sing",
  });
}

// 基頻唱歌: sing the loaded 基頻分析's rich notes verbatim, following the measured
// pitch contour / 抑揚頓挫 / 氣音 (no 簡譜 transposition).
async function runPitchSing() {
  if (!state.selectedId) {
    setStatus("請先在聲音庫選擇或新增一個目標聲音", "error");
    return;
  }
  if (!state.pitchNotes || !state.pitchNotes.length) {
    setStatus("請先載入基頻分析 (.csv)", "error");
    return;
  }
  await performSing({
    notes: state.pitchNotes,
    useMeasured: true,
    tonicHz: state.pitchTonic,
    speed: positiveNumber(els.pitchSpeed.value),
    resultKind: "pitch",
  });
}

// 文字轉換 / 簡譜唱歌 / 基頻唱歌 each have one output player.
const RESULT_TARGETS = {
  tts: { audio: "ttsResult", wrap: "ttsResultWrap", url: "ttsResultUrl", b64: "ttsResultB64" },
  sing: { audio: "singResult", wrap: "singResultWrap", url: "singResultUrl", b64: "singResultB64" },
  pitch: {
    audio: "pitchResult",
    wrap: "pitchResultWrap",
    url: "pitchResultUrl",
    b64: "pitchResultB64",
  },
};

function showResult(kind, audioBase64) {
  const target = RESULT_TARGETS[kind];
  const audioEl = els[target.audio];
  const wrapEl = els[target.wrap];
  const urlKey = target.url;
  const b64Key = target.b64;
  if (state[urlKey]) {
    URL.revokeObjectURL(state[urlKey]);
  }
  const blob = base64ToBlob(audioBase64, "audio/wav");
  state[urlKey] = URL.createObjectURL(blob);
  state[b64Key] = audioBase64;
  audioEl.src = state[urlKey];
  audioEl.load();
  wrapEl.hidden = false;
  // Make sure the freshly revealed player is actually on screen so the user can
  // listen before deciding to download / save. "nearest" avoids the big viewport
  // jump that "center" can cause on phones.
  try {
    wrapEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch {
    /* scrollIntoView options unsupported — ignore */
  }
}

// 聲音轉換 shows the original input clip next to the converted one so they can be
// compared. Per-clip 下載/雲端儲存 live in each player's ⋮ menu; the buttons below
// act on both as a 打包 (bundle).
function showConvertResult(sourceBase64, convertedBase64) {
  if (state.cvSourceUrl) {
    URL.revokeObjectURL(state.cvSourceUrl);
  }
  if (state.cvResultUrl) {
    URL.revokeObjectURL(state.cvResultUrl);
  }
  state.cvSourceB64 = sourceBase64;
  state.cvResultB64 = convertedBase64;
  state.cvSourceUrl = URL.createObjectURL(base64ToBlob(sourceBase64, "audio/wav"));
  state.cvResultUrl = URL.createObjectURL(base64ToBlob(convertedBase64, "audio/wav"));
  els.cvSource.src = state.cvSourceUrl;
  els.cvResult.src = state.cvResultUrl;
  els.cvSource.load();
  els.cvResult.load();
  closeAllPlayerMenus();
  els.cvResultWrap.hidden = false;
  try {
    els.cvResultWrap.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch {
    /* scrollIntoView options unsupported — ignore */
  }
}

// Low-level remote save: POST one WAV to the outputs endpoint, return its server
// filename, or throw with a readable message.
async function saveAudioRemote(audioBase64, kind, { text } = {}) {
  const body = { kind, audioBase64, voiceId: state.selectedId };
  if (text) {
    body.text = text;
  }
  const response = await fetch("/api/voice/outputs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data.ok) {
    throw new Error(data.detail || "雲端儲存失敗");
  }
  return data.filename;
}

// Save one clip with button-disable + status feedback (used by the single-clip
// 雲端儲存 actions in the TTS/sing players and the convert ⋮ menus).
async function saveSingleOutput(audioBase64, kind, button, { text } = {}) {
  if (!audioBase64) {
    return;
  }
  if (button) {
    button.disabled = true;
  }
  setStatus("雲端儲存中…", "live");
  try {
    const filename = await saveAudioRemote(audioBase64, kind, { text });
    setStatus(`已雲端儲存:${filename}`);
  } catch (error) {
    setStatus(error?.message || "雲端儲存失敗", "error");
  } finally {
    if (button) {
      button.disabled = false;
    }
  }
}

// Which clips to include — asked via the checkbox dialog (selection remembered).
// The converted clip defaults on; the original defaults off but stays available.
function convertBundleItems() {
  const items = [];
  if (state.cvSourceB64) {
    items.push({ key: "source", label: "原始音檔", defaultChecked: false });
  }
  items.push({ key: "converted", label: "轉換音檔", defaultChecked: true });
  return items;
}

// 打包下載 — ask which clips to include, then download (zip when >1, else the
// single wav directly).
async function bundleDownloadConvert() {
  if (!state.cvResultB64) {
    return;
  }
  const keys = await pickItems({
    title: "打包下載",
    storageKey: "voice-convert-download",
    items: convertBundleItems(),
  });
  if (!keys || keys.length === 0) {
    return;
  }
  const stamp = Date.now();
  const files = [];
  if (keys.includes("source") && state.cvSourceB64) {
    files.push({ name: `breeze-voice-original-${stamp}.wav`, bytes: base64ToBytes(state.cvSourceB64) });
  }
  if (keys.includes("converted")) {
    files.push({ name: `breeze-voice-converted-${stamp}.wav`, bytes: base64ToBytes(state.cvResultB64) });
  }
  if (files.length === 0) {
    return;
  }
  if (files.length === 1) {
    const url = URL.createObjectURL(new Blob([files[0].bytes], { type: "audio/wav" }));
    downloadUrl(url, files[0].name);
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  } else {
    const url = URL.createObjectURL(filesToZipBlob(files));
    downloadUrl(url, `breeze-voice-convert-${stamp}.zip`);
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  setStatus("已下載所選音檔");
}

// 打包雲端儲存 — ask which clips to include, then save the selected ones.
async function bundleSaveConvert() {
  if (!state.cvResultB64) {
    return;
  }
  const keys = await pickItems({
    title: "打包雲端儲存",
    storageKey: "voice-convert-save",
    items: convertBundleItems(),
  });
  if (!keys || keys.length === 0) {
    return;
  }
  els.cvBundleSave.disabled = true;
  setStatus("打包雲端儲存中…", "live");
  try {
    const names = [];
    if (keys.includes("source") && state.cvSourceB64) {
      names.push(await saveAudioRemote(state.cvSourceB64, "convert-source"));
    }
    if (keys.includes("converted")) {
      names.push(await saveAudioRemote(state.cvResultB64, "convert"));
    }
    setStatus(names.length ? `已雲端儲存:${names.join("、")}` : "未選擇任何項目");
  } catch (error) {
    setStatus(error?.message || "打包雲端儲存失敗", "error");
  } finally {
    els.cvBundleSave.disabled = false;
  }
}

// --------------------------------------------------------------------------- //
// per-player functions (⋮) menus
// --------------------------------------------------------------------------- //

function closeAllPlayerMenus(except) {
  document.querySelectorAll("#page-voice .player-menu.open").forEach((menu) => {
    if (menu === except) {
      return;
    }
    menu.classList.remove("open");
    menu.querySelector(".player-menu-btn")?.setAttribute("aria-expanded", "false");
    const list = menu.querySelector(".player-menu-list");
    if (list) {
      list.hidden = true;
    }
  });
}

function wirePlayerMenu(menu) {
  const button = menu.querySelector(".player-menu-btn");
  const list = menu.querySelector(".player-menu-list");
  if (!button || !list) {
    return;
  }
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    const willOpen = !menu.classList.contains("open");
    closeAllPlayerMenus(menu);
    menu.classList.toggle("open", willOpen);
    button.setAttribute("aria-expanded", String(willOpen));
    list.hidden = !willOpen;
  });
}

function base64ToBlob(base64, type) {
  return new Blob([base64ToBytes(base64)], { type });
}

function downloadUrl(url, name) {
  if (!url) {
    return;
  }
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
}

// --------------------------------------------------------------------------- //
// shared helpers
// --------------------------------------------------------------------------- //

function refreshButtons() {
  const idle = !state.busy;
  // The library + convert buttons gate on the page not being busy; TTS also
  // needs text and a target. Model readiness is awaited inside each job, so the
  // buttons stay usable and trigger a load on demand.
  els.vcRecord.disabled = !idle;
  els.vcUpload.disabled = !idle;
  els.cvRecord.disabled = !idle;
  els.cvUpload.disabled = !idle;
  els.ttsRun.disabled = !(idle && els.ttsText.value.trim() && state.selectedId);
  const hasScore = state.singBlocks.some(
    (block) => (block.text || "").trim() || (block.jianpu || "").trim(),
  );
  els.singRun.disabled = !(idle && hasScore && state.selectedId);
  els.pitchRun.disabled = !(idle && state.pitchNotes && state.pitchNotes.length && state.selectedId);
}

function setStatus(text, mode = "") {
  els.status.textContent = text;
  els.status.classList.toggle("live", mode === "live");
  els.status.classList.toggle("error", mode === "error");
}

// Parse a 基音 Hz / 速度 input: a finite positive number, or null when blank /
// invalid so the field falls back to its server-side default (auto / 1.0×).
function positiveNumber(value) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

// --------------------------------------------------------------------------- //
// multi-file picker dialog (shared markup in index.html, remembers selection)
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
    /* storage unavailable — selection just won't be remembered */
  }
}

// Show the shared checkbox dialog; resolves to the chosen keys (locked items are
// always included) or null when cancelled. Selection is remembered per key.
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

function formatDuration(seconds) {
  const value = Number(seconds) || 0;
  return `${value.toFixed(1)} 秒`;
}

function formatDate(iso) {
  if (!iso) {
    return "";
  }
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return `${date.getMonth() + 1}/${date.getDate()}`;
}

async function handleUpload(file, onDone) {
  if (!file) {
    return;
  }
  setStatus("解碼音檔中…");
  let blob;
  try {
    const decoded = await decodeFileToInt16(file, VOICE_SAMPLE_RATE);
    if (!decoded.pcm.length) {
      setStatus("音檔沒有聲音", "error");
      return;
    }
    blob = int16ToWavBlob(decoded.pcm, decoded.sampleRate);
  } catch (error) {
    setStatus(error?.message || "音檔解碼失敗", "error");
    return;
  }
  await onDone(blob, file.name);
}

// --------------------------------------------------------------------------- //
// event wiring
// --------------------------------------------------------------------------- //

els.vcRecord.addEventListener("click", () =>
  toggleRecording(els.vcRecord, (blob) => addVoiceFromBlob(blob)),
);
els.vcUpload.addEventListener("click", () => els.vcFile.click());
els.vcFile.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  event.target.value = "";
  void handleUpload(file, (blob, name) => addVoiceFromBlob(blob, name));
});

els.cvRecord.addEventListener("click", () =>
  toggleRecording(els.cvRecord, (blob) => convertFromBlob(blob)),
);
els.cvUpload.addEventListener("click", () => els.cvFile.click());
els.cvFile.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  event.target.value = "";
  void handleUpload(file, (blob) => convertFromBlob(blob));
});
els.cvSrcDownload.addEventListener("click", () =>
  downloadUrl(state.cvSourceUrl, `breeze-voice-original-${Date.now()}.wav`),
);
els.cvSrcSave.addEventListener("click", () =>
  saveSingleOutput(state.cvSourceB64, "convert-source", els.cvSrcSave),
);
els.cvDownload.addEventListener("click", () =>
  downloadUrl(state.cvResultUrl, `breeze-voice-converted-${Date.now()}.wav`),
);
els.cvSave.addEventListener("click", () =>
  saveSingleOutput(state.cvResultB64, "convert", els.cvSave),
);
els.cvBundleDownload.addEventListener("click", bundleDownloadConvert);
els.cvBundleSave.addEventListener("click", bundleSaveConvert);

els.ttsUpload.addEventListener("click", () => els.ttsFile.click());
els.ttsFile.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  event.target.value = "";
  void loadTextFile(file);
});
els.ttsText.addEventListener("input", refreshButtons);
els.ttsRun.addEventListener("click", runTts);
els.ttsDownload.addEventListener("click", () =>
  downloadUrl(state.ttsResultUrl, `breeze-voice-tts-${Date.now()}.wav`),
);
els.ttsSave.addEventListener("click", () =>
  saveSingleOutput(state.ttsResultB64, "tts", els.ttsSave, { text: els.ttsText.value.trim() }),
);

els.singUpload.addEventListener("click", () => els.singFile.click());
els.singFile.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  event.target.value = "";
  void loadTranscriptFile(file);
});
els.singAdd.addEventListener("click", () => {
  state.singBlocks.push({ text: "", jianpu: "", durations: [], glides: [], leadRest: 0 });
  renderSingBlocks();
  refreshButtons();
});
els.singRun.addEventListener("click", runSing);
els.singDownload.addEventListener("click", () =>
  downloadUrl(state.singResultUrl, `breeze-voice-sing-${Date.now()}.wav`),
);
els.singSave.addEventListener("click", () =>
  saveSingleOutput(state.singResultB64, "sing", els.singSave),
);

els.pitchUpload.addEventListener("click", () => els.pitchFile.click());
els.pitchFile.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  event.target.value = "";
  void loadAnalysisFile(file);
});
els.pitchRun.addEventListener("click", runPitchSing);
els.pitchDownload.addEventListener("click", () =>
  downloadUrl(state.pitchResultUrl, `breeze-voice-sing-${Date.now()}.wav`),
);
els.pitchSave.addEventListener("click", () =>
  saveSingleOutput(state.pitchResultB64, "sing", els.pitchSave),
);
// Redraw the relationship chart on resize so it stays crisp / correctly sized.
window.addEventListener("resize", () => {
  if (state.pitchRows && !els.pitchVis.hidden) {
    renderPitchRelation(state.pitchRows);
  }
});

// Long 說明 paragraphs sit behind a ⓘ next to the section title — toggle them.
document.querySelectorAll("#page-voice .voice-desc-toggle").forEach((button) => {
  button.addEventListener("click", () => {
    const desc = button.closest(".voice-card")?.querySelector(".voice-desc");
    if (!desc) {
      return;
    }
    const show = desc.hidden;
    desc.hidden = !show;
    button.setAttribute("aria-expanded", String(show));
    button.setAttribute("aria-label", show ? "隱藏說明" : "顯示說明");
    button.title = show ? "隱藏說明" : "顯示說明";
  });
});

// Seed the 簡譜唱歌 editor with one empty sentence row so manual entry works.
renderSingBlocks();

// Wire each player's ⋮ menu and close any open menu on an outside click.
document.querySelectorAll("#page-voice .player-menu").forEach(wirePlayerMenu);
document.addEventListener("click", () => closeAllPlayerMenus());

// Open the voice page directly when deep-linked (?page=voice / #voice), which is
// how the phone PWA shortcut and a reloaded voice tab land here.
const requestedPage =
  new URLSearchParams(location.search).get("page") || location.hash.replace("#", "");
if (requestedPage === "voice" || document.body.dataset.page === "voice") {
  showPage("voice");
}
