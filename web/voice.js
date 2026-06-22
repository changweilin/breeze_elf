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
  ttsRun: document.querySelector("#tts-run"),
  ttsResultWrap: document.querySelector("#tts-result-wrap"),
  ttsResult: document.querySelector("#tts-result"),
  ttsDownload: document.querySelector("#tts-download"),
  ttsSave: document.querySelector("#tts-save"),
  // 簡譜 singing
  singUpload: document.querySelector("#sing-upload"),
  singFile: document.querySelector("#sing-file"),
  singScore: document.querySelector("#sing-score"),
  singTonic: document.querySelector("#sing-tonic"),
  singRun: document.querySelector("#sing-run"),
  singResultWrap: document.querySelector("#sing-result-wrap"),
  singResult: document.querySelector("#sing-result"),
  singDownload: document.querySelector("#sing-download"),
  singSave: document.querySelector("#sing-save"),
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

// Sub-tabs inside the voice page: 聲音轉換 / 文字轉語音 / 簡譜唱歌 share the one
// 聲音庫 above them, so only the active operation panel is shown at a time.
const subtabs = Array.from(document.querySelectorAll("#page-voice .subtab"));

function showVoicePanel(panel) {
  closeAllPlayerMenus();
  document.querySelectorAll("#page-voice .vpanel").forEach((element) => {
    element.hidden = element.id !== `vpanel-${panel}`;
  });
  subtabs.forEach((tab) => {
    tab.setAttribute("aria-selected", String(tab.dataset.panel === panel));
  });
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
    setStatus("轉換完成,可試聽後打包下載或打包遠端儲存");
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
    const response = await fetch("/api/voice/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ voiceId: state.selectedId, text }),
      signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      throw new Error(data.detail || "合成失敗");
    }
    showResult("tts", data.audioBase64);
    setStatus("合成完成,可試聽後下載或遠端儲存");
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

// Turn a transcript JSON (the 逐字稿 download/remote-save bundle) into the
// editable "字 簡譜 秒" score. 主音 is left on 自動 so the song is sung at the
// target voice's pitch; the original key is shown only as a placeholder hint.
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
  const lines = [];
  let tonic = 0;
  for (const block of blocks) {
    if (!tonic && block.pitch && block.pitch.medianHz) {
      tonic = Math.round(block.pitch.medianHz);
    }
    const chars = Array.isArray(block.characters) ? block.characters : [];
    for (const entry of chars) {
      const char = (entry.char || "").trim() || "-";
      const jianpu = (entry.jianpu || "").trim() || "0";
      const seconds = entry.durationSeconds ? Number(entry.durationSeconds).toFixed(2) : "0.40";
      lines.push(`${char} ${jianpu} ${seconds}`);
    }
  }
  if (!lines.length) {
    setStatus("逐字稿裡找不到音高資料", "error");
    return;
  }
  els.singScore.value = lines.join("\n");
  // Keep 主音 empty so the song defaults to the *target* voice's pitch — that is
  // what makes it sung in the chosen voice. The 簡譜 is relative, so the melody is
  // preserved and just transposed into the target's range. Auto-filling the
  // original recording's pitch (as before) forced the song into the original
  // singer's key and ignored the target voice. Surface it as a hint instead.
  els.singTonic.value = "";
  els.singTonic.placeholder = tonic ? `自動 · 目標聲音(原曲約 ${tonic} Hz)` : "自動";
  refreshButtons();
  setStatus(`已載入逐字稿「${file.name}」,共 ${lines.length} 個音,將以目標聲音音高演唱`);
}

function parseScore(text) {
  const notes = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) {
      continue;
    }
    const parts = line.split(/\s+/);
    const duration = parts[2] ? Number.parseFloat(parts[2]) : null;
    notes.push({
      char: parts[0] || "",
      jianpu: parts[1] || "",
      durationSeconds: Number.isFinite(duration) ? duration : null,
    });
  }
  return notes;
}

async function runSing() {
  if (!state.selectedId) {
    setStatus("請先在聲音庫選擇或新增一個目標聲音", "error");
    return;
  }
  const notes = parseScore(els.singScore.value);
  if (!notes.some((note) => note.jianpu && note.jianpu !== "0")) {
    setStatus("沒有可唱的簡譜,請先上傳逐字稿或輸入簡譜", "error");
    return;
  }
  const tonic = Number.parseFloat(els.singTonic.value);
  await runExclusive("合成歌聲中…", async (signal) => {
    await waitForModelReady();
    setStatus("唱歌合成中…", "live");
    const body = { voiceId: state.selectedId, notes };
    if (Number.isFinite(tonic) && tonic > 0) {
      body.tonicHz = tonic;
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
    showResult("sing", data.audioBase64);
    setStatus("歌聲完成,可試聽後下載或遠端儲存");
  });
}

// 文字轉語音 / 簡譜唱歌 each have one output player.
const RESULT_TARGETS = {
  tts: { audio: "ttsResult", wrap: "ttsResultWrap", url: "ttsResultUrl", b64: "ttsResultB64" },
  sing: { audio: "singResult", wrap: "singResultWrap", url: "singResultUrl", b64: "singResultB64" },
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
// compared. Per-clip 下載/遠端儲存 live in each player's ⋮ menu; the buttons below
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
    throw new Error(data.detail || "遠端儲存失敗");
  }
  return data.filename;
}

// Save one clip with button-disable + status feedback (used by the single-clip
// 遠端儲存 actions in the TTS/sing players and the convert ⋮ menus).
async function saveSingleOutput(audioBase64, kind, button, { text } = {}) {
  if (!audioBase64) {
    return;
  }
  if (button) {
    button.disabled = true;
  }
  setStatus("遠端儲存中…", "live");
  try {
    const filename = await saveAudioRemote(audioBase64, kind, { text });
    setStatus(`已遠端儲存:${filename}`);
  } catch (error) {
    setStatus(error?.message || "遠端儲存失敗", "error");
  } finally {
    if (button) {
      button.disabled = false;
    }
  }
}

// 打包下載 — zip the original + converted clips into one .zip download.
function bundleDownloadConvert() {
  if (!state.cvResultB64) {
    return;
  }
  const stamp = Date.now();
  const files = [];
  if (state.cvSourceB64) {
    files.push({ name: `breeze-voice-original-${stamp}.wav`, bytes: base64ToBytes(state.cvSourceB64) });
  }
  files.push({ name: `breeze-voice-converted-${stamp}.wav`, bytes: base64ToBytes(state.cvResultB64) });
  const url = URL.createObjectURL(filesToZipBlob(files));
  downloadUrl(url, `breeze-voice-convert-${stamp}.zip`);
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  setStatus("已打包下載原始與轉換音檔");
}

// 打包遠端儲存 — save both clips remotely as a pair.
async function bundleSaveConvert() {
  if (!state.cvResultB64) {
    return;
  }
  els.cvBundleSave.disabled = true;
  setStatus("打包遠端儲存中…", "live");
  try {
    const names = [];
    if (state.cvSourceB64) {
      names.push(await saveAudioRemote(state.cvSourceB64, "convert-source"));
    }
    names.push(await saveAudioRemote(state.cvResultB64, "convert"));
    setStatus(`已打包遠端儲存:${names.join("、")}`);
  } catch (error) {
    setStatus(error?.message || "打包遠端儲存失敗", "error");
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
  els.singRun.disabled = !(idle && els.singScore.value.trim() && state.selectedId);
}

function setStatus(text, mode = "") {
  els.status.textContent = text;
  els.status.classList.toggle("live", mode === "live");
  els.status.classList.toggle("error", mode === "error");
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
els.singScore.addEventListener("input", refreshButtons);
els.singRun.addEventListener("click", runSing);
els.singDownload.addEventListener("click", () =>
  downloadUrl(state.singResultUrl, `breeze-voice-sing-${Date.now()}.wav`),
);
els.singSave.addEventListener("click", () =>
  saveSingleOutput(state.singResultB64, "sing", els.singSave),
);

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
