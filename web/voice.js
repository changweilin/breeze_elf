// 變聲工作室 page controller. A single 聲音庫 holds every saved voice (add /
// favorite / rename / delete / pick-as-target are all in one place). Recording
// or uploading runs the matching job immediately — building a voiceprint for a
// new library voice, or re-voicing your clip as the target — behind a busy
// overlay that locks the page and offers a 中斷 button. Finished audio lands in
// a player so you can listen before downloading or saving it remotely.

import {
  VOICE_SAMPLE_RATE,
  blobToBase64,
  decodeFileToInt16,
  int16ToWavBlob,
  pcm16BytesToWavBlob,
} from "./audio-utils.js";

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
  cvResult: document.querySelector("#cv-result"),
  cvDownload: document.querySelector("#cv-download"),
  cvSave: document.querySelector("#cv-save"),
  // text -> target voice
  ttsText: document.querySelector("#tts-text"),
  ttsRun: document.querySelector("#tts-run"),
  ttsResultWrap: document.querySelector("#tts-result-wrap"),
  ttsResult: document.querySelector("#tts-result"),
  ttsDownload: document.querySelector("#tts-download"),
  ttsSave: document.querySelector("#tts-save"),
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
  cvResultB64: "",
  cvResultUrl: "",
  ttsResultB64: "",
  ttsResultUrl: "",
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
    void stopActiveRecording("切換頁面");
  }
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
  els.targetName.textContent = selected ? selected.name : "尚未選擇";
}

function renderVoiceList() {
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

  const pick = document.createElement("input");
  pick.type = "radio";
  pick.name = "voice-pick";
  pick.className = "voice-pick";
  pick.checked = voice.id === state.selectedId;
  pick.setAttribute("aria-label", `設為目標聲音:${voice.name}`);
  pick.addEventListener("change", () => selectVoice(voice.id));

  const body = document.createElement("div");
  body.style.minWidth = "0";
  const name = document.createElement("div");
  name.className = "voice-item-name";
  name.textContent = voice.name;
  name.title = "點擊重新命名";
  name.style.cursor = "pointer";
  name.addEventListener("click", () => renameVoice(voice));
  const meta = document.createElement("div");
  meta.className = "voice-item-meta";
  meta.textContent = `${formatDuration(voice.durationSeconds)} · ${formatDate(voice.createdAt)}`;
  body.append(name, meta);

  const actions = document.createElement("div");
  actions.className = "voice-item-actions";

  const star = document.createElement("button");
  star.type = "button";
  star.className = `icon-btn star${voice.favorite ? " on" : ""}`;
  star.textContent = voice.favorite ? "★" : "☆";
  star.title = voice.favorite ? "取消我的最愛" : "加入我的最愛";
  star.setAttribute("aria-label", star.title);
  star.addEventListener("click", () => toggleFavorite(voice));
  actions.appendChild(star);

  if (voice.hasSample) {
    const play = document.createElement("button");
    play.type = "button";
    play.className = "icon-btn play";
    play.textContent = "▶";
    play.title = "試聽這個聲音";
    play.setAttribute("aria-label", play.title);
    play.addEventListener("click", () => playSample(voice.id));
    actions.appendChild(play);
  }

  const del = document.createElement("button");
  del.type = "button";
  del.className = "icon-btn del";
  del.textContent = "🗑";
  del.title = "刪除這個聲音";
  del.setAttribute("aria-label", del.title);
  del.addEventListener("click", () => deleteVoice(voice));
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

function playSample(voiceId) {
  if (samplePlayer) {
    samplePlayer.pause();
  }
  samplePlayer = new Audio(`/api/voices/${encodeURIComponent(voiceId)}/sample`);
  samplePlayer.play().catch(() => setStatus("無法播放試聽", "error"));
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
    showResult("cv", data.audioBase64);
    setStatus("轉換完成,可試聽後下載或遠端儲存");
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

function showResult(kind, audioBase64) {
  const audioEl = kind === "cv" ? els.cvResult : els.ttsResult;
  const wrapEl = kind === "cv" ? els.cvResultWrap : els.ttsResultWrap;
  const urlKey = kind === "cv" ? "cvResultUrl" : "ttsResultUrl";
  const b64Key = kind === "cv" ? "cvResultB64" : "ttsResultB64";
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
  // listen before deciding to download / save.
  try {
    wrapEl.scrollIntoView({ behavior: "smooth", block: "center" });
  } catch {
    /* scrollIntoView options unsupported — ignore */
  }
}

async function saveOutput(kind) {
  const audioBase64 = kind === "convert" ? state.cvResultB64 : state.ttsResultB64;
  if (!audioBase64) {
    return;
  }
  const button = kind === "convert" ? els.cvSave : els.ttsSave;
  button.disabled = true;
  setStatus("遠端儲存中…", "live");
  try {
    const body = { kind, audioBase64, voiceId: state.selectedId };
    if (kind === "tts") {
      body.text = els.ttsText.value.trim();
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
    setStatus(`已遠端儲存:${data.filename}`);
  } catch (error) {
    setStatus(error?.message || "遠端儲存失敗", "error");
  } finally {
    button.disabled = false;
  }
}

function base64ToBlob(base64, type) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Blob([bytes], { type });
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
els.cvDownload.addEventListener("click", () =>
  downloadUrl(state.cvResultUrl, `breeze-voice-convert-${Date.now()}.wav`),
);
els.cvSave.addEventListener("click", () => saveOutput("convert"));

els.ttsText.addEventListener("input", refreshButtons);
els.ttsRun.addEventListener("click", runTts);
els.ttsDownload.addEventListener("click", () =>
  downloadUrl(state.ttsResultUrl, `breeze-voice-tts-${Date.now()}.wav`),
);
els.ttsSave.addEventListener("click", () => saveOutput("tts"));

// Open the voice page directly when deep-linked (?page=voice / #voice), which is
// how the phone PWA shortcut and a reloaded voice tab land here.
const requestedPage =
  new URLSearchParams(location.search).get("page") || location.hash.replace("#", "");
if (requestedPage === "voice" || document.body.dataset.page === "voice") {
  showPage("voice");
}
