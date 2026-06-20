// 變聲工作室 page controller: record/upload a target voice A, manage favorites,
// re-voice B's speech as A, and read text in A's voice. Talks to the /api/voice*
// endpoints and shows a progress bar while the model warms up.

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
  target: document.querySelector("#voice-target"),
  empty: document.querySelector("#voice-empty"),
  // progress
  progress: document.querySelector("#voice-progress"),
  progressLabel: document.querySelector("#voice-progress-label"),
  progressPct: document.querySelector("#voice-progress-pct"),
  progressBar: document.querySelector("#voice-progress-bar"),
  // create A
  vcRecord: document.querySelector("#vc-record"),
  vcUpload: document.querySelector("#vc-upload"),
  vcFile: document.querySelector("#vc-file"),
  vcPreview: document.querySelector("#vc-preview"),
  vcName: document.querySelector("#vc-name"),
  vcFav: document.querySelector("#vc-fav"),
  vcSave: document.querySelector("#vc-save"),
  // convert B -> A
  cvRecord: document.querySelector("#cv-record"),
  cvUpload: document.querySelector("#cv-upload"),
  cvFile: document.querySelector("#cv-file"),
  cvInput: document.querySelector("#cv-input"),
  cvRun: document.querySelector("#cv-run"),
  cvResultWrap: document.querySelector("#cv-result-wrap"),
  cvResult: document.querySelector("#cv-result"),
  cvDownload: document.querySelector("#cv-download"),
  // text -> A
  ttsText: document.querySelector("#tts-text"),
  ttsRun: document.querySelector("#tts-run"),
  ttsResultWrap: document.querySelector("#tts-result-wrap"),
  ttsResult: document.querySelector("#tts-result"),
  ttsDownload: document.querySelector("#tts-download"),
  // library
  list: document.querySelector("#voice-list"),
  refresh: document.querySelector("#vl-refresh"),
};

const state = {
  initialized: false,
  modelReady: false,
  loading: false,
  pollTimer: 0,
  voices: [],
  selectedId: "",
  vcBlob: null,
  vcUrl: "",
  cvBlob: null,
  cvUrl: "",
  cvResultUrl: "",
  ttsResultUrl: "",
  busy: false,
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
  document.dispatchEvent(new CustomEvent("breeze:page", { detail: { page } }));
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => showPage(tab.dataset.page));
});

document.addEventListener("breeze:page", (event) => {
  if (event.detail?.page === "voice") {
    void initVoicePage();
  } else {
    stopActiveRecording("切換頁面");
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

async function toggleRecording(button, label, onDone) {
  if (activeRecorder && activeRecordButton !== button) {
    await stopActiveRecording("先停止其他錄音");
  }

  if (activeRecorder && activeRecordButton === button) {
    const recorder = activeRecorder;
    activeRecorder = null;
    activeRecordButton = null;
    button.classList.remove("recording");
    button.textContent = label;
    try {
      const blob = await recorder.stop();
      onDone(blob);
      setStatus("錄音完成");
    } catch (error) {
      setStatus(error?.message || "錄音失敗", "error");
    }
    refreshButtons();
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
    button.textContent = button === els.vcRecord ? "● 開始錄音" : "● 錄製 B 的聲音";
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
  // Keep polling only while the model is still warming up; applyStatus already
  // stops the loop once it reaches ready/error.
  if (state.loading) {
    startPolling();
  }
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
    setStatus("待命");
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
// create voice A
// --------------------------------------------------------------------------- //

function setVcAudio(blob) {
  if (state.vcUrl) {
    URL.revokeObjectURL(state.vcUrl);
  }
  state.vcBlob = blob;
  state.vcUrl = URL.createObjectURL(blob);
  els.vcPreview.src = state.vcUrl;
  els.vcPreview.hidden = false;
  refreshButtons();
}

async function saveVoice() {
  const name = els.vcName.value.trim();
  if (!state.vcBlob || !name || state.busy || !state.modelReady) {
    if (!state.modelReady) {
      void ensureModelLoaded();
    }
    return;
  }
  state.busy = true;
  refreshButtons();
  setStatus("擷取聲音特徵中…", "live");
  try {
    const audioBase64 = await blobToBase64(state.vcBlob);
    const response = await fetch("/api/voices", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, audioBase64, favorite: els.vcFav.checked }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      throw new Error(data.detail || "儲存失敗");
    }
    setStatus(`已儲存「${data.voice.name}」`);
    // reset the create form
    els.vcName.value = "";
    els.vcFav.checked = false;
    clearVcAudio();
    await refreshVoices(data.voice.id);
  } catch (error) {
    setStatus(error?.message || "儲存失敗", "error");
  } finally {
    state.busy = false;
    refreshButtons();
  }
}

function clearVcAudio() {
  if (state.vcUrl) {
    URL.revokeObjectURL(state.vcUrl);
    state.vcUrl = "";
  }
  state.vcBlob = null;
  els.vcPreview.removeAttribute("src");
  els.vcPreview.hidden = true;
}

// --------------------------------------------------------------------------- //
// convert B -> A
// --------------------------------------------------------------------------- //

function setCvAudio(blob) {
  if (state.cvUrl) {
    URL.revokeObjectURL(state.cvUrl);
  }
  state.cvBlob = blob;
  state.cvUrl = URL.createObjectURL(blob);
  els.cvInput.src = state.cvUrl;
  els.cvInput.hidden = false;
  refreshButtons();
}

async function runConvert() {
  if (!state.cvBlob || !state.selectedId || state.busy || !state.modelReady) {
    if (!state.modelReady) {
      void ensureModelLoaded();
    }
    return;
  }
  state.busy = true;
  refreshButtons();
  setStatus("轉換中…", "live");
  try {
    const audioBase64 = await blobToBase64(state.cvBlob);
    const response = await fetch("/api/voice/convert", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ voiceId: state.selectedId, audioBase64 }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      throw new Error(data.detail || "轉換失敗");
    }
    showResult(els.cvResult, els.cvResultWrap, "cvResultUrl", data.audioBase64);
    setStatus("轉換完成");
  } catch (error) {
    setStatus(error?.message || "轉換失敗", "error");
  } finally {
    state.busy = false;
    refreshButtons();
  }
}

// --------------------------------------------------------------------------- //
// text -> A
// --------------------------------------------------------------------------- //

async function runTts() {
  const text = els.ttsText.value.trim();
  if (!text || !state.selectedId || state.busy || !state.modelReady) {
    if (!state.modelReady) {
      void ensureModelLoaded();
    }
    return;
  }
  state.busy = true;
  refreshButtons();
  setStatus("合成語音中…", "live");
  try {
    const response = await fetch("/api/voice/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ voiceId: state.selectedId, text }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      throw new Error(data.detail || "合成失敗");
    }
    showResult(els.ttsResult, els.ttsResultWrap, "ttsResultUrl", data.audioBase64);
    setStatus("合成完成");
  } catch (error) {
    setStatus(error?.message || "合成失敗", "error");
  } finally {
    state.busy = false;
    refreshButtons();
  }
}

function showResult(audioEl, wrapEl, urlKey, audioBase64) {
  if (state[urlKey]) {
    URL.revokeObjectURL(state[urlKey]);
  }
  const blob = base64ToBlob(audioBase64, "audio/wav");
  state[urlKey] = URL.createObjectURL(blob);
  audioEl.src = state[urlKey];
  wrapEl.hidden = false;
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
// library / favorites
// --------------------------------------------------------------------------- //

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
  renderTargetSelect();
  renderVoiceList();
  refreshButtons();
}

function renderTargetSelect() {
  els.target.innerHTML = "";
  state.voices.forEach((voice) => {
    const option = document.createElement("option");
    option.value = voice.id;
    option.textContent = voice.favorite ? `★ ${voice.name}` : voice.name;
    if (voice.id === state.selectedId) {
      option.selected = true;
    }
    els.target.appendChild(option);
  });
  els.target.disabled = state.voices.length === 0;
  els.empty.hidden = state.voices.length !== 0;
}

function renderVoiceList() {
  els.list.innerHTML = "";
  state.voices.forEach((voice) => {
    els.list.appendChild(renderVoiceItem(voice));
  });
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
  pick.setAttribute("aria-label", `選為目標聲音:${voice.name}`);
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
    play.title = "試聽 A 的聲音";
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
  els.target.value = voiceId;
  renderVoiceList();
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
// shared helpers
// --------------------------------------------------------------------------- //

function refreshButtons() {
  const ready = state.modelReady && !state.busy;
  const hasTarget = Boolean(state.selectedId);
  els.vcSave.disabled = !(ready && state.vcBlob && els.vcName.value.trim());
  els.cvRun.disabled = !(ready && state.cvBlob && hasTarget);
  els.ttsRun.disabled = !(ready && els.ttsText.value.trim() && hasTarget);
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
  try {
    const decoded = await decodeFileToInt16(file, VOICE_SAMPLE_RATE);
    if (!decoded.pcm.length) {
      setStatus("音檔沒有聲音", "error");
      return;
    }
    onDone(int16ToWavBlob(decoded.pcm, decoded.sampleRate));
    setStatus("已載入音檔");
  } catch (error) {
    setStatus(error?.message || "音檔解碼失敗", "error");
  }
}

// --------------------------------------------------------------------------- //
// event wiring
// --------------------------------------------------------------------------- //

els.vcRecord.addEventListener("click", () =>
  toggleRecording(els.vcRecord, "● 開始錄音", setVcAudio),
);
els.vcUpload.addEventListener("click", () => els.vcFile.click());
els.vcFile.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  event.target.value = "";
  void handleUpload(file, setVcAudio);
});
els.vcName.addEventListener("input", refreshButtons);
els.vcSave.addEventListener("click", saveVoice);

els.cvRecord.addEventListener("click", () =>
  toggleRecording(els.cvRecord, "● 錄製 B 的聲音", setCvAudio),
);
els.cvUpload.addEventListener("click", () => els.cvFile.click());
els.cvFile.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  event.target.value = "";
  void handleUpload(file, setCvAudio);
});
els.cvRun.addEventListener("click", runConvert);
els.cvDownload.addEventListener("click", () =>
  downloadUrl(state.cvResultUrl, `breeze-voice-convert-${Date.now()}.wav`),
);

els.ttsText.addEventListener("input", refreshButtons);
els.ttsRun.addEventListener("click", runTts);
els.ttsDownload.addEventListener("click", () =>
  downloadUrl(state.ttsResultUrl, `breeze-voice-tts-${Date.now()}.wav`),
);

els.target.addEventListener("change", () => selectVoice(els.target.value));
els.refresh.addEventListener("click", () => refreshVoices(state.selectedId));

// If the page is loaded directly on the voice tab (e.g. via a deep link that
// pre-set the body attribute), kick off initialization.
if (document.body.dataset.page === "voice") {
  void initVoicePage();
}
