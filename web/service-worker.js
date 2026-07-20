const CACHE_NAME = "breeze-elf-v59";
const ASSETS = [
  "./",
  "./app.js",
  "./app.js?v=37",
  "./voice.js",
  "./voice.js?v=23",
  "./audio-utils.js",
  "./audio-utils.js?v=2",
  "./audio-worklet.js",
  "./manifest.webmanifest",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((names) => Promise.all(names.filter((name) => name !== CACHE_NAME).map((name) => caches.delete(name))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") {
    return;
  }
  event.respondWith(fetch(event.request).catch(() => caches.match(event.request)));
});
