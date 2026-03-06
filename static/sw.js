const CACHE_NAME = 'mmd-v8';
const ASSETS = [
  '/',
  'https://raw.githubusercontent.com/mahadevbk/mmd/main/static/mmdlogo-192.png',
  'https://raw.githubusercontent.com/mahadevbk/mmd/main/static/mmdlogo-512.png',
  'https://raw.githubusercontent.com/mahadevbk/mmd/main/static/manifest.json'
];

self.addEventListener('install', (e) => {
  console.log('[Service Worker] Install');
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[Service Worker] Caching all assets');
      return cache.addAll(ASSETS);
    })
  );
});

self.addEventListener('activate', (e) => {
  console.log('[Service Worker] Activate');
});

self.addEventListener('fetch', (e) => {
  e.respondWith(
    fetch(e.request).catch(() => caches.match(e.request))
  );
});
