const CACHE_NAME = 'mmd-v3';
const ASSETS = [
  '/',
  'https://raw.githubusercontent.com/mahadevbk/mmd/main/assets/logo/mmdlogo.png'
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
  );
});

self.addEventListener('fetch', (e) => {
  e.respondWith(
    fetch(e.request).catch(() => caches.match(e.request))
  );
});
