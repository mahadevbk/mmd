const CACHE_NAME = 'mmd-v14';
const ASSETS = [
  '/',
  '/static/manifest.json',
  '/static/mmdlogo-192.png',
  '/static/mmdlogo-512.png'
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(ASSETS);
    })
  );
});

self.addEventListener('fetch', (e) => {
  e.respondWith(
    caches.match(e.request).then((response) => {
      return response || fetch(e.request);
    })
  );
});
