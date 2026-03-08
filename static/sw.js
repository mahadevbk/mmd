const CACHE_NAME = 'mmd-v12'; // Incremented version
const ASSETS = [
  '/',
  './manifest.json',        // Relative to the sw.js location
  './mmdlogo-192.png',
  './mmdlogo-512.png'
];

self.addEventListener('install', (e) => {
  console.log('[Service Worker] Install');
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[Service Worker] Caching all assets');
      // Using map to catch individual errors so one failure doesn't kill the whole SW
      return Promise.all(
        ASSETS.map(url => {
          return cache.add(url).catch(err => console.error('Failed to cache:', url, err));
        })
      );
    })
  );
});

self.addEventListener('activate', (e) => {
  console.log('[Service Worker] Activate');
  // Clean up old caches
  e.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(keys.map((key) => {
        if (key !== CACHE_NAME) return caches.delete(key);
      }));
    })
  );
});

self.addEventListener('fetch', (e) => {
  e.respondWith(
    fetch(e.request).catch(() => caches.match(e.request))
  );
});
