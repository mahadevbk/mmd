const CACHE_NAME = 'mmd-v25';
const ASSETS = [
  'https://mmdtennis.streamlit.app/',
  'https://raw.githubusercontent.com/mahadevbk/mmd/main/static/manifest.json',
  'https://raw.githubusercontent.com/mahadevbk/mmd/main/static/mmdlogo-192.png',
  'https://raw.githubusercontent.com/mahadevbk/mmd/main/static/mmdlogo-512.png'
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
  );
});

self.addEventListener('fetch', (e) => {
  e.respondWith(
    caches.match(e.request).then((res) => res || fetch(e.request))
  );
});
