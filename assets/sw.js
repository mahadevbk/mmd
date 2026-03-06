self.addEventListener('install', (e) => {
  console.log('MMD Service Worker Installed');
});

self.addEventListener('fetch', (e) => {
  e.respondWith(fetch(e.request));
});
