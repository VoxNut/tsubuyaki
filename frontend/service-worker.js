const CACHE_NAME = 'tsubuyaki-v6';
const ASSETS = [
  './',
  './index.html',
  './manifest.json?v=6',
  './maskable-icon.png?v=6',
  './regular-icon.png?v=6'
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(ASSETS);
    }).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            return caches.delete(key);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  // Do not cache API server requests
  if (e.request.url.includes('/api/')) {
    return;
  }
  
  // Network-first keeps deployed UI updates from being hidden by a stale PWA
  // cache while still allowing the shell to open offline.
  e.respondWith(
    fetch(e.request).then((networkResponse) => {
      if (networkResponse && networkResponse.status === 200 && networkResponse.type === 'basic') {
        const responseToCache = networkResponse.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(e.request, responseToCache));
      }
      return networkResponse;
    }).catch(() => caches.match(e.request))
  );
});
