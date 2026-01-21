const CACHE_NAME = 'visioguard-v1';
const ASSETS_TO_CACHE = [
  './',
  './index.html',
  './home.html',
  './style.css',
  './script.js',
  './detection.js',
  './dashboard.js',
  './burger-btn.js',
  './img/white-logo.png',
  './img/blue-logo.png',
  './img/icon-192.png',
  './img/icon-512.png'
  // No cacheamos archivos de Supabase ni CDNs externos, esos se cargan por red
];

// 1. Instalación: Guardar archivos estáticos
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[Service Worker] Cacheando archivos locales');
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
});

// 2. Activación: Limpiar cachés viejas si actualizas la versión
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keyList) => {
      return Promise.all(keyList.map((key) => {
        if (key !== CACHE_NAME) {
          return caches.delete(key);
        }
      }));
    })
  );
});

// 3. Interceptar peticiones: Servir caché si existe, sino buscar en red
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request);
    })
  );
});