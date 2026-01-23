// detection.js
// SISTEMA DE DETECCIÓN: ARQUITECTURA SERVERLESS (JS -> SUPABASE)
// VERSIÓN: Matemática Corregida (Estabilidad) + Bostezos Sensibles

import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm'

// --- CONFIGURACIÓN SUPABASE ---
const supabaseUrl = 'https://roogjmgxghbuiogpcswy.supabase.co'
const supabaseKey = 'sb_publishable_RTN2PXvdWOQFfUySAaTa_g_LLe-T_NU'
const supabase = createClient(supabaseUrl, supabaseKey)

// --- AUDIO & UI ---
const alarmAudio = document.getElementById('alarmSound');
const notifyAudio = document.getElementById('notifySound');
const warningPopup = document.getElementById('warningPopup');

// --- VARIABLES GLOBALES ---
let moderateAlertCooldown = false;
let moderateWarningCount = 0; 
let lastWarningTime = 0; 
let lastCaptureMinute = 0; 
let wakeLock = null; 

// --- VARIABLES MODO NOCTURNO ---
let isNightMode = false;
let processingCanvas = null; 
let processingCtx = null;

// --- VARIABLES LÓGICAS ---
let blinkTimestamps = [];
let slowBlinksBuffer = []; 
let yawnsBuffer = [];       
let yawnCountTotal = 0; 

let earHistory = [];
let mouthHistory = [];
let baselineSamples = [];
let baselineEMA = null;
let initialCalibrationDone = false;

let eyeState = 'open';
let mouthState = 'closed';
let closedFrameCounter = 0;
let reopenGraceCounter = 0;
let prevSmoothedEAR = 0;
let eyeClosedStartTime = 0;

let dynamicEARBaseline = null;
let dynamicMARBaseline = 0.1; // Baseline boca cerrada

let lastModerateTimestamp = 0;
let microsleepTriggered = false; 
let yawnFrameCounter = 0;
let lastYawnTimestamp = 0;

export function toggleNightMode(active) {
    isNightMode = active;
    console.log(`🌙 Modo Nocturno: ${isNightMode ? 'ON' : 'OFF'}`);
}

function resetDetectionState() {
    console.log("🔄 Reiniciando detección...");
    blinkTimestamps = []; slowBlinksBuffer = []; yawnsBuffer = []; yawnCountTotal = 0; 
    earHistory = []; mouthHistory = []; baselineSamples = []; baselineEMA = null;
    initialCalibrationDone = false; 
    eyeState = 'open'; mouthState = 'closed'; closedFrameCounter = 0; reopenGraceCounter = 0;
    prevSmoothedEAR = 0; eyeClosedStartTime = 0;
    dynamicEARBaseline = null; dynamicMARBaseline = 0.1;
    lastModerateTimestamp = 0; microsleepTriggered = false; yawnFrameCounter = 0;
    moderateWarningCount = 0; lastCaptureMinute = 0; moderateAlertCooldown = false;
    lastYawnTimestamp = 0;
}

export async function startDetection({ rol, videoElement, canvasElement, estado, cameraRef, sessionId, onRiskUpdate }) {
    resetDetectionState();

    processingCanvas = document.createElement('canvas');
    processingCtx = processingCanvas.getContext('2d');

    const canvasCtx = canvasElement.getContext('2d');
    const isDev = rol === 'Dev';

    videoElement.style.display = 'block';
    canvasElement.style.display = isDev ? 'block' : 'none';

    try {
        if ('wakeLock' in navigator) wakeLock = await navigator.wakeLock.request('screen');
    } catch (err) { console.error(err); }

    // =========================================================
    // PARÁMETROS
    // =========================================================
    const SMOOTHING_WINDOW = 3; 
    const BASELINE_FRAMES_INIT = 45; 
    const EMA_ALPHA = 0.05; 
    const BASELINE_MULTIPLIER = 0.65; // Ajustado para EAR puro
    const CLOSED_FRAMES_THRESHOLD = 1; 
    const DERIVATIVE_THRESHOLD = -0.0025;
    
    // --- UMBRALES BOSTEZO (ALTA SENSIBILIDAD) ---
    const MIN_YAWN_MAR = 0.35; // Umbral absoluto mínimo
    const YAWN_DURATION_THRESHOLD = 0.8; // Segundos
    const YAWN_COOLDOWN_MS = 4000;

    const MICROSUEÑO_THRESHOLD = 2.0; 
    const MIN_SLOW_BLINK_DURATION = 0.5; 
    const FPS = 30; 
    const EYE_REOPEN_GRACE_FRAMES = 3;

    // --- MATEMÁTICAS ---
    // Usamos las dimensiones de la imagen, no del canvas, para mayor precisión
    function toPixel(l, w, h) { return { x: l.x * w, y: l.y * h }; }
    function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
    function movingAverage(arr) { return arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0; }
    function median(arr) {
        const a = [...arr].sort((x, y) => x - y);
        const m = Math.floor(a.length / 2);
        return arr.length % 2 === 0 ? (a[m - 1] + a[m]) / 2 : a[m];
    }

    function calculateEAR_px(landmarks, indices, w, h) {
        const [p0, p1, p2, p3, p4, p5] = indices.map(i => toPixel(landmarks[i], w, h));
        const v1 = dist(p1, p5); const v2 = dist(p2, p4); const horiz = dist(p0, p3);
        return horiz === 0 ? 0 : (v1 + v2) / (2 * horiz);
    }

    function calculateMAR_px(landmarks, indices, w, h) {
        const p = indices.map(i => toPixel(landmarks[i], w, h));
        const horiz = dist(p[0], p[1]); 
        const v = (dist(p[2], p[3]) + dist(p[4], p[5]) + dist(p[6], p[7])) / 3;
        return horiz === 0 ? 0 : v / horiz;
    }

    const RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144];
    const LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380];
    const MOUTH_IDX = [61, 291, 13, 14, 81, 178, 311, 402];

    const faceMesh = new FaceMesh({
        locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    });

    faceMesh.setOptions({
        maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5
    });

    faceMesh.onResults((results) => {
        if (!results.image) return;

        // Dimensiones reales de la imagen analizada
        const imgW = results.image.width;
        const imgH = results.image.height;

        const imagenFinal = isNightMode ? processingCanvas : results.image;

        if (isDev) {
            canvasElement.width = imgW;
            canvasElement.height = imgH;
            canvasCtx.save();
            canvasCtx.clearRect(0, 0, imgW, imgH);
            canvasCtx.drawImage(imagenFinal, 0, 0, imgW, imgH);
        }

        if (!results.multiFaceLandmarks?.length) {
            estado.innerHTML = `<p>❌ No se detecta rostro</p>`;
            if (isDev) canvasCtx.restore();
            return;
        }

        const lm = results.multiFaceLandmarks[0];

        // 1. CÁLCULO EAR/MAR (CORREGIDO: Sin depender de distancia)
        // Usamos EAR puro. El valor normal de ojo abierto es ~0.25 - 0.35
        const rightEAR = calculateEAR_px(lm, RIGHT_EYE_IDX, imgW, imgH);
        const leftEAR = calculateEAR_px(lm, LEFT_EYE_IDX, imgW, imgH);
        const earRaw = (rightEAR + leftEAR) / 2; // Ya no dividimos por faceWidth
        
        const mar = calculateMAR_px(lm, MOUTH_IDX, imgW, imgH);

        // 2. CALIBRACIÓN
        if (!initialCalibrationDone) {
            const progress = Math.round((baselineSamples.length / BASELINE_FRAMES_INIT) * 100);
            estado.innerHTML = `<p>🔄 Calibrando... ${progress}%</p>`;
            if (earRaw > 0) baselineSamples.push(earRaw);
            
            if (baselineSamples.length >= BASELINE_FRAMES_INIT) {
                baselineEMA = median(baselineSamples);
                // Protección: Si el baseline es imposiblemente bajo (ojos cerrados), usamos un estándar
                if (baselineEMA < 0.15) baselineEMA = 0.25; 
                
                dynamicEARBaseline = baselineEMA;
                initialCalibrationDone = true;
                console.log("Baseline EAR Fijado:", dynamicEARBaseline);
            }
            return;
        }

        // 3. SUAVIZADO
        earHistory.push(earRaw); if (earHistory.length > SMOOTHING_WINDOW) earHistory.shift();
        mouthHistory.push(mar); if (mouthHistory.length > SMOOTHING_WINDOW) mouthHistory.shift();

        const smoothedEAR = movingAverage(earHistory);
        const instantEAR = earRaw;
        const smoothedMAR = movingAverage(mouthHistory);

        // =====================================================================
        // LÓGICA DE BOSTEZOS (ALTA SENSIBILIDAD)
        // =====================================================================
        // Umbral dinámico: Boca cerrada actual + un delta pequeño (0.15)
        // O el mínimo absoluto 0.35 (lo que sea mayor)
        const personalYawnThreshold = dynamicMARBaseline + 0.15; 
        const FINAL_YAWN_THRESHOLD = Math.max(personalYawnThreshold, MIN_YAWN_MAR);

        let isYawningNow = false;

        if (Date.now() - lastYawnTimestamp > YAWN_COOLDOWN_MS) {
            if (smoothedMAR > FINAL_YAWN_THRESHOLD) {
                isYawningNow = true;
                yawnFrameCounter++;
                if (yawnFrameCounter / FPS >= YAWN_DURATION_THRESHOLD && mouthState === 'closed') {
                    yawnsBuffer.push(Date.now());
                    yawnCountTotal++;
                    mouthState = 'open'; 
                    lastYawnTimestamp = Date.now(); 
                    console.log(`🥱 Bostezo: ${smoothedMAR.toFixed(2)} > ${FINAL_YAWN_THRESHOLD.toFixed(2)}`);
                }
            } else {
                if (smoothedMAR < FINAL_YAWN_THRESHOLD * 0.85) {
                    yawnFrameCounter = 0;
                    mouthState = 'closed';
                }
            }
        }

        // Adaptación Baseline (Lenta)
        if (smoothedEAR > 0 && eyeState === 'open' && !isYawningNow && smoothedMAR < FINAL_YAWN_THRESHOLD) {
            dynamicEARBaseline = EMA_ALPHA * smoothedEAR + (1 - EMA_ALPHA) * dynamicEARBaseline;
        }
        if (mouthState === 'closed' && !isYawningNow) {
            dynamicMARBaseline = EMA_ALPHA * smoothedMAR + (1 - EMA_ALPHA) * dynamicMARBaseline;
        }

        const now = Date.now();
        blinkTimestamps = blinkTimestamps.filter(ts => ts > now - 60000);
        slowBlinksBuffer = slowBlinksBuffer.filter(ts => ts > now - 60000); 
        yawnsBuffer = yawnsBuffer.filter(ts => ts > now - 60000);
        const totalBlinksLastMinute = blinkTimestamps.length;
        const recentSlowBlinks = slowBlinksBuffer.length;
        const recentYawns = yawnsBuffer.length;

        // =====================================================================
        // LÓGICA DE OJOS
        // =====================================================================
        // Umbrales relativos al baseline calibrado
        const EAR_CLOSE_THRESH = dynamicEARBaseline * 0.70; // Cerrar al 70% del abierto
        const EAR_OPEN_THRESH = dynamicEARBaseline * 0.80;  // Abrir al 80% (Histéresis)

        // Usamos instantáneo O suavizado para cierre rápido
        const isEyeClosed = (instantEAR < EAR_CLOSE_THRESH) || (smoothedEAR < EAR_CLOSE_THRESH);
        const isEyeOpen = (instantEAR > EAR_OPEN_THRESH);

        if (eyeState === 'open') {
            if (isEyeClosed) {
                // Si bosteza, ignoramos cierre de ojos
                if (isYawningNow) {
                    closedFrameCounter = 0; eyeClosedStartTime = 0; reopenGraceCounter = 0;
                } else {
                    if (eyeClosedStartTime === 0) eyeClosedStartTime = Date.now();
                    closedFrameCounter++;
                }
                eyeState = 'closed';
            }
        } 
        else if (eyeState === 'closed') {
            if (isEyeOpen) {
                const blinkDuration = (Date.now() - eyeClosedStartTime) / 1000;
                
                if (microsleepTriggered) {
                    console.log(`🚨 Fin Microsueño. ${blinkDuration.toFixed(2)}s`);
                    sendDetectionEvent({ type: 'ALERTA', sessionId, blinkRate: totalBlinksLastMinute, ear: smoothedEAR, riskLevel: 'Alto riesgo', immediate: true, realDuration: blinkDuration });
                    microsleepTriggered = false;
                } else if (blinkDuration > MIN_SLOW_BLINK_DURATION) {
                    slowBlinksBuffer.push(Date.now());
                    blinkTimestamps.push(Date.now()); 
                } else {
                    blinkTimestamps.push(Date.now());
                }
                eyeState = 'open';
                eyeClosedStartTime = 0;
            }
        }

        // =====================================================================
        // GESTIÓN DE RIESGO
        // =====================================================================
        let riskLevel = 'Normal';
        let currentClosureDuration = 0;
        if (eyeState === 'closed' && eyeClosedStartTime > 0) {
            currentClosureDuration = (Date.now() - eyeClosedStartTime) / 1000;
        }

        if (currentClosureDuration >= MICROSUEÑO_THRESHOLD) {
            riskLevel = 'Alto riesgo';
            warningPopup.className = "warning-popup alert-red active";
            const popupContent = document.getElementById('popupTextContent');
            if (popupContent) popupContent.innerHTML = `<h3>🚨 ¡PELIGRO! 🚨</h3><p>Ojos cerrados: ${currentClosureDuration.toFixed(1)}s</p>`;
            if (alarmAudio && alarmAudio.paused) { alarmAudio.currentTime = 0; alarmAudio.play().catch(e => {}); }
            if (!microsleepTriggered) microsleepTriggered = true;
        } 
        else if (recentSlowBlinks >= 2 || recentYawns >= 2 || (recentYawns >= 1 && recentSlowBlinks >= 1)) {
            riskLevel = 'Moderado';
            if (!microsleepTriggered && alarmAudio && !alarmAudio.paused) { alarmAudio.pause(); alarmAudio.currentTime = 0; }
            if (!moderateAlertCooldown) {
                if (Date.now() - lastWarningTime > 120000) moderateWarningCount = 0;
                moderateWarningCount++;
                lastWarningTime = Date.now();
                if (notifyAudio) { notifyAudio.currentTime = 0; notifyAudio.play().catch(e => {}); }
                
                const popupContent = document.getElementById('popupTextContent');
                if (popupContent) {
                    let razon = recentYawns >= 2 ? "Bostezos frecuentes." : "Somnolencia detectada.";
                    warningPopup.className = moderateWarningCount >= 3 ? "warning-popup alert-red active" : "warning-popup alert-orange active";
                    popupContent.innerHTML = `<h3>⚠️ Atención</h3><p>${razon}</p>`;
                }
                sendDetectionEvent({ type: 'ALERTA', sessionId, blinkRate: totalBlinksLastMinute, slowBlinks: recentSlowBlinks, ear: smoothedEAR, riskLevel, yawnDetected: (recentYawns > 0), totalYawns: recentYawns });
                slowBlinksBuffer = []; yawnsBuffer = []; moderateAlertCooldown = true;
                setTimeout(() => { if (riskLevel !== 'Alto riesgo') warningPopup.classList.remove('active'); }, 5000);
                setTimeout(() => moderateAlertCooldown = false, 15000);
            }
        } 
        else {
            if (totalBlinksLastMinute > 25) riskLevel = 'Leve'; else riskLevel = 'Normal';
            if (!microsleepTriggered) {
                if (alarmAudio && !alarmAudio.paused) { alarmAudio.pause(); alarmAudio.currentTime = 0; }
                if (!moderateAlertCooldown && riskLevel === 'Normal') warningPopup.classList.remove('active');
            }
        }

        if (onRiskUpdate) onRiskUpdate(riskLevel);

        if (isDev) {
            drawConnectors(canvasCtx, lm, FACEMESH_TESSELATION, { color: '#00C853', lineWidth: 0.5 });
            drawConnectors(canvasCtx, lm, FACEMESH_RIGHT_EYE, { color: '#FF5722', lineWidth: 1 });
            drawConnectors(canvasCtx, lm, FACEMESH_LEFT_EYE, { color: '#FF5722', lineWidth: 1 });
            drawConnectors(canvasCtx, lm, FACEMESH_LIPS, { color: '#FF4081', lineWidth: 1 });
            canvasCtx.restore();
        }

        // --- SNAPSHOTS ---
        const currentMinute = Math.floor(now / 60000);
        if (currentMinute > lastCaptureMinute && sessionId) {
            let prob = riskLevel === 'Leve' ? 0.3 : riskLevel === 'Moderado' ? 0.7 : riskLevel === 'Alto riesgo' ? 1.0 : 0.0;
            sendDetectionEvent({ type: 'CAPTURA', sessionId, blinkRate: totalBlinksLastMinute, slowBlinks: recentSlowBlinks, ear: smoothedEAR, riskLevel, probabilidad: prob, totalBlinks: totalBlinksLastMinute, totalYawns: yawnCountTotal });
            if (riskLevel === 'Leve') sendDetectionEvent({ type: 'ALERTA', sessionId, blinkRate: totalBlinksLastMinute, ear: smoothedEAR, riskLevel: 'Leve', immediate: false });
            lastCaptureMinute = currentMinute;
        }

        let colorEstado = riskLevel === 'Leve' ? '#facc15' : riskLevel === 'Moderado' ? '#fbbf24' : riskLevel === 'Alto riesgo' ? '#ef4444' : '#4ade80'; 
        estado.innerHTML = `
            <p style="font-size:14px">Parpadeos/min: ${totalBlinksLastMinute}</p>
            <p style="font-size:14px">P. Lentos: ${recentSlowBlinks} | Bostezos: ${recentYawns}</p>
            <p style="font-size:12px; color:#aaa">MAR: ${smoothedMAR.toFixed(2)} | Meta: ${FINAL_YAWN_THRESHOLD.toFixed(2)}</p>
            <p style="font-weight:bold; color:${colorEstado}">Estado: ${riskLevel}</p>
            ${isNightMode ? '<p style="font-size:12px; color:#60a5fa">🌙 Filtro Nocturno: ACTIVO</p>' : ''}
        `;
    });

    cameraRef.current = new Camera(videoElement, {
        onFrame: async () => {
            if (isNightMode) {
                processingCanvas.width = videoElement.videoWidth;
                processingCanvas.height = videoElement.videoHeight;
                processingCtx.filter = 'brightness(1.5) contrast(1.3) grayscale(0.5)';
                processingCtx.drawImage(videoElement, 0, 0, processingCanvas.width, processingCanvas.height);
                await faceMesh.send({ image: processingCanvas });
            } else {
                await faceMesh.send({ image: videoElement });
            }
        },
        width: 480,
        height: 360
    });

    cameraRef.current.start();
}

export async function stopDetection(cameraRef) {
    if (cameraRef.current) { cameraRef.current.stop(); cameraRef.current = null; }
    if (alarmAudio) { alarmAudio.pause(); alarmAudio.currentTime = 0; }
    if (warningPopup) warningPopup.classList.remove('active');
    try { if (wakeLock !== null) { await wakeLock.release(); wakeLock = null; } } catch (err) {}
}

async function sendDetectionEvent({ type, sessionId, blinkRate, slowBlinks = 0, ear, riskLevel, probabilidad = 0, yawnDetected, totalBlinks, totalYawns, immediate = false, realDuration = 0 }) {
    if (!sessionId) return;
    try {
        const captureData = {
            id_sesion: sessionId, hora_captura: new Date().toISOString(), frecuencia_parpadeo: blinkRate, parpadeos_lentos: slowBlinks, bostezos: totalYawns, promedio_ear: Number(ear.toFixed(6)), probabilidad_somnolencia: probabilidad, nivel_riesgo_calculado: riskLevel
        };
        if (type === 'CAPTURA') {
            await supabase.from('Capturas').insert([captureData]);
        } else if (type === 'ALERTA') {
            const { data: snapshotData } = await supabase.from('Capturas').insert([captureData]).select();
            if (!snapshotData) return;
            const relatedCaptureId = snapshotData[0].id_captura;
            let causa = "Fatiga General", valor = probabilidad;
            if (riskLevel === 'Alto riesgo') { causa = "Microsueño"; valor = realDuration > 0 ? parseFloat(realDuration.toFixed(2)) : 2.0; }
            else if (yawnDetected) { causa = "Bostezos"; valor = parseFloat(totalYawns); }
            else if (riskLevel === 'Moderado' && slowBlinks >= 2) { causa = "Somnolencia"; valor = parseFloat(slowBlinks); }
            else if (riskLevel === 'Leve') { causa = "Fatiga"; valor = parseFloat(blinkRate); }
            await supabase.from('Alertas').insert([{ id_sesion: sessionId, id_captura: relatedCaptureId, tipo_alerta: riskLevel === 'Leve' ? "Registro Silencioso" : "Sonora/Visual", nivel_riesgo: riskLevel, causa_detonante: causa, valor_medido: valor, fecha_alerta: new Date().toISOString() }]);
        }
    } catch (err) { console.error(err); }
}