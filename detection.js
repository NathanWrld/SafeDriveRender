// detection.js
// SISTEMA DE DETECCIÓN: ARQUITECTURA SERVERLESS (JS -> SUPABASE)
// CORRECCIÓN: Reseteo de estado al reiniciar la cámara

import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm'

// --- CONFIGURACIÓN SUPABASE ---
const supabaseUrl = 'https://roogjmgxghbuiogpcswy.supabase.co'
const supabaseKey = 'sb_publishable_RTN2PXvdWOQFfUySAaTa_g_LLe-T_NU'
const supabase = createClient(supabaseUrl, supabaseKey)

// --- AUDIO & UI ---
const alarmAudio = document.getElementById('alarmSound');
const notifyAudio = document.getElementById('notifySound');
const warningPopup = document.getElementById('warningPopup');

// --- VARIABLES DE CONTROL (Estado Global del Módulo) ---
let moderateAlertCooldown = false;
let moderateWarningCount = 0; 
let lastWarningTime = 0; 
let lastCaptureMinute = 0; 
let wakeLock = null; 

// Variables de Lógica
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

let dynamicEARBaseline = null;
let dynamicMARBaseline = 0.65;

let lastModerateTimestamp = 0;
let microsleepTriggered = false; 
let yawnFrameCounter = 0;

// --- FUNCIÓN DE RESETEO (NUEVO) ---
function resetDetectionState() {
    console.log("🔄 Reseteando variables de detección...");
    blinkTimestamps = [];
    slowBlinksBuffer = []; 
    yawnsBuffer = [];       
    yawnCountTotal = 0; 

    earHistory = [];
    mouthHistory = [];
    baselineSamples = [];
    baselineEMA = null;
    initialCalibrationDone = false; // Forzar recalibración

    eyeState = 'open';
    mouthState = 'closed';
    closedFrameCounter = 0;
    reopenGraceCounter = 0;
    prevSmoothedEAR = 0;

    dynamicEARBaseline = null;
    dynamicMARBaseline = 0.65;

    lastModerateTimestamp = 0;
    microsleepTriggered = false; 
    yawnFrameCounter = 0;
    
    moderateWarningCount = 0;
    lastCaptureMinute = 0;
    moderateAlertCooldown = false;
}

export async function startDetection({ rol, videoElement, canvasElement, estado, cameraRef, sessionId, onRiskUpdate }) {
    // 1. LIMPIEZA INICIAL: Borramos todo el "sucio" de la sesión anterior
    resetDetectionState();

    const canvasCtx = canvasElement.getContext('2d');
    const isDev = rol === 'Dev';

    videoElement.style.display = 'block';
    canvasElement.style.display = isDev ? 'block' : 'none';

    // --- ACTIVAR WAKE LOCK (MANTENER PANTALLA ENCENDIDA) ---
    try {
        if ('wakeLock' in navigator) {
            wakeLock = await navigator.wakeLock.request('screen');
            console.log('💡 Pantalla mantenida encendida (Wake Lock activo)');
        } else {
            console.warn('⚠️ Wake Lock no soportado en este navegador.');
        }
    } catch (err) {
        console.error(`Error activando Wake Lock: ${err.name}, ${err.message}`);
    }

    // ===============================
    // PARÁMETROS CONSTANTES
    // ===============================
    const SMOOTHING_WINDOW = 5;
    const BASELINE_FRAMES_INIT = 60;
    const EMA_ALPHA = 0.03;
    const BASELINE_MULTIPLIER = 0.60; 
    const CLOSED_FRAMES_THRESHOLD = 1;
    const DERIVATIVE_THRESHOLD = -0.0025;
    
    const MICROSUEÑO_THRESHOLD = 2.0; 
    const MIN_SLOW_BLINK_DURATION = 0.5; 
    
    const FPS = 30;
    const MIN_YAWN_DURATION = 0.8; 
    const EYE_REOPEN_GRACE_FRAMES = 3;

    // ===============================
    // UTILIDADES
    // ===============================
    function toPixel(l) { return { x: l.x * canvasElement.width, y: l.y * canvasElement.height }; }
    function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
    function movingAverage(arr) { return arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0; }
    function median(arr) {
        const a = [...arr].sort((x, y) => x - y);
        const m = Math.floor(a.length / 2);
        return arr.length % 2 === 0 ? (a[m - 1] + a[m]) / 2 : a[m];
    }

    function calculateEAR_px(landmarks, indices) {
        const [p0, p1, p2, p3, p4, p5] = indices.map(i => toPixel(landmarks[i]));
        const vertical1 = dist(p1, p5);
        const vertical2 = dist(p2, p4);
        const horizontal = dist(p0, p3);
        return horizontal === 0 ? 0 : (vertical1 + vertical2) / (2 * horizontal);
    }

    function calculateMAR_px(landmarks, indices) {
        const p = indices.map(i => toPixel(landmarks[i]));
        const horizontal = dist(p[0], p[1]);
        const vAvg = (dist(p[2], p[3]) + dist(p[4], p[5]) + dist(p[6], p[7])) / 3;
        return horizontal === 0 ? 0 : vAvg / horizontal;
    }

    const RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144];
    const LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380];
    const MOUTH_IDX = [61, 291, 13, 14, 81, 178, 311, 402];

    const faceMesh = new FaceMesh({
        locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    });

    faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.55,
        minTrackingConfidence: 0.55
    });

    faceMesh.onResults((results) => {
        if (!results.image) return;

        if (isDev) {
            canvasElement.width = results.image.width;
            canvasElement.height = results.image.height;
            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
            canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
        }

        if (!results.multiFaceLandmarks?.length) {
            estado.innerHTML = `<p>❌ No se detecta rostro</p>`;
            if (isDev) canvasCtx.restore();
            return;
        }

        const lm = results.multiFaceLandmarks[0];

        // --- CÁLCULOS ---
        const rightEAR = calculateEAR_px(lm, RIGHT_EYE_IDX);
        const leftEAR = calculateEAR_px(lm, LEFT_EYE_IDX);
        const earPx = (rightEAR + leftEAR) / 2;
        const mar = calculateMAR_px(lm, MOUTH_IDX);

        const xs = lm.map(p => p.x * canvasElement.width);
        const faceWidthPx = Math.max(...xs) - Math.min(...xs);
        const earRel = faceWidthPx > 0 ? earPx / faceWidthPx : earPx;

        // --- CALIBRACIÓN ---
        if (!initialCalibrationDone) {
            estado.innerHTML = `<p>🔄 Calibrando... (${baselineSamples.length}/${BASELINE_FRAMES_INIT})</p>`; // Feedback visual
            if (earRel > 0) baselineSamples.push(earRel);
            if (baselineSamples.length >= BASELINE_FRAMES_INIT) {
                baselineEMA = median(baselineSamples) || 0.01;
                dynamicEARBaseline = baselineEMA;
                initialCalibrationDone = true;
            }
            return;
        }

        // --- SUAVIZADO ---
        earHistory.push(earRel);
        if (earHistory.length > SMOOTHING_WINDOW) earHistory.shift();
        mouthHistory.push(mar);
        if (mouthHistory.length > SMOOTHING_WINDOW) mouthHistory.shift();

        const smoothedEAR = movingAverage(earHistory);
        const smoothedMAR = movingAverage(mouthHistory);
        const derivative = smoothedEAR - prevSmoothedEAR;
        prevSmoothedEAR = smoothedEAR;

        // --- DETECCIÓN DE BOSTEZO ---
        const MIN_YAWN_MAR = 0.50; 
        const CURRENT_YAWN_THRESHOLD = Math.max(dynamicMARBaseline * 1.4, MIN_YAWN_MAR);
        const isYawningNow = smoothedMAR > CURRENT_YAWN_THRESHOLD;

        // --- ADAPTACIÓN ---
        if (smoothedEAR > 0 && eyeState === 'open' && !isYawningNow) {
            dynamicEARBaseline = EMA_ALPHA * smoothedEAR + (1 - EMA_ALPHA) * dynamicEARBaseline;
        }
        if (mouthState === 'closed') {
            dynamicMARBaseline = EMA_ALPHA * smoothedMAR + (1 - EMA_ALPHA) * dynamicMARBaseline;
        }

        const EAR_THRESHOLD = dynamicEARBaseline * BASELINE_MULTIPLIER;

        // =========================================================================
        // ANÁLISIS 60s (VARIABLES DEFINIDAS AQUÍ PARA EVITAR CRASH)
        // =========================================================================
        const now = Date.now();
        blinkTimestamps = blinkTimestamps.filter(ts => ts > now - 60000);
        slowBlinksBuffer = slowBlinksBuffer.filter(ts => ts > now - 60000); 
        yawnsBuffer = yawnsBuffer.filter(ts => ts > now - 60000);

        const totalBlinksLastMinute = blinkTimestamps.length;
        const recentSlowBlinks = slowBlinksBuffer.length;
        const recentYawns = yawnsBuffer.length;

        // =========================================================================
        // LÓGICA PRINCIPAL DE OJOS
        // =========================================================================
        const consideredClosed = smoothedEAR < EAR_THRESHOLD || derivative < DERIVATIVE_THRESHOLD;

        if (consideredClosed) {
            if (isYawningNow) {
                closedFrameCounter = 0; 
                reopenGraceCounter = 0;
            } else {
                closedFrameCounter++;
            }
            reopenGraceCounter = 0;
            if (eyeState === 'open' && closedFrameCounter >= CLOSED_FRAMES_THRESHOLD) {
                eyeState = 'closed';
            }
        } else {
            // El usuario está abriendo los ojos
            reopenGraceCounter++;
            
            if (reopenGraceCounter >= EYE_REOPEN_GRACE_FRAMES) {
                if (eyeState === 'closed') {
                    // --- EVENTO: LOS OJOS SE ACABAN DE ABRIR ---
                    const totalClosedDuration = closedFrameCounter / FPS;
                    
                    if (microsleepTriggered) {
                        console.log(`🚨 Microsueño finalizado. Total: ${totalClosedDuration.toFixed(2)}s`);
                        sendDetectionEvent({
                            type: 'ALERTA',
                            sessionId,
                            blinkRate: totalBlinksLastMinute, 
                            ear: smoothedEAR,
                            riskLevel: 'Alto riesgo',
                            immediate: true,
                            realDuration: totalClosedDuration 
                        });
                        microsleepTriggered = false; 
                    }
                    else if (totalClosedDuration > MIN_SLOW_BLINK_DURATION && totalClosedDuration < MICROSUEÑO_THRESHOLD) {
                        slowBlinksBuffer.push(Date.now());
                        console.log(`🐢 Parpadeo Lento: ${totalClosedDuration.toFixed(2)}s`);
                    }

                    blinkTimestamps.push(Date.now());
                    eyeState = 'open';
                }
                closedFrameCounter = 0;
            }
        }

        // --- LÓGICA DE BOSTEZOS ---
        if (isYawningNow) {
            yawnFrameCounter++;
            if (yawnFrameCounter / FPS >= MIN_YAWN_DURATION && mouthState === 'closed') {
                yawnsBuffer.push(Date.now());
                yawnCountTotal++;
                mouthState = 'open';
                console.log("🥱 Bostezo detectado");
            }
        } else {
            if (smoothedMAR < CURRENT_YAWN_THRESHOLD * 0.9) {
                yawnFrameCounter = 0;
                mouthState = 'closed';
            }
        }

        // --- GESTIÓN DE RIESGO ---
        let riskLevel = 'Normal';
        const closureDuration = closedFrameCounter / FPS; 
        const popupContent = document.getElementById('popupTextContent');

        // A. ALTO RIESGO
        if (closureDuration >= MICROSUEÑO_THRESHOLD) {
            riskLevel = 'Alto riesgo';
            warningPopup.className = "warning-popup alert-red active";
            if (popupContent) popupContent.innerHTML = `<h3>🚨 ¡PELIGRO! 🚨</h3><p>Mantenga los ojos abiertos.</p>`;

            if (alarmAudio && alarmAudio.paused) {
                alarmAudio.currentTime = 0;
                alarmAudio.play().catch(e => console.log(e));
            }

            if (!microsleepTriggered) {
                microsleepTriggered = true;
                console.log("⚠️ Alerta activada (esperando cierre para guardar)");
            }
        } 
        
        // B. MODERADO
        else if (recentSlowBlinks >= 3 || recentYawns >= 2 || (recentYawns >= 1 && recentSlowBlinks >= 2)) {
            riskLevel = 'Moderado';
            
            if (!microsleepTriggered && alarmAudio && !alarmAudio.paused) {
                alarmAudio.pause();
                alarmAudio.currentTime = 0;
            }

            if (!moderateAlertCooldown) {
                if (Date.now() - lastWarningTime > 120000) moderateWarningCount = 0;
                moderateWarningCount++;
                lastWarningTime = Date.now();

                if (notifyAudio) {
                    notifyAudio.currentTime = 0;
                    notifyAudio.play().catch(e => console.error(e));
                }

                if (popupContent) {
                    let razon = recentYawns >= 2 ? "Bostezos frecuentes." : "Somnolencia detectada.";
                    warningPopup.className = moderateWarningCount >= 3 ? "warning-popup alert-red active" : "warning-popup alert-orange active";
                    popupContent.innerHTML = moderateWarningCount >= 3 
                        ? `<h3>🛑 DESCANSO SUGERIDO</h3><p>${razon}</p><p>Fatiga persistente.</p>`
                        : `<h3>⚠️ Atención</h3><p>${razon}</p><p>Manténgase alerta.</p>`;
                }

                sendDetectionEvent({
                    type: 'ALERTA',
                    sessionId,
                    blinkRate: totalBlinksLastMinute,
                    slowBlinks: recentSlowBlinks,
                    ear: smoothedEAR,
                    riskLevel,
                    yawnDetected: (recentYawns > 0),
                    totalYawns: recentYawns
                });

                slowBlinksBuffer = []; 
                yawnsBuffer = [];

                moderateAlertCooldown = true;
                setTimeout(() => { if (riskLevel !== 'Alto riesgo') warningPopup.classList.remove('active'); }, 6000);
                setTimeout(() => moderateAlertCooldown = false, 15000);
            }
            lastModerateTimestamp = now;
        } 

        // C. LEVE (Umbral de prueba: 20 parpadeos)
        else {
            if (totalBlinksLastMinute > 20) riskLevel = 'Leve'; 
            else riskLevel = 'Normal';

            if (!microsleepTriggered) {
                if (alarmAudio && !alarmAudio.paused) {
                    alarmAudio.pause();
                    alarmAudio.currentTime = 0;
                }
                if (!moderateAlertCooldown && riskLevel === 'Normal') {
                     warningPopup.classList.remove('active');
                }
            }
        }

        // --- ENVIAR ESTADO AL SCRIPT PRINCIPAL ---
        if (onRiskUpdate && typeof onRiskUpdate === 'function') {
            onRiskUpdate(riskLevel);
        }

        // --- DEV DRAW ---
        if (isDev) {
            drawConnectors(canvasCtx, lm, FACEMESH_TESSELATION, { color: '#00C853', lineWidth: 0.5 });
            drawConnectors(canvasCtx, lm, FACEMESH_RIGHT_EYE, { color: '#FF5722', lineWidth: 1 });
            drawConnectors(canvasCtx, lm, FACEMESH_LEFT_EYE, { color: '#FF5722', lineWidth: 1 });
            drawConnectors(canvasCtx, lm, FACEMESH_LIPS, { color: '#FF4081', lineWidth: 1 });
            canvasCtx.restore();
        }

        // =========================================================================
        // ENVÍO PERIÓDICO (CAPTURA CADA MINUTO)
        // =========================================================================
        const currentMinute = Math.floor(now / 60000);
        if (currentMinute > lastCaptureMinute && sessionId) {
            
            let prob = 0.0;
            if (riskLevel === 'Leve') prob = 0.3;
            if (riskLevel === 'Moderado') prob = 0.7;
            if (riskLevel === 'Alto riesgo') prob = 1.0;

            // 1. Guardamos la captura (siempre)
            sendDetectionEvent({
                type: 'CAPTURA',
                sessionId,
                blinkRate: totalBlinksLastMinute,
                slowBlinks: recentSlowBlinks,
                ear: smoothedEAR,
                riskLevel,
                probabilidad: prob,
                totalBlinks: totalBlinksLastMinute,
                totalYawns: yawnCountTotal
            });

            // 2. Si es 'Leve', también lo guardamos en ALERTAS
            if (riskLevel === 'Leve') {
                sendDetectionEvent({
                    type: 'ALERTA',
                    sessionId,
                    blinkRate: totalBlinksLastMinute,
                    ear: smoothedEAR,
                    riskLevel: 'Leve',
                    immediate: false
                });
            }

            lastCaptureMinute = currentMinute;
        }

        // Color condicional en el texto de estado
        let colorEstado = '#4ade80'; 
        if (riskLevel === 'Leve') colorEstado = '#facc15'; 
        if (riskLevel === 'Moderado') colorEstado = '#fbbf24'; 
        if (riskLevel === 'Alto riesgo') colorEstado = '#ef4444'; 

        estado.innerHTML = `
            <p style="font-size:14px">Parpadeos/min: ${totalBlinksLastMinute}</p>
            <p style="font-size:14px">P. Lentos (1min): ${recentSlowBlinks} (Min: 3)</p>
            <p style="font-size:14px">Bostezos (1min): ${recentYawns} (Min: 2)</p>
            <p style="font-weight:bold; color:${colorEstado}">Estado: ${riskLevel}</p>
        `;
    });

    cameraRef.current = new Camera(videoElement, {
        onFrame: async () => await faceMesh.send({ image: videoElement }),
        width: 480,
        height: 360
    });

    cameraRef.current.start();
}

export async function stopDetection(cameraRef) {
    if (cameraRef.current) {
        cameraRef.current.stop();
        cameraRef.current = null;
    }
    if (alarmAudio) { alarmAudio.pause(); alarmAudio.currentTime = 0; }
    if (warningPopup) warningPopup.classList.remove('active');

    // --- LIBERAR WAKE LOCK (PERMITIR QUE LA PANTALLA SE APAGUE) ---
    try {
        if (wakeLock !== null) {
            await wakeLock.release();
            wakeLock = null;
            console.log('💡 Wake Lock liberado');
        }
    } catch (err) {
        console.error(`Error liberando Wake Lock: ${err.name}, ${err.message}`);
    }
}

// --- FUNCIÓN DE ENVÍO DIRECTO A SUPABASE ---
async function sendDetectionEvent({ 
    type, 
    sessionId, 
    blinkRate, 
    slowBlinks = 0, 
    ear, 
    riskLevel, 
    probabilidad = 0, 
    yawnDetected, 
    totalBlinks, 
    totalYawns, 
    immediate = false, 
    realDuration = 0
}) {
    if (!sessionId) return;

    try {
        const captureData = {
            id_sesion: sessionId,
            hora_captura: new Date().toISOString(),
            frecuencia_parpadeo: blinkRate,
            parpadeos_lentos: slowBlinks,
            bostezos: totalYawns, 
            promedio_ear: Number(ear.toFixed(6)),
            probabilidad_somnolencia: probabilidad,
            nivel_riesgo_calculado: riskLevel
        };

        if (type === 'CAPTURA') {
            const { error } = await supabase.from('Capturas').insert([captureData]);
            if (error) console.error('Error guardando Captura:', error.message);
            else console.log(`💾 Captura guardada (${riskLevel})`);
        } 
        
        else if (type === 'ALERTA') {
            
            const { data: snapshotData, error: snapError } = await supabase
                .from('Capturas')
                .insert([captureData])
                .select();

            if (snapError) {
                console.error('Error creando snapshot:', snapError.message);
                return;
            }

            const relatedCaptureId = snapshotData[0].id_captura;

            let causa = "Fatiga General";
            let valor = probabilidad;

            if (riskLevel === 'Alto riesgo') { 
                causa = "Microsueño"; 
                valor = realDuration > 0 ? parseFloat(realDuration.toFixed(2)) : 2.0; 
            }
            else if (yawnDetected) { 
                causa = "Bostezos"; 
                valor = parseFloat(totalYawns); 
            }
            else if (riskLevel === 'Moderado' && slowBlinks >= 2) { 
                causa = "Somnolencia"; 
                valor = parseFloat(slowBlinks); 
            }
            // CAMBIO AQUI: Fatiga y valor = blinkRate
            else if (riskLevel === 'Leve') {
                causa = "Fatiga"; 
                valor = parseFloat(blinkRate); 
            }

            const { error: alertError } = await supabase.from('Alertas').insert([{
                id_sesion: sessionId,
                id_captura: relatedCaptureId,
                tipo_alerta: riskLevel === 'Leve' ? "Registro Silencioso" : "Sonora/Visual",
                nivel_riesgo: riskLevel,
                causa_detonante: causa,
                valor_medido: valor,
                fecha_alerta: new Date().toISOString()
            }]);

            if (alertError) console.error('Error guardando Alerta:', alertError.message);
            else console.log(`🚨 Alerta guardada: ${causa} (Valor: ${valor})`);
        }

    } catch (err) {
        console.error('Error crítico en envío:', err);
    }
}