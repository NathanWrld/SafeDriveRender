// detection.js
// SISTEMA DE DETECCIÓN: ARQUITECTURA SERVERLESS (JS -> SUPABASE)

import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm'

// --- CONFIGURACIÓN SUPABASE ---
const supabaseUrl = 'https://roogjmgxghbuiogpcswy.supabase.co'
const supabaseKey = 'sb_publishable_RTN2PXvdWOQFfUySAaTa_g_LLe-T_NU'
const supabase = createClient(supabaseUrl, supabaseKey)

// --- AUDIO & UI ---
const alarmAudio = document.getElementById('alarmSound');
const notifyAudio = document.getElementById('notifySound');
const warningPopup = document.getElementById('warningPopup');

// --- VARIABLES DE CONTROL ---
let moderateAlertCooldown = false;
let moderateWarningCount = 0; 
let lastWarningTime = 0; 
let lastCaptureMinute = 0; // Control para guardar cada minuto

export function startDetection({ rol, videoElement, canvasElement, estado, cameraRef, sessionId }) {
    const canvasCtx = canvasElement.getContext('2d');
    const isDev = rol === 'Dev';

    videoElement.style.display = 'block';
    canvasElement.style.display = isDev ? 'block' : 'none';

    // ===============================
    // PARÁMETROS
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
    // ESTADO
    // ===============================
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

        // --- LÓGICA DE OJOS ---
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
            reopenGraceCounter++;
            if (reopenGraceCounter >= EYE_REOPEN_GRACE_FRAMES) {
                if (eyeState === 'closed') {
                    const duration = closedFrameCounter / FPS;
                    if (duration > MIN_SLOW_BLINK_DURATION && duration < MICROSUEÑO_THRESHOLD) {
                        slowBlinksBuffer.push(Date.now());
                        console.log(`🐢 Parpadeo Lento: ${duration.toFixed(2)}s`);
                    }
                    blinkTimestamps.push(Date.now());
                    eyeState = 'open';
                }
                closedFrameCounter = 0;
                microsleepTriggered = false;
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

        // --- ANÁLISIS 60s ---
        const now = Date.now();
        blinkTimestamps = blinkTimestamps.filter(ts => ts > now - 60000);
        slowBlinksBuffer = slowBlinksBuffer.filter(ts => ts > now - 60000); 
        yawnsBuffer = yawnsBuffer.filter(ts => ts > now - 60000);

        const totalBlinksLastMinute = blinkTimestamps.length;
        const recentSlowBlinks = slowBlinksBuffer.length;
        const recentYawns = yawnsBuffer.length;

        // --- RIESGO ---
        let riskLevel = 'Normal';
        const closureDuration = closedFrameCounter / FPS;
        const popupContent = document.getElementById('popupTextContent');

        // A. ALTO RIESGO
        if (closureDuration >= MICROSUEÑO_THRESHOLD) {
            riskLevel = 'Alto riesgo';
            
            warningPopup.className = "warning-popup alert-red active";
            if (popupContent) {
                popupContent.innerHTML = `<h3>🚨 ¡PELIGRO! 🚨</h3><p>Mantenga los ojos abiertos.</p>`;
            }

            if (alarmAudio && alarmAudio.paused) {
                alarmAudio.currentTime = 0;
                alarmAudio.play().catch(e => console.log(e));
            }

            if (!microsleepTriggered) {
                microsleepTriggered = true;
                // Envío inmediato de alerta a BD
                sendDetectionEvent({
                    type: 'ALERTA',
                    sessionId,
                    blinkRate: totalBlinksLastMinute,
                    ear: smoothedEAR,
                    riskLevel,
                    immediate: true
                });
            }
        } 
        
        // B. MODERADO
        else if (recentSlowBlinks >= 3 || recentYawns >= 2 || (recentYawns >= 1 && recentSlowBlinks >= 2)) {
            riskLevel = 'Moderado';
            
            if (alarmAudio && !alarmAudio.paused) {
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
                    let razon = recentYawns >= 2 ? "Bostezos frecuentes." : "Parpadeos lentos.";
                    warningPopup.className = moderateWarningCount >= 3 ? "warning-popup alert-red active" : "warning-popup alert-orange active";
                    popupContent.innerHTML = moderateWarningCount >= 3 
                        ? `<h3>🛑 DESCANSO SUGERIDO</h3><p>${razon}</p><p>Fatiga persistente.</p>`
                        : `<h3>⚠️ Atención</h3><p>${razon}</p><p>Manténgase alerta.</p>`;
                }

                // Envío de alerta a BD
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

                // BORRÓN Y CUENTA NUEVA
                slowBlinksBuffer = []; 
                yawnsBuffer = [];

                moderateAlertCooldown = true;
                setTimeout(() => { if (riskLevel !== 'Alto riesgo') warningPopup.classList.remove('active'); }, 6000);
                setTimeout(() => moderateAlertCooldown = false, 15000);
            }
            lastModerateTimestamp = now;
        } 

        // C. NORMAL
        else {
            if (totalBlinksLastMinute > 25) riskLevel = 'Leve'; 
            else riskLevel = 'Normal';

            if (alarmAudio && !alarmAudio.paused) {
                alarmAudio.pause();
                alarmAudio.currentTime = 0;
            }
            if (!moderateAlertCooldown && riskLevel === 'Normal') {
                 warningPopup.classList.remove('active');
            }
        }

        // --- DEV DRAW ---
        if (isDev) {
            drawConnectors(canvasCtx, lm, FACEMESH_TESSELATION, { color: '#00C853', lineWidth: 0.5 });
            drawConnectors(canvasCtx, lm, FACEMESH_RIGHT_EYE, { color: '#FF5722', lineWidth: 1 });
            drawConnectors(canvasCtx, lm, FACEMESH_LEFT_EYE, { color: '#FF5722', lineWidth: 1 });
            drawConnectors(canvasCtx, lm, FACEMESH_LIPS, { color: '#FF4081', lineWidth: 1 });
            canvasCtx.restore();
        }

        // --- ENVÍO PERIÓDICO (CAPTURA CADA MINUTO) ---
        const currentMinute = Math.floor(now / 60000);
        if (currentMinute > lastCaptureMinute && sessionId) {
            
            let prob = 0.0;
            if (riskLevel === 'Leve') prob = 0.3;
            if (riskLevel === 'Moderado') prob = 0.7;
            if (riskLevel === 'Alto riesgo') prob = 1.0;

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

            lastCaptureMinute = currentMinute;
        }

        estado.innerHTML = `
            <p style="font-size:14px">Parpadeos/min: ${totalBlinksLastMinute}</p>
            <p style="font-size:14px">P. Lentos (1min): ${recentSlowBlinks} (Min: 3)</p>
            <p style="font-size:14px">Bostezos (1min): ${recentYawns} (Min: 2)</p>
            <p style="font-weight:bold; color:${riskLevel === 'Normal' ? '#4ade80' : '#fbbf24'}">Estado: ${riskLevel}</p>
        `;
    });

    cameraRef.current = new Camera(videoElement, {
        onFrame: async () => await faceMesh.send({ image: videoElement }),
        width: 480,
        height: 360
    });

    cameraRef.current.start();
}

export function stopDetection(cameraRef) {
    if (cameraRef.current) {
        cameraRef.current.stop();
        cameraRef.current = null;
    }
    if (alarmAudio) { alarmAudio.pause(); alarmAudio.currentTime = 0; }
    if (warningPopup) warningPopup.classList.remove('active');
}

// --- FUNCIÓN DE ENVÍO DIRECTO A SUPABASE ---
async function sendDetectionEvent({ type, sessionId, blinkRate, slowBlinks = 0, ear, riskLevel, probabilidad = 0, yawnDetected, totalBlinks, totalYawns, immediate = false }) {
    if (!sessionId) return;

    try {
        if (type === 'CAPTURA') {
            await supabase.from('Capturas').insert([{
                id_sesion: sessionId,
                hora_captura: new Date().toISOString(),
                frecuencia_parpadeo: blinkRate,
                parpadeos_lentos: slowBlinks,
                bostezos: totalYawns, 
                promedio_ear: Number(ear.toFixed(6)),
                probabilidad_somnolencia: probabilidad,
                nivel_riesgo_calculado: riskLevel
            }]);
            console.log("💾 Captura guardada");
        } 
        else if (type === 'ALERTA') {
            let causa = "Fatiga General";
            let valor = probabilidad;

            if (riskLevel === 'Alto riesgo') { causa = "Microsueño"; valor = 2.0; }
            else if (yawnDetected) { causa = "Bostezos"; valor = parseFloat(totalYawns); }
            else if (slowBlinks >= 2) { causa = "Parpadeos Lentos"; valor = parseFloat(slowBlinks); }

            await supabase.from('Alertas').insert([{
                id_sesion: sessionId,
                tipo_alerta: "Sonora/Visual",
                nivel_riesgo: riskLevel,
                causa_detonante: causa,
                valor_medido: valor,
                fecha_alerta: new Date().toISOString()
            }]);
            console.log("🚨 Alerta guardada");
        }
    } catch (err) {
        console.error('Error Supabase:', err);
    }
}