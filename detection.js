// detection.js
// SISTEMA DE DETECCIÓN CON TOLERANCIA Y PATRONES ACUMULATIVOS

const BACKEND_URL = 'https://safe-drive-backend.onrender.com';
const alarmAudio = document.getElementById('alarmSound');
const notifyAudio = document.getElementById('notifySound');
const warningPopup = document.getElementById('warningPopup');

// Variables de control de notificaciones
let moderateAlertCooldown = false;
let moderateWarningCount = 0; 
let lastWarningTime = 0; 

export function startDetection({ rol, videoElement, canvasElement, estado, cameraRef }) {
    const canvasCtx = canvasElement.getContext('2d');
    const isDev = rol === 'Dev';

    videoElement.style.display = 'block';
    canvasElement.style.display = isDev ? 'block' : 'none';

    // ===============================
    // PARÁMETROS AJUSTADOS (MENOS SENSIBLES)
    // ===============================
    const SMOOTHING_WINDOW = 5;
    const BASELINE_FRAMES_INIT = 60;
    const EMA_ALPHA = 0.03;
    const BASELINE_MULTIPLIER = 0.60; // Bajamos un poco para exigir un cierre más claro
    const CLOSED_FRAMES_THRESHOLD = 1;
    const DERIVATIVE_THRESHOLD = -0.0025;
    
    // --- UMBRALES DE TIEMPO (TOLERANTES) ---
    // Aumentamos microsueño a 2.0s para evitar falsos positivos por frotarse los ojos
    const MICROSUEÑO_THRESHOLD = 2.0; 
    
    // Parpadeo lento ahora es > 0.5s (0.35s era muy rápido y común)
    const MIN_SLOW_BLINK_DURATION = 0.5; 
    
    const FPS = 30;
    const MIN_YAWN_DURATION = 0.8; 
    const EYE_REOPEN_GRACE_FRAMES = 3;

    // ===============================
    // ESTADO
    // ===============================
    let blinkTimestamps = [];
    
    // BUFFERS PARA DETECCIÓN DE PATRONES
    let slowBlinksBuffer = []; // Guarda timestamps de parpadeos lentos
    let yawnsBuffer = [];      // Guarda timestamps de bostezos
    
    let yawnCountTotal = 0; // Histórico total

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

    // ===============================
    // ÍNDICES MEDIAPIPE
    // ===============================
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

    // ===============================
    // PROCESAMIENTO
    // ===============================
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

        // --- CÁLCULOS MATEMÁTICOS ---
        const rightEAR = calculateEAR_px(lm, RIGHT_EYE_IDX);
        const leftEAR = calculateEAR_px(lm, LEFT_EYE_IDX);
        const earPx = (rightEAR + leftEAR) / 2;
        const mar = calculateMAR_px(lm, MOUTH_IDX);

        const xs = lm.map(p => p.x * canvasElement.width);
        const faceWidthPx = Math.max(...xs) - Math.min(...xs);
        const earRel = faceWidthPx > 0 ? earPx / faceWidthPx : earPx;

        // --- CALIBRACIÓN INICIAL ---
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

        // ===============================
        // 1. DETECCIÓN DE BOSTEZO (ESTADO ACTUAL)
        // ===============================
        const MIN_YAWN_MAR = 0.50; 
        const CURRENT_YAWN_THRESHOLD = Math.max(dynamicMARBaseline * 1.4, MIN_YAWN_MAR);
        const isYawningNow = smoothedMAR > CURRENT_YAWN_THRESHOLD;

        // --- CALIBRACIÓN DINÁMICA ---
        if (smoothedEAR > 0 && eyeState === 'open' && !isYawningNow) {
            dynamicEARBaseline = EMA_ALPHA * smoothedEAR + (1 - EMA_ALPHA) * dynamicEARBaseline;
        }
        if (mouthState === 'closed') {
            dynamicMARBaseline = EMA_ALPHA * smoothedMAR + (1 - EMA_ALPHA) * dynamicMARBaseline;
        }

        const EAR_THRESHOLD = dynamicEARBaseline * BASELINE_MULTIPLIER;

        // ===============================
        // 2. LÓGICA DE OJOS Y PARPADEOS LENTOS
        // ===============================
        const consideredClosed = smoothedEAR < EAR_THRESHOLD || derivative < DERIVATIVE_THRESHOLD;

        if (consideredClosed) {
            if (isYawningNow) {
                // Si bosteza, ignoramos los ojos. No cuenta para nada.
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
                    // --- OJOS ABIERTOS: ANALIZAR DURACIÓN ---
                    const duration = closedFrameCounter / FPS;
                    
                    // Solo consideramos "Lento" si supera 0.5 segundos (antes era 0.35)
                    // Y si es menor al umbral de microsueño (2.0)
                    if (duration > MIN_SLOW_BLINK_DURATION && duration < MICROSUEÑO_THRESHOLD) {
                        slowBlinksBuffer.push(Date.now());
                        console.log(`🐢 Parpadeo Lento registrado: ${duration.toFixed(2)}s`);
                    }

                    blinkTimestamps.push(Date.now());
                    eyeState = 'open';
                }
                closedFrameCounter = 0;
                microsleepTriggered = false;
            }
        }

        // ===============================
        // 3. LÓGICA DE BOSTEZOS (EVENTOS)
        // ===============================
        if (isYawningNow) {
            yawnFrameCounter++;
            if (yawnFrameCounter / FPS >= MIN_YAWN_DURATION && mouthState === 'closed') {
                // ¡BOSTEZO DETECTADO! 
                // Pero NO activamos la alerta aquí directamente. Solo lo guardamos en el historial.
                yawnsBuffer.push(Date.now());
                yawnCountTotal++;
                mouthState = 'open';
                console.log("🥱 Bostezo registrado en historial");
            }
        } else {
            if (smoothedMAR < CURRENT_YAWN_THRESHOLD * 0.9) {
                yawnFrameCounter = 0;
                mouthState = 'closed';
            }
        }

        // ===============================
        // 4. ANÁLISIS DE PATRONES (ULTIMOS 60 SEGUNDOS)
        // ===============================
        const now = Date.now();
        blinkTimestamps = blinkTimestamps.filter(ts => ts > now - 60000);
        
        // Ventana de análisis de 60 segundos para fatiga moderada
        slowBlinksBuffer = slowBlinksBuffer.filter(ts => ts > now - 60000); 
        yawnsBuffer = yawnsBuffer.filter(ts => ts > now - 60000);

        const totalBlinksLastMinute = blinkTimestamps.length;
        const recentSlowBlinks = slowBlinksBuffer.length;
        const recentYawns = yawnsBuffer.length;

        // ===============================
        // 5. DETERMINACIÓN DE RIESGO
        // ===============================
        let riskLevel = 'Normal';
        const closureDuration = closedFrameCounter / FPS;
        const popupContent = document.getElementById('popupTextContent');

        // --- A. ALTO RIESGO (MICROSUEÑO - ACCIÓN INMEDIATA) ---
        // Solo si los ojos están CERRADOS AHORA por más de 2 segundos
        if (closureDuration >= MICROSUEÑO_THRESHOLD) {
            riskLevel = 'Alto riesgo';
            
            warningPopup.className = "warning-popup alert-red active";
            if (popupContent) {
                popupContent.innerHTML = `
                    <h3>🚨 ¡PELIGRO! 🚨</h3>
                    <p>Mantenga los ojos abiertos.</p>
                `;
            }

            if (alarmAudio && alarmAudio.paused) {
                alarmAudio.currentTime = 0;
                let playPromise = alarmAudio.play();
                if (playPromise !== undefined) playPromise.catch(e => console.log(e));
            }

            if (!microsleepTriggered) {
                microsleepTriggered = true;
                sendDetectionEvent({
                    blinkRate: totalBlinksLastMinute,
                    ear: smoothedEAR,
                    riskLevel,
                    yawnDetected: false, 
                    totalBlinks: totalBlinksLastMinute,
                    totalYawns: yawnCountTotal,
                    immediate: true
                });
            }
        } 
        
        // --- B. MODERADO (SOLO POR ACUMULACIÓN DE FACTORES) ---
        // REGLAS DE DISPARO:
        // 1. Al menos 3 parpadeos lentos en 1 minuto.
        // 2. Al menos 2 bostezos en 1 minuto.
        // 3. Combinación: 1 Bostezo + Al menos 2 parpadeos lentos.
        else if (
            recentSlowBlinks >= 3 || 
            recentYawns >= 2 || 
            (recentYawns >= 1 && recentSlowBlinks >= 2)
        ) {
            
            riskLevel = 'Moderado';
            
            // Apagar alarma fuerte
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
                    // Mensaje explicativo de POR QUÉ suena
                    let razon = "";
                    if (recentYawns >= 2) razon = "Bostezos frecuentes detectados.";
                    else if (recentSlowBlinks >= 3) razon = "Varios parpadeos lentos seguidos.";
                    else razon = "Signos combinados de fatiga.";

                    if (moderateWarningCount >= 3) {
                        warningPopup.className = "warning-popup alert-red active";
                        popupContent.innerHTML = `
                            <h3>🛑 DESCANSO SUGERIDO</h3>
                            <p>${razon}</p>
                            <p>La fatiga es persistente.</p>
                        `;
                    } else {
                        warningPopup.className = "warning-popup alert-orange active";
                        popupContent.innerHTML = `
                            <h3>⚠️ Atención</h3>
                            <p>${razon}</p>
                            <p>Manténgase alerta.</p>
                        `;
                    }
                }

                // Aumentamos el cooldown a 15 segundos para no ser pesados
                moderateAlertCooldown = true;
                setTimeout(() => {
                    if (riskLevel !== 'Alto riesgo') warningPopup.classList.remove('active');
                }, 6000);
                setTimeout(() => moderateAlertCooldown = false, 15000);
            }
            lastModerateTimestamp = now;
        } 

        // --- C. NORMAL / LEVE ---
        else {
            if (totalBlinksLastMinute > 25) riskLevel = 'Leve'; // Subimos umbral de fatiga leve
            else riskLevel = 'Normal';

            if (alarmAudio && !alarmAudio.paused) {
                alarmAudio.pause();
                alarmAudio.currentTime = 0;
            }
            if (!moderateAlertCooldown && riskLevel === 'Normal') {
                 warningPopup.classList.remove('active');
            }
        }

        // --- DIBUJO ---
        if (isDev) {
            drawConnectors(canvasCtx, lm, FACEMESH_TESSELATION, { color: '#00C853', lineWidth: 0.5 });
            drawConnectors(canvasCtx, lm, FACEMESH_RIGHT_EYE, { color: '#FF5722', lineWidth: 1 });
            drawConnectors(canvasCtx, lm, FACEMESH_LEFT_EYE, { color: '#FF5722', lineWidth: 1 });
            drawConnectors(canvasCtx, lm, FACEMESH_LIPS, { color: '#FF4081', lineWidth: 1 });
            canvasCtx.restore();
        }

        // --- ENVÍO DE DATOS ---
        if (now % 10000 < 60) {
            sendDetectionEvent({
                blinkRate: totalBlinksLastMinute,
                ear: smoothedEAR,
                riskLevel,
                yawnDetected: (isYawningNow), 
                totalBlinks: totalBlinksLastMinute,
                totalYawns: yawnCountTotal
            });
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
    if (alarmAudio) {
        alarmAudio.pause();
        alarmAudio.currentTime = 0;
    }
    if (warningPopup) {
        warningPopup.classList.remove('active');
    }
}

async function sendDetectionEvent({ blinkRate, ear, riskLevel, yawnDetected, totalBlinks, totalYawns, immediate = false }) {
    try {
        await fetch(`${BACKEND_URL}/api/detection-event`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                blink_rate: Number(blinkRate.toFixed(2)),
                ear: Number(ear.toFixed(6)),
                risk_level: riskLevel,
                yawn_detected: yawnDetected,
                total_blinks: totalBlinks,
                total_yawns: totalYawns,
                immediate_alert: immediate,
                timestamp: new Date().toISOString()
            })
        });
    } catch (err) {
        console.error('Error enviando evento:', err);
    }
}