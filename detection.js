// detection.js
// Detección de parpadeos, fatiga, somnolencia y microsueño

const BACKEND_URL = 'https://safe-drive-backend.onrender.com';
const alarmAudio = document.getElementById('alarmSound');
const notifyAudio = document.getElementById('notifySound');
const warningPopup = document.getElementById('warningPopup');

// Variables de control para la notificación preventiva
let moderateAlertCooldown = false;
let moderateWarningCount = 0; 
let lastWarningTime = 0; 

export function startDetection({ rol, videoElement, canvasElement, estado, cameraRef }) {
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
    const BASELINE_MULTIPLIER = 0.62;
    const CLOSED_FRAMES_THRESHOLD = 1;
    const DERIVATIVE_THRESHOLD = -0.0025;
    
    // UMBRALES DE TIEMPO
    const MICROSUEÑO_THRESHOLD = 1.5; // segundos (Alarma inmediata)
    const MIN_SLOW_BLINK_DURATION = 0.35; // segundos (Para considerar un parpadeo como "lento")
    const FPS = 30;
    
    const MIN_YAWN_DURATION = 0.8; // segundos
    const EYE_REOPEN_GRACE_FRAMES = 3;

    // ===============================
    // ESTADO
    // ===============================
    let blinkTimestamps = [];
    let yawnCount = 0;
    
    // NUEVO: Historial de parpadeos lentos
    let slowBlinksBuffer = []; // Guardará los timestamps de los parpadeos lentos

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
    // UTILIDADES (One-liners)
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
    // ÍNDICES
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

        // --- CÁLCULOS ---
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
        // 1. ESTADO BOSTEZO (Ahora mismo)
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
        // 2. LÓGICA DE OJOS (CON DETECCIÓN DE PARPADEO LENTO)
        // ===============================
        const consideredClosed = smoothedEAR < EAR_THRESHOLD || derivative < DERIVATIVE_THRESHOLD;

        if (consideredClosed) {
            if (isYawningNow) {
                // Si bosteza, no acumulamos cierre de ojos (evita falsos positivos por entrecerrar)
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
                    // --- AQUÍ EL USUARIO ACABA DE ABRIR LOS OJOS ---
                    // Calculamos cuánto duró el cierre
                    const duration = closedFrameCounter / FPS;
                    
                    // Si duró más de 0.35s pero menos de 1.5s (no fue microsueño, fue parpadeo lento)
                    if (duration > MIN_SLOW_BLINK_DURATION && duration < MICROSUEÑO_THRESHOLD) {
                        slowBlinksBuffer.push(Date.now());
                        console.log(`🐢 Parpadeo Lento detectado: ${duration.toFixed(2)}s`);
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
        let yawnDetected = false;
        if (isYawningNow) {
            yawnFrameCounter++;
            if (yawnFrameCounter / FPS >= MIN_YAWN_DURATION && mouthState === 'closed') {
                yawnDetected = true;
                yawnCount++;
                mouthState = 'open';
            }
        } else {
            if (smoothedMAR < CURRENT_YAWN_THRESHOLD * 0.9) {
                yawnFrameCounter = 0;
                mouthState = 'closed';
            }
        }

        // --- LIMPIEZA DE BUFFERS (Últimos 30 seg para parpadeos lentos) ---
        const now = Date.now();
        blinkTimestamps = blinkTimestamps.filter(ts => ts > now - 60000);
        slowBlinksBuffer = slowBlinksBuffer.filter(ts => ts > now - 30000); // 30 segundos ventana
        
        const totalBlinksLastMinute = blinkTimestamps.length;
        const recentSlowBlinks = slowBlinksBuffer.length;

        // ===============================
        // NIVEL DE RIESGO
        // ===============================
        let riskLevel = 'Normal';
        const closureDuration = closedFrameCounter / FPS; // Duración en tiempo real (si sigue cerrado)
        const popupContent = document.getElementById('popupTextContent');

        // --------------------------------------------------------
        // A. ALTO RIESGO (MICROSUEÑO - PRIORIDAD MÁXIMA E INMEDIATA)
        // --------------------------------------------------------
        if (closureDuration >= MICROSUEÑO_THRESHOLD) {
            riskLevel = 'Alto riesgo';
            
            warningPopup.className = "warning-popup alert-red active";
            if (popupContent) {
                popupContent.innerHTML = `
                    <h3>🚨 ¡PELIGRO! DESPIERTE 🚨</h3>
                    <p>Ojos cerrados por tiempo prolongado.</p>
                    <p>Mantenga la vista en el camino.</p>
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
                    yawnDetected,
                    totalBlinks: totalBlinksLastMinute,
                    totalYawns: yawnCount,
                    immediate: true
                });
            }
        } 
        
        // --------------------------------------------------------
        // B. MODERADO (SOMNOLENCIA - ACUMULATIVA O BOSTEZO)
        // --------------------------------------------------------
        // Condición: Bostezo Activo O (Al menos 2 parpadeos lentos en 30s)
        else if ((yawnDetected && mouthState === 'open') || recentSlowBlinks >= 2) {
            
            riskLevel = 'Moderado';
            
            // Apagar alarma crítica
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
                    if (moderateWarningCount >= 3) {
                        warningPopup.className = "warning-popup alert-red active";
                        popupContent.innerHTML = `
                            <h3>🛑 ALTO RIESGO DE SUEÑO</h3>
                            <p>Patrones de sueño persistentes.</p>
                            <p>Busque un área de descanso segura.</p>
                        `;
                    } else {
                        // Mensaje diferente según la causa
                        let mensaje = recentSlowBlinks >= 2 ? "Parpadeo lento detectado." : "Bostezos detectados.";
                        warningPopup.className = "warning-popup alert-orange active";
                        popupContent.innerHTML = `
                            <h3>⚠️ Signos de cansancio</h3>
                            <p>${mensaje}</p>
                            <p>Ventile el vehículo o hidrátese.</p>
                        `;
                    }
                }

                moderateAlertCooldown = true;
                setTimeout(() => {
                    if (riskLevel !== 'Alto riesgo') warningPopup.classList.remove('active');
                }, 5000);
                setTimeout(() => moderateAlertCooldown = false, 10000);
            }
            lastModerateTimestamp = now;
        } 

        // --------------------------------------------------------
        // C. NIVELES BAJOS
        // --------------------------------------------------------
        else {
            // Riesgo Leve si hay mucho parpadeo rápido
            if (totalBlinksLastMinute > 20) riskLevel = 'Leve';
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
                yawnDetected,
                totalBlinks: totalBlinksLastMinute,
                totalYawns: yawnCount
            });
        }

        estado.innerHTML = `
            <p>Parpadeos último minuto: ${totalBlinksLastMinute}</p>
            <p>Parpadeos Lentos (30s): ${recentSlowBlinks}</p>
            <p>Bostezos: ${yawnCount}</p>
            <p>Riesgo: ${riskLevel}</p>
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

async function sendDetectionEvent({
    blinkRate,
    ear,
    riskLevel,
    yawnDetected,
    totalBlinks,
    totalYawns,
    immediate = false
}) {
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