// detection.js
// Detección de parpadeos, fatiga, somnolencia y microsueño

const BACKEND_URL = 'https://safe-drive-backend.onrender.com';
const alarmAudio = document.getElementById('alarmSound');
const notifyAudio = document.getElementById('notifySound');
const warningPopup = document.getElementById('warningPopup');

// Variables de control para la notificación preventiva
let moderateAlertCooldown = false;

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
    const MICROSUEÑO_THRESHOLD = 1.5; // segundos
    const FPS = 30;
    const MIN_BLINKS_NORMAL = 20;
    const MIN_CLOSURE_MODERATE = 0.4;

    // FIX: persistencia mínima del riesgo moderado
    const MODERATE_PERSISTENCE_MS = 3000;

    // FIX BOSTEZOS
    const MIN_YAWN_DURATION = 0.8; // segundos

    // FIX MICROFLUCTUACIONES
    const EYE_REOPEN_GRACE_FRAMES = 3;

    // ===============================
    // ESTADO
    // ===============================
    let blinkTimestamps = [];
    let yawnCount = 0;

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

    // FIX BOSTEZOS
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
        if (horizontal === 0) return 0;
        return (vertical1 + vertical2) / (2 * horizontal);
    }

    function calculateMAR_px(landmarks, indices) {
        const [p0, p1, p2, p3, p4, p5] = indices.map(i => toPixel(landmarks[i]));
        const vertical = dist(p2, p4);
        const horizontal = dist(p0, p5);
        if (horizontal === 0) return 0;
        return vertical / horizontal;
    }

    // ===============================
    // ÍNDICES
    // ===============================
    const RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144];
    const LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380];
    const MOUTH_IDX = [78, 308, 13, 14, 311, 402];

    // ===============================
    // MEDIAPIPE
    // ===============================
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
    // RESULTADOS
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

        // ===============================
        // EAR / MAR
        // ===============================
        const rightEAR = calculateEAR_px(lm, RIGHT_EYE_IDX);
        const leftEAR = calculateEAR_px(lm, LEFT_EYE_IDX);
        const earPx = (rightEAR + leftEAR) / 2;
        const mar = calculateMAR_px(lm, MOUTH_IDX);

        const xs = lm.map(p => p.x * canvasElement.width);
        const faceWidthPx = Math.max(...xs) - Math.min(...xs);
        const earRel = faceWidthPx > 0 ? earPx / faceWidthPx : earPx;

        // ===============================
        // CALIBRACIÓN INICIAL
        // ===============================
        if (!initialCalibrationDone) {
            if (earRel > 0) baselineSamples.push(earRel);
            if (baselineSamples.length >= BASELINE_FRAMES_INIT) {
                baselineEMA = median(baselineSamples) || 0.01;
                dynamicEARBaseline = baselineEMA;
                initialCalibrationDone = true;
            }
            // NO RESTAURAR AQUÍ
            return;
        }

        // ===============================
        // SUAVIZADO
        // ===============================
        earHistory.push(earRel);
        if (earHistory.length > SMOOTHING_WINDOW) earHistory.shift();
        mouthHistory.push(mar);
        if (mouthHistory.length > SMOOTHING_WINDOW) mouthHistory.shift();

        const smoothedEAR = movingAverage(earHistory);
        const smoothedMAR = movingAverage(mouthHistory);
        const derivative = smoothedEAR - prevSmoothedEAR;
        prevSmoothedEAR = smoothedEAR;

        // ===============================
        // CALIBRACIÓN DINÁMICA (FIX)
        // ===============================
        if (smoothedEAR > 0 && eyeState === 'open') {
            dynamicEARBaseline = EMA_ALPHA * smoothedEAR + (1 - EMA_ALPHA) * dynamicEARBaseline;
        }

        if (mouthState === 'closed') {
            dynamicMARBaseline = EMA_ALPHA * smoothedMAR + (1 - EMA_ALPHA) * dynamicMARBaseline;
        }

        const EAR_THRESHOLD = dynamicEARBaseline * BASELINE_MULTIPLIER;
        const YAWN_THRESHOLD = dynamicMARBaseline * 1.3;

        // ===============================
        // OJOS (FIX ROBUSTO)
        // ===============================
        const consideredClosed = smoothedEAR < EAR_THRESHOLD || derivative < DERIVATIVE_THRESHOLD;

        if (consideredClosed) {
            closedFrameCounter++;
            reopenGraceCounter = 0;
            if (eyeState === 'open' && closedFrameCounter >= CLOSED_FRAMES_THRESHOLD) {
                eyeState = 'closed';
            }
        } else {
            reopenGraceCounter++;
            if (reopenGraceCounter >= EYE_REOPEN_GRACE_FRAMES) {
                if (eyeState === 'closed') {
                    blinkTimestamps.push(Date.now());
                    eyeState = 'open';
                }
                closedFrameCounter = 0;
                microsleepTriggered = false;
            }
        }

        // ===============================
        // BOSTEZOS (FIX TIEMPO)
        // ===============================
        let yawnDetected = false;
        if (smoothedMAR > YAWN_THRESHOLD) {
            yawnFrameCounter++;
            if (
                yawnFrameCounter / FPS >= MIN_YAWN_DURATION &&
                mouthState === 'closed'
            ) {
                yawnDetected = true;
                yawnCount++;
                mouthState = 'open';
            }
        } else {
            yawnFrameCounter = 0;
            mouthState = 'closed';
        }

        // ===============================
        // PARPADEOS ÚLTIMO MINUTO
        // ===============================
        const now = Date.now();
        blinkTimestamps = blinkTimestamps.filter(ts => ts > now - 60000);
        const totalBlinksLastMinute = blinkTimestamps.length;


        // ===============================
        // NIVEL DE RIESGO + ALERTAS
        // ===============================
        let riskLevel = 'Normal';
        const closureDuration = closedFrameCounter / FPS;

        // --- 1. NIVEL ALTO (ALARMA CRÍTICA) ---
        if (closureDuration >= MICROSUEÑO_THRESHOLD) {
            riskLevel = 'Alto riesgo';
            
            // Ocultar preventiva si estaba activa para dar paso a la crítica
            warningPopup.classList.remove('active');

            if (alarmAudio && alarmAudio.paused) {
                alarmAudio.currentTime = 0;
                alarmAudio.play().catch(e => console.log("Audio play error", e));
            }

            if (!microsleepTriggered) {
                microsleepTriggered = true;
                // ... (tu código de envío de evento) ...
            }
        } 
        
        // --- 2. NIVEL MODERADO (ALERTA PREVENTIVA) ---
        else if (closureDuration >= MIN_CLOSURE_MODERATE || (yawnDetected && mouthState === 'open')) {
            riskLevel = 'Moderado';
            
            // Apagar alarma crítica si sonaba
            if (alarmAudio && !alarmAudio.paused) {
                alarmAudio.pause();
                alarmAudio.currentTime = 0;
            }

            // Lógica del Pop-up y Sonido "Ding" (SOLO UNA VEZ cada cierto tiempo)
            if (!moderateAlertCooldown) {
                // 1. Reproducir sonido
                if (notifyAudio) {
                    notifyAudio.currentTime = 0;
                    notifyAudio.play().catch(e => console.error(e));
                }

                // 2. Mostrar Pop-up
                warningPopup.classList.add('active');

                // 3. Activar Cooldown para no spammear
                moderateAlertCooldown = true;

                // 4. Ocultar Pop-up automáticamente después de 4 segundos
                setTimeout(() => {
                    warningPopup.classList.remove('active');
                }, 4000);

                // 5. Resetear el Cooldown después de 10 segundos
                // (Para que si sigue con sueño, le vuelva a avisar en 10s)
                setTimeout(() => {
                    moderateAlertCooldown = false;
                }, 10000);
            }
            
            lastModerateTimestamp = now;
        } 

        // --- 3. OTROS NIVELES (RESET) ---
        else {
            // Persistencia visual del estado en el texto (sin sonido)
            if (now - lastModerateTimestamp < MODERATE_PERSISTENCE_MS) {
                riskLevel = 'Moderado';
            } else if (totalBlinksLastMinute > 20) {
                riskLevel = 'Leve';
            } else {
                riskLevel = 'Normal';
            }

            // Si el riesgo baja, aseguramos silencio total
            if (alarmAudio && !alarmAudio.paused) {
                alarmAudio.pause();
                alarmAudio.currentTime = 0;
            }
        }

        // ===============================
        // DIBUJO DE MALLA PARA DEVS
        // ===============================
        if (isDev) {
            drawConnectors(canvasCtx, lm, FACEMESH_TESSELATION, { color: '#00C853', lineWidth: 0.5 });
            drawConnectors(canvasCtx, lm, FACEMESH_RIGHT_EYE, { color: '#FF5722', lineWidth: 1 });
            drawConnectors(canvasCtx, lm, FACEMESH_LEFT_EYE, { color: '#FF5722', lineWidth: 1 });
            drawConnectors(canvasCtx, lm, FACEMESH_LIPS, { color: '#FF4081', lineWidth: 1 });
            canvasCtx.restore(); // SOLO AQUÍ
        }

        // ===============================
        // ENVÍO NORMAL (~10s)
        // ===============================
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
            <p>Bostezos: ${yawnCount}</p>
            <p>EAR: ${smoothedEAR.toFixed(6)}</p>
            <p>MAR: ${smoothedMAR.toFixed(3)}</p>
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

    // APAGAR ALARMA AL DETENER
    if (alarmAudio) {
        alarmAudio.pause();
        alarmAudio.currentTime = 0;
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
