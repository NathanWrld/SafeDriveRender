// detection.js
// Lógica de detección de parpadeos, fatiga y bostezos con alerta inmediata de microsueño

const BACKEND_URL = 'https://safe-drive-backend.onrender.com';

// ===============================
// FUNCIÓN PRINCIPAL
// ===============================
export function startDetection({
    rol,
    videoElement,
    canvasElement,
    estado,
    cameraRef
}) {
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
    const EMA_ALPHA_DYNAMIC = 0.01; // calibración dinámica
    const BASELINE_MULTIPLIER = 0.62;
    const CLOSED_FRAMES_THRESHOLD = 1;
    const MIN_TIME_BETWEEN_BLINKS = 150;
    const DERIVATIVE_THRESHOLD = -0.0025;
    const MICROSUEÑO_THRESHOLD = 1.5; // segundos
    const INITIAL_YAWN_THRESHOLD = 0.65;

    // ===============================
    // ESTADO
    // ===============================
    let blinkCount = 0;
    let yawnCount = 0;
    let blinkStartTime = Date.now();
    let lastBlinkTime = 0;

    let earHistory = [];
    let mouthHistory = [];
    let baselineSamples = [];
    let baselineEMA = null;
    let initialCalibrationDone = false;

    let eyeState = 'open';
    let mouthState = 'closed';
    let closedFrameCounter = 0;
    let prevSmoothedEAR = 0;
    let closureDurations = [];

    let dynamicEARBaseline = null;
    let dynamicMARBaseline = INITIAL_YAWN_THRESHOLD;

    // ===============================
    // UTILIDADES
    // ===============================
    function toPixel(l) {
        return { x: l.x * canvasElement.width, y: l.y * canvasElement.height };
    }

    function dist(a, b) {
        return Math.hypot(a.x - b.x, a.y - b.y);
    }

    function movingAverage(arr) {
        if (!arr.length) return 0;
        return arr.reduce((s, v) => s + v, 0) / arr.length;
    }

    function median(arr) {
        if (!arr.length) return 0;
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
    // ÍNDICES DE OJOS Y BOCA
    // ===============================
    const RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144];
    const LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380];
    const MOUTH_IDX     = [78, 308, 13, 14, 311, 402];

    // ===============================
    // MEDIAPIPE FACE MESH
    // ===============================
    const faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
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

        if (isDev) {
            drawConnectors(canvasCtx, lm, FACEMESH_TESSELATION, { color: '#00C853', lineWidth: 0.5 });
            drawConnectors(canvasCtx, lm, FACEMESH_RIGHT_EYE, { color:'#FF5722', lineWidth:1 });
            drawConnectors(canvasCtx, lm, FACEMESH_LEFT_EYE, { color:'#FF5722', lineWidth:1 });
            drawConnectors(canvasCtx, lm, FACEMESH_LIPS, { color:'#FF4081', lineWidth:1 });
        }

        // ===============================
        // EAR Y MAR
        // ===============================
        const rightEAR = calculateEAR_px(lm, RIGHT_EYE_IDX);
        const leftEAR  = calculateEAR_px(lm, LEFT_EYE_IDX);
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
            const remaining = BASELINE_FRAMES_INIT - baselineSamples.length;
            estado.innerHTML = `<p>✅ Rostro detectado — calibrando...</p>
                                <p>Frames restantes: ${Math.max(0, remaining)}</p>`;
            if (baselineSamples.length >= BASELINE_FRAMES_INIT) {
                baselineEMA = median(baselineSamples) || 0.01;
                dynamicEARBaseline = baselineEMA;
                initialCalibrationDone = true;
            }
            if (isDev) canvasCtx.restore();
            return;
        }

        // ===============================
        // SUAVIZADO Y DERIVADA
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
        // CALIBRACIÓN DINÁMICA
        // ===============================
        if (smoothedEAR > 0) {
            dynamicEARBaseline = EMA_ALPHA_DYNAMIC * smoothedEAR + (1 - EMA_ALPHA_DYNAMIC) * dynamicEARBaseline;
        }

        if (mouthState === 'closed') {
            dynamicMARBaseline = EMA_ALPHA_DYNAMIC * smoothedMAR + (1 - EMA_ALPHA_DYNAMIC) * dynamicMARBaseline;
        }

        const EAR_THRESHOLD_DYNAMIC = dynamicEARBaseline * BASELINE_MULTIPLIER;
        const YAWN_THRESHOLD_DYNAMIC = dynamicMARBaseline * 1.3;

        // ===============================
        // DETECCIÓN DE OJOS
        // ===============================
        const consideredClosed = smoothedEAR < EAR_THRESHOLD_DYNAMIC || derivative < DERIVATIVE_THRESHOLD;

        if (consideredClosed) {
            closedFrameCounter++;
            if (eyeState === 'open' && closedFrameCounter >= CLOSED_FRAMES_THRESHOLD) {
                eyeState = 'closed';
            }
        } else {
            if (eyeState === 'closed') {
                const closureDuration = closedFrameCounter / 30; // ~30fps
                closureDurations.push(closureDuration);
                eyeState = 'open';

                if (Date.now() - lastBlinkTime > MIN_TIME_BETWEEN_BLINKS) {
                    blinkCount++;
                    lastBlinkTime = Date.now();
                }

                closedFrameCounter = 0;
            }
        }

        // ===============================
        // DETECCIÓN DE BOSTEZO
        // ===============================
        let yawnDetected = false;
        if (smoothedMAR > YAWN_THRESHOLD_DYNAMIC && mouthState === 'closed') {
            yawnDetected = true;
            yawnCount++;
            mouthState = 'open';
        } else if (smoothedMAR <= YAWN_THRESHOLD_DYNAMIC) {
            mouthState = 'closed';
        }

        // ===============================
        // BPM Y RESET POR MINUTO
        // ===============================
        const elapsedMinutes = (Date.now() - blinkStartTime) / 60000;
        const bpm = blinkCount / (elapsedMinutes || 1);

        if (elapsedMinutes >= 1) {
            blinkCount = 0;
            yawnCount = 0;
            closureDurations = [];
            blinkStartTime = Date.now();
        }

        // ===============================
        // NIVEL DE RIESGO
        // ===============================
        const avgClosure = closureDurations.length ? movingAverage(closureDurations) : 0;
        const riskLevel = getRiskLevel(bpm, avgClosure);

        // ===============================
        // ALERTA INMEDIATA MICROSUEÑO
        // ===============================
        if (avgClosure >= MICROSUEÑO_THRESHOLD) {
            sendDetectionEvent({
                blinkRate: bpm,
                ear: smoothedEAR,
                riskLevel: 'Alto',
                yawnDetected,
                totalBlinks: blinkCount,
                totalYawns: yawnCount,
                immediate: true
            });
        }

        // ===============================
        // ENVÍO AL BACKEND (~10s)
        // ===============================
        const now = Date.now();
        if (now % 10000 < 60) {
            sendDetectionEvent({
                blinkRate: bpm,
                ear: smoothedEAR,
                riskLevel,
                yawnDetected,
                totalBlinks: blinkCount,
                totalYawns: yawnCount
            });
        }

        estado.innerHTML = `
            <p>✅ Rostro detectado</p>
            <p>Parpadeos/min: ${bpm.toFixed(1)}</p>
            <p>Total parpadeos: ${blinkCount}</p>
            <p>Bostezos: ${yawnCount}</p>
            <p>EAR: ${smoothedEAR.toFixed(6)}</p>
            <p>MAR: ${smoothedMAR.toFixed(3)}</p>
            <p>Riesgo: ${riskLevel}</p>
        `;

        if (isDev) canvasCtx.restore();
    });

    // ===============================
    // CÁMARA
    // ===============================
    cameraRef.current = new Camera(videoElement, {
        onFrame: async () => await faceMesh.send({ image: videoElement }),
        width: 480,
        height: 360
    });
    cameraRef.current.start();
}

// ===============================
// DETENER DETECCIÓN
// ===============================
export function stopDetection(cameraRef) {
    if (cameraRef.current) {
        cameraRef.current.stop();
        cameraRef.current = null;
    }
}

// ===============================
// BACKEND
// ===============================
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

// ===============================
// RIESGO
// ===============================
function getRiskLevel(blinksPerMinute, avgClosureTime) {
    if (avgClosureTime >= 1.5) return 'Alto'; // microsueño
    if (blinksPerMinute <= 20 && avgClosureTime < 0.25) return 'Normal';
    if (blinksPerMinute > 20 && avgClosureTime < 0.25) return 'Leve';
    if (blinksPerMinute > 20 && avgClosureTime >= 0.25 && avgClosureTime < 1) return 'Moderado';
    return 'Moderado'; // caso intermedio
}
