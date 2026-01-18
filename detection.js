// detection.js
// Lógica de detección de parpadeos y fatiga (modularizada)

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
    const BASELINE_MULTIPLIER = 0.62;
    const CLOSED_FRAMES_THRESHOLD = 1;
    const MIN_TIME_BETWEEN_BLINKS = 150;
    const DERIVATIVE_THRESHOLD = -0.0025;

    // ===============================
    // ESTADO
    // ===============================
    let blinkCount = 0;
    let blinkStartTime = Date.now();
    let lastBlinkTime = 0;

    let earHistory = [];
    let baselineSamples = [];
    let baselineEMA = null;
    let initialCalibrationDone = false;

    let eyeState = 'open';
    let closedFrameCounter = 0;
    let prevSmoothedEAR = 0;

    // ===============================
    // UTILIDADES
    // ===============================
    function toPixel(l) {
        return {
            x: l.x * canvasElement.width,
            y: l.y * canvasElement.height
        };
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
        return a.length % 2 === 0
            ? (a[m - 1] + a[m]) / 2
            : a[m];
    }

    function calculateEAR_px(landmarks, indices) {
        const [p0, p1, p2, p3, p4, p5] =
            indices.map(i => toPixel(landmarks[i]));

        const vertical1 = dist(p1, p5);
        const vertical2 = dist(p2, p4);
        const horizontal = dist(p0, p3);

        if (horizontal === 0) return 0;
        return (vertical1 + vertical2) / (2 * horizontal);
    }

    // ===============================
    // ÍNDICES DE OJOS
    // ===============================
    const RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144];
    const LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380];

    // ===============================
    // MEDIAPIPE FACE MESH
    // ===============================
    const faceMesh = new FaceMesh({
        locateFile: (file) =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
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
            canvasCtx.drawImage(
                results.image,
                0,
                0,
                canvasElement.width,
                canvasElement.height
            );
        }

        if (!results.multiFaceLandmarks?.length) {
            estado.innerHTML = `<p>❌ No se detecta rostro</p>`;
            if (isDev) canvasCtx.restore();
            return;
        }

        const lm = results.multiFaceLandmarks[0];

        if (isDev) {
            drawConnectors(canvasCtx, lm, FACEMESH_TESSELATION, {
                color: '#00C853',
                lineWidth: 0.5
            });
            drawConnectors(canvasCtx, lm, FACEMESH_RIGHT_EYE, {
                color: '#FF5722',
                lineWidth: 1
            });
            drawConnectors(canvasCtx, lm, FACEMESH_LEFT_EYE, {
                color: '#FF5722',
                lineWidth: 1
            });
        }

        // ===============================
        // EAR NORMALIZADO
        // ===============================
        const rightEAR = calculateEAR_px(lm, RIGHT_EYE_IDX);
        const leftEAR  = calculateEAR_px(lm, LEFT_EYE_IDX);
        const earPx = (rightEAR + leftEAR) / 2;

        const xs = lm.map(p => p.x * canvasElement.width);
        const faceWidthPx = Math.max(...xs) - Math.min(...xs);
        const earRel = faceWidthPx > 0 ? earPx / faceWidthPx : earPx;

        // ===============================
        // CALIBRACIÓN INICIAL
        // ===============================
        if (!initialCalibrationDone) {
            if (earRel > 0) baselineSamples.push(earRel);

            const remaining =
                BASELINE_FRAMES_INIT - baselineSamples.length;

            estado.innerHTML = `
                <p>✅ Rostro detectado — calibrando...</p>
                <p>Frames restantes: ${Math.max(0, remaining)}</p>
            `;

            if (baselineSamples.length >= BASELINE_FRAMES_INIT) {
                baselineEMA = median(baselineSamples) || 0.01;
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

        const smoothedEAR = movingAverage(earHistory);
        const derivative = smoothedEAR - prevSmoothedEAR;
        prevSmoothedEAR = smoothedEAR;

        baselineEMA =
            EMA_ALPHA * smoothedEAR +
            (1 - EMA_ALPHA) * baselineEMA;

        const EAR_THRESHOLD = baselineEMA * BASELINE_MULTIPLIER;
        const rapidDrop = derivative < DERIVATIVE_THRESHOLD;
        const consideredClosed =
            smoothedEAR < EAR_THRESHOLD || rapidDrop;

        const now = Date.now();

        if (consideredClosed) {
            closedFrameCounter++;
            if (
                eyeState === 'open' &&
                closedFrameCounter >= CLOSED_FRAMES_THRESHOLD
            ) {
                eyeState = 'closed';
            }
        } else {
            if (eyeState === 'closed') {
                if (now - lastBlinkTime > MIN_TIME_BETWEEN_BLINKS) {
                    blinkCount++;
                    lastBlinkTime = now;
                }
                eyeState = 'open';
            }
            closedFrameCounter = 0;
        }

        // ===============================
        // BPM
        // ===============================
        const elapsedMinutes =
            (now - blinkStartTime) / 60000;

        if (elapsedMinutes >= 1) {
            blinkCount = 0;
            blinkStartTime = now;
        }

        const bpm =
            blinkCount / (elapsedMinutes || 1);

        const riskLevel = getRiskLevel(bpm);

        // ===============================
        // ENVÍO AL BACKEND (cada ~10s)
        // ===============================
        if (now % 10000 < 60) {
            sendDetectionEvent({
                blinkRate: bpm,
                ear: smoothedEAR,
                riskLevel
            });
        }

        estado.innerHTML = `
            <p>✅ Rostro detectado</p>
            <p>Parpadeos/min: ${bpm.toFixed(1)}</p>
            <p>EAR: ${smoothedEAR.toFixed(6)}</p>
            <p>Riesgo: ${riskLevel}</p>
        `;

        if (isDev) canvasCtx.restore();
    });

    // ===============================
    // CÁMARA
    // ===============================
    cameraRef.current = new Camera(videoElement, {
        onFrame: async () => {
            await faceMesh.send({ image: videoElement });
        },
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
async function sendDetectionEvent({ blinkRate, ear, riskLevel }) {
    try {
        await fetch(`${BACKEND_URL}/api/detection-event`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                blink_rate: Number(blinkRate.toFixed(2)),
                ear: Number(ear.toFixed(6)),
                risk_level: riskLevel,
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
function getRiskLevel(bpm) {
    if (bpm < 10) return 'Normal';
    if (bpm < 15) return 'Leve';
    if (bpm < 20) return 'Moderado';
    return 'Alto';
}
