// detection.js
// Detección de parpadeos, fatiga, somnolencia y microsueño

const BACKEND_URL = 'https://safe-drive-backend.onrender.com';

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
    const MIN_CLOSURE_MODERATE = 0.25;

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
    let prevSmoothedEAR = 0;

    let dynamicEARBaseline = null;
    let dynamicMARBaseline = 0.65;

    // 🔴 NUEVO (microsueño real)
    let eyeClosedStartTime = null;
    let microsleepActive = false;

    // ===============================
    // UTILIDADES
    // ===============================
    function toPixel(l){ return { x:l.x*canvasElement.width, y:l.y*canvasElement.height }; }
    function dist(a,b){ return Math.hypot(a.x-b.x, a.y-b.y); }
    function movingAverage(arr){ return arr.length ? arr.reduce((s,v)=>s+v,0)/arr.length : 0; }
    function median(arr){
        const a=[...arr].sort((x,y)=>x-y);
        const m=Math.floor(a.length/2);
        return a.length%2===0 ? (a[m-1]+a[m])/2 : a[m];
    }

    function calculateEAR_px(landmarks, idx){
        const [p0,p1,p2,p3,p4,p5] = idx.map(i=>toPixel(landmarks[i]));
        const v1 = dist(p1,p5);
        const v2 = dist(p2,p4);
        const h = dist(p0,p3);
        return h===0 ? 0 : (v1+v2)/(2*h);
    }

    function calculateMAR_px(landmarks, idx){
        const [p0,p1,p2,p3,p4,p5] = idx.map(i=>toPixel(landmarks[i]));
        const v = dist(p2,p4);
        const h = dist(p0,p5);
        return h===0 ? 0 : v/h;
    }

    // ===============================
    // ÍNDICES
    // ===============================
    const RIGHT_EYE_IDX=[33,160,158,133,153,144];
    const LEFT_EYE_IDX=[362,385,387,263,373,380];
    const MOUTH_IDX=[78,308,13,14,311,402];

    // ===============================
    // MEDIAPIPE
    // ===============================
    const faceMesh=new FaceMesh({
        locateFile:f=>`https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`
    });

    faceMesh.setOptions({
        maxNumFaces:1,
        refineLandmarks:true,
        minDetectionConfidence:0.55,
        minTrackingConfidence:0.55
    });

    // ===============================
    // RESULTADOS
    // ===============================
    faceMesh.onResults(results=>{
        if(!results.image) return;

        if(isDev){
            canvasElement.width=results.image.width;
            canvasElement.height=results.image.height;
            canvasCtx.save();
            canvasCtx.clearRect(0,0,canvasElement.width,canvasElement.height);
            canvasCtx.drawImage(results.image,0,0,canvasElement.width,canvasElement.height);
        }

        if(!results.multiFaceLandmarks?.length){
            estado.innerHTML=`<p>❌ No se detecta rostro</p>`;
            if(isDev) canvasCtx.restore();
            return;
        }

        const lm=results.multiFaceLandmarks[0];

        // ===============================
        // EAR / MAR
        // ===============================
        const earPx=(calculateEAR_px(lm,RIGHT_EYE_IDX)+calculateEAR_px(lm,LEFT_EYE_IDX))/2;
        const mar=calculateMAR_px(lm,MOUTH_IDX);

        const xs=lm.map(p=>p.x*canvasElement.width);
        const faceWidthPx=Math.max(...xs)-Math.min(...xs);
        const earRel=faceWidthPx>0?earPx/faceWidthPx:earPx;

        // ===============================
        // CALIBRACIÓN INICIAL
        // ===============================
        if(!initialCalibrationDone){
            if(earRel>0) baselineSamples.push(earRel);
            if(baselineSamples.length>=BASELINE_FRAMES_INIT){
                baselineEMA=median(baselineSamples)||0.01;
                dynamicEARBaseline=baselineEMA;
                initialCalibrationDone=true;
            }
            if(isDev) canvasCtx.restore();
            return;
        }

        // ===============================
        // SUAVIZADO
        // ===============================
        earHistory.push(earRel); if(earHistory.length>SMOOTHING_WINDOW) earHistory.shift();
        mouthHistory.push(mar); if(mouthHistory.length>SMOOTHING_WINDOW) mouthHistory.shift();

        const smoothedEAR=movingAverage(earHistory);
        const smoothedMAR=movingAverage(mouthHistory);
        const derivative=smoothedEAR-prevSmoothedEAR;
        prevSmoothedEAR=smoothedEAR;

        dynamicEARBaseline=EMA_ALPHA*smoothedEAR+(1-EMA_ALPHA)*dynamicEARBaseline;
        if(mouthState==='closed') dynamicMARBaseline=EMA_ALPHA*smoothedMAR+(1-EMA_ALPHA)*dynamicMARBaseline;

        const EAR_THRESHOLD=dynamicEARBaseline*BASELINE_MULTIPLIER;
        const YAWN_THRESHOLD=dynamicMARBaseline*1.3;

        const consideredClosed = smoothedEAR<EAR_THRESHOLD || derivative<DERIVATIVE_THRESHOLD;
        const now=Date.now();

        // ===============================
        // OJOS (MICROSUEÑO REAL)
        // ===============================
        if(consideredClosed){
            closedFrameCounter++;
            if(!eyeClosedStartTime) eyeClosedStartTime=now;

            const closedTime=(now-eyeClosedStartTime)/1000;
            if(closedTime>=MICROSUEÑO_THRESHOLD && !microsleepActive){
                microsleepActive=true;
                sendDetectionEvent({
                    blinkRate:blinkTimestamps.length,
                    ear:smoothedEAR,
                    riskLevel:'Alto',
                    yawnDetected:false,
                    totalBlinks:blinkTimestamps.length,
                    totalYawns:yawnCount,
                    immediate:true
                });
            }

            eyeState='closed';
        } else {
            if(eyeState==='closed'){
                blinkTimestamps.push(now);
            }
            eyeState='open';
            closedFrameCounter=0;
            eyeClosedStartTime=null;
            microsleepActive=false;
        }

        // ===============================
        // BOSTEZO
        // ===============================
        if(smoothedMAR>YAWN_THRESHOLD && mouthState==='closed'){
            yawnCount++;
            mouthState='open';
        } else if(smoothedMAR<=YAWN_THRESHOLD){
            mouthState='closed';
        }

        // ===============================
        // PARPADEOS ÚLTIMO MINUTO
        // ===============================
        blinkTimestamps=blinkTimestamps.filter(t=>now-t<60000);
        const totalBlinksLastMinute=blinkTimestamps.length;

        // ===============================
        // RIESGO
        // ===============================
        let riskLevel='Normal';
        if(microsleepActive){
            riskLevel='Alto';
        } else if(totalBlinksLastMinute>MIN_BLINKS_NORMAL && closedFrameCounter/FPS>=MIN_CLOSURE_MODERATE){
            riskLevel='Moderado';
        } else if(totalBlinksLastMinute>MIN_BLINKS_NORMAL){
            riskLevel='Leve';
        }

        estado.innerHTML=`
            <p>Parpadeos último minuto: ${totalBlinksLastMinute}</p>
            <p>Bostezos: ${yawnCount}</p>
            <p>Riesgo: <b>${riskLevel}</b></p>
        `;

        if(isDev) canvasCtx.restore();
    });

    cameraRef.current=new Camera(videoElement,{
        onFrame:async()=>await faceMesh.send({image:videoElement}),
        width:480,
        height:360
    });
    cameraRef.current.start();
}

export function stopDetection(cameraRef){
    if(cameraRef.current){
        cameraRef.current.stop();
        cameraRef.current=null;
    }
}

async function sendDetectionEvent(data){
    try{
        await fetch(`${BACKEND_URL}/api/detection-event`,{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({...data, timestamp:new Date().toISOString()})
        });
    }catch(e){
        console.error(e);
    }
}
