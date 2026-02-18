// packages/face/src/index.ts
import WebcamClient from './Core/WebcamClient.ts';
import WebEyeTrackProxy from './WebEyeTrack/WebEyeTrackProxy.ts';
import { drawMesh } from './drawMesh.ts';

// --- 1. Setup Biometrics ---
const client = new WebcamClient('webcam');
const proxy = new WebEyeTrackProxy({
    maxPoints: 3,
});
await client.startWebcam(async (frame, context) => {
    void proxy.processFrame(frame, context);
});

// --- 2. Visuals Setup ---
const cursor = document.createElement('div');
Object.assign(cursor.style, {
    position: 'fixed',
    width: '20px', height: '20px',
    borderRadius: '50%', backgroundColor: 'red',
    pointerEvents: 'none', zIndex: '9999',
    transform: 'translate(-50%, -50%)',
    transition: 'background-color 0.1s'
});
document.body.appendChild(cursor);

const canvasElement = document.getElementById('output_canvas') as HTMLCanvasElement;
const videoElement = document.getElementById('webcam') as HTMLVideoElement;
let isPaused = false;

window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
        e.preventDefault(); // Prevent page scrolling
        isPaused = !isPaused;

        if (isPaused) {
            console.log("⏸ Paused");
            videoElement.pause(); // Freezes the raw camera feed
        } else {
            console.log("▶ Resumed");
            videoElement.play();  // Resumes camera feed
        }
    }
});

// OPTIMIZATION: Handle resize efficiently
let resizeTimeout: any;
const handleResize = () => {
    // 1. Existing canvas resize logic
    if (videoElement.videoWidth > 0) {
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
    }

    // 2. NEW: Reset Calibration on significant resize
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        console.log("📏 Window resized - Resetting calibration buffers to maintain accuracy.");
        // We need to expose this method in proxy first (see below)
        proxy.resetAllBuffers();
    }, 500); // Debounce to avoid spamming during drag
};

videoElement.addEventListener('resize', handleResize);
videoElement.addEventListener('loadedmetadata', handleResize);

// --- DEBUG DASHBOARD ---
const debugContainer = document.createElement('div');
Object.assign(debugContainer.style, {
    position: 'fixed',
    top: '10px',
    left: '10px',
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    color: '#0f0',
    padding: '10px',
    borderRadius: '4px',
    fontFamily: 'monospace',
    fontSize: '12px',
    zIndex: '10000',
    pointerEvents: 'none'
});
debugContainer.innerHTML = `
    <strong>Debug Metrics</strong>
    <ul style="padding-left: 15px; margin: 5px 0 0 0;">
        <li>FPS: <span id="dbg-fps">0</span></li>
        <li>Process (ms): <span id="dbg-ms">0</span></li>
        <li>Gaze Jitter (SD): <span id="dbg-jitter" style="color:cyan">0.000</span></li>
        <li>Face Dist WET (Z): <span id="dbg-z">0</span> cm</li>
        <li>Face Dist LM (Z): <span id="dbg-zlm">0</span> cm</li>
        <li>Face Width: <span id="dbg-width">0</span> cm</li>
        <li>Confidence: <span id="dbg-conf">0</span></li>
    </ul>
`;
document.body.appendChild(debugContainer);

// Metrics State
let frameCount = 0;
let lastFpsTime = performance.now();
const gazeHistoryX: number[] = [];
const gazeHistoryY: number[] = [];
const HISTORY_SIZE = 30; // 1 second @ 30fps

function updateDebugMetrics(result: any) {
    // 1. FPS
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
        document.getElementById('dbg-fps')!.innerText = frameCount.toString();
        frameCount = 0;
        lastFpsTime = now;
    }

    // 2. Processing Time
    if (result.durations) {
        document.getElementById('dbg-ms')!.innerText = result.durations.total.toFixed(1);
    }

    // 3. Jitter (Standard Deviation)
    if (result.normPog) {
        gazeHistoryX.push(result.normPog[0]);
        gazeHistoryY.push(result.normPog[1]);
        if (gazeHistoryX.length > HISTORY_SIZE) {
            gazeHistoryX.shift();
            gazeHistoryY.shift();
        }

        if (gazeHistoryX.length > 2) {
            // Compute SD
            const meanX = gazeHistoryX.reduce((a, b) => a + b) / gazeHistoryX.length;
            const variance = gazeHistoryX.reduce((a, b) => a + Math.pow(b - meanX, 2), 0) / gazeHistoryX.length;
            const sd = Math.sqrt(variance);

            const el = document.getElementById('dbg-jitter')!;
            el.innerText = sd.toFixed(4);
            // Visual Warning if unstable
            el.style.color = sd > 0.005 ? 'red' : 'cyan';
        }
    }

    // 4. Geometry
    if (result.faceOrigin3D) {
        // This is the calculated origin from your WebEyeTrack utils
        document.getElementById('dbg-z')!.innerText = result.faceOrigin3D[2].toFixed(1);
    }
    if (result.faceRt && result.faceRt.data.length > 0) {
        // Matrix is 4x4 flat array. Index 14 is Z-translation (Depth in cm)
        const rawDistanceCm = result.faceRt.data[14];
        document.getElementById('dbg-zlm')!.innerText = rawDistanceCm.toFixed(1) + " cm";
    }

    // Note: You might need to expose faceWidthCm in result to log it here,
    // or infer it from faceBlendshapes if available.
}

// --- 3. DEBUG: Eye Patch Visualizer ---
// This canvas shows exactly what the Neural Network sees
const debugCanvas = document.createElement('canvas');
Object.assign(debugCanvas.style, {
    position: 'fixed',
    top: '10px',
    right: '10px',
    width: '256px', // Scaled up for visibility
    height: '64px',
    border: '2px solid cyan',
    backgroundColor: '#000',
    zIndex: '10000',
    imageRendering: 'pixelated' // Keep it raw to spot blurring
});
document.body.appendChild(debugCanvas);
const debugCtx = debugCanvas.getContext('2d')!;

// Label for the debug view
const debugLabel = document.createElement('div');
debugLabel.innerText = "Model Input (Eye Patch)";
Object.assign(debugLabel.style, {
    position: 'fixed',
    top: '78px',
    right: '10px',
    color: 'cyan',
    fontFamily: 'monospace',
    fontSize: '12px',
    zIndex: '10000'
});
document.body.appendChild(debugLabel);


// --- 4. Signal Graph Setup ---
class RealtimeGraph {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private data: number[] = [];
    private maxDataPoints = 300;
    private bpmDisplay: HTMLDivElement;

    constructor(color: string = '#00ff00') {
        this.canvas = document.createElement('canvas');
        this.canvas.width = 300;
        this.canvas.height = 100;
        Object.assign(this.canvas.style, {
            position: 'fixed', bottom: '10px', right: '10px',
            backgroundColor: '#111', border: '1px solid #333',
            zIndex: '9998'
        });
        document.body.appendChild(this.canvas);

        this.bpmDisplay = document.createElement('div');
        Object.assign(this.bpmDisplay.style, {
            position: 'fixed', bottom: '115px', right: '10px',
            color: color, fontFamily: 'monospace', fontSize: '24px',
            fontWeight: 'bold', zIndex: '9998'
        });
        this.bpmDisplay.innerText = "-- BPM";
        document.body.appendChild(this.bpmDisplay);

        this.ctx = this.canvas.getContext('2d')!;
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2;
    }

    update(signal: number, bpm: number) {
        if (bpm > 0) this.bpmDisplay.innerText = `${Math.round(bpm)} BPM`;
        else this.bpmDisplay.innerText = "Processing...";
        this.data.push(signal);
        if (this.data.length > this.maxDataPoints) this.data.shift();
        this.draw();
    }

    private draw() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        this.ctx.clearRect(0, 0, w, h);
        if (this.data.length < 2) return;
        let min = Math.min(...this.data);
        let max = Math.max(...this.data);
        let range = max - min || 1;
        min -= range * 0.1; max += range * 0.1; range = max - min;
        this.ctx.beginPath();
        this.data.forEach((val, i) => {
            const x = (i / (this.maxDataPoints - 1)) * w;
            const y = h - ((val - min) / range) * h;
            if (i === 0) this.ctx.moveTo(x, y);
            else this.ctx.lineTo(x, y);
        });
        this.ctx.stroke();
    }
}

const foreheadGraph = new RealtimeGraph('#00ff00');

// --- 5. Main Loop ---
proxy.onGazeResults = (gazeResult: any) => {
    if (isPaused) return;
    if (!gazeResult || !gazeResult.normPog) return;

    // A. Update Cursor
    const x = (gazeResult.normPog[0] + 0.5) * (document.documentElement.clientWidth || window.innerWidth);
    const y = (gazeResult.normPog[1] + 0.5) * (document.documentElement.clientHeight || window.innerHeight);
    cursor.style.left = `${x}px`;
    cursor.style.top = `${y}px`;
    cursor.style.backgroundColor = gazeResult.gazeState === 'closed' ? 'gray' : 'red';

    // B. DEBUG: Draw Eye Patch
    if (gazeResult.eyePatch) {
        // Ensure debug canvas matches incoming patch size
        if (debugCanvas.width !== gazeResult.eyePatch.width || debugCanvas.height !== gazeResult.eyePatch.height) {
            debugCanvas.width = gazeResult.eyePatch.width;
            debugCanvas.height = gazeResult.eyePatch.height;
        }
        debugCtx.putImageData(gazeResult.eyePatch, 0, 0);
    }

    // C. Draw Mesh
    drawMesh(gazeResult, canvasElement);

    // D. Update Graph
    if (gazeResult.vitals) {
        foreheadGraph.update(gazeResult.vitals.wave, gazeResult.vitals.bpm);
    }

    // Jitter
    updateDebugMetrics(gazeResult);
}
