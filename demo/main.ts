import type {BiometricsResult} from "../src";
import { BiometricsClient } from '../src';
import { drawPulseMask } from "./drawing/drawPulseMask";
import { PulseGraph } from './drawing/PulseGraph';
import { AccuracyTracker } from './drawing/AccuracyTracker';
import { TraceDisplay } from './drawing/TraceDisplay';

// ─── Pulse Graph ────────────────────────────────────────────────────────────
const cursor = document.getElementById('cursor') as HTMLDivElement;
const video = document.getElementById('webcam') as HTMLVideoElement;

const pulseGraphCanvas = document.getElementById('heartRate') as HTMLCanvasElement;
const pulseGraph = new PulseGraph(pulseGraphCanvas);
const bpmDisplay = document.getElementById('BPM') as HTMLParagraphElement;

const eyePatchCanvas = document.getElementById('eyePatch') as HTMLCanvasElement;
const eyePatchCtx = eyePatchCanvas.getContext('2d');

const webcamOverlayCanvas = document.getElementById('webcamCanvas') as HTMLCanvasElement;
const webcamOverlayCtx = webcamOverlayCanvas.getContext('2d')!;

const accuracyText = document.getElementById('accuracy') as HTMLDivElement;

let displayedFFT = '';
let displayedPeak = '';

const traceDisplay = new TraceDisplay(
    document.getElementById('stats') as HTMLDivElement,
    60 // last 60 frames
);

// Track output canvas size to avoid unnecessary resets
let lastOutputW = 0;
let lastOutputH = 0;

function syncWebcamCanvas(): void {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (lastOutputW !== vw || lastOutputH !== vh) {
        webcamOverlayCanvas.width = vw;
        webcamOverlayCanvas.height = vh;
        lastOutputW = vw;
        lastOutputH = vh;
    }
}
// ─── Results callback ───────────────────────────────────────────────────────
const showResults = (result: BiometricsResult) => {
    if (!result.face?.detected) return;
    if (paused) return;

    if(result.frameMetadata?.trace){
        traceDisplay.update(result.frameMetadata?.trace);
    }

    // ── Gaze cursor ─────────────────────────────────────────────────
    const normPog = result.gaze?.normPog;
    if (normPog) {
        const vw = document.documentElement.clientWidth || window.innerWidth;
        const vh = document.documentElement.clientHeight || window.innerHeight;
        cursor.style.left = `${(normPog[0] + 0.5) * vw}px`;
        cursor.style.top = `${(normPog[1] + 0.5) * vh}px`;
        cursor.style.backgroundColor = result.gaze!.gazeState === 'closed' ? 'gray' : 'purple';
    }

    if (normPog && result.gaze?.gazeState === 'open') {
        const stats = tracker.sample(normPog as [number, number]);
        if (stats) {
            accuracyText.innerText = `Dist: ${stats.dist.toFixed(4)} | Avg: ${stats.avg.toFixed(4)} | Jitter: ${stats.stddev.toFixed(4)} | Range: ${stats.min.toFixed(4)}-${stats.max.toFixed(4)} | n=${stats.count}`;
        }
    }

    // ── eye patch ───────────────────────────────────────────────
    const eyePatch = result.gaze?.eyePatch;
    if (eyePatch && eyePatchCtx) {
        eyePatchCtx.putImageData(eyePatch, 0, 0);
    }

    // ── Heart rate regions + graph ──────────────────────────────────
    if (result.heart) {
        // console.log(result.heart);
        syncWebcamCanvas();
        const heartRateResult = result.heart;
        if (heartRateResult.signal.raw !== null) {
            pulseGraph.update(
                heartRateResult.signal.raw,
                heartRateResult.signal.filtered ?? null,
                heartRateResult.signal.peakDetected ?? false
            );
        }
        drawPulseMask(heartRateResult.regions, webcamOverlayCtx, webcamOverlayCanvas.width, webcamOverlayCanvas.height)
        // console.log(heartRateResult)
        const fftBPM = heartRateResult.estimators.fft?.bpm
        const newFFT = fftBPM ? `FFT: ${Math.round(fftBPM)}` : 'Processing...';
        if (newFFT !== displayedFFT) displayedFFT = newFFT;
        if (heartRateResult.signal.peakDetected && heartRateResult.estimators.peak?.bpm) {
            displayedPeak = ` | Live: ${Math.round(heartRateResult.estimators.peak.bpm)}`;
        }

        const text = `BPM ${displayedFFT}${displayedPeak}`;
        if (bpmDisplay.innerText !== text) bpmDisplay.innerText = text;
    }
};

// ─── Accuracy tracker ───────────────────────────────────────────────────────
const tracker = new AccuracyTracker();
const target = document.getElementById('target') as HTMLDivElement;

window.addEventListener('keydown', (e) => {
    if (e.key === 'b') pulseGraph.beep = !pulseGraph.beep;
    if (e.key !== 't' && e.key !== 'T') return;
    if (tracker.active) {
        const finalStats = tracker.stop();
        target.style.display = 'none';
        console.log('Accuracy results:', finalStats);
    } else {
        tracker.start([0, 0]);
        target.style.display = 'block';
    }
});

// ─── Toggle state ───────────────────────────────────────────────────────────
const toggleGaze = document.getElementById('toggleGaze') as HTMLInputElement;
const toggleHeart = document.getElementById('toggleHeart') as HTMLInputElement;

let client: BiometricsClient | null = null;
let restarting = false;

const assets = {
    wasmBasePath: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm",
    faceLandmarkerModelPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    gazeModelPath: import.meta.env.BASE_URL + 'models/model.json',
};

function createClient(): BiometricsClient {
    const c = new BiometricsClient('webcam', {
        pipeline: {
            gaze: toggleGaze.checked ? undefined : false,
            heart: toggleHeart.checked ? undefined : false,
        },
        assets,
    });

    c.onResult = (result) => {
        result?.frameMetadata?.trace?.push({ step: 'frame_received', timestamp: performance.now() });
        showResults(result);
    };

    c.onWebcamStatus = (status, msg) => {
        console.log(`Webcam: ${status}`, msg);
    };

    return c;
}

function clearDisabledUI(): void {
    if (!toggleGaze.checked) {
        cursor.style.left = '-100px';
        cursor.style.top = '-100px';
        accuracyText.innerText = '';
        eyePatchCtx?.clearRect(0, 0, eyePatchCanvas.width, eyePatchCanvas.height);
    }
    if (!toggleHeart.checked) {
        bpmDisplay.innerText = '';
        displayedFFT = '';
        displayedPeak = '';
        webcamOverlayCtx.clearRect(0, 0, webcamOverlayCanvas.width, webcamOverlayCanvas.height);
        const pulseGraphCtx = pulseGraphCanvas?.getContext('2d')
        pulseGraphCtx?.clearRect(0, 0, pulseGraphCanvas.width, pulseGraphCanvas.height);
        pulseGraphCanvas.style.display = 'none';
    } else {
        pulseGraphCanvas.style.display = 'inline';
    }
}

async function restart(): Promise<void> {
    if (restarting) return;
    restarting = true;

    try {
        if (client) {
            client.dispose();
            client = null;
        }

        clearDisabledUI();

        client = createClient();
        await client.start();
        console.log('Restarted with gaze=' + toggleGaze.checked + ' heart=' + toggleHeart.checked);
    } catch (error) {
        console.error('Restart failed:', error);
    } finally {
        restarting = false;
    }
}

toggleGaze.addEventListener('change', () => restart());
toggleHeart.addEventListener('change', () => restart());

// ─── Initial start ─────────────────────────────────────────────────────────
client = createClient();

try {
    // Attempt the frictionless auto-start
    await client.start();
    console.log("Auto-start successful!");
} catch (error) {
    // Auto-start was blocked (likely iOS/Safari requiring a gesture)
    console.warn("Auto-start blocked. Waiting for user interaction...", error);
    const startButton = document.getElementById('startButton') as HTMLButtonElement;
    // Reveal the button to the user
    startButton.style.display = 'block';
    // Attach the click listener for the manual fallback
    startButton.addEventListener('click', async () => {
        startButton.style.display = 'none'; // Hide button immediately on click
        try {
            await client!.start();
            console.log("Manual start successful!");
        } catch (manualError) {
            console.error("Camera access denied after click:", manualError);
            startButton.style.display = 'block'; // Bring the button back if they denied permission
            alert("Please grant camera permissions to use this feature.");
        }
    });
}

const pauseButton = document.getElementById('pauseButton') as HTMLButtonElement;
let paused = false;

function setPaused(p: boolean): void {
    paused = p;
    pauseButton.innerText = paused ? 'Resume' : 'Pause';
    if(paused){
        video.pause()
    } else {
        video.play()
    }
}

pauseButton.addEventListener('click', () => setPaused(!paused));

window.addEventListener('keydown', (e) => {
    if (e.key === ' ' || e.key === 'P') setPaused(!paused);
});
