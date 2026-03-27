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
// ─── Client setup ───────────────────────────────────────────────────────────
const client = new BiometricsClient('webcam', {
    assets: {
        wasmBasePath: import.meta.env.BASE_URL + 'wasm',
        faceLandmarkerModelPath: import.meta.env.BASE_URL + 'wasm/face_landmarker.task',
        gazeModelPath: import.meta.env.BASE_URL + 'models/model.json',
    }
});

client.onResult = (result) => {
    result?.frameMetadata?.trace?.push({ step: 'frame_received', timestamp: performance.now() });
    showResults(result);
};

client.onWebcamStatus = (status, msg) => {
    console.log(`Webcam: ${status}`, msg);
};

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
            await client.start();
            console.log("Manual start successful!");
        } catch (manualError) {
            console.error("Camera access denied after click:", manualError);
            startButton.style.display = 'block'; // Bring the button back if they denied permission
            alert("Please grant camera permissions to use this feature.");
        }
    });
}
