import type {BiometricsResult} from "./pipeline/types.ts";
import { BiometricsClient } from './pipeline/BiometricsClient';
import { drawPulseMask } from "./drawingUtils/drawPulseMask.ts";
import { PulseGraph } from './drawingUtils/PulseGraph';
import { AccuracyTracker } from './drawingUtils/AccuracyTracker';
import { TraceDisplay } from './drawingUtils/TraceDisplay';

// ─── Pulse Graph ────────────────────────────────────────────────────────────
const cursor = document.getElementById('cursor') as HTMLDivElement;

const pulseGraphCanvas = document.getElementById('heartRate') as HTMLCanvasElement;
const pulseGraph = new PulseGraph(pulseGraphCanvas);
const bpmDisplay = document.getElementById('BPM') as HTMLCanvasElement;

const eyePatchCanvas = document.getElementById('eyePatch') as HTMLCanvasElement;
const eyePatchCtx = eyePatchCanvas.getContext('2d');

const webcamOverlayCanvas = document.getElementById('webcamCanvas') as HTMLCanvasElement;
const webcamOverlayCtx = webcamOverlayCanvas.getContext('2d')!;

const traceDisplay = new TraceDisplay(
    document.getElementById('stats') as HTMLDivElement,
    60 // last 60 frames
);

// Track output canvas size to avoid unnecessary resets
let lastOutputW = 0;
let lastOutputH = 0;

function syncWebcamCanvas(): void {
    const cw = webcamOverlayCanvas.clientWidth;
    const ch = webcamOverlayCanvas.clientHeight;
    if (lastOutputW !== cw || lastOutputH !== ch) {
        webcamOverlayCanvas.width = cw;
        webcamOverlayCanvas.height = ch;
        lastOutputW = cw;
        lastOutputH = ch;
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
        if (stats) updateOverlay(stats);
    }

    // ── eye patch ───────────────────────────────────────────────
    const eyePatch = result.gaze?.eyePatch;
    if (eyePatch && eyePatchCtx) {
        eyePatchCtx.putImageData(eyePatch, 0, 0);
    }

    // ── Heart rate regions + graph ──────────────────────────────────
    if (result.heart) {
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
        console.log(heartRateResult)
        const fft = heartRateResult.bpm
        const peakBPM = heartRateResult.estimators.peak?.bpm
        bpmDisplay.innerText = fft ? `BPM FFT: ${Math.round(fft)}` : "Processing...";
        bpmDisplay.innerText += peakBPM ? ` | Live: ${Math.round(peakBPM)}` : ''
    }
};

// ─── Accuracy tracker ───────────────────────────────────────────────────────
const tracker = new AccuracyTracker();
const updateOverlay = AccuracyTracker.createOverlay();
const target = document.getElementById('target') as HTMLDivElement;

window.addEventListener('keydown', (e) => {
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
const client = new BiometricsClient('webcam');

client.onResult = (result) => {
    result?.frameMetadata?.trace?.push({ step: 'frame_received', timestamp: performance.now() });
    showResults(result);
};

client.onWebcamStatus = (status, msg) => {
    console.log(`Webcam: ${status}`, msg);
};

await client.start();
