import type {BiometricsResult} from "./pipeline/types.ts";
import { BiometricsClient } from './pipeline/BiometricsClient';

// ─── POS Graph ──────────────────────────────────────────────────────────────
const posGraph = document.getElementById('pos_graph') as HTMLCanvasElement;
const bpmDisplay = document.getElementById('bpm_display') as HTMLDivElement;
posGraph.width = 600;
posGraph.height = 200;
const posCtx = posGraph.getContext('2d')!;

const POS_MAX_POINTS = 300;

// Ring buffers instead of shifting arrays
const posHistory = new Float64Array(POS_MAX_POINTS);
const filteredHistory = new Float64Array(POS_MAX_POINTS);
let posHead = 0;       // write index for posHistory
let posCount = 0;      // how many values written so far
let filtHead = 0;      // write index for filteredHistory
let filtCount = 0;     // how many values written so far

// Peak tracking — ring buffer of frame numbers
const MAX_PEAKS = 64;
const peakFramesBuf = new Int32Array(MAX_PEAKS);
let peakHead = 0;
let peakCount = 0;
let frameCounter = 0;

function ringPush(buf: Float64Array | Int32Array, head: number, count: number, value: number): [number, number] {
    buf[head] = value;
    return [(head + 1) % buf.length, Math.min(count + 1, buf.length)];
}

function ringGet(buf: Float64Array | Int32Array, head: number, count: number, i: number): number {
    // i=0 is oldest, i=count-1 is newest
    const start = (head - count + buf.length) % buf.length;
    return buf[(start + i) % buf.length];
}

function drawPosGraph(signal: number, filteredSignal: number | null, bpm: number | null, peakBPM: number | null, peakDetected: boolean) {
    bpmDisplay.innerText = bpm && bpm > 0 ? `${Math.round(bpm)} BPM` : "Processing...";
    if (peakBPM) bpmDisplay.innerText += ` | ${Math.round(peakBPM)}`;

    frameCounter++;
    if (peakDetected) {
        [peakHead, peakCount] = ringPush(peakFramesBuf, peakHead, peakCount, frameCounter);
    }

    [posHead, posCount] = ringPush(posHistory, posHead, posCount, signal);
    if (filteredSignal !== null) {
        [filtHead, filtCount] = ringPush(filteredHistory, filtHead, filtCount, filteredSignal);
    }

    if (posCount < 2) return;

    const w = posGraph.width, h = posGraph.height;
    posCtx.clearRect(0, 0, w, h);

    // Compute min/max in one pass over both buffers — no array spread
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < posCount; i++) {
        const v = ringGet(posHistory, posHead, posCount, i);
        if (v < min) min = v;
        if (v > max) max = v;
    }
    for (let i = 0; i < filtCount; i++) {
        const v = ringGet(filteredHistory, filtHead, filtCount, i);
        if (v < min) min = v;
        if (v > max) max = v;
    }
    let range = max - min || 1;
    min -= range * 0.1;
    max += range * 0.1;
    range = max - min;

    // Raw POS (green)
    drawRingTrace(posHistory, posHead, posCount, 'green', w, h, min, range);
    // Filtered POS (red)
    if (filtCount >= 2) {
        drawRingTrace(filteredHistory, filtHead, filtCount, 'red', w, h, min, range);
    }

    // Draw peak markers
    if (filtCount > 0) {
        posCtx.fillStyle = '#ffff00';
        const peakOffset = POS_MAX_POINTS - filtCount;
        for (let i = 0; i < peakCount; i++) {
            const peakFrame = ringGet(peakFramesBuf, peakHead, peakCount, i);
            const age = frameCounter - peakFrame;
            if (age >= POS_MAX_POINTS) continue;

            const idx = filtCount - 1 - age;
            if (idx < 0 || idx >= filtCount) continue;

            const x = ((idx + peakOffset) / (POS_MAX_POINTS - 1)) * w;
            const val = ringGet(filteredHistory, filtHead, filtCount, idx);
            const y = h - ((val - min) / range) * h;
            posCtx.beginPath();
            posCtx.arc(x, y, 4, 0, Math.PI * 2);
            posCtx.fill();
        }
    }
}

function drawRingTrace(
    buf: Float64Array, head: number, count: number,
    color: string, w: number, h: number, min: number, range: number
) {
    posCtx.strokeStyle = color;
    posCtx.lineWidth = 2;
    posCtx.beginPath();
    // Offset so newest value is always at the right edge
    const offset = POS_MAX_POINTS - count;
    for (let i = 0; i < count; i++) {
        const x = ((i + offset) / (POS_MAX_POINTS - 1)) * w;
        const y = h - ((ringGet(buf, head, count, i) - min) / range) * h;
        if (i === 0) posCtx.moveTo(x, y); else posCtx.lineTo(x, y);
    }
    posCtx.stroke();
}

// ─── Cursor ─────────────────────────────────────────────────────────────────
const cursor = document.getElementById('cursor') as HTMLDivElement;

// ─── Timing helpers ─────────────────────────────────────────────────────────
// Ring buffer for frame times — avoids growing/clearing arrays
const FRAME_TIME_BUF_SIZE = 100;
const frameTimes = new Float64Array(FRAME_TIME_BUF_SIZE);
let ftHead = 0;
let ftCount = 0;
let ftMin = Infinity;
let ftMax = -Infinity;
let ftSum = 0;

function pushFrameTime(t: number) {
    // If buffer is full, subtract the oldest value before overwriting
    if (ftCount === FRAME_TIME_BUF_SIZE) {
        const oldest = frameTimes[ftHead];
        ftSum -= oldest;
    }
    frameTimes[ftHead] = t;
    ftSum += t;
    ftHead = (ftHead + 1) % FRAME_TIME_BUF_SIZE;
    if (ftCount < FRAME_TIME_BUF_SIZE) ftCount++;

    // Recompute min/max only when buffer wraps (every 100 frames)
    // Otherwise maintain running values
    if (t < ftMin) ftMin = t;
    if (t > ftMax) ftMax = t;
    if (ftHead === 0) {
        // Full wrap — recompute exact min/max
        ftMin = Infinity;
        ftMax = -Infinity;
        for (let i = 0; i < ftCount; i++) {
            if (frameTimes[i] < ftMin) ftMin = frameTimes[i];
            if (frameTimes[i] > ftMax) ftMax = frameTimes[i];
        }
    }
}

// ─── Inference times ────────────────────────────────────────────────────────
const INFERENCE_BUF_SIZE = 30;
const inferenceTimes = new Float64Array(INFERENCE_BUF_SIZE);
let infHead = 0;
let infCount = 0;
let infSum = 0;

// ─── Canvas refs (cached once) ──────────────────────────────────────────────
const cpuCanvas = document.getElementById('cpu_patch') as HTMLCanvasElement;
const cpuCtx = cpuCanvas.getContext('2d');
const gpuCanvas = document.getElementById('gpu_patch') as HTMLCanvasElement;
const gpuCtx = gpuCanvas.getContext('2d');
const outputCanvas = document.getElementById('output_canvas') as HTMLCanvasElement;
const outputCtx = outputCanvas.getContext('2d')!;

// Track output canvas size to avoid unnecessary resets
let lastOutputW = 0;
let lastOutputH = 0;

function syncOutputCanvas(): void {
    const cw = outputCanvas.clientWidth;
    const ch = outputCanvas.clientHeight;
    if (lastOutputW !== cw || lastOutputH !== ch) {
        outputCanvas.width = cw;
        outputCanvas.height = ch;
        lastOutputW = cw;
        lastOutputH = ch;
    }
}

// ─── Results callback ───────────────────────────────────────────────────────
const showResults = (result: BiometricsResult) => {
    if (!result.face?.detected) return;

    // ── Gaze cursor ─────────────────────────────────────────────────
    const normPog = result.gaze?.normPog;
    if (normPog) {
        const vw = document.documentElement.clientWidth || window.innerWidth;
        const vh = document.documentElement.clientHeight || window.innerHeight;
        cursor.style.left = `${(normPog[0] + 0.5) * vw}px`;
        cursor.style.top = `${(normPog[1] + 0.5) * vh}px`;
        cursor.style.backgroundColor = result.gaze!.gazeState === 'closed' ? 'gray' : 'red';
    }

    // ── CPU eye patch ───────────────────────────────────────────────
    const cpuPatch = result.gaze?.debug?.eyePatch;
    if (cpuPatch && cpuCtx) {
        cpuCanvas.width = cpuPatch.width;
        cpuCanvas.height = cpuPatch.height;
        cpuCtx.putImageData(cpuPatch, 0, 0);
    }

    // ── GPU eye patch ───────────────────────────────────────────────
    // @ts-ignore
    const gpuPatch = result.gaze?.debug?.eyePatchGPU;
    if (gpuPatch && gpuCtx) {
        const gpuImgData = new ImageData(
            // @ts-ignore
            new Uint8ClampedArray(gpuPatch),
            512, 128
        );
        gpuCanvas.width = 512;
        gpuCanvas.height = 128;
        // gpuCtx.putImageData(gpuImgData, 0, 0);
    }

    // ── Output canvas — clear once, draw all overlays ───────────────
    syncOutputCanvas();
    outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);

    // @ts-ignore — debug new patch outline
    if (result.debug?.newPatch) {
        // @ts-ignore
        const pts = result.debug.newPatch;
        outputCtx.strokeStyle = 'cyan';
        outputCtx.lineWidth = 2;
        outputCtx.beginPath();
        outputCtx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) outputCtx.lineTo(pts[i][0], pts[i][1]);
        outputCtx.closePath();
        outputCtx.stroke();
    }

    // ── Heart rate regions + graph ──────────────────────────────────
    if (result.heart) {
        const heartRateResult = result.heart;

        if (heartRateResult.signal.raw !== null) {
            drawPosGraph(
                heartRateResult.signal.raw,
                heartRateResult.signal.filtered ?? null,
                heartRateResult.estimators.fft?.bpm ?? 0,
                heartRateResult.estimators.peak?.bpm ?? 0,
                heartRateResult.signal.peakDetected ?? false
            );
        }

        // Draw heart regions on output canvas (already cleared above)
        outputCtx.strokeStyle = 'purple';
        outputCtx.lineWidth = 2;

        const regions = heartRateResult.regions;
        for (const regionName in regions) {
            const regionData = regions[regionName];
            const polygon = regionData.polygon;
            if (polygon.length === 0) continue;
            outputCtx.beginPath();
            outputCtx.moveTo(polygon[0].x, polygon[0].y);
            for (let i = 1; i < polygon.length; i++) outputCtx.lineTo(polygon[i].x, polygon[i].y);
            outputCtx.closePath();

            const posH = regionData.pulse ?? 0;
            const alpha = Math.min(1, Math.max(0, (posH + 0.009) / 0.015));
            outputCtx.fillStyle = `rgba(255, 0, 0, ${alpha})`;
            outputCtx.fill();
        }
    }

    // @ts-ignore — debug patch metrics outline
    if (result.debug?.newPatchMetrics) {
        // @ts-ignore
        const pts = result.debug.newPatchMetrics;
        outputCtx.strokeStyle = 'purple';
        outputCtx.lineWidth = 2;
        outputCtx.beginPath();
        outputCtx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) outputCtx.lineTo(pts[i][0], pts[i][1]);
        outputCtx.closePath();
        outputCtx.stroke();
    }
};

// ─── Client setup ───────────────────────────────────────────────────────────
const client = new BiometricsClient('webcam');

client.onResult = (result) => {
    // console.log({result});
    result.frameMetadata.trace.push({ step: 'frame_recieved', timestamp: performance.now() });
    const trace = result.frameMetadata.trace;
    // console.log({trace});
    const time = trace[trace.length - 1].timestamp - trace[0].timestamp;

    pushFrameTime(time);
    if (ftCount > 0) {
        console.log('Frame time:', Math.round(ftSum / ftCount), Math.floor(ftMin), '-', Math.ceil(ftMax));
    }

    // Inference logging
    const ctx = (result as any).context;
    if (ctx?.trace) {
        const totalLatency = performance.now() - ctx.trace[0].timestamp;
        if (infCount === INFERENCE_BUF_SIZE) {
            infSum -= inferenceTimes[infHead];
        }
        inferenceTimes[infHead] = totalLatency;
        infSum += totalLatency;
        infHead = (infHead + 1) % INFERENCE_BUF_SIZE;
        if (infCount < INFERENCE_BUF_SIZE) infCount++;

        if (infHead === 0 && infCount === INFERENCE_BUF_SIZE) {
            console.log('INFERENCE TIMES:', (infSum / INFERENCE_BUF_SIZE).toFixed(1));
        }
    }

    showResults(result);
};

client.onWebcamStatus = (status, msg) => {
    console.log(`Webcam: ${status}`, msg);
};

await client.start();
