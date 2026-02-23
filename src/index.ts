import WebcamClient from './Core/WebcamClient.ts';
import WebEyeTrackProxy from './WebEyeTrack/WebEyeTrackProxy.ts';
import type { BiometricsResult } from './WebEyeTrack';

// After the imports, add a simple graph
const posGraph = document.getElementById('pos_graph') as HTMLCanvasElement;
const bpmDisplay = document.getElementById('bpm_display') as HTMLDivElement;
posGraph.width = 300;
posGraph.height = 100;
const posCtx = posGraph.getContext('2d')!;
const posHistory: number[] = [];
const POS_MAX_POINTS = 300;

function drawPosGraph(signal: number, bpm: number | null, peakBPM?: number | null) {
    bpmDisplay.innerText = bpm && bpm > 0 ? `${Math.round(bpm)} BPM` : "Processing...";
    bpmDisplay.innerText += peakBPM ? `| ${Math.round(peakBPM)}` : ''
    posHistory.push(signal);
    if (posHistory.length > POS_MAX_POINTS) posHistory.shift();
    if (posHistory.length < 2) return;

    const w = posGraph.width, h = posGraph.height;
    posCtx.clearRect(0, 0, w, h);

    let min = Math.min(...posHistory);
    let max = Math.max(...posHistory);
    let range = max - min || 1;
    min -= range * 0.1; max += range * 0.1; range = max - min;

    posCtx.strokeStyle = '#00ff00';
    posCtx.lineWidth = 2;
    posCtx.beginPath();
    posHistory.forEach((val, i) => {
        const x = (i / (POS_MAX_POINTS - 1)) * w;
        const y = h - ((val - min) / range) * h;
        if (i === 0) posCtx.moveTo(x, y); else posCtx.lineTo(x, y);
    });
    posCtx.stroke();
}

// 1. Initialize the passive tracker
const tracker = new WebEyeTrackProxy({
    clickTTL: 30,
    maxPoints: 10
});

const cursor = document.getElementById('cursor') as HTMLDivElement;
// Setup results callback

function getTimeDiff(arr: object[], t1: string, t2: string): number {
    // @ts-ignore
    const start = arr.find((t: { step: string; }) => t.step === t1)?.timestamp;
    // @ts-ignore
    const end = arr.find((t: { step: string; }) => t.step === t2)?.timestamp;
    return end - start;
}

const inferenceTimes: number[] = []

tracker.onGazeResults = (result: BiometricsResult) => {
    if(!result.summary.faceDetected) return;
    // console.log("Result:", result);
    // WebEyeTrack - TODO: add to helper file in WebEyeTrack, pass in window, document, webeyetrack results, cursor element
    const normPog = result.webEyeTrack.normPog
    const x = (normPog[0] + 0.5) * (document.documentElement.clientWidth || window.innerWidth);
    const y = (normPog[1] + 0.5) * (document.documentElement.clientHeight || window.innerHeight);
    cursor.style.left = `${x}px`;
    cursor.style.top = `${y}px`;
    cursor.style.backgroundColor = result.webEyeTrack.gazeState === 'closed' ? 'gray' : 'red';


    // --- DEBUG VISUALIZATION ---

    // 1. Draw Existing CPU Patch (It's already an ImageData object)
    const cpuPatch = result.webEyeTrack.eyePatch;
    if (cpuPatch) {
        const cpuCanvas = document.getElementById('cpu_patch') as HTMLCanvasElement;
        const cpuCtx = cpuCanvas.getContext('2d');
        // Clear and draw
        cpuCanvas.width = cpuPatch.width;
        cpuCanvas.height = cpuPatch.height;
        cpuCtx?.putImageData(cpuPatch, 0, 0);
    }

    // 2. Draw New GPU Patch (Coming as raw pixels)
    // @ts-ignore
    if (result.gpuPixels) {
        const gpuCanvas = document.getElementById('gpu_patch') as HTMLCanvasElement;
        const gpuCtx = gpuCanvas.getContext('2d');


        // Create ImageData from the raw pixel array
        const gpuImgData = new ImageData(
            new Uint8ClampedArray(result.gpuPixels),
            512, 128 // Ensure these match your getEyePatchGPU output dimensions
        );

        gpuCanvas.width = 512;
        gpuCanvas.height = 128;
        gpuCtx?.putImageData(gpuImgData, 0, 0);
    }
// @ts-ignore
    if (result.debug?.newPatch) {
        // console.log(result.debug?.newPatch)
        const canvas = document.getElementById('output_canvas') as HTMLCanvasElement;
        const ctx = canvas.getContext('2d');

        if (ctx) {
            if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
                canvas.width = canvas.clientWidth;
                canvas.height = canvas.clientHeight;
            }
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.strokeStyle = 'cyan';
            ctx.lineWidth = 2;
            ctx.beginPath();
            // @ts-ignore
            const pts = result.debug.newPatch;
            ctx.moveTo(pts[0][0], pts[0][1]);
            pts.forEach((p: number[]) => ctx.lineTo(p[0], p[1]));
            ctx.closePath();
            ctx.stroke();
        }
    }
    if (result.debug?.heartRateResult) {
        const heartRateResult = result.debug?.heartRateResult;
        console.log('H', heartRateResult);
        if (heartRateResult.fusedSample !== null && heartRateResult.fusedSample !== undefined) {
            drawPosGraph(heartRateResult.fusedSample, heartRateResult.raw.fft?.bpm??0, heartRateResult.raw.peak?.bpm??0);
        }
        const canvas = document.getElementById('output_canvas') as HTMLCanvasElement;
        const ctx = canvas.getContext('2d');
        if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
            canvas.width = canvas.clientWidth;
            canvas.height = canvas.clientHeight;
        }
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'purple';
        ctx.lineWidth = 2;

        // console.log(heartRateResult.posH);

        // Iterate over regions object
        Object.entries(heartRateResult.regions).forEach(([regionName, regionData]) => {
            const polygon = regionData.polygon;
            if (polygon.length === 0) return;
            ctx.beginPath();
            ctx.moveTo(polygon[0].x, polygon[0].y);
            polygon.forEach((p) => ctx.lineTo(p.x, p.y));
            ctx.closePath();

            const posH = regionData.pulse ?? 0;
            // Normalize: -0.008 (transparent) → +0.003 (fully opaque)
            const alpha = Math.min(1, Math.max(0, (posH + 0.009) / 0.015));
            ctx.fillStyle = `rgba(255, 0, 0, ${alpha})`;
            ctx.fill();
        });
    }


    if (result.debug?.newPatchMetrics) {
        // console.log(result.debug?.newPatchMetrics)
        const canvas = document.getElementById('output_canvas') as HTMLCanvasElement;
        const ctx = canvas.getContext('2d');

        if (ctx) {
            if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
                canvas.width = canvas.clientWidth;
                canvas.height = canvas.clientHeight;
            }
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.strokeStyle = 'purple';
            ctx.lineWidth = 2;
            ctx.beginPath();
            // @ts-ignore
            const pts = result.debug.newPatchMetrics;
            ctx.moveTo(pts[0][0], pts[0][1]);
            pts.forEach((p: number[]) => ctx.lineTo(p[0], p[1]));
            ctx.closePath();
            ctx.stroke();
        }
    }

    // //// DEBUGGING METRIC TRANSFORMS
    const W_fo3d = result.webEyeTrack.faceOrigin3D
    // const W_HV = result.webEyeTrack.headVector
    // const FL_FAT = result.faceLandmarker.facialTransformationMatrixes[0].data
    //
    // console.log(`
    // Vector:
    //     FL ${FL_FAT[2]},${FL_FAT[6]},${FL_FAT[10]},
    //     WET ${W_HV}.
    // Origin:
    //     FL ${FL_FAT[12]},${FL_FAT[13]},${FL_FAT[14]},
    //     WET ${W_fo3d}.
    // `)
    // console.log(result.summary.headPosition[2].toFixed(1), W_fo3d[2].toFixed(1), ((((result.summary.headPosition[2]+W_fo3d[2])/2)+W_fo3d[2])/2).toFixed(1))
    // Tracking performance
    const now = performance.now();
    // Calculate Latency
    const ctx = (result as any).context;
    if (ctx && ctx.trace) {
        const t0 = ctx.trace[0].timestamp;
        const totalLatency = now - t0;
        // Find worker time if available
        // const inferenceTime = getTimeDiff(ctx.trace,'worker_start', 'worker_end')
        inferenceTimes.push(totalLatency);
        if(inferenceTimes.length > 30){
            console.log('INFERENCE TIMES: ', inferenceTimes.reduce(function (avg, value, _, { length }) {
                    return avg + value / length;
                }, 0).toFixed(1));
            inferenceTimes.length = 0
        }
        // console.log(`${totalLatency.toFixed(1)}, ${inferenceTime.toFixed(1)}, ${(totalLatency - inferenceTime).toFixed(1)}`)

        // console.log(`⏱
        //     Total: ${totalLatency.toFixed(1)}ms |
        //     Inference Total: ${inferenceTime.toFixed(1)}ms |
        //     FaceLandmarker: ${getTimeDiff(ctx.trace,'facelandmarker_start', 'facelandmarker_end').toFixed(1)}ms |
        //     Webeyetrack: ${getTimeDiff(ctx.trace,'webeyetrack_start', 'webeyetrack_end').toFixed(1)}ms |
        //     Overhead: ${(totalLatency - inferenceTime).toFixed(1)}ms
        // `);
    }
};

const webcam = new WebcamClient('webcam');

async function start() {
    try {
        await webcam.startWebcam(async (frame, context) => {
            context.trace = [{ step: 'main_receive', timestamp: performance.now() }];
            void tracker.processFrame(frame, context);
        });
    } catch (e) {
        console.error("Failed to start tracking:", e);
    }
}

// Bind to a button or run immediately
// document.getElementById('startBtn')?.addEventListener('click', start);
start();
