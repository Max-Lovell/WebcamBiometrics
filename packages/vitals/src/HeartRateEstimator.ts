import type {VideoFrameData} from "./types";
import type {FaceLandmarkerResult, NormalizedLandmark} from "@mediapipe/tasks-vision";
import {calculatePOS} from "./signal/POS";

export interface Point {
    x: number;
    y: number;
}

interface RGBBuffer { // TODO: should this hold interpolated or raw values?
    index: number;
    ready: boolean;
    times: Float32Array;
    regions: Record<string, { // keyed by region name
        r: Float32Array;
        g: Float32Array;
        b: Float32Array;
    }>;
}

interface POSHBuffer {
    index: number;
    ready: boolean; // min buffer length reached
    h: Float32Array; // fused H over all regions
    regions: Record<string, Float32Array>; // Per-region H
}

export interface HeartRateResult {
    timestamp: number;
    posH: number | null
    bpm: number | null;
    confidence: number;
    regions: Record<string, {
        polygon: Point[];
        averageRGB: { r: number, g: number, b: number } | null;
        posH: number | null;
    }>;
}

export type FaceRegion = 'forehead' | 'leftCheek' | 'rightCheek';
export type LandmarkerROIs = Record<string, number[]>;
export const FACE_ROIS: LandmarkerROIs = {
    // See https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
    // Or https://github.com/google-ai-edge/mediapipe/blob/e0eef9791ebb84825197b49e09132d3643564ee2/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    // Center forehead (Avoids hair/eyebrows)
    forehead: [9, 107, 66, 105, 63, 68, 54, 103, 67, 109, 10, 338, 297, 332, 284, 298, 293, 334, 296, 336],
    // Left Cheek (Subject's Left - Indices > 200)
    leftCheek: [350, 349, 348, 347, 346, 352, 376, 411, 427, 436, 426, 423, 331, 279, 429, 437, 343],
    // Right Cheek (Subject's Right - Indices < 200)
    rightCheek: [121, 120, 119, 118, 117, 123, 147, 187, 207, 216, 206, 203, 129, 49, 198, 217, 114]
};

export class HeartRateEstimator {
    private offscreenCanvas: OffscreenCanvas | null = null;
    private ctx: OffscreenCanvasRenderingContext2D | null = null;

    private landmarkerROIs: LandmarkerROIs;

    private MAX_RGB_SAMPLES: number = 32;
    private rgbRingBuffer: RGBBuffer; // RGB samples for each region - store POS too?

    private MAX_POS_SAMPLES: number = 300; // ~10 seconds at 30fps
    private posHRingBuffer: POSHBuffer;


    constructor(landmarkerROIs?: LandmarkerROIs, fps: number = 30) {
        this.landmarkerROIs = landmarkerROIs ?? FACE_ROIS;
        this.MAX_RGB_SAMPLES = Math.ceil(fps*1.6) // 1.6 from POS paper where l=20fps*1.6 = 32 frames window size, probably arbitrary?

        // Initialize region buffers
        const regionRgbBuffers: RGBBuffer['regions'] = {};
        const regionPOSBuffers: POSHBuffer['regions'] = {};
        for (let key of Object.keys(this.landmarkerROIs)) {
            regionRgbBuffers[key] = {
                r: new Float32Array(this.MAX_RGB_SAMPLES),
                g: new Float32Array(this.MAX_RGB_SAMPLES),
                b: new Float32Array(this.MAX_RGB_SAMPLES),
            }
            regionPOSBuffers[key] = new Float32Array(this.MAX_POS_SAMPLES)
        }

        // Instantiate RGB ring buffer
        this.rgbRingBuffer = {
            index: 0,
            ready: false,
            times: new Float32Array(this.MAX_RGB_SAMPLES),
            regions: regionRgbBuffers
        }

        // Make POS samples array
        this.MAX_POS_SAMPLES = fps*10 // 10 seconds of POS samples for FFT signal to be calculated
        this.posHRingBuffer = {
            index: 0,
            ready: false,
            h: new Float32Array(this.MAX_POS_SAMPLES),
            regions: regionPOSBuffers
        }
    }

    private addSample(region: string, sample: {r: number, g: number, b: number}, time: number) {
        // Write at the current index
        const idx = this.rgbRingBuffer.index;

        this.rgbRingBuffer.regions[region].r[idx] = sample.r;
        this.rgbRingBuffer.regions[region].g[idx] = sample.g;
        this.rgbRingBuffer.regions[region].b[idx] = sample.b;
        this.rgbRingBuffer.times[idx] = time;

        // Note: index advancement happens once per frame for all regions
        // This should be called from a common location after all regions are processed
    }

    private advanceRGBBuffer() {
        // Advance the index
        this.rgbRingBuffer.index++;

        // Wrap around if we hit the limit (Circular Buffer)
        if (this.rgbRingBuffer.index >= this.MAX_RGB_SAMPLES) {
            this.rgbRingBuffer.index = 0;
            this.rgbRingBuffer.ready = true;
        }
    }

    private getUnrolledSignal(region: string): { r: Float32Array, g: Float32Array, b: Float32Array, times: Float32Array } {
        // TODO: this returns new arrays - pre-allocate Work Buffers in constructor and use buffer.set() to copy data.
        const n = this.rgbRingBuffer.ready ? this.MAX_RGB_SAMPLES : this.rgbRingBuffer.index;
        const r = new Float32Array(n);
        const g = new Float32Array(n);
        const b = new Float32Array(n);
        const t = new Float32Array(n);

        // If buffer is full, start reading from 'index' (the oldest data)
        // If not full, start reading from 0
        const start = this.rgbRingBuffer.ready ? this.rgbRingBuffer.index : 0;

        for (let i = 0; i < n; i++) {
            const idx = (start + i) % this.MAX_RGB_SAMPLES;
            r[i] = this.rgbRingBuffer.regions[region].r[idx];
            g[i] = this.rgbRingBuffer.regions[region].g[idx];
            b[i] = this.rgbRingBuffer.regions[region].b[idx];
            t[i] = this.rgbRingBuffer.times[idx];
        }
        return { r, g, b, times: t };
    }

    private addPOSValue(region: string, hValue: number) {
        const idx = this.posHRingBuffer.index;

        // Overlap-add the mean-centered value
        this.posHRingBuffer.regions[region][idx] += hValue;
    }

    private advancePOSBuffer() {
        this.posHRingBuffer.index++;

        if (this.posHRingBuffer.index >= this.MAX_POS_SAMPLES) {
            this.posHRingBuffer.index = 0;
            this.posHRingBuffer.ready = true;
        }
    }

    private initCanvases(width: number, height: number) {
        if (!this.offscreenCanvas) {
            this.offscreenCanvas = new OffscreenCanvas(width, height);
            this.ctx = this.offscreenCanvas.getContext('2d', { willReadFrequently: true });
        }
        // Note resize auto-clears context
        if (this.offscreenCanvas.width !== width || this.offscreenCanvas.height !== height) {
            this.offscreenCanvas.width = width;
            this.offscreenCanvas.height = height;
        } else {
            // Dimensions are the same, so we must clear it manually
            this.ctx?.clearRect(0, 0, width, height);
        }
    }

    getAverageRgb(frame: VideoFrameData, polygon: Point[]) {
        const width = 'displayWidth' in frame ? frame.displayWidth : frame.width;
        const height = 'displayHeight' in frame ? frame.displayHeight : frame.height;

        // Bounding Box with Safety Clamping to loop through less pixels
        let minX = width, minY = height, maxX = 0, maxY = 0;
        for (const p of polygon) {
            if (p.x < minX) minX = p.x;
            if (p.y < minY) minY = p.y;
            if (p.x > maxX) maxX = p.x;
            if (p.y > maxY) maxY = p.y;
        }

        minX = Math.max(0, Math.floor(minX));
        minY = Math.max(0, Math.floor(minY));
        const bWidth = Math.min(width - minX, Math.ceil(maxX - minX));
        const bHeight = Math.min(height - minY, Math.ceil(maxY - minY));
        if (bWidth <= 0 || bHeight <= 0) return null;
        // console.log({minX, minY, maxX, maxY, bWidth, bHeight});

        this.initCanvases(bWidth, bHeight);
        if(!this.ctx) return null;
        // Consider bitmap approach:
        //const roiBitmap = await createImageBitmap(frame, minX, minY, bWidth, bHeight);
        //const roiCanvas = new OffscreenCanvas(bWidth, bHeight); // Small canvas, would need to make coords relative to this
        //const roiCtx = roiCanvas.getContext('2d', { willReadFrequently: true });

        // Clip polygon region
        // TODO: need to check output of this vs other approaches - consider Point-in-polygon or ray-casting.
        // this.ctx.clearRect(0, 0, width, height);
        this.ctx.save(); // So can return to same state later
        // Draw mask TODO: use Path2D() instead
        this.ctx.beginPath();
        this.ctx.moveTo(polygon[0].x - minX, polygon[0].y - minY);
        for (let i = 1; i < polygon.length; i++) {
            this.ctx.lineTo(polygon[i].x - minX, polygon[i].y - minY);
        }
        // this.ctx.closePath();
        this.ctx.clip();
        // Draw frame to mask. CanvasImageSource = (Video, Image, or another Canvas)
        this.ctx.drawImage(frame as CanvasImageSource, minX, minY, bWidth, bHeight, 0, 0, bWidth, bHeight);
        const imageData = this.ctx.getImageData(0, 0, bWidth, bHeight).data;
        this.ctx.restore(); // Return canvas state to what it was at save() (removes clip region)
        // this.logCanvas()
        // TODO: consider https://github.com/fast-average-color for averaging
        let rSum = 0, gSum = 0, bSum = 0, count = 0;
        for (let i = 0; i < imageData.length; i += 4) {
            // note border gets aliased hence translucent alpha - factor in here to reduce pixel popping in and out affecting mean
            const alpha = imageData[i + 3];
            if (alpha > 0) {
                const weight = alpha / 255;
                rSum += imageData[i] * weight;
                gSum += imageData[i + 1] * weight;
                bSum += imageData[i + 2] * weight;
                count += weight;
            }
        }
        // console.log({rSum, gSum, bSum, count});
        if (count === 0) return null;

        // Return floats to preserve the decimal precision for the signal processor
        return {
            r: rSum / count,
            g: gSum / count,
            b: bSum / count,
        };
    }

    // TODO separate function to process given coordinates too, e.g. for doing finger-phone rPPG with no landmarks
    // processFrame(frame: VideoFrameData, frameContext: TrackingContext, coordinates?: Point[]) {}

    getRoiPixelCoordinates(frame: VideoFrameData, faceLandmarkerResult: FaceLandmarkerResult) {
        const width = 'displayWidth' in frame ? frame.displayWidth : frame.width;
        const height = 'displayHeight' in frame ? frame.displayHeight : frame.height;

        const allLandmarks: NormalizedLandmark[] = faceLandmarkerResult.faceLandmarks[0]
        return Object.values(this.landmarkerROIs).map(indices =>
            indices.map(idx => ({
                x: Math.floor(allLandmarks[idx].x * width),
                y: Math.floor(allLandmarks[idx].y * height)
            }))
        );
    }

    processLandmarks(frame: VideoFrameData, time: number, faceLandmarkerResult: FaceLandmarkerResult): HeartRateResult {
        // TODO: PLAN
        //  Optionally run skin detector, landmark occlusion detection
        //  Get RGB average, Add sliding window, get POS estimate

        // Extract useful info
        const allLandmarks: NormalizedLandmark[] = faceLandmarkerResult.faceLandmarks[0];
        const width = 'displayWidth' in frame ? frame.displayWidth : frame.width;
        const height = 'displayHeight' in frame ? frame.displayHeight : frame.height;

        const regionResults: HeartRateResult['regions'] = {};

        // Use each region separately - Multi-Site/ Maximum Ratio Combination (MRC).
        // TODO: Occlusion detection using z to find points with other landmarks ontop of them, occupying same x/y?
        //  Might even be able to extract a visible face outline for use in a pull request? Furthest extreme x/y which is visible.
        //  Actually if landmarker already has a camera estimate and geometry then a ray trace would work best
        Object.keys(this.landmarkerROIs).forEach((region) => {
            const polygon = this.landmarkerROIs[region].map(i =>({
                x: Math.floor(allLandmarks[i].x * width),
                y: Math.floor(allLandmarks[i].y * height)
            }))

            if (!polygon || polygon.length === 0) {
                regionResults[region] = {
                    polygon: [],
                    averageRGB: null,
                    posH: null
                };
                return;
            }
            // TODO: deal with obscured landmarks or ones out of frame - check what effect out of frame has?

            const rgbData = this.getAverageRgb(frame, polygon);

            // Initialize region result
            regionResults[region] = {
                polygon: polygon,
                averageRGB: rgbData,
                posH: null // TODO: calculate POS H value for this region
            };

            // Store in window
            if (rgbData) { // TODO: replace sliding window with better approach.
                this.addSample(region, rgbData, time);

                // Only process the signal if we have enough data (e.g., at least 32 frames)
                if (this.rgbRingBuffer.ready || this.rgbRingBuffer.index >= this.MAX_RGB_SAMPLES) {
                    const unrolled = this.getUnrolledSignal(region); // Puts circular buffer in order for POS
                    // TODO: interpolate
                    const h = calculatePOS(unrolled.r,unrolled.g,unrolled.b)
                    // CLAUDE LOOK HERE
                    regionResults[region].posH = h;
                    this.addPOSValue(region, h);


                    // TODO: consider interpolate each time vs batch interpolation - doing batch interpolation here for ease.
                    // pos(regionSamples[region])
                    // regionResults[region].posH = calculatedPosH; // Placeholder for actual calculation
                }
            }
        });

        // Calculate fused POS by averaging all valid region values
        const validRegionPOSValues = Object.values(regionResults)
            .map(r => r.posH)
            .filter((h): h is number => h !== null);

        if (validRegionPOSValues.length > 0) {
            const fusedH = validRegionPOSValues.reduce((sum, h) => sum + h, 0) / validRegionPOSValues.length;
            this.posHRingBuffer.h[this.posHRingBuffer.index] += fusedH;
            this.advancePOSBuffer();
        }

        // Advance buffer index once after processing all regions
        this.advanceRGBBuffer();

        // Return result with placeholders for BPM and overall POS
        return {
            timestamp: time,
            posH: validRegionPOSValues.length > 0
                ? this.posHRingBuffer.h[(this.posHRingBuffer.index - 1 + this.MAX_POS_SAMPLES) % this.MAX_POS_SAMPLES]
                : null,
            bpm: null, // TODO: calculate BPM from POS signal via FFT
            confidence: 1.0, // TODO: calculate actual confidence metric
            regions: regionResults
        };
    }

    private interpolateRGB(
        unrolled: { r: Float32Array, g: Float32Array, b: Float32Array, times: Float64Array },
        targetFps: number
    ): { r: Float32Array, g: Float32Array, b: Float32Array, times: Float64Array } {
        // Linear interpolation
        const { r, g, b, times } = unrolled;

        // Need at least 2 samples to interpolate
        if (times.length < 2) return unrolled;

        // Calculate time range
        const startTime = times[0];
        const endTime = times[times.length - 1];
        const duration = endTime - startTime;

        // Calculate number of evenly-spaced samples we want
        const targetSamples = Math.ceil(duration * targetFps / 1000) + 1; // +1 for fenceposting
        const dt = duration / (targetSamples - 1);

        // Allocate output arrays
        const interpR = new Float32Array(targetSamples);
        const interpG = new Float32Array(targetSamples);
        const interpB = new Float32Array(targetSamples);
        const interpTimes = new Float64Array(targetSamples);

        let sourceIdx = 0;

        for (let i = 0; i < targetSamples; i++) {
            const targetTime = startTime + i * dt;
            interpTimes[i] = targetTime;

            // Find bracket: move sourceIdx forward until times[sourceIdx+1] >= targetTime
            while (sourceIdx < times.length - 1 && times[sourceIdx + 1] < targetTime) {
                sourceIdx++;
            }

            if (sourceIdx >= times.length - 1) {
                // Past the end, use last values
                interpR[i] = r[r.length - 1];
                interpG[i] = g[g.length - 1];
                interpB[i] = b[b.length - 1];
            } else {
                // Linear interpolation
                const t0 = times[sourceIdx];
                const t1 = times[sourceIdx + 1];
                const alpha = (targetTime - t0) / (t1 - t0);

                interpR[i] = r[sourceIdx] + alpha * (r[sourceIdx + 1] - r[sourceIdx]);
                interpG[i] = g[sourceIdx] + alpha * (g[sourceIdx + 1] - g[sourceIdx]);
                interpB[i] = b[sourceIdx] + alpha * (b[sourceIdx + 1] - b[sourceIdx]);
            }
        }

        // console.log({times, interpTimes, inputFPS: targetFps,
        //     interpolatedFPS: Math.round(((interpTimes.length-1)/duration)*1000),
        //     actualFPS: Math.round((times.length/duration)*1000)})

        return { r: interpR, g: interpG, b: interpB, times: interpTimes };
    }
    
    logCanvas() {
        if (Math.random() < 0.01) { // run basically never.
            this.offscreenCanvas?.convertToBlob().then(blob => {
                const reader = new FileReader();
                reader.onload = () => console.log("%c ROI SNAPSHOT ",
                    "background: #222; color: #bada55; font-size: 20px;",
                    reader.result); // Click this data-url in console to view image
                reader.readAsDataURL(blob);
            });
        }
    }
    // Hard reset (e.g. if tracking is lost or scene changes)
    reset() {
    }
}
