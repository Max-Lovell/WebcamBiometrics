import type {VideoFrameData} from "./types";
import type {FaceLandmarkerResult, NormalizedLandmark} from "@mediapipe/tasks-vision";

export interface Point {
    x: number;
    y: number;
}

interface RGBSamples {
    r: Float32Array;
    g: Float32Array;
    b: Float32Array;
    times: Float32Array;
    // TODO: these two below should be global to the object as no need to duplicate for each channel
    ptr: number;         // The write head
    isFull: boolean;     // To know if we have enough data to run POS
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

    private regionSamples: Record<string, RGBSamples>; // RGB samples for each region - store POS too?
    private MAX_RGB_SAMPLES: number = 32;

    private MAX_POS_SAMPLES: number = 300; // ~10 seconds at 30fps
    private landmarkerROIs: LandmarkerROIs;

    constructor(landmarkerROIs?: LandmarkerROIs, fps: number = 30) {
        this.landmarkerROIs = landmarkerROIs ?? FACE_ROIS;
        // Number of samples of POS
        this.MAX_POS_SAMPLES = fps*10 // 10 seconds of POS samples for FFT signal to be calculated
        this.regionSamples = {} as Record<string, RGBSamples>;
        this.MAX_RGB_SAMPLES = Math.ceil(fps*1.6) // 1.6 from POS paper where l=20fps*1.6 = 32 frames window size, probably arbitrary?
        Object.keys(this.landmarkerROIs).forEach((region) => {
            this.regionSamples[region as FaceRegion] = {
                r: new Float32Array(this.MAX_POS_SAMPLES),
                g: new Float32Array(this.MAX_POS_SAMPLES),
                b: new Float32Array(this.MAX_POS_SAMPLES),
                times: new Float32Array(this.MAX_POS_SAMPLES),
                ptr: 0,
                isFull: false
            };
        });

    }

    private addSample(region: string, sample: {r: number, g: number, b: number, time: number}) {
        // Note doing this way stops any garbage collection spikes by not recreating
        const data = this.regionSamples[region];

        // Write at the current pointer
        data.r[data.ptr] = sample.r;
        data.g[data.ptr] = sample.g;
        data.b[data.ptr] = sample.b;
        data.times[data.ptr] = sample.time;

        // Advance the pointer
        data.ptr++;

        // Wrap around if we hit the limit (Circular Buffer)
        if (data.ptr >= this.MAX_RGB_SAMPLES) {
            data.ptr = 0;
            data.isFull = true;
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

    getLandmarkPixelCoordinates(landmark: NormalizedLandmark, width: number, height: number): Point {
        return {
            x: Math.floor(landmark.x * width),
            y: Math.floor(landmark.y * height),
        }
    }

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

    processLandmarks(frame: VideoFrameData, time: number, faceLandmarkerResult: FaceLandmarkerResult): Point[][] {
        // TODO: PLAN
        //  Optionally run skin detector, landmark occlusion detection
        //  Get RGB average, Add sliding window, get POS estimate

        // Extract useful info
        const allLandmarks: NormalizedLandmark[] = faceLandmarkerResult.faceLandmarks[0];
        const width = 'displayWidth' in frame ? frame.displayWidth : frame.width;
        const height = 'displayHeight' in frame ? frame.displayHeight : frame.height;

        const polygons = this.getRoiPixelCoordinates(frame, faceLandmarkerResult);
        // Get polygon of ROI
        // Use each region separately - Multi-Site/ Maximum Ratio Combination (MRC).

        // TODO: Occlusion detection using z to find points with other landmarks ontop of them, occupying same x/y?
        //  Might even be able to extract a visible face outline for use in a pull request? Furthest extreme x/y which is visible.
        //  Actually if landmarker already has a camera estimate and geometry then a ray trace would work best
        Object.keys(this.landmarkerROIs).forEach((region) => {
            const polygon = this.landmarkerROIs[region].map(i =>({
                    x: Math.floor(allLandmarks[i].x * width),
                    y: Math.floor(allLandmarks[i].y * height)
            }))
            if (!polygon || polygon.length === 0) return;

            const rgbData = this.getAverageRgb(frame, polygon);
            // Store in window
            // Calc POS
            if (rgbData) { // TODO: replace sliding window with better approach.
                this.addSample(region, rgbData);
                // Only process the signal if we have enough data (e.g., at least 32 frames)
                if (this.regionSamples[region].isFull || this.regionSamples[region].ptr > this.MAX_RGB_SAMPLES) {
                    // TODO: interpolate and calc POS, or calc POS and interpolate?
                    // pos(regionSamples[region])
                }
            }
        });

        // console.log(this.regionSamples)
        // TODO: return statement { regions: {forehead: {path: [], averageColour: , POS: }}, BPM: 0, POS: 0}
        const recentSample = Object.values(this.regionSamples).map((region)=> {
            // Note flashes when the bucket resets - not an issue though as values should only
            return {r: region.r[region.ptr-1], g: region.g[region.ptr-1], b: region.b[region.ptr-1], ptr:region.ptr}
        })

        return {polygons, regionSamples: recentSample}
    }
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
