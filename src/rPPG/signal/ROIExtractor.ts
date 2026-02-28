/**
 * ROI Extractor - average RGB from regions defined using MediaPipe's FaceLandmarker
 * For each region, returns input polygon in pixels and RGB average for that region.
 *
 * Two extraction strategies:
 *   'scanline' (default) — One frame readback, scanline masks for polygon averaging.
 *                           Uses VideoFrame.copyTo() when available, else single drawImage + getImageData.
 *   'canvas'  (legacy)   — Per-region drawImage with canvas clip. Slower but kept for comparison.
 *
 * TODO: This is fine speed wise, but slowest part of the rPPG at time of writing. Consider TensorFlow or WebGPU version of this.
 */

import type { FaceLandmarkerResult, NormalizedLandmark } from '@mediapipe/tasks-vision';
import type { VideoFrameData, Point } from "../../types.ts";
import type { RGB } from "../types.ts";

// ─── Types ──────────────────────────────────────────────────────────────────
// ROIS: keyed by region name, with arrays of MediaPipe Face Landmark indices defining region boundary
// See: https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
export type LandmarkerROIs = Record<string, number[]>;

export type ExtractionStrategy = 'scanline' | 'canvas';

// Per-region result
export interface RegionResult {
    polygon: Point[];  // Pixel coordinates of the polygon boundary
    rgb: RGB | null; // Average RGB within the polygon, or null if extraction failed
}

// Scanline span — a horizontal run of pixels inside a polygon at a given y
interface ScanlineSpan {
    y: number;
    xStart: number;
    xEnd: number;
}

// Cached mask for a region — spans + the polygon that generated them
interface CachedMask {
    polygon: Point[];   // The polygon vertices used to build this mask
    spans: ScanlineSpan[];
}

// ─── Default Face ROIs ──────────────────────────────────────────────────────

// Default face regions for rPPG, selected to maximise skin visibility while avoiding hair, eyebrows, eyes, and mouth.
// canonical face mesh: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
// Face mesh points: https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
// optimal regions: https://www.nature.com/articles/s44325-024-00033-7 - NOTE not aligned perfectly, I went for maximising pixels
export const FACE_ROIS: LandmarkerROIs = {
    // Center forehead — avoids hair and eyebrows
    forehead: [107, 66, 105, 63, 68, 54, 103, 67, 109, 10, 338, 297, 332, 284, 298, 293, 334, 296, 336, 285, 417, 351, 419, 197, 196, 122, 193, 55],
    // Left cheek (subject's left, indices > 200)
    leftCheek: [350, 349, 348, 347, 346, 345, 352, 376, 411, 427, 436, 426, 423, 331, 279, 429, 437, 343],
    // Right cheek (subject's right, indices < 200)
    rightCheek: [121, 120, 119, 118, 117, 116, 123, 147, 187, 207, 216, 206, 203, 129, 49, 198, 217, 114],
};

// ─── ROIExtractor Class ─────────────────────────────────────────────────────

export class ROIExtractor {
    // The landmark index arrays defining each face region
    private readonly rois: LandmarkerROIs;
    private readonly strategy: ExtractionStrategy;

    // Canvas — used for legacy strategy, and as fallback frame readback for scanline when frame is not a VideoFrame
    private canvas: OffscreenCanvas | null = null;
    private ctx: OffscreenCanvasRenderingContext2D | null = null;

    // Scanline mask cache — keyed by region name, invalidated when polygon moves
    private maskCache: Map<string, CachedMask> = new Map();
    // Pixel movement threshold before rebuilding a scanline mask (in px)
    private static readonly MASK_INVALIDATION_THRESHOLD = 1;

    constructor(rois: LandmarkerROIs = FACE_ROIS, strategy: ExtractionStrategy = 'scanline') {
        this.rois = rois;
        this.strategy = strategy;
    }

    // ─── Public API ─────────────────────────────────────────────────────

    // Main entry point — produces per-region RGB averages
    extract(frame: VideoFrameData, faceLandmarks: FaceLandmarkerResult, logRegion?: boolean): Record<string, RegionResult> {
        const landmarks: NormalizedLandmark[] = faceLandmarks.faceLandmarks[0];
        const width = this.getFrameWidth(frame); // Note these could be made static and persistent
        const height = this.getFrameHeight(frame);

        if (this.strategy === 'scanline') { // alternatives: ray-casting (point-in-polygon test per pixel) https://aykevl.nl/2024/02/tinygl-polygon/
            return this.extractScanline(frame, landmarks, width, height);
        } else {
            return this.extractCanvas(frame, landmarks, width, height, logRegion); // TODO: probably just remove this as it's worse?
        }
    }

    // Get region names
    get regionNames(): string[] {
        return Object.keys(this.rois);
    }

    // ─── Scanline Strategy ──────────────────────────────────────────────
    // One frame readback (VideoFrame.copyTo() or single drawImage), then scanline masks for each region.
    // Avoids per-region drawImage calls and canvas clip overhead.
    private extractScanline(
        frame: VideoFrameData,
        landmarks: NormalizedLandmark[],
        frameWidth: number,
        frameHeight: number,
    ): Record<string, RegionResult> {
        // C.f.: 'scanline fill algorithm', 'scanline polygon rasterization', 'active edge table'
            // https://cssdeck.com/labs/scan-line-algorithm
            // https://www.tutorialspoint.com/computer_graphics/computer_graphics_scan_line_algorithm.htm
            // https://observablehq.com/@floledermann/scanline-fill-algorithm
            // https://dl.acm.org/doi/pdf/10.1145/245.248
            // https://web.cs.ucdavis.edu/~ma/ECS175_S00/Notes/0411_b.pdf
            // https://www.geeksforgeeks.org/c/scan-line-polygon-filling-using-opengl-c/

        // console.time('EXTRACTION')
        // Get raw RGBA pixel buffer for the entire frame (one readback)
        const pixelBuffer = this.getFramePixels(frame, frameWidth, frameHeight);
        if (!pixelBuffer) {
            // Fallback: return null RGB for all regions
            return this.emptyRegions(landmarks, frameWidth, frameHeight);
        }

        // For each region, build/reuse scanline mask and average
        const regions: Record<string, RegionResult> = {};
        for (const regionName of Object.keys(this.rois)) {
            const polygon = this.landmarksToPolygon(this.rois[regionName], landmarks, frameWidth, frameHeight);

            if (polygon.length === 0) {
                regions[regionName] = { polygon: [], rgb: null };
                continue;
            }

            // TODO: occlusion detection — use landmark z-values?
            // TODO: check signal quality metrics downstream to flag bad data.
            // Get or rebuild scanline mask
            const spans = this.getOrBuildMask(regionName, polygon);
            // Average RGB from the full-frame buffer using scanline spans
            const rgb = this.averageRGBFromSpans(pixelBuffer, spans, frameWidth, 0, 0);

            regions[regionName] = { polygon, rgb };
        }
        // console.timeEnd('EXTRACTION')
        return regions;
    }

    // Get raw RGBA pixels from the frame - single drawImage to full-frame canvas + getImageData.
    private getFramePixels(frame: VideoFrameData, width: number, height: number): Uint8ClampedArray | Uint8Array | null {
        // TODO: consider using VideoFrame.copyTo() as a direct buffer access
        //  but then need parent functions to be async, adding overheads/ sync complexity - benchmark first. (also 'Format negotiation')
        // Canvas fallback — draw full frame once, read pixels once
        this.initFullFrameCanvas(width, height);
        if (!this.ctx) return null;

        this.ctx.drawImage(frame as CanvasImageSource, 0, 0);
        return this.ctx.getImageData(0, 0, width, height).data;
    }

    // ─── Scanline Mask ──────────────────────────────────────────────────
    // Get cached scanline mask for a region, or rebuild if the polygon has moved.
    private getOrBuildMask(regionName: string, polygon: Point[]): ScanlineSpan[] {
        const cached = this.maskCache.get(regionName);
        // console.log(cached ? this.polygonMoved(cached.polygon, polygon) : undefined);
        if (cached && !this.polygonMoved(cached.polygon, polygon)) {
            return cached.spans;
        }

        // Compute bounding box
        let minY = Infinity, maxY = -Infinity;
        for (const p of polygon) {
            if (p.y < minY) minY = p.y;
            if (p.y > maxY) maxY = p.y;
        }
        minY = Math.floor(minY);
        maxY = Math.ceil(maxY);

        const spans = this.buildScanlineMask(polygon, minY, maxY);

        this.maskCache.set(regionName, { polygon: [...polygon], spans });
        return spans;
    }

    // Check if a polygon has moved enough to warrant rebuilding the scanline mask - basically a polygon comparison of each point
    private polygonMoved(cached: Point[], current: Point[]): boolean {
        if (cached.length !== current.length) return true;
        const threshold = ROIExtractor.MASK_INVALIDATION_THRESHOLD;
        for (let i = 0; i < cached.length; i++) {
            if (Math.abs(cached[i].x - current[i].x) > threshold ||
                Math.abs(cached[i].y - current[i].y) > threshold) {
                return true;
            }
        }
        return false;
    }

    // Build scanline mask for a polygon — for each row y in [minY, maxY], find horizontal spans inside the polygon using edge intersection + even-odd fill.
    private buildScanlineMask(polygon: Point[], minY: number, maxY: number): ScanlineSpan[] {
        const spans: ScanlineSpan[] = [];

        for (let y = minY; y <= maxY; y++) { // For each scanline y from the polygon's top to bottom:
            // Find all x-intersections of polygon edges with this scanline
            const intersections: number[] = [];

            for (let i = 0; i < polygon.length; i++) { // Walk every edge of the polygon.
                // Each edge connects two vertices - meaning one vertex is above and the other is below (or at) this y
                const a = polygon[i];
                const b = polygon[(i + 1) % polygon.length];

                // Does this edge cross the scanline (current y value)?
                if ((a.y <= y && b.y > y) || (b.y <= y && a.y > y)) {
                    // Linear interpolation for x at this y - where exactly does this edge cross y?
                    const x = a.x + (y - a.y) / (b.y - a.y) * (b.x - a.x);
                    intersections.push(x);
                }
            }

            // Sort and pair up (even-odd fill rule): every time we cross an edge, toggle between inside and outside.
            intersections.sort((a, b) => a - b); // Collect all intersections for this row, sort them left to right,
            for (let i = 0; i < intersections.length - 1; i += 2) {
                spans.push({ // pair them up: pixels between the 1st and 2nd intersection are inside, between 3rd and 4th are inside, etc.
                    y,
                    xStart: Math.ceil(intersections[i]), // Ceil/floor so only include pixels whose centers are inside polygon.
                    xEnd: Math.floor(intersections[i + 1]),
                });
            }
        }

        return spans; // List of pixels to read from the getImageData pixel buffer
    }

    // Sum RGB over scanline spans from a pixel buffer.
    private averageRGBFromSpans(
        data: Uint8Array | Uint8ClampedArray,
        spans: ScanlineSpan[],
        stride: number, // row width in pixels (frame width for full-frame buffer)
        offsetX: number, // 0 if buffer is full frame
        offsetY: number,
    ): RGB | null {
        let rSum = 0, gSum = 0, bSum = 0, count = 0;

        for (const { y, xStart, xEnd } of spans) {
            const rowBase = ((y - offsetY) * stride + (xStart - offsetX)) * 4;
            for (let x = xStart; x <= xEnd; x++) {
                const i = rowBase + (x - xStart) * 4;
                rSum += data[i];
                gSum += data[i + 1];
                bSum += data[i + 2];
                count++;
            }
        }

        if (count === 0) return null;
        return { r: rSum / count, g: gSum / count, b: bSum / count };
    }

    // ─── Canvas Strategy (Legacy) ────────────────────────────────── CONSIDER REMOVING AT THIS POINT?
    // Per-region drawImage with canvas clip. Slower due to repeated drawImage calls
    // from video source, but kept for comparison and validation.
    private extractCanvas(
        frame: VideoFrameData,
        landmarks: NormalizedLandmark[],
        frameWidth: number,
        frameHeight: number,
        logRegion?: boolean,
    ): Record<string, RegionResult> {
        const regions: Record<string, RegionResult> = {};
        for (const regionName of Object.keys(this.rois)) {
            const polygon = this.landmarksToPolygon(this.rois[regionName], landmarks, frameWidth, frameHeight);

            if (polygon.length === 0) {
                regions[regionName] = { polygon: [], rgb: null };
                continue;
            }

            const rgb = this.extractRegionRGB_Canvas(frame, polygon, frameWidth, frameHeight, logRegion);
            regions[regionName] = { polygon, rgb };
        }
        return regions;
    }

    // LEGACY - Extract the average RGB color within a polygon region of a frame using canvas clip.
    // Note if using this approach, consider refactoring to single-canvas across ROIs approach above, to reduce drawImage calls.
    private extractRegionRGB_Canvas(
        frame: VideoFrameData,
        polygon: Point[],
        frameWidth: number,
        frameHeight: number,
        logRegion?: boolean
    ): RGB | null {
        // Bounding box with safety clamping - only read the pixels we need, not the entire frame.
        let minX = frameWidth, minY = frameHeight, maxX = 0, maxY = 0;
        for (const p of polygon) {
            if (p.x < minX) minX = p.x;
            if (p.y < minY) minY = p.y;
            if (p.x > maxX) maxX = p.x;
            if (p.y > maxY) maxY = p.y;
        }

        minX = Math.max(0, Math.floor(minX));
        minY = Math.max(0, Math.floor(minY));
        const bWidth = Math.min(frameWidth - minX, Math.ceil(maxX - minX));
        const bHeight = Math.min(frameHeight - minY, Math.ceil(maxY - minY));
        if (bWidth <= 0 || bHeight <= 0) return null;

        // prep canvas - Resize offscreen canvas to the bounding box size
        this.initBoundingBoxCanvas(bWidth, bHeight);
        if (!this.ctx) return null;

        // Clip to polygon and draw frame - [0,0,0] everything except region
        this.ctx.save();
        this.ctx.beginPath();
        this.ctx.moveTo(polygon[0].x - minX, polygon[0].y - minY);
        for (let i = 1; i < polygon.length; i++) {
            this.ctx.lineTo(polygon[i].x - minX, polygon[i].y - minY);
        }
        this.ctx.clip();

        // Draw only bounding box region of the frame — most expensive step (~1ms per region)
        this.ctx.drawImage(
            frame as CanvasImageSource,
            minX, minY, bWidth, bHeight, // source rect
            0, 0, bWidth, bHeight // dest rect
        );

        if (logRegion) this.logCanvas();

        // Read pixels
        const imageData = this.ctx.getImageData(0, 0, bWidth, bHeight).data;
        this.ctx.restore();

        // Alpha-weighted RGB average
        // Note clip path produces anti-aliased edges with <255 alpha.
        // Could just skip if a<255 but this reduces pixels popping in and out at border - probably worth testing speed though.
        let rSum = 0, gSum = 0, bSum = 0, count = 0;
        for (let i = 0; i < imageData.length; i += 4) {
            const alpha = imageData[i + 3];
            if (alpha > 0) {
                const weight = alpha / 255;
                rSum += imageData[i] * weight;
                gSum += imageData[i + 1] * weight;
                bSum += imageData[i + 2] * weight;
                count += weight;
            }
        }

        if (count === 0) return null;

        return {
            r: rSum / count,
            g: gSum / count,
            b: bSum / count,
        };
    }

    // ─── Canvas Management ──────────────────────────────────────────────
    // Init/resize canvas to full frame dimensions (for scanline strategy) - 1 canvas draw per frame, getImageData reads regions from it.
    private initFullFrameCanvas(width: number, height: number): void {
        if (!this.canvas) {
            this.canvas = new OffscreenCanvas(width, height);
            this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
            return;
        }
        if (this.canvas.width !== width || this.canvas.height !== height) {
            this.canvas.width = width;
            this.canvas.height = height;
        }
        // No need to clear — drawImage(frame, 0, 0) overwrites entire canvas
    }

    // Init/resize canvas to bounding box dimensions (for canvas/legacy strategy).
    private initBoundingBoxCanvas(width: number, height: number): void {
        if (!this.canvas) {
            this.canvas = new OffscreenCanvas(width, height);
            this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
            return;
        }
        if (this.canvas.width !== width || this.canvas.height !== height) {
            // Resize clears automatically
            this.canvas.width = width;
            this.canvas.height = height;
        } else {
            // Same size — must clear manually
            this.ctx?.clearRect(0, 0, width, height);
        }
    }

    // Logs extracted region to console - expensive so use sparingly, maybe just on first valid frame
    logCanvas(): void {
        this.canvas?.convertToBlob().then(blob => {
            const reader = new FileReader();
            reader.onload = () => console.log(
                '%c ROI SNAPSHOT ',
                'background: #222; color: #bada55; font-size: 20px;',
                reader.result
            );
            reader.readAsDataURL(blob);
        });
    }

    // ─── Shared Utilities ───────────────────────────────────────────────
    // Convert landmark indices (landmarks are [0,1]) to pixel polygon points
    private landmarksToPolygon(indices: number[], landmarks: NormalizedLandmark[], frameWidth: number, frameHeight: number): Point[] {
        if (!landmarks || landmarks.length === 0) return [];

        return indices.map(i => ({
            x: Math.floor(landmarks[i].x * frameWidth),
            y: Math.floor(landmarks[i].y * frameHeight),
        }));
    }

    // Return empty results for all regions (used when frame readback fails)
    private emptyRegions(landmarks: NormalizedLandmark[], frameWidth: number, frameHeight: number): Record<string, RegionResult> {
        const regions: Record<string, RegionResult> = {};
        for (const regionName of Object.keys(this.rois)) {
            const polygon = this.landmarksToPolygon(this.rois[regionName], landmarks, frameWidth, frameHeight);
            regions[regionName] = { polygon, rgb: null };
        }
        return regions;
    }

    // Frame width, handling both VideoFrame (displayWidth) and other CanvasImageSource types (width).
    private getFrameWidth(frame: VideoFrameData): number {
        if ('displayWidth' in frame) return frame.displayWidth;   // VideoFrame
        if ('videoWidth' in frame) return frame.videoWidth;        // HTMLVideoElement
        return frame.width;                                        // ImageData, ImageBitmap
    }

    private getFrameHeight(frame: VideoFrameData): number {
        if ('displayHeight' in frame) return frame.displayHeight;   // VideoFrame
        if ('videoHeight' in frame) return frame.videoHeight;        // HTMLVideoElement
        return frame.height;                                        // ImageData, ImageBitmap
    }
}
