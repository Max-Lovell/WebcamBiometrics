/**
 * ROI Extractor — Tier 1 of the rPPG pipeline
 *
 * Extracts average RGB color from face regions of interest using
 * MediaPipe face landmarks and canvas-based pixel sampling.
 *
 * This is the only class in the pipeline that touches pixel data or
 * the Canvas API. Everything downstream works with numeric RGB values.
 *
 * Stateless per-frame — no signal buffers, no temporal memory.
 * Call extract() once per frame, get back RGB averages per region.
 *
 * Usage:
 *   const roi = new ROIExtractor();
 *   const result = roi.extract(videoFrame, faceLandmarks, performance.now());
 *   console.log(result.regions.forehead?.rgb); // { r: 182.3, g: 141.7, b: 122.1 }
 */

import type { FaceLandmarkerResult, NormalizedLandmark } from '@mediapipe/tasks-vision';

// ─── Types ──────────────────────────────────────────────────────────────────

/** A 2D pixel coordinate */
export interface Point {
    x: number;
    y: number;
}

/** Average RGB color as floats (not clamped to integers, preserving precision for signal processing) */
export interface RegionRGB {
    r: number;
    g: number;
    b: number;
}

/**
 * ROI landmark index definitions.
 *
 * Each key is a region name, each value is an array of MediaPipe face
 * landmark indices that define the polygon boundary of that region.
 *
 * See: https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
 */
export type LandmarkerROIs = Record<string, number[]>;

/** Per-region extraction result */
export interface RegionResult {
    /** Pixel coordinates of the polygon boundary (for visualization) */
    polygon: Point[];
    /** Average RGB within the polygon, or null if extraction failed */
    rgb: RegionRGB | null;
}

/** Full extraction result for one frame */
export interface ROIResult {
    /** Per-region results, keyed by region name */
    regions: Record<string, RegionResult>;
    /** Timestamp of the frame (passed through for downstream use) */
    timestamp: number;
}

/**
 * Frame dimensions — we need width and height from the video frame,
 * but VideoFrame uses displayWidth/displayHeight while other sources
 * use width/height. This type covers both.
 */
export type VideoFrameData = CanvasImageSource & (
    | { width: number; height: number }
    | { displayWidth: number; displayHeight: number }
    );

// ─── Default Face ROIs ──────────────────────────────────────────────────────

/**
 * Default face regions for rPPG, selected to maximise skin visibility
 * while avoiding hair, eyebrows, eyes, and mouth.
 *
 * Uses MediaPipe Face Landmarker indices (468 landmarks).
 * Reference: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
 */
export const FACE_ROIS: LandmarkerROIs = {
    // Center forehead — avoids hair and eyebrows
    forehead: [9, 107, 66, 105, 63, 68, 54, 103, 67, 109, 10, 338, 297, 332, 284, 298, 293, 334, 296, 336],
    // Left cheek (subject's left, indices > 200)
    leftCheek: [350, 349, 348, 347, 346, 352, 376, 411, 427, 436, 426, 423, 331, 279, 429, 437, 343],
    // Right cheek (subject's right, indices < 200)
    rightCheek: [121, 120, 119, 118, 117, 123, 147, 187, 207, 216, 206, 203, 129, 49, 198, 217, 114],
};

// ─── ROIExtractor Class ─────────────────────────────────────────────────────

export class ROIExtractor {
    /**
     * OffscreenCanvas used for pixel extraction.
     *
     * Why OffscreenCanvas and not a regular Canvas?
     *   - Works in Web Workers (important if you move processing off the main thread)
     *   - No DOM attachment needed
     *   - Same getImageData() API for pixel access
     *
     * Lazily initialized on first extract() call, then resized as needed.
     * Sized to the bounding box of the current region (not the full frame),
     * so we're only reading pixels we actually need.
     */
    private canvas: OffscreenCanvas | null = null;
    private ctx: OffscreenCanvasRenderingContext2D | null = null;

    /** The landmark index arrays defining each face region */
    private readonly rois: LandmarkerROIs;

    /**
     * @param rois - Region definitions mapping names to landmark indices.
     *   Defaults to FACE_ROIS (forehead + both cheeks).
     *   Pass custom ROIs if you want different regions or a single region.
     */
    constructor(rois: LandmarkerROIs = FACE_ROIS) {
        this.rois = rois;
    }

    // ─── Public API ─────────────────────────────────────────────────────

    /**
     * Extract average RGB for each face region from a video frame.
     *
     * This is the only public method. One frame in, one result out.
     * No state is carried between calls.
     *
     * @param frame - Video frame (VideoFrame, HTMLVideoElement, ImageBitmap, etc.)
     * @param faceLandmarks - MediaPipe face landmarker result
     * @param time - Frame timestamp in ms (DOMHighResTimeStamp from performance.now())
     * @returns Per-region RGB averages and polygon coordinates
     */
    extract(frame: VideoFrameData, faceLandmarks: FaceLandmarkerResult, time: number): ROIResult {
        const landmarks: NormalizedLandmark[] = faceLandmarks.faceLandmarks[0];
        const width = this.getFrameWidth(frame);
        const height = this.getFrameHeight(frame);

        const regions: Record<string, RegionResult> = {};

        for (const regionName of Object.keys(this.rois)) {
            // Convert normalized landmarks (0–1) to pixel coordinates
            const polygon = this.landmarksToPolygon(
                this.rois[regionName],
                landmarks,
                width,
                height
            );

            if (polygon.length === 0) {
                regions[regionName] = { polygon: [], rgb: null };
                continue;
            }

            // TODO: occlusion detection — check landmark z-values to detect
            // regions where other face parts (hand, hair) overlap the ROI.
            // For now we extract regardless and rely on signal quality metrics
            // downstream to flag bad data.

            const rgb = this.extractRegionRGB(frame, polygon, width, height);
            regions[regionName] = { polygon, rgb };
        }

        return { regions, timestamp: time };
    }

    /**
     * Get the current region names (useful for iterating results).
     */
    get regionNames(): string[] {
        return Object.keys(this.rois);
    }

    /**
     * Debug utility: log a snapshot of the canvas to the browser console.
     * The logged data URL can be clicked in DevTools to view the image.
     *
     * Call sparingly — convertToBlob() is async and relatively expensive.
     */
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

    // ─── Internals ──────────────────────────────────────────────────────

    /**
     * Convert an array of landmark indices to pixel-space polygon points.
     *
     * MediaPipe landmarks are normalized to [0, 1] relative to the frame
     * dimensions. We multiply by width/height and floor to get pixel coords.
     */
    private landmarksToPolygon(
        indices: number[],
        landmarks: NormalizedLandmark[],
        frameWidth: number,
        frameHeight: number
    ): Point[] {
        if (!landmarks || landmarks.length === 0) return [];

        return indices.map(i => ({
            x: Math.floor(landmarks[i].x * frameWidth),
            y: Math.floor(landmarks[i].y * frameHeight),
        }));
    }

    /**
     * Extract the average RGB color within a polygon region of a frame.
     *
     * Pipeline:
     *   1. Compute bounding box of the polygon (with clamping to frame edges)
     *   2. Resize the offscreen canvas to the bounding box size
     *   3. Set a clip path to the polygon shape
     *   4. Draw the relevant portion of the frame onto the canvas
     *   5. Read pixels with getImageData()
     *   6. Average the RGB values, weighted by alpha
     *
     * Why alpha-weighted averaging?
     *   The canvas clip path produces anti-aliased edges — pixels along the
     *   polygon boundary have partial alpha (0 < α < 255). Without weighting,
     *   these edge pixels would contribute fully despite being partly outside
     *   the region. Weighting by α/255 means edge pixels contribute
     *   proportionally, reducing the "pixel popping" artifact where a pixel
     *   at the polygon edge jumps in and out between frames as landmarks
     *   shift slightly.
     */
    private extractRegionRGB(
        frame: VideoFrameData,
        polygon: Point[],
        frameWidth: number,
        frameHeight: number
    ): RegionRGB | null {
        // ── 1. Bounding box with safety clamping ──
        // Only read the pixels we need, not the entire frame.
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

        // ── 2. Prepare canvas ──
        this.initCanvas(bWidth, bHeight);
        if (!this.ctx) return null;

        // ── 3. Clip to polygon and draw frame ──
        this.ctx.save();

        // TODO: consider using Path2D for the clip — it can be cached and reused if the polygon doesn't change much between frames.
        // TODO: NOTE using canvas clip method here. consider using ray-casting on data to test if pixel in polygon on raw data
        //  would allow for single getImageData or VideoFrame.copyTo() on full frame. or, just use 2d tensor or webGPU
        this.ctx.beginPath();
        this.ctx.moveTo(polygon[0].x - minX, polygon[0].y - minY);
        for (let i = 1; i < polygon.length; i++) {
            this.ctx.lineTo(polygon[i].x - minX, polygon[i].y - minY);
        }
        this.ctx.clip();

        // Draw only the bounding box region of the frame
        this.ctx.drawImage(
            frame as CanvasImageSource,
            minX, minY, bWidth, bHeight,   // source rect
            0, 0, bWidth, bHeight           // dest rect
        );

        // ── 4. Read pixels ──
        const imageData = this.ctx.getImageData(0, 0, bWidth, bHeight).data;
        this.ctx.restore();

        // ── 5. Alpha-weighted RGB average ──
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

        // Return as floats — the decimal precision matters for POS signal processing.
        // Rounding to integers would lose sub-pixel color variation that carries
        // the pulse signal.
        return {
            r: rSum / count,
            g: gSum / count,
            b: bSum / count,
        };
    }

    /**
     * Initialize or resize the offscreen canvas.
     *
     * The canvas is sized to the bounding box of the current region,
     * NOT the full video frame. This means we only allocate and read
     * the pixels we actually need.
     *
     * Note: setting canvas.width/height clears the canvas content and
     * resets the context state (transforms, clip paths, etc.), so we
     * only resize when dimensions actually change.
     */
    private initCanvas(width: number, height: number): void {
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

    /**
     * Get frame width, handling both VideoFrame (displayWidth) and
     * other CanvasImageSource types (width).
     */
    private getFrameWidth(frame: VideoFrameData): number {
        return 'displayWidth' in frame ? (frame as any).displayWidth : (frame as any).width;
    }

    private getFrameHeight(frame: VideoFrameData): number {
        return 'displayHeight' in frame ? (frame as any).displayHeight : (frame as any).height;
    }
}
