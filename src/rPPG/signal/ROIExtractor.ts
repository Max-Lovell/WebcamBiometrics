/**
 * ROI Extractor - average RGB from regions defined using MediaPipe's FaceLandmarker
 * For each region, returns input polygon in pixels and RGB average for that region
 * Only the canvas is reused here.
 */

import type { FaceLandmarkerResult, NormalizedLandmark } from '@mediapipe/tasks-vision';
import type { VideoFrameData, Point } from "../../types.ts";
import type { RGB } from "../types.ts";

// ─── Types ──────────────────────────────────────────────────────────────────
// ROIS: keyed by region name, with arrays of MediaPipe Face Landmark indices defining region boundary
    // See: https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
export type LandmarkerROIs = Record<string, number[]>;

// Per-region result
export interface RegionResult {
    polygon: Point[];  // Pixel coordinates of the polygon boundary
    rgb: RGB | null; // Average RGB within the polygon, or null if extraction failed
}

// ─── Default Face ROIs ──────────────────────────────────────────────────────

// Default face regions for rPPG, selected to maximise skin visibility while avoiding hair, eyebrows, eyes, and mouth.
    // See: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
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
    // OffscreenCanvas used for pixel extraction - permanent, created once
        // lazy init on first extract() call, then resized as needed to bounding box of extracted region
    private canvas: OffscreenCanvas | null = null;
    private ctx: OffscreenCanvasRenderingContext2D | null = null;
    // The landmark index arrays defining each face region
    private readonly rois: LandmarkerROIs;
    // Optional ROIs, defaults to ROIs defined above
    constructor(rois: LandmarkerROIs = FACE_ROIS) {
        this.rois = rois;
    }

    // ─── Public API ─────────────────────────────────────────────────────

    // This is the public method here, produces per region rgb averages
    extract(frame: VideoFrameData, faceLandmarks: FaceLandmarkerResult, logRegion?: boolean): Record<string, RegionResult> {
        const landmarks: NormalizedLandmark[] = faceLandmarks.faceLandmarks[0];
        const width = this.getFrameWidth(frame); // Note these could be made static and persistent
        const height = this.getFrameHeight(frame);

        // Per-region result, keyed by region name
        const regions: Record<string, RegionResult> = {};
        for (const regionName of Object.keys(this.rois)) {
            // Convert normalized landmarks (0–1) to pixel coordinates
            const polygon = this.landmarksToPolygon(this.rois[regionName], landmarks, width, height);

            if (polygon.length === 0) { // Handle empty region - TODO: might also want to check if region too small?
                regions[regionName] = { polygon: [], rgb: null };
                continue;
            }

            // TODO: occlusion detection — use landmark z-values?
            // TODO: check signal quality metrics downstream to flag bad data.
            const rgb = this.extractRegionRGB(frame, polygon, width, height, logRegion);
            regions[regionName] = { polygon, rgb };
        }

        return regions;
    }

    // Get region names
    get regionNames(): string[] {
        return Object.keys(this.rois);
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

    // ─── Internals ──────────────────────────────────────────────────────

    // Convert landmark indices(landmarks are [0,1]) to pixel polygon points
    private landmarksToPolygon(indices: number[], landmarks: NormalizedLandmark[], frameWidth: number, frameHeight: number): Point[] {
        if (!landmarks || landmarks.length === 0) return [];

        return indices.map(i => ({
            x: Math.floor(landmarks[i].x * frameWidth),
            y: Math.floor(landmarks[i].y * frameHeight),
        }));
    }

    // Extract the average RGB color within a polygon region of a frame.
    private extractRegionRGB(
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
        this.initCanvas(bWidth, bHeight);
        if (!this.ctx) return null;

        // Clip to polygon and draw frame - [0,0,0] everything except region
        // TODO: consider using Path2D for the clip — it can be cached and reused if the polygon doesn't change much between frames.
        // TODO: NOTE using canvas clip method here. consider using ray-casting on data to test if pixel in polygon on raw data
        //  - would allow for single getImageData or VideoFrame.copyTo() on full frame. or, just use 2d tensor or webGPU
        this.ctx.save();
        this.ctx.beginPath();
        this.ctx.moveTo(polygon[0].x - minX, polygon[0].y - minY);
        for (let i = 1; i < polygon.length; i++) {
            this.ctx.lineTo(polygon[i].x - minX, polygon[i].y - minY);
        }
        this.ctx.clip();

        // Draw only bounding box region of the frame
        this.ctx.drawImage(
            frame as CanvasImageSource,
            minX, minY, bWidth, bHeight, // source rect
            0, 0, bWidth, bHeight // dest rect
        );

        if(logRegion) this.logCanvas()
        // Read pixels
        const imageData = this.ctx.getImageData(0, 0, bWidth, bHeight).data;
        this.ctx.restore();

        // Alpha-weighted RGB average ──
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

        // Return as floats for decimal precision
        return {
            r: rSum / count,
            g: gSum / count,
            b: bSum / count,
        };
    }

    // Init/resize offscreen canvas.
    private initCanvas(width: number, height: number): void {
        if (!this.canvas) {
            this.canvas = new OffscreenCanvas(width, height);
            this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
            return;
        }
        // Resized to bounding box of region.
        if (this.canvas.width !== width || this.canvas.height !== height) {
            // Resize clears automatically
            this.canvas.width = width;
            this.canvas.height = height;
        } else {
            // Same size — must clear manually - maybe !this.ctx above?
            this.ctx?.clearRect(0, 0, width, height);
        }
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
