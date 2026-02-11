import type {VideoFrameData} from "./types";
import type {FaceLandmarkerResult, NormalizedLandmark} from "@mediapipe/tasks-vision";

export interface Point {
    x: number;
    y: number;
}

interface RGBSample {
    r: number;
    g: number;
    b: number;
    time: number; // for POS to handle varying frame rates
}

export type FaceRegion = 'forehead' | 'leftCheek' | 'rightCheek';
export type FaceROIs = Record<FaceRegion, number[]>;
export const FACE_ROIS: FaceROIs = {
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
    private regionSamples: Record<FaceRegion, RGBSample[]> = {
        forehead: [],
        leftCheek: [],
        rightCheek: []
    };
    private MAX_SAMPLES = 300; // ~10 seconds at 30fps

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

    getAverageRgb(frame: VideoFrameData, polygon: Point[]): RGBSample | null {
        const width = 'displayWidth' in frame ? frame.displayWidth : frame.width;
        const height = 'displayHeight' in frame ? frame.displayHeight : frame.height;

        this.initCanvases(width, height);
        const { ctx } = this;
        if (!ctx) return null;

        // Bounding Box with Safety Clamping
        const xs = polygon.map(p => p.x);
        const ys = polygon.map(p => p.y);
        const minX = Math.max(0, Math.floor(Math.min(...xs)));
        const minY = Math.max(0, Math.floor(Math.min(...ys)));
        const maxX = Math.min(width, Math.ceil(Math.max(...xs)));
        const maxY = Math.min(height, Math.ceil(Math.max(...ys)));
        const bWidth = maxX - minX;
        const bHeight = maxY - minY;
        if (bWidth <= 0 || bHeight <= 0) return null;

        // Clip polygon region
        // TODO: need to check output of this vs other approaches - consider Point-in-polygon or ray-casting.
        ctx.clearRect(0, 0, width, height);
        ctx.save(); // So can return to same state later
        // Draw mask TODO: use Path2D() instead
        ctx.beginPath();
        ctx.moveTo(polygon[0].x, polygon[0].y);
        polygon.forEach(p => ctx.lineTo(p.x, p.y));
        ctx.clip(); //
        // Draw frame to mask. CanvasImageSource = (Video, Image, or another Canvas)
        ctx.drawImage(frame as CanvasImageSource, 0, 0, width, height);
        ctx.restore(); // Return canvas state to what it was at save() (removes clip region)

        // Extract bounding box to loop through less pixels
        const imageData = ctx.getImageData(minX, minY, bWidth, bHeight).data;
        // const imageData = ctx.getImageData(0, 0, width, height).data;

        // TODO: consider https://github.com/fast-average-color for averaging
        let rSum = 0, gSum = 0, bSum = 0, count = 0;
        for (let i = 0; i < imageData.length; i += 4) {
            const alpha = imageData[i + 3];
            if (alpha === 255) { // Ignore transparent/translucent pixels outside (or on border) of polygon clip
                rSum += imageData[i];
                gSum += imageData[i + 1];
                bSum += imageData[i + 2];
                count++;
            }
        }
        if (count === 0) return null;
        // Shrink masked ROI into a 1x1 pixel - evil hack for mathematical average of all pixels in the polygon - doesn't work?
        // averageCtx.drawImage(this.offscreenCanvas, 0, 0, width, height, 0, 0, 1, 1);

        // Return floats to preserve the decimal precision for the signal processor
        return {
            r: rSum / count,
            g: gSum / count,
            b: bSum / count,
            time: performance.now()
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

    getRoiPixelCoordinates(frame: VideoFrameData, faceLandmarkerResult: FaceLandmarkerResult, landmarkerROIs: FaceROIs) {
        const width = 'displayWidth' in frame ? frame.displayWidth : frame.width;
        const height = 'displayHeight' in frame ? frame.displayHeight : frame.height;

        const allLandmarks: NormalizedLandmark[] = faceLandmarkerResult.faceLandmarks[0]
        return Object.values(landmarkerROIs).map(indices =>
            indices.map(idx => this.getLandmarkPixelCoordinates(allLandmarks[idx], width, height))
        );
        // TODO: Occlusion detection using z to find points with other landmarks ontop of them, occupying same x/y?
        //  Might even be able to extract a visible face outline for use in a pull request? Furthest extreme x/y which is visible.
        //  Actually if landmarker already has a camera estimate and geometry then a ray trace would work best
    }

    processLandmarks(frame: VideoFrameData, time: number, faceLandmarkerResult: FaceLandmarkerResult, landmarkerROIs: FaceROIs = FACE_ROIS): Point[][] {
        // TODO: PLAN
        //  Extract point-in-polygon region from landmarks, using z to find obscured landmarks to avoid resampling
        //  Optionally run skin detector
        //  Calculate average RGB for pixels
        const polygons = this.getRoiPixelCoordinates(frame, faceLandmarkerResult, landmarkerROIs);

        // Get polygon of ROI
        const regionKeys = Object.keys(landmarkerROIs) as FaceRegion[];

        regionKeys.forEach((regionName, index) => {
            const polygon = polygons[index];
            if (!polygon || polygon.length === 0) return;

            const rgbData = this.getAverageRgb(frame, polygon);
            console.log({rgbData})
            if (rgbData) {
                this.regionSamples[regionName].push(rgbData);

                // Maintain the sliding window for the signal processing phase
                if (this.regionSamples[regionName].length > this.MAX_SAMPLES) {
                    this.regionSamples[regionName].shift();
                }
            }
        });
        const nSamples= this.regionSamples.forehead.length
        // console.log(this.regionSamples)
        return {polygons, regionSamples: [this.regionSamples.forehead[nSamples-1], this.regionSamples.leftCheek[nSamples-1], this.regionSamples.rightCheek[nSamples-1]]};
    }

    // Hard reset (e.g. if tracking is lost or scene changes)
    reset() {
    }
}
