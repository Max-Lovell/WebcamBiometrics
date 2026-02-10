import type {VideoFrameData} from "./types";
import type {FaceLandmarkerResult, NormalizedLandmark} from "@mediapipe/tasks-vision";

export interface Point {
    x: number;
    y: number;
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

    constructor() {
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
        let width, height;
        if (frame instanceof VideoFrame) {
            width = frame.displayWidth;
            height = frame.displayHeight;
        } else {
            width = frame.width;
            height = frame.height;
        }

        const allLandmarks: NormalizedLandmark[] = faceLandmarkerResult.faceLandmarks[0]

        const roiPixels = []
        let region: keyof FaceROIs;
        for(region in landmarkerROIs) {
            const regionPixels = landmarkerROIs[region].map((landmark)=>
                this.getLandmarkPixelCoordinates(allLandmarks[landmark], width, height)
            );
            roiPixels.push(regionPixels);
        }

        // Extract pixel coords
        return roiPixels;
        // TODO: Occlusion detection using z to find points with other landmarks ontop of them, occupying same x/y?
        //  Might even be able to extract a visible face outline for use in a pull request? Furthest extreme x/y which is visible.
        //  Actually if landmarker already has a camera estimate and geometry then a ray trace would work best
    }

    processLandmarks(frame: VideoFrameData, faceLandmarkerResult: FaceLandmarkerResult, landmarkerROIs: FaceROIs = FACE_ROIS): Point[][] {
        // TODO: PLAN
        //  Extract pixel coordinates from frame using faceLandmarker
        //  Extract point-in-polygon region from landmarks, using z to find obscured landmarks to avoid resampling
        //  Optionally run skin detector
        //  Calculate average RGB for pixels

        return this.getRoiPixelCoordinates(frame, faceLandmarkerResult, landmarkerROIs)
    }

    // Hard reset (e.g. if tracking is lost or scene changes)
    reset() {
    }
}
