import type {FaceContext, FrameMetadata} from '../pipeline/types';
import type { MiscResult } from './types';
import {irisDistance} from "./utils/distance";
import {getFrameDimensions} from "../utils/frameUtils";
import type {VideoFrameData} from "../types";

export class MiscProcessor {
    process(frame: VideoFrameData, face: FaceContext): MiscResult {// frameMetadata: FrameMetadata,
        const landmarks = face.faceLandmarkerResult.faceLandmarks[0]
        const dims = getFrameDimensions(frame)
        const irisDist = irisDistance(landmarks, dims.width, dims.height)
        // TODO: blink, eyePatch, calibration, camera intrinsics
        return {
            distance: irisDist,
        };
    }



    dispose(): void {
        // clean up if needed
    }
}
