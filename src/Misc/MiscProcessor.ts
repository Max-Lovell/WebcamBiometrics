import type {FaceContext, FrameMetadata} from '../pipeline/types';
import type { MiscResult } from './types';
import {irisDistance} from "./utils/distance.ts";
import {getFrameDimensions} from "../utils/frameUtils.ts";
import type {VideoFrameData} from "../types.ts";

export class MiscProcessor {
    process(frame: VideoFrameData, frameMetadata: FrameMetadata, face: FaceContext): MiscResult {
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
