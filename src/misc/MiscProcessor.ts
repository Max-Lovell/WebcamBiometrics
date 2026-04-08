import type {FaceContext} from '../pipeline/types';
import type { MiscResult } from './types';
import {irisDistance} from "./utils/distance";
import {getFrameDimensions} from "../utils/frameUtils";
import type {VideoFrameData} from "../types";
import {irisUnitGaze} from "./utils/irisUnit.ts";

export class MiscProcessor {
    process(frame: VideoFrameData, face: FaceContext): MiscResult {// frameMetadata: FrameMetadata,
        const landmarks = face.faceLandmarkerResult.faceLandmarks[0]
        const dims = getFrameDimensions(frame)
        const irisDist = irisDistance(landmarks, dims.width, dims.height)
        // TODO: blink, eyePatch, calibration, camera intrinsics
        const gaze = irisUnitGaze(face.faceLandmarkerResult, dims.width, dims.height)

        return {
            distance: irisDist,
            gaze: gaze
        };
    }

    dispose(): void {
        // clean up if needed
    }
}
