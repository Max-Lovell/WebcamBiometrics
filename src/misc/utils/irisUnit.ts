import type {FaceLandmarkerResult, NormalizedLandmark, Matrix} from "@mediapipe/tasks-vision";
import type {Point} from "../../types.ts";

// -------------
interface IrisLandmarks {
    inner: number;
    outer: number;
    top: number;
    bottom: number;
}

interface EyeCorners {
    inner: number;
    outer: number;
}

interface EyeLandmark {
    pupil: number;
    iris: IrisLandmarks;
    corners: EyeCorners;
}

interface EyeLandmarks {
    left: EyeLandmark;
    right: EyeLandmark
}

// -------------
interface Location {
    x: number;
    y: number;
    z: number;
}

const eyeLandmarks: EyeLandmarks = {
    // person/stage left/right
    // https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    // https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
    left: {
        pupil: 473,
        iris: {
            inner: 476,
            outer: 474,
            top: 475,
            bottom: 477,
        },
        corners: {
            inner: 362,
            outer: 263
        }
    },
    right: {
        pupil: 468,
        iris: {
            inner: 469,
            outer: 471,
            top: 470,
            bottom: 472,
        },
        corners: {
            inner: 133,
            outer: 33
        }
    }
};

export function irisUnitGaze (
    faceLandmarkerResult: FaceLandmarkerResult,
    frameWidth: number,
    frameHeight: number,
    focalLengthPx?: number,
){
    // Extract facelandmarker
    const facialTransformationMatrix: number[] = faceLandmarkerResult.facialTransformationMatrixes[0].data
    const faceLandmarks: NormalizedLandmark[] = faceLandmarkerResult.faceLandmarks[0]

    // Fall back to a rough focal length estimate if none provided
    const fx = focalLengthPx ?? Math.max(frameWidth, frameHeight);

    const leftPupilZ = irisDepth(faceLandmarks, 'left', frameWidth, frameHeight, fx)
    const leftPupil = faceLandmarks[eyeLandmarks['left'].pupil]
    const leftPupilXY = pixel2Metric(leftPupil.x, leftPupil.y, frameWidth, frameHeight, fx, leftPupilZ)
    const leftEyeballCenter = getEyeballCenter(faceLandmarks, facialTransformationMatrix, 'left', frameWidth, frameHeight, fx, leftPupilZ)

    const rightPupil = faceLandmarks[eyeLandmarks['right'].pupil]
    const rightPupilZ = irisDepth(faceLandmarks, 'right', frameWidth, frameHeight, fx)
    const rightPupilXY = pixel2Metric(rightPupil.x, rightPupil.y, frameWidth, frameHeight, fx, rightPupilZ)
    const rightEyeballCenter = getEyeballCenter(faceLandmarks, facialTransformationMatrix, 'right', frameWidth, frameHeight, fx, rightPupilZ)

    // Get gaze vector next
    console.log({...rightPupilXY, z:-rightPupilZ}, {x: facialTransformationMatrix[12],y: facialTransformationMatrix[13],z: facialTransformationMatrix[14]})
    // getEyeballCenter(faceLandmarks, 'left')

}

function getEyeballCenter(
    landmarks: NormalizedLandmark[],
    facialTransformationMatrix: number[],
    side: 'left' | 'right',
    frameWidth: number,
    frameHeight: number,
    fx: number,
    z: number
): Location {
    const EYEBALL_AXIAL_RADIUS = 11.75 // https://pmc.ncbi.nlm.nih.gov/articles/PMC4238270/
    const innerEyeCorner = landmarks[eyeLandmarks[side].corners.inner]
    const outerEyeCorner = landmarks[eyeLandmarks[side].corners.outer]
    const eyeMiddle = midpoint(innerEyeCorner, outerEyeCorner)
    const eyeMiddleMetric = pixel2Metric(eyeMiddle.x, eyeMiddle.y, frameWidth, frameHeight, fx, z)
    // Work backwards along head vector
    const headForward = {
        x: facialTransformationMatrix[8],   // row 0, col 2
        y: facialTransformationMatrix[9],   // row 1, col 2
        z: facialTransformationMatrix[10],  // row 2, col 2
    };

    return {
        x: eyeMiddleMetric.x - EYEBALL_AXIAL_RADIUS * headForward.x,
        y: eyeMiddleMetric.y - EYEBALL_AXIAL_RADIUS * headForward.y,
        z: z - EYEBALL_AXIAL_RADIUS * headForward.z,
    };
}

function midpoint (a: Point, b: Point): Point {
    return {
        x: (a.x+b.x)/2,
        y: (a.y+b.y)/2,
    }
}

function landmarkDistancePx(
    a: { x: number; y: number },
    b: { x: number; y: number },
    frameWidth: number,
    frameHeight: number,
): number {
    const dx = (a.x - b.x) * frameWidth;
    const dy = (a.y - b.y) * frameHeight;
    return Math.sqrt(dx * dx + dy * dy);
}

function irisDepth(
    landmarks: NormalizedLandmark[],
    side: 'left' | 'right',
    frameWidth: number,
    frameHeight: number,
    fx: number,
): number {
    // https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/
    const IRIS_DIAMETER_CM = 1.17; // HVID

    // Extract largest of iris height/width
    const irisTop = landmarks[eyeLandmarks[side].iris.top]
    const irisBottom = landmarks[eyeLandmarks[side].iris.bottom]
    const irisHeight = landmarkDistancePx(irisTop, irisBottom, frameWidth, frameHeight);

    const irisInner = landmarks[eyeLandmarks[side].iris.inner]
    const irisOuter = landmarks[eyeLandmarks[side].iris.outer]
    const irisWidth = landmarkDistancePx(irisInner, irisOuter, frameWidth, frameHeight);

    const apparentDiameterPx = Math.max(irisHeight, irisWidth);

    // Depth is based on focal length
    return (fx * IRIS_DIAMETER_CM) / apparentDiameterPx; // flip for same coordinates as facelandmarker
    // const pxPerCm = apparentDiameterPx/IRIS_DIAMETER_CM
    // const frameWidthCm = (frameWidth/apparentDiameterPx)*IRIS_DIAMETER_CM
    // const frameHeightCm = (frameHeight/apparentDiameterPx)*IRIS_DIAMETER_CM
}

function pixel2Metric(
    x: number,
    y: number,
    frameWidth: number,
    frameHeight: number,
    fx: number,
    z: number
): Point {
    // x and y relative to centre of camera at current depth using number of irises
    // centered and unnormalized
    const pixelX = (x - 0.5) * frameWidth;
    const pixelY = ((y - 0.5) * -1) * frameHeight; // Flip for same coordinates as facelandmarker
    return {
        x: (pixelX / fx) * z,
        y: (pixelY / fx) * z,
    }
}

