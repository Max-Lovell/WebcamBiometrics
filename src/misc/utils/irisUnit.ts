import type {FaceLandmarkerResult, NormalizedLandmark} from "@mediapipe/tasks-vision";
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
export interface Coordinate3D {
    x: number;
    y: number;
    z: number;
}

interface GazeInfo {
    pixel: Point;
    origin: Coordinate3D;
    vector: Coordinate3D;
}

interface IrisGazeOutput {
    left: GazeInfo;
    right: GazeInfo;
    cyclopean: GazeInfo
    screenPog: Point | null;  // cm in screen plane, null if ray misses
    fx: number;
    debug?: any;
}

// ----------------
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
): IrisGazeOutput {
    // Screen config — pass into irisUnitGaze or keep as module const for now
    // const SCREEN = {
    //     widthCm: 61,
    //     heightCm: 34.3,
    //     // Camera position on screen: (0,0) = camera at top-center
    //     // Screen extends from x ∈ [-widthCm/2, widthCm/2], y ∈ [-heightCm, 0]
    // };

    // Extract facelandmarker
    const facialTransformationMatrix: number[] = faceLandmarkerResult.facialTransformationMatrixes[0].data
    const faceLandmarks: NormalizedLandmark[] = faceLandmarkerResult.faceLandmarks[0]
    // Fall back to a rough focal length estimate if none provided
    const fx = focalLengthPx ?? Math.max(frameWidth, frameHeight);
    const fxFacelandmarker = (frameHeight / 2) / Math.tan((63 / 2) * Math.PI / 180);

    // Left eye
    const leftPupilZ = irisDepth(faceLandmarks, 'left', frameWidth, frameHeight, fx)
    const leftPupil = faceLandmarks[eyeLandmarks['left'].pupil]
    const leftPupilXY = landmark2Metric(leftPupil.x, leftPupil.y, frameWidth, frameHeight, fx, leftPupilZ)

    const leftEyeballCenter = getEyeballCenter(faceLandmarks, 'left', frameWidth, frameHeight, fx, leftPupilZ)
    // const leftEyeballCenter = getCanonicalEyeballCenter('left',facialTransformationMatrix,fx,fxFacelandmarker,frameWidth,frameHeight,leftPupilZ)
    const leftGazeCartesian = gazeCartesian(leftEyeballCenter, {...leftPupilXY, z:leftPupilZ});

    // Right eye
    const rightPupilZ = irisDepth(faceLandmarks, 'right', frameWidth, frameHeight, fx)
    const rightPupil = faceLandmarks[eyeLandmarks['right'].pupil]
    const rightPupilXY = landmark2Metric(rightPupil.x, rightPupil.y, frameWidth, frameHeight, fx, rightPupilZ)

    const rightEyeballCenter = getEyeballCenter(faceLandmarks, 'right', frameWidth, frameHeight, fx, rightPupilZ)
    // const rightEyeballCenter = getCanonicalEyeballCenter('right',facialTransformationMatrix,fx,fxFacelandmarker,frameWidth,frameHeight,rightPupilZ)
    const rightGazeCartesian = gazeCartesian(rightEyeballCenter, {...rightPupilXY, z: rightPupilZ});

    // Cyclopean
    const cyclopeanGaze = averageGaze(leftGazeCartesian, rightGazeCartesian)
    const cyclopeanEyePixel = midpoint(faceLandmarks[eyeLandmarks.left.corners.inner], faceLandmarks[eyeLandmarks.right.corners.inner])
    const averageIrisDepth = (leftPupilZ + rightPupilZ)/2
    const cyclopeanEyeDepth = averageIrisDepth+1.175
    const cyclopeanEyeOrigin = landmark2Metric(cyclopeanEyePixel.x, cyclopeanEyePixel.y, frameWidth, frameHeight, fx, cyclopeanEyeDepth)
    const cyclopeanOrigin: Coordinate3D = { ...cyclopeanEyeOrigin, z: averageIrisDepth };

    const screenPog = intersectScreenPlane(cyclopeanOrigin, cyclopeanGaze);
    
    const debug = {
        landmarkPupilLeft: metric2Pixel({...leftPupilXY, z: leftPupilZ}, frameWidth, frameHeight, fx),
        landmarkPupilRight: metric2Pixel({...rightPupilXY, z: rightPupilZ}, frameWidth, frameHeight, fx),
        canonicalPupilLeft: metric2Pixel(leftEyeballCenter, frameWidth, frameHeight, fx),
        canonicalPupilRight: metric2Pixel(rightEyeballCenter, frameWidth, frameHeight, fx)
        // canonicalPupilLeft: leftEyeballCenterPixel,
        // canonicalPupilRight: rightEyeballCenterPixel
    };

    // TODO: calculating point of gaze with intersection of two gaze vectors might be better here
    return {
        left: {
            origin: leftEyeballCenter,
            vector: leftGazeCartesian,
            pixel: denormaliseLandmark(leftPupil, frameWidth, frameHeight)
        },
        right: {
            origin: rightEyeballCenter,
            vector: rightGazeCartesian,
            pixel: denormaliseLandmark(rightPupil, frameWidth, frameHeight)
        },
        cyclopean: {
            origin: cyclopeanOrigin,
            vector: cyclopeanGaze,
            pixel: denormaliseLandmark(cyclopeanEyePixel, frameWidth, frameHeight)
        },
        screenPog,
        fx,
        debug
    };
}


// EXTRACT EYE LOCATIONS --------------------------------------
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

// Eyeball Centre using midpoint between eye corners
function getEyeballCenter(
    landmarks: NormalizedLandmark[],
    side: 'left' | 'right',
    frameWidth: number,
    frameHeight: number,
    fx: number,
    z: number
): Coordinate3D {
    const EYEBALL_AXIAL_RADIUS = 1.175 // https://pmc.ncbi.nlm.nih.gov/articles/PMC4238270/
    const innerEyeCorner = landmarks[eyeLandmarks[side].corners.inner]
    const outerEyeCorner = landmarks[eyeLandmarks[side].corners.outer]
    const eyeMiddle = midpoint(innerEyeCorner, outerEyeCorner)
    const eyeMiddleMetric = landmark2Metric(eyeMiddle.x, eyeMiddle.y, frameWidth, frameHeight, fx, z)
    // Work backwards along head vector

    // Note tried using xyz (idx 8,9,10) from facelandmarker transformation matrix and walking backwards into the head but seems too noisy?
    // TODO: can also get angular rotations from iris: roll is angle between eyes, yaw is size difference, pitch can't get though...
    return {
        x: eyeMiddleMetric.x,
        y: eyeMiddleMetric.y + .1, // TODO: Temp fix as midpoint of eyes is below centre and biases upwards?
        z: z + EYEBALL_AXIAL_RADIUS
    };
}

function walkIntoHead(
    matrix: number[],
    start: Coordinate3D,
    distance: number
): Coordinate3D {
    // Column-major: the Z basis vector of the face's local frame lives in
    // matrix indices 8, 9, 10 (third column, ignoring the w row).
    const zx = matrix[8];
    const zy = matrix[9];
    const zz = matrix[10];

    // Normalize in case of any numerical drift (rotation columns should already be unit length).
    const len = Math.hypot(zx, zy, zz) || 1;
    const nx = zx / len;
    const ny = zy / len;
    const nz = zz / len;

    // "Backwards into the head" = negative local Z direction.
    return {
        x: start.x - nx * distance,
        y: start.y - ny * distance,
        z: start.z - nz * distance,
    };
}

// Eyeball Centre using iris in canonical facemesh estimate
export function getCanonicalEyeballCenter(
    side: 'left' | 'right',
    transformationMatrix: number[],
    fx: number,
    mediapipeFx: number,
    frameWidth: number,
    frameHeight: number,
    z: number
): Coordinate3D {
    // Move from mediapipe's facial transformation matrix at 'center of mass of face' to the canonical iris position at center of eye
    // vertices from: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/face_model_with_iris.obj
    const CANONICAL_VERTICES = {
        left: {x: 3.181751, y: 2.635786, z: 3.826339}, // 473
        right: {x: -3.18175, y: 2.635786, z: 3.826339} // 468
    }
    const v = CANONICAL_VERTICES[side]
    const m = transformationMatrix;
    const eyeCenter3D = {
        x: (m[0]*v.x + m[4]*v.y + m[8]*v.z  + m[12]),
        y: -(m[1]*v.x + m[5]*v.y + m[9]*v.z  + m[13]), // invert y inline with mediapipe convention: up = positive
        z: -(m[2]*v.x + m[6]*v.y + m[10]*v.z + m[14]) // invert z inline with mediapipe convention: far = negative
    };

    // Project to pixel on screen so we can covert to metric the same way as the iris position
    const eyeCenterSurfacePixel: Point = {
        x: mediapipeFx * (eyeCenter3D.x / eyeCenter3D.z) + frameWidth / 2,
        y: mediapipeFx * (eyeCenter3D.y / eyeCenter3D.z) + frameHeight / 2,
    }

    // Convert to 3D metric - assume same depth as iris, should roughly hold
    const eyeCenterIrisMetric = pixel2metric(eyeCenterSurfacePixel.x, eyeCenterSurfacePixel.y, frameWidth, frameHeight, fx, z)
    const eyeCenterIris3D = {...eyeCenterIrisMetric, z}
    // to be honest just adding 1.175 to z works well ... maybe better?
    return walkIntoHead(transformationMatrix, eyeCenterIris3D, -1.175);
}


// GAZE CALCULATION --------------------------------------
function intersectScreenPlane(
    origin: Coordinate3D,
    gaze: Coordinate3D,
): Point | null {
    // This is the point in cm where the gaze ray intersects z=0
    // Screen is at z = 0. Ray: p = origin + t * gaze. Solve origin.z + t*gaze.z = 0.
    if (Math.abs(gaze.z) < 1e-6) return null;
    const t = -origin.z / gaze.z;
    if (t <= 0) return null; // looking away from screen
    return {
        x: origin.x + t * gaze.x,
        y: origin.y + t * gaze.y,
    };
}

function averageGaze(a: Coordinate3D, b: Coordinate3D) {
    const gx = a.x + b.x;
    const gy = a.y + b.y;
    const gz = a.z + b.z;
    const length = Math.sqrt(gx*gx + gy*gy + gz*gz);
    return {x: gx/length, y: gy/length, z: gz/length};
}

function gazeCartesian(
    eyeballCenter: {x: number; y: number; z: number},
    pupil: {x: number; y: number; z: number},
): {x: number; y: number; z: number} {
    const dx = pupil.x - eyeballCenter.x;
    const dy = pupil.y - eyeballCenter.y;
    const dz = pupil.z - eyeballCenter.z;
    const length = Math.sqrt(dx*dx + dy*dy + dz*dz);
    return {
        x: dx / length,
        y: dy / length,
        z: dz / length,
    };
}


// LANDMARK UTILITIES --------------------------------------

function denormaliseLandmark(landmark: Point, width: number, height: number): Point {
    // Note Point {x,y} is technically a subset of NormalizedLandmark {x,y,z} from FaceLandmarker
    return {
        x: landmark.x * width,
        y: landmark.y * height,
    }
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


// PIXEL/METRIC CONVERSION ----------------------------------

function pixel2metric(
    x: number,
    y: number,
    frameWidth: number,
    frameHeight: number,
    fx: number,
    z: number
): Point {
    return {
        x: ((x-(frameWidth/2)) / fx) * z,
        y: ((-y+(frameHeight/2)) / fx) * z,
    }
}


function landmark2Metric(
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

export function metric2Pixel(
    p: Coordinate3D,
    frameWidth: number,
    frameHeight: number,
    fx: number,
): Point {
    const pixelX = (p.x / p.z) * fx;
    const pixelY = (p.y / p.z) * fx;
    return {
        x: pixelX + frameWidth / 2,
        y: -pixelY + frameHeight / 2,
    };
}