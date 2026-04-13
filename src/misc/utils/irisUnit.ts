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
    const leftPupil3D = landmark2Metric(leftPupil.x, leftPupil.y, frameWidth, frameHeight, fx, leftPupilZ)

    const leftEyeballCenter = getEyeballCenter(faceLandmarks, 'left', frameWidth, frameHeight, fx, leftPupilZ)
    // const leftEyeballCenter = getCanonicalEyeballCenter('left',facialTransformationMatrix,fx,fxFacelandmarker,frameWidth,frameHeight,leftPupilZ)
    const leftGazeCartesian = gazeCartesian(leftEyeballCenter, leftPupil3D);

    // Right eye
    const rightPupilZ = irisDepth(faceLandmarks, 'right', frameWidth, frameHeight, fx)
    const rightPupil = faceLandmarks[eyeLandmarks['right'].pupil]
    const rightPupil3D = landmark2Metric(rightPupil.x, rightPupil.y, frameWidth, frameHeight, fx, rightPupilZ)
    const rightEyeballCenter = getEyeballCenter(faceLandmarks, 'right', frameWidth, frameHeight, fx, rightPupilZ)
    // const rightEyeballCenter = getCanonicalEyeballCenter('right',facialTransformationMatrix,fx,fxFacelandmarker,frameWidth,frameHeight,rightPupilZ)
    const rightGazeCartesian = gazeCartesian(rightEyeballCenter, rightPupil3D);

    // Cyclopean
    const cyclopeanGaze = averageGaze(leftGazeCartesian, rightGazeCartesian)
    const cyclopeanOrigin: Coordinate3D = {
        x: (leftEyeballCenter.x + rightEyeballCenter.x) / 2,
        y: (leftEyeballCenter.y + rightEyeballCenter.y) / 2,
        z: (leftEyeballCenter.z + rightEyeballCenter.z) / 2,
    };

    const screenPog = intersectScreenPlane(cyclopeanOrigin, cyclopeanGaze);

    const debug = {
        landmarkPupilLeft: metric2Pixel(leftPupil3D, frameWidth, frameHeight, fx),
        landmarkPupilRight: metric2Pixel(rightPupil3D, frameWidth, frameHeight, fx),
        canonicalPupilLeft: metric2Pixel(leftEyeballCenter, frameWidth, frameHeight, fx),
        canonicalPupilRight: metric2Pixel(rightEyeballCenter, frameWidth, frameHeight, fx),
        // canonicalPupilLeft: leftEyeballCenterPixel,
        // canonicalPupilRight: rightEyeballCenterPixel
        headAxes: computeHeadAxesDisplay(facialTransformationMatrix, frameWidth, frameHeight, fxFacelandmarker)
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
            pixel: metric2Pixel(cyclopeanOrigin, frameWidth, frameHeight, fx)
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
}

function irisDepthWidthHeight(
    landmarks: NormalizedLandmark[],
    side: 'left' | 'right',
    frameWidth: number,
    frameHeight: number,
    fx: number,
): {z: number, width: number, height: number} {
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

    // TODO: why not return x,y here, - y is negative down x is negative right
    // const pxPerCm = apparentDiameterPx/IRIS_DIAMETER_CM
    const frameWidthCm = (frameWidth/apparentDiameterPx)*IRIS_DIAMETER_CM
    const frameHeightCm = (frameHeight/apparentDiameterPx)*IRIS_DIAMETER_CM
    // Depth is based on focal length
    return {
        z: (fx * IRIS_DIAMETER_CM) / apparentDiameterPx, // flip for same coordinates as facelandmarker
        width: frameWidthCm,
        height: frameHeightCm
    }
}

function landmark2cm(landmark: Point, frameWidthCm: number, frameHeightCm: number): Point {
    return {
        x: (landmark.x-.5) * frameWidthCm,
        y: (-landmark.y+.5) * frameHeightCm
    }
}

function getEyeballCenterNew(
    landmarks: NormalizedLandmark[],
    side: 'left' | 'right',
    frameWidth: number,
    frameHeight: number,
    fx: number,
    z: number,                          // iris depth from irisDepth()
    transformationMatrix: number[],
): Coordinate3D {
    const EYEBALL_AXIAL_RADIUS = 1.175;
    const HALF_CORNER_SEPARATION = 1.5;
    const CORNER_PLANE_OFFSET = 1.0;

    const innerCorner = landmarks[eyeLandmarks[side].corners.inner];
    const outerCorner = landmarks[eyeLandmarks[side].corners.outer];
    const m = transformationMatrix;
    const headX = { x: m[0], y: m[1], z: m[2] };
    const headZ = { x: m[8], y: m[9], z: m[10] };
    const zBase = z + CORNER_PLANE_OFFSET * Math.abs(headZ.z);
    const cornerHalfDepth = HALF_CORNER_SEPARATION * headX.z;
    const innerSign = side === 'left' ? -1 : +1;

    const zInner = zBase + innerSign * cornerHalfDepth;
    const zOuter = zBase - innerSign * cornerHalfDepth;

    // Back-project each corner at its own depth.
    const innerMetric = landmark2Metric(
        innerCorner.x, innerCorner.y, frameWidth, frameHeight, fx, zInner
    );
    const outerMetric = landmark2Metric(
        outerCorner.x, outerCorner.y, frameWidth, frameHeight, fx, zOuter
    );

    // 3D midpoint on the corner plane.
    const midpoint3D: Coordinate3D = {
        x: (innerMetric.x + outerMetric.x) / 2,
        y: (innerMetric.y + outerMetric.y) / 2,
        z: (zInner + zOuter) / 2,
    };

    return {
        x: midpoint3D.x - headZ.x * EYEBALL_AXIAL_RADIUS,
        y: midpoint3D.y - headZ.y * EYEBALL_AXIAL_RADIUS,
        z: midpoint3D.z - headZ.z * EYEBALL_AXIAL_RADIUS,
    };
}

// Eyeball Centre using midpoint between eye corners
function getEyeballCenterCm(
    landmarks: NormalizedLandmark[],
    side: 'left' | 'right',
    frameWidthCm: number,
    frameHeightCm: number,
    z: number,
): Coordinate3D {
    // This is kinda nuts but also kinda works?
    const EYEBALL_AXIAL_RADIUS = 1.175 // https://pmc.ncbi.nlm.nih.gov/articles/PMC4238270/
    const innerEyeCorner = landmarks[eyeLandmarks[side].corners.inner]
    const outerEyeCorner = landmarks[eyeLandmarks[side].corners.outer]
    const eyeMiddle = midpoint(innerEyeCorner, outerEyeCorner)
    const eyeMiddle3D = landmark2cm(eyeMiddle, frameWidthCm, frameHeightCm)
    // Work backwards along head vector

    // Note tried using xyz (idx 8,9,10) from facelandmarker transformation matrix and walking backwards into the head but seems too noisy?
    // TODO: can also get angular rotations from iris: roll is angle between eyes, yaw is size difference, pitch can't get though...
    // return walkIntoHead(transformationMatrix, eyeMiddle3D, EYEBALL_AXIAL_RADIUS)
    return {...eyeMiddle3D, z:z+EYEBALL_AXIAL_RADIUS};
}

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
    const eyeMiddleMetric = landmark2Metric(eyeMiddle.x, eyeMiddle.y, frameWidth, frameHeight, fx, z+ EYEBALL_AXIAL_RADIUS)
    // Work backwards along head vector

    // Note tried using xyz (idx 8,9,10) from facelandmarker transformation matrix and walking backwards into the head but seems too noisy?
    // TODO: can also get angular rotations from iris: roll is angle between eyes, yaw is size difference, pitch can't get though...
    return {
        ...eyeMiddleMetric,
        y: eyeMiddleMetric.y + .1, // TODO: Temp fix as midpoint of eyes is below centre and biases upwards?
    };
}

function walkIntoHead(
    matrix: number[],
    start: Coordinate3D,
    distance: number
): Coordinate3D {
    const fx_basis =  matrix[8];
    const fy_basis =  matrix[9];
    const fz_basis = -matrix[10];

    const len = Math.hypot(fx_basis, fy_basis, fz_basis) || 1;
    const nx = fx_basis / len;
    const ny = fy_basis / len;
    const nz = fz_basis / len;

    // "Backwards into the head" = negative head-forward direction.
    // distance > 0 means walk backward by `distance` cm.
    return {
        x: start.x - nx * distance,
        y: start.y - ny * distance* 0.5,
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
    // TODO: weird routine here - couldn't figure out how to just -eyeball_radius in canon position, and project to same metric space as my iris estimate...
    const EYEBALL_AXIAL_RADIUS = 1.175;

    // Canonical iris SURFACE vertices (473 left, 468 right) from
    // face_model_with_iris.obj. We do NOT walk into the head in canonical
    // space — we want the iris surface pixel, then add the axial offset to
    // z at the end (matching the eye-corner-midpoint approach).
    const CANONICAL_IRIS = {
        left:  { x:  3.181751, y: 2.635786, z: 3.826339 },
        right: { x: -3.181751, y: 2.635786, z: 3.826339 },
    };
    const v = CANONICAL_IRIS[side];
    const m = transformationMatrix;

    // Canonical -> MediaPipe camera space, applying MediaPipe's y/z sign
    // convention (negate y and z so the projection formula below matches
    // the version in the original code that was visually validated).
    const eyeCenter3D = {
        x:  (m[0]*v.x + m[4]*v.y + m[8]*v.z  + m[12]),
        y: -(m[1]*v.x + m[5]*v.y + m[9]*v.z  + m[13]),
        z: -(m[2]*v.x + m[6]*v.y + m[10]*v.z + m[14]),
    };

    // Project to pixel under MediaPipe's intrinsics.
    const eyeCenterSurfacePixel: Point = {
        x: mediapipeFx * (eyeCenter3D.x / eyeCenter3D.z) + frameWidth / 2,
        y: mediapipeFx * (eyeCenter3D.y / eyeCenter3D.z) + frameHeight / 2,
    };

    // Back-project to metric using our fx at the iris depth — this puts the
    // point in the same coordinate frame as the pupil from landmark2Metric.
    const eyeCenterIrisMetric = pixel2metric(eyeCenterSurfacePixel.x, eyeCenterSurfacePixel.y, frameWidth, frameHeight, fx, z);

    // return {...eyeCenterIrisMetric, z: z+EYEBALL_AXIAL_RADIUS}
    return walkIntoHead(transformationMatrix, eyeCenterIrisMetric, EYEBALL_AXIAL_RADIUS);
}

// HOLDING THESE HERE FOR ANALYSIS, some ideas on how to properly extract eyeball centre
export function getCanonicalEyeballCenterV2(
    side: 'left' | 'right',
    transformationMatrix: number[],
    fx: number,
    mediapipeFx: number,
    frameWidth: number,   // unused
    frameHeight: number,  // unused
    z: number
): Coordinate3D {
    void frameWidth; void frameHeight;

    const EYEBALL_AXIAL_RADIUS = 1.175;
    const CANONICAL_IRIS = {
        left:  { x:  3.181751, y: 2.635786, z: 3.826339 - EYEBALL_AXIAL_RADIUS },
        right: { x: -3.181751, y: 2.635786, z: 3.826339 - EYEBALL_AXIAL_RADIUS },
    };
    const v = CANONICAL_IRIS[side];
    const m = transformationMatrix;

    const mpX = m[0]*v.x + m[4]*v.y + m[8]*v.z  + m[12];
    const mpY = m[1]*v.x + m[5]*v.y + m[9]*v.z  + m[13];
    const mpZ = m[2]*v.x + m[6]*v.y + m[10]*v.z + m[14];

    const k = (mediapipeFx / fx) * (z / -mpZ);

    return {
        x: mpX * k,
        y: mpY * k,
        z: z,
    };
}

export function getCanonicalEyeballCenterV3(
    side: 'left' | 'right',
    transformationMatrix: number[],
    mediapipeFx: number,
    frameWidth: number,
    frameHeight: number,
    z: number,
    pupilMetric: { x: number; y: number },  // pass in the pupil's metric x,y from landmark2Metric
): Coordinate3D {
    void mediapipeFx; void frameWidth; void frameHeight;

    const EYEBALL_AXIAL_RADIUS = 1.175;
    const CANONICAL_IRIS_SURFACE = {
        left:  { x:  3.181751, y: 2.635786, z: 3.826339 },
        right: { x: -3.181751, y: 2.635786, z: 3.826339 },
    };
    const irisSurf = CANONICAL_IRIS_SURFACE[side];
    const eyeCenter = {
        x: irisSurf.x,
        y: irisSurf.y,
        z: irisSurf.z - EYEBALL_AXIAL_RADIUS,
    };

    const m = transformationMatrix;

    // Project both points through the matrix into MediaPipe space.
    const irisMpX = m[0]*irisSurf.x + m[4]*irisSurf.y + m[8]*irisSurf.z  + m[12];
    const irisMpY = m[1]*irisSurf.x + m[5]*irisSurf.y + m[9]*irisSurf.z  + m[13];
    const irisMpZ = m[2]*irisSurf.x + m[6]*irisSurf.y + m[10]*irisSurf.z + m[14];

    const ecMpX = m[0]*eyeCenter.x + m[4]*eyeCenter.y + m[8]*eyeCenter.z  + m[12];
    const ecMpY = m[1]*eyeCenter.x + m[5]*eyeCenter.y + m[9]*eyeCenter.z  + m[13];
    const ecMpZ = m[2]*eyeCenter.x + m[6]*eyeCenter.y + m[10]*eyeCenter.z + m[14];

    // Relative offset from iris to eyeball center, in MediaPipe camera space.
    // This is purely a rotation of (0, 0, -EYEBALL_AXIAL_RADIUS) in canonical
    // space, so it has length EYEBALL_AXIAL_RADIUS in cm — no focal-length
    // scaling needed because it's already a metric 3D vector in physical cm.
    // The rotation is the same in any pinhole camera frame; the matrix's
    // translation cancels out in the subtraction.
    const dx = ecMpX - irisMpX;
    const dy = ecMpY - irisMpY;
    const dz = ecMpZ - irisMpZ;  // negative when face looks at camera (eyeball is further from camera)

    // Convert to our space: +Y up matches, but our +Z points AWAY from the
    // camera (positive forward) while MediaPipe's +Z points TOWARD the
    // camera (face is at negative z). So flip the sign of dz for our frame.
    const offset = { x: dx, y: dy, z: -dz };

    // Anchor to the pupil in our space and add the offset. The pupil's
    // metric (x, y) at depth z is already in our `fx`-based frame, so
    // adding a physical-cm offset gives us the eyeball center in the
    // same frame, with the correct axial depth difference preserved.
    return {
        x: pupilMetric.x + offset.x,
        y: pupilMetric.y + offset.y,
        z: z + offset.z,
    };
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
): Coordinate3D {
    return {
        x: ((x-(frameWidth/2)) / fx) * z,
        y: ((-y+(frameHeight/2)) / fx) * z,
        z
    }
}


function landmark2Metric(
    x: number,
    y: number,
    frameWidth: number,
    frameHeight: number,
    fx: number,
    z: number
): Coordinate3D {
    // x and y relative to centre of camera at current depth using number of irises
    // centered and unnormalized
    const pixelX = (x - 0.5) * frameWidth;
    const pixelY = (-y + 0.5) * frameHeight; // Flip for same coordinates as facelandmarker
    return {
        x: (pixelX / fx) * z,
        y: (pixelY / fx) * z,
        z
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

// Compute head axes for display - just used for a sense check -------
export function computeHeadAxesDisplay(
    transformationMatrix: ArrayLike<number>,
    frameWidth: number,
    frameHeight: number,
    fx: number,
    axisLengthCm = 4,
    eyeballCenter: boolean = true,
    side: 'left'|'right' = 'right',
): { origin: Point; xAxis: Point; yAxis: Point; zAxis: Point } {
    const m = transformationMatrix;

    let ox = m[12];
    let oy = m[13];
    let oz = m[14];
    if(eyeballCenter){
        // Origin of the head frame in camera space = translation column.
        const EYEBALL_AXIAL_RADIUS = 1.175;
        const CANONICAL_IRIS = {
            left:  { x:  3.181751, y: 2.635786, z: 3.826339-EYEBALL_AXIAL_RADIUS },
            right: { x: -3.181751, y: 2.635786, z: 3.826339-EYEBALL_AXIAL_RADIUS },
        };
        const v = CANONICAL_IRIS[side];

        // Canonical -> MediaPipe camera space, applying MediaPipe's y/z sign
        // convention (negate y and z so the projection formula below matches
        // the version in the original code that was visually validated).
        const eyeCenter3D = {
            x:  (m[0]*v.x + m[4]*v.y + m[8]*v.z  + m[12]),
            y: (m[1]*v.x + m[5]*v.y + m[9]*v.z  + m[13]),
            z: (m[2]*v.x + m[6]*v.y + m[10]*v.z + m[14]),
        };

        ox = eyeCenter3D.x//m[12];
        oy = eyeCenter3D.y//m[13];
        oz = eyeCenter3D.z//m[14];
    }


    // Tips of each axis = origin + axisLength * (rotation column).
    // Column 0 = model X in camera space, column 1 = model Y, column 2 = model Z.
    const xTip: [number, number, number] = [ox + m[0]*axisLengthCm, oy + m[1]*axisLengthCm, oz + m[2]*axisLengthCm];
    const yTip: [number, number, number] = [ox + m[4]*axisLengthCm, oy + m[5]*axisLengthCm, oz + m[6]*axisLengthCm];
    const zTip: [number, number, number] = [ox + m[8]*axisLengthCm, oy + m[9]*axisLengthCm, oz + m[10]*axisLengthCm];

    const project = (cx: number, cy: number, cz: number): Point => {
        const depth = -cz;
        // +y-down projection, directly to canvas coords.
        return {
            x: fx * (cx / depth) + frameWidth / 2,
            y: fx * (-cy / depth) + frameHeight / 2,
        };
    };

    const origin = project(ox, oy, oz);
    const xAxis = project(xTip[0], xTip[1], xTip[2]);
    const yAxis = project(yTip[0], yTip[1], yTip[2]);
    const zAxis = project(zTip[0], zTip[1], zTip[2]);

    return { origin, xAxis, yAxis, zAxis };
}
