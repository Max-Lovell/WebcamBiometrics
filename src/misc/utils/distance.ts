import type {FaceLandmarkerResult, NormalizedLandmark} from "@mediapipe/tasks-vision";

export function miscStuff(faceLandmarkerResult: FaceLandmarkerResult, frameWidth: number, frameHeight: number){
    const landmarks = faceLandmarkerResult.faceLandmarks[0];
    // const altZ = getAlternativeDepths(faceLandmarkerResult, frameWidth, frameHeight)
    // Can get face and iris distance from this easily.
    return irisDistance(landmarks, frameWidth, frameHeight);
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

export function irisDistance(
    landmarks: NormalizedLandmark[],
    frameWidth: number,
    frameHeight: number,
    focalLengthPx: number | null = null,
): number | null {
    // https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/
    const IRIS_DIAMETER_CM = 1.17; // HVID
    // Fall back to a rough focal length estimate if none provided
    const fx = focalLengthPx ?? Math.max(frameWidth, frameHeight);

    const rightH = landmarkDistancePx(landmarks[469], landmarks[471], frameWidth, frameHeight);
    const rightV = landmarkDistancePx(landmarks[470], landmarks[472], frameWidth, frameHeight);
    const leftH = landmarkDistancePx(landmarks[474], landmarks[476], frameWidth, frameHeight);
    const leftV = landmarkDistancePx(landmarks[475], landmarks[477], frameWidth, frameHeight);

    const apparentDiameterPx = Math.max(rightH, rightV, leftH, leftV);

    if (apparentDiameterPx < 1) return null; // too small / degenerate
    return (fx * IRIS_DIAMETER_CM) / apparentDiameterPx;
}

// ------------
// Alternative distance calculations
// -------------
function getEuclidianDistance(x: number, y: number, z: number): number {
    return Math.sqrt(x*x + y*y + z*z);
}

// https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/geometry_pipeline_metadata_including_iris_landmarks.pbtxt
    // procrustes_landmark_basis with highest weight are probably most stable landmarks…
export function cyclopeanEyeMetric(matrix: ArrayLike<number>): number | null {
    // See iris files here for cyclopean eye point: https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/data
    // projecting a canonical midpoint of eyes through the matrix
    const CYCLOPEAN_CANONICAL = [0.0, 2.585245, 3.757904]
    const [cx, cy, cz] = CYCLOPEAN_CANONICAL;
    const z = matrix[2] * cx + matrix[6] * cy + matrix[10] * cz + matrix[14];
    return Math.abs(z);
}

export function getAlternativeDepths(faceLandmarkerResult: FaceLandmarkerResult, frameWidth: number, frameHeight: number){
    // See WebEyeTrack for a Proscutes analysis approach estimated using iris ratio to face width
    const transformation = faceLandmarkerResult.facialTransformationMatrixes[0].data
    const distanceCalculators = {
        iris: irisDistance(faceLandmarkerResult.faceLandmarks[0], frameWidth, frameHeight)?.toFixed(2),
        euclid: getEuclidianDistance(transformation[12], transformation[13], transformation[14]).toFixed(2),
        faceTransZ: Math.abs(transformation[14]).toFixed(2),
        canonical: cyclopeanEyeMetric(transformation)?.toFixed(2),
    }
    console.log({distanceCalculators})
    return distanceCalculators;
    // Can get face and iris distance from this easily.
}
