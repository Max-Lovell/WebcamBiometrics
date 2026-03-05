// ============================================================================
// Face Origin and Head Vector
// ============================================================================
import { Matrix, inverse} from 'ml-matrix'; // TODO: this in faceReconstruction and computeFaceOrigin3D probably should use preallocated mats
import type {Matrix as MediaPipeMatrix} from '@mediapipe/tasks-vision';
import type {Point} from '../types.ts';

// Depth radial parameters
const MAX_STEP_CM = 5

// According to https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/graphs/face_effect/face_effect_gpu.pbtxt#L61-L65
const VERTICAL_FOV_DEGREES = 60;
const NEAR = 1.0; // 1cm
const FAR = 10000; // 100m

// Used to determine the width of the face
const LEFTMOST_LANDMARK = 356
const RIGHTMOST_LANDMARK = 127
const RIGHT_IRIS_LANDMARKS = [468, 470, 469, 472, 471] // center, top, right, bottom, left
const LEFT_IRIS_LANDMARKS = [473, 475, 474, 477, 476] // center, top, right, bottom, left
const AVERAGE_IRIS_SIZE_CM = 1.2;
const LEFT_EYE_HORIZONTAL_LANDMARKS = [362, 263]
const RIGHT_EYE_HORIZONTAL_LANDMARKS = [33, 133]
// const ORIGIN_POINT_LOCATION = 'BOTTOM_LEFT_CORNER';


export function translateMatrix(
    matrix: MediaPipeMatrix,
): Matrix {
    // Convert MediaPipeMatrix to ml-matrix format
    const data = matrix.data;
    const translatedMatrix = new Matrix(matrix.rows, matrix.columns);
    for (let i = 0; i < matrix.rows; i++) {
        for (let j = 0; j < matrix.columns; j++) {
            translatedMatrix.set(i, j, data[i * matrix.columns + j]);
        }
    }
    return translatedMatrix;
}

export function createPerspectiveMatrix(aspectRatio: number): Matrix {
    // Creates 4*4 projection matrix,
    // maps camera coordinates to Normalized Device Coordinates (NDC: cube from -1 to 1).
    // used for faceReconstruction to un-project 2D landmarks back into 3D metric space.
    // TODO: cleanup how many new matrices are created here to stop triggering GC? preallocate into single ImageData buffer using data.set()

    const kDegreesToRadians = Math.PI / 180.0;

    // Standard perspective projection matrix calculations
    // VERTICAL_FOV_DEGREES = 60. f estimate.
    const f = 1.0 / Math.tan(kDegreesToRadians * VERTICAL_FOV_DEGREES / 2.0);
    const denom = 1.0 / (NEAR - FAR); // Maps depth so objects near=-1 vs far=100, so reconstruct depth

    // Create and populate the matrix
    const perspectiveMatrix = new Matrix(4, 4).fill(0);
    // X,Y here flattens the camera cone of vision into a square.
    perspectiveMatrix.set(0, 0, f / aspectRatio); //X is scaled by f/aspectRatio
    perspectiveMatrix.set(1, 1, f); // Y scaled by f
    perspectiveMatrix.set(2, 2, (NEAR + FAR) * denom);
    perspectiveMatrix.set(2, 3, -1.0);
    perspectiveMatrix.set(3, 2, 2.0 * FAR * NEAR * denom);

    return perspectiveMatrix;
}

export function createIntrinsicsMatrix(
    width: number, height: number,
    fovX?: number // in degrees
): Matrix {
    // Intrinsics Matrix is a computer Vision matrix (K) describing internal properties of the camera (focal length/ optical center)
    // Maps 3D camera coordinates (X,Y,Z) to 2D image pixel coordinates (u, v)
    // Note K is not really necessary for eye patch warp, but is for 3d head vector/face origin
    const w = width;
    const h = height;

    // Camera center
    const cX = w / 2;
    const cY = h / 2;

    // Estimate camera FOV - note fovX isn't passed in from tracker so f always = w (standard estimate)
    let fX: number, fY: number;
    if (fovX !== undefined) {
        const fovXRad = (fovX * Math.PI) / 180;
        fX = w / (2 * Math.tan(fovXRad / 2));
        fY = fX; // Assume square pixels
    } else {
        fX = fY = w; // Fallback estimate - note though VERTICAL_FOV_DEGREES=60 in perspectiveMat
    }

    // Construct the intrinsic matrix
    const K = new Matrix([
        [fX, 0, cX],
        [0, fY, cY],
        [0, 0, 1],
    ]);

    return K;
}

function distance2D(p1: number[], p2: number[]): number {
    const dx = p1[0] - p2[0];
    const dy = p1[1] - p2[1];
    return Math.sqrt(dx * dx + dy * dy);
}

export function estimateFaceWidth(
    faceLandmarks: Point[],
): number {
    // Estimate the size of the face by how many irises it is
    const irisDist: number[] = [];

    for (const side of ['left', 'right']) {
        const eyeIrisLandmarks = side === 'left' ? LEFT_IRIS_LANDMARKS : RIGHT_IRIS_LANDMARKS;
        const leftmost = faceLandmarks[eyeIrisLandmarks[4]].slice(0, 2);
        const rightmost = faceLandmarks[eyeIrisLandmarks[2]].slice(0, 2);
        const horizontalDist = distance2D(leftmost, rightmost);
        irisDist.push(horizontalDist);
    }

    const avgIrisDist = irisDist.reduce((a, b) => a + b, 0) / irisDist.length;

    const leftmostFace = faceLandmarks[LEFTMOST_LANDMARK];
    const rightmostFace = faceLandmarks[RIGHTMOST_LANDMARK];
    const faceWidthPx = distance2D(leftmostFace, rightmostFace);

    const faceIrisRatio = avgIrisDist / faceWidthPx;
    const faceWidthCm = AVERAGE_IRIS_SIZE_CM / faceIrisRatio;

    return faceWidthCm;
}

export function convertUvToXyz(
    perspectiveMatrix: Matrix,
    u: number,
    v: number,
    zRelative: number
): [number, number, number] {
    // Unprojects pixel back into the world

    // Step 1: Convert to Normalized Device Coordinates (NDC)
    const ndcX = 2 * u - 1;
    const ndcY = 1 - 2 * v;

    // Step 2: Create NDC point in homogeneous coordinates
    const ndcPoint = new Matrix([[ndcX], [ndcY], [-1.0], [1.0]]);

    // Step 3: Invert the perspective matrix
    const invPerspective = inverse(perspectiveMatrix);

    // Step 4: Multiply to get world point in homogeneous coords
    const worldHomogeneous = invPerspective.mmul(ndcPoint);

    // Step 5: Dehomogenize - Divide resulting xy by w to get coordinates in camera space.
    const w = worldHomogeneous.get(3, 0);
    const x = worldHomogeneous.get(0, 0) / w;
    const y = worldHomogeneous.get(1, 0) / w;
    // const z = worldHomogeneous.get(2, 0) / w;

    // Step 6: Scale using the provided zRelative
    // apply zRelative (depth in cm) to place the point at the correct distance from camera
    const xRelative = -x; // negated to match original convention
    const yRelative = y;
    // zRelative stays as-is (external input)

    return [xRelative, yRelative, zRelative];
}

/**
 * Convert UVZ coordinates to XYZ using pre-computed inverse perspective matrix.
 *
 * This is an optimized version of convertUvToXyz() that accepts a pre-inverted
 * perspective matrix, eliminating redundant matrix inversions when processing
 * multiple landmarks.
 *
 * **Performance**: When processing N landmarks, this approach computes the matrix
 * inverse once instead of N times, providing ~N-fold speedup for the inversion
 * operation (O(n³) complexity for 4x4 matrices).
 *
 * **Accuracy**: Mathematically equivalent to convertUvToXyz(). Computing the
 * inverse once actually improves numerical stability by avoiding accumulation
 * of rounding errors from multiple inversion operations.
 *
 * @param invPerspective - Pre-inverted 4x4 perspective matrix
 * @param u - Normalized horizontal coordinate [0, 1]
 * @param v - Normalized vertical coordinate [0, 1]
 * @param zRelative - Relative depth value
 * @returns 3D coordinates [xRelative, yRelative, zRelative]
 *
 * @example
 * ```typescript
 * // Efficient processing of multiple landmarks
 * const invPerspective = inverse(perspectiveMatrix);
 * const xyzCoords = faceLandmarks.map(([u, v]) =>
 *   convertUvToXyzWithInverse(invPerspective, u, v, initialZ)
 * );
 * ```
 *
 * @see convertUvToXyz - Original function (kept for backwards compatibility)
 * @see MATRIX_INVERSION_ANALYSIS.md - Detailed performance analysis
 */
export function convertUvToXyzWithInverse(
    invPerspective: Matrix,
    u: number,
    v: number,
    zRelative: number
): [number, number, number] {
    // Step 1: Convert to Normalized Device Coordinates (NDC)
    const ndcX = 2 * u - 1;
    const ndcY = 1 - 2 * v;

    // Step 2: Create NDC point in homogeneous coordinates
    const ndcPoint = new Matrix([[ndcX], [ndcY], [-1.0], [1.0]]);

    // Step 3: Use pre-inverted matrix (OPTIMIZATION: skip inversion)
    // This is the key optimization - matrix is already inverted by caller

    // Step 4: Multiply to get world point in homogeneous coords
    const worldHomogeneous = invPerspective.mmul(ndcPoint);

    // Step 5: Dehomogenize
    const w = worldHomogeneous.get(3, 0);
    const x = worldHomogeneous.get(0, 0) / w;
    const y = worldHomogeneous.get(1, 0) / w;
    // const z = worldHomogeneous.get(2, 0) / w;

    // Step 6: Scale using the provided zRelative
    const xRelative = -x; // negated to match original convention
    const yRelative = y;
    // zRelative stays as-is (external input)

    return [xRelative, yRelative, zRelative];
}

export function imageShiftTo3D(shift2d: [number, number], depthZ: number, K: Matrix): [number, number, number] {
    const fx = K.get(0, 0);
    const fy = K.get(1, 1);

    const dx3D = shift2d[0] * (depthZ / fx);
    const dy3D = shift2d[1] * (depthZ / fy);

    return [dx3D, dy3D, 0.0];
}

export function transform3DTo3D(
    point: [number, number, number],
    rtMatrix: Matrix
): [number, number, number] {
    const homogeneous = [point[0], point[1], point[2], 1];
    const result = rtMatrix.mmul(Matrix.columnVector(homogeneous)).to1DArray();
    return [result[0], result[1], result[2]];
}


export function transform3DTo2D(
    point3D: [number, number, number],
    K: Matrix
): [number, number] {
    // Flatten 3d point - multiply xyz by intrinsics matrix (k)

    const eps = 1e-6;
    const [x, y, z] = point3D;
    // divide by z so xy smaller if further away and round to nearest pixel

    const projected = K.mmul(Matrix.columnVector([x, y, z])).to1DArray();

    const zVal = Math.abs(projected[2]) < eps ? eps : projected[2];
    const u = Math.round(projected[0] / zVal);
    const v = Math.round(projected[1] / zVal);

    return [u, v];
}


export function partialProcrustesTranslation2D(
    canonical2D: [number, number][],
    detected2D: [number, number][]
): [number, number] {
    const [cx, cy] = canonical2D[4];
    const [dx, dy] = detected2D[4];
    return [dx - cx, dy - cy];
}

export function refineDepthByRadialMagnitude(
    finalProjectedPts: [number, number][],
    detected2D: [number, number][],
    oldZ: number,
    // alpha = 0.5
): number {
    // Compare spread of points, if model is smaller than face, must be further away
    const numPts = finalProjectedPts.length;

    // Compute centroid of detected 2D
    const detectedCenter = detected2D.reduce(
        (acc, [x, y]) => [acc[0] + x / numPts, acc[1] + y / numPts],
        [0, 0]
    );

    let totalDistance = 0;

    for (let i = 0; i < numPts; i++) {
        const p1 = finalProjectedPts[i];
        const p2 = detected2D[i];

        const v: [number, number] = [p2[0] - p1[0], p2[1] - p1[1]];
        const vNorm = Math.hypot(v[0], v[1]);

        const c: [number, number] = [detectedCenter[0] - p1[0], detectedCenter[1] - p1[1]];
        const dotProduct = v[0] * c[0] + v[1] * c[1];

        totalDistance += dotProduct < 0 ? -vNorm : vNorm;
    }

    const distancePerPoint = totalDistance / numPts;
    const delta = 1e-1 * distancePerPoint;
    const safeDelta = Math.max(-MAX_STEP_CM, Math.min(MAX_STEP_CM, delta));

    const newZ = oldZ + safeDelta;
    return newZ;
}

export function faceReconstruction(
    perspectiveMatrix: Matrix,
    faceLandmarks: [number, number][],
    faceRT: Matrix,
    intrinsicsMatrix: Matrix,
    faceWidthCm: number,
    videoWidth: number,
    videoHeight: number,
    initialZGuess = 60
): [Matrix, [number, number, number][]] {
    // PERFORMANCE OPTIMIZATION: Perspective matrix is constant across all 478 landmarks.
    // Computing inverse once instead of 478 times provides ~10-50x speedup for this
    // operation with zero impact on accuracy. See MATRIX_INVERSION_ANALYSIS.md for details.

    // Model-Based Pose Solver - Uses cm face width to estimate depth and predict angle of gaze/head
    // take flat 2D landmarks from camera, find where a physical ?cm-wide human face would have to be standing in 3D space to create that 2D image.
    // Uses virtual "canonical" face and moves around until lines up with camera video.

    // unproject 2D landmarks (u,v) into 3D using the inverse perspective matrix.
    const invPerspective = inverse(perspectiveMatrix); // returns 3d mesh floating in space
    // Step 1: Convert UVZ to XYZ using pre-inverted matrix (OPTIMIZED)
    const relativeFaceMesh = faceLandmarks.map(([u, v]) => convertUvToXyzWithInverse(invPerspective, u, v, initialZGuess));

    // Step 2: Center to nose (index 4 is assumed nose) - shift facemesh so nose is at (0,0,0)
    const nose = relativeFaceMesh[4];
    const centered = relativeFaceMesh.map(([x, y, z]) => [-(x - nose[0]), -(y - nose[1]), z - nose[2]]) as [number, number, number][];

    // Step 3: Normalize by width - measure width of mesh and force to faceWidthCm to get cm
    const left = centered[LEFTMOST_LANDMARK];
    const right = centered[RIGHTMOST_LANDMARK];
    // Gets 3D distance between left/rightmost points of face mesh
    const euclideanDistance = Math.hypot(
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2]
    );
    // Make face 1 unit wide by dividing each coordinate by xyz, multiply by face width for cm
    const normalized = centered.map(([x, y, z]) => [x / euclideanDistance * faceWidthCm, y / euclideanDistance * faceWidthCm, z / euclideanDistance * faceWidthCm]) as [number, number, number][];

    // Step 4: Extract + invert MediaPipe face rotation, convert to euler, flip pitch/yaw, back to rotmat
    // Invert the rotation matrix from MediaPipe
    const faceR = faceRT.subMatrix(0, 2, 0, 2); // extact 3*3 rotation from mediapipe matrix
    let [pitch, yaw, roll] = matrixToEuler(faceR); // Convert rotation to angles
    [pitch, yaw] = [-yaw, pitch]; //coordinate swizzle - just flip to coordinates expected by tracker
    // turn corrected angles back to rotation matrix
    const finalR = eulerToMatrix(pitch, yaw, roll); // can be used to un-rotate face so it's looking straight ahead for the canonical model.

    // Step 5: Derotate face - apply inverse rotation mat to mesh
    // creates standardised 3D face mesh centered at the origin, scaled to Xcm wide, and looking straight ahead ('Canonical Face.')
    const canonical = normalized.map(p => multiplyVecByMat(p, finalR.transpose()));

    // Step 6: Scale from R columns
    const scales = [0, 1, 2].map(i => Math.sqrt(faceR.get(0, i) ** 2 + faceR.get(1, i) ** 2 + faceR.get(2, i) ** 2));
    const faceS = scales.reduce((a, b) => a + b, 0) / 3;

    // Step 7: Initial transform
    // First go - place at default distance initialZGuess
    const initTransform = Matrix.eye(4);
    initTransform.setSubMatrix(finalR.div(faceS), 0, 0);
    initTransform.set(0, 3, 0);
    initTransform.set(1, 3, 0);
    initTransform.set(2, 3, initialZGuess);

    // project 60cm-away face back onto 2D screen
    const cameraPts3D = canonical.map(p => transform3DTo3D(p, initTransform));
    const canonicalProj2D = cameraPts3D.map(p => transform3DTo2D(p, intrinsicsMatrix));
    // Check if centers don't match (e.g. virtual face is center, but user is in top-right corner).
    const detected2D = faceLandmarks.map(([x, y]) => [x * videoWidth, y * videoHeight]) as [number, number][];
    // Align Centers (Procrustes Analysis) - get 2D shift needed to align the centers and convert that to a 3D shift
    // so face is now in the right X/Y position, but the depth (Z) likely wrong
    const shift2D = partialProcrustesTranslation2D(canonicalProj2D, detected2D);

    const shift3D = imageShiftTo3D(shift2D, initialZGuess, intrinsicsMatrix);
    const finalTransform = initTransform.clone();
    finalTransform.set(0, 3, finalTransform.get(0, 3) + shift3D[0]);
    finalTransform.set(1, 3, finalTransform.get(1, 3) + shift3D[1]);
    finalTransform.set(2, 3, finalTransform.get(2, 3) + shift3D[2]);
    const firstFinalTransform = finalTransform.clone();

    // Slide face back and forth along the Z-axis until it fits
    // snaps X and Y to the image center, and zoom mask along Z-axis until mask size matches face on screen
    let newZ = initialZGuess;
    for (let i = 0; i < 10; i++) {
        // Project the 3D model onto screen and compare to observed landmarks
        const projectedPts = canonical.map(p => transform3DTo2D(transform3DTo3D(p, finalTransform), intrinsicsMatrix));
        // Compare spread of points, if model is smaller than face, must be further away
        // calculate correction factor (delta) and moves the face along Z-axis
        newZ = refineDepthByRadialMagnitude(projectedPts, detected2D, finalTransform.get(2, 3));
        if (Math.abs(newZ - finalTransform.get(2, 3)) < 0.25) break;
        // Perspective Correction: Crucially, apparent X/Y changes with z
        // so recalculate X,Y so face stays aligned with pixels as it moves
        const newX = firstFinalTransform.get(0, 3) * (newZ / initialZGuess);
        const newY = firstFinalTransform.get(1, 3) * (newZ / initialZGuess);

        finalTransform.set(0, 3, newX);
        finalTransform.set(1, 3, newY);
        finalTransform.set(2, 3, newZ);
    }

    const finalFacePts = canonical.map(p => transform3DTo3D(p, finalTransform));
    return [finalTransform, finalFacePts];
}

export function computeFaceOrigin3D(
    metricFace: [number, number, number][],
): [number, number, number] {
    // The Cyclopean Eye - center between eyes in 3d space.
    // TODO: why not use nose bridge? guess slightly less accurate and eyes are a bit further back...
    // Take the metric coordinates from faceReconstruction for corners of eyelid and average to find center of left/right eye, and middle of them.
    const computeMean = (indices: number[]): [number, number, number] => {
        const points = indices.map(idx => metricFace[idx]);
        const sum = points.reduce(
            (acc, [x, y, z]) => [acc[0] + x, acc[1] + y, acc[2] + z],
            [0, 0, 0]
        );
        return [sum[0] / points.length, sum[1] / points.length, sum[2] / points.length];
    };

    const leftEyeCenter = computeMean(LEFT_EYE_HORIZONTAL_LANDMARKS);
    const rightEyeCenter = computeMean(RIGHT_EYE_HORIZONTAL_LANDMARKS);

    const face_origin_3d: [number, number, number] = [
        (leftEyeCenter[0] + rightEyeCenter[0]) / 2,
        (leftEyeCenter[1] + rightEyeCenter[1]) / 2,
        (leftEyeCenter[2] + rightEyeCenter[2]) / 2
    ];

    return face_origin_3d;
}

function multiplyVecByMat(v: [number, number, number], m: Matrix): [number, number, number] {
    const [x, y, z] = v;
    const res = m.mmul(Matrix.columnVector([x, y, z])).to1DArray();
    return [res[0], res[1], res[2]];
}

export function matrixToEuler(matrix: Matrix, degrees: boolean = true): [number, number, number] {
    // Extract human readable rotation angles for gaze model requirements

    if (matrix.rows !== 3 || matrix.columns !== 3) {
        throw new Error('Rotation matrix must be 3x3.');
    }

    const pitch = Math.asin(-matrix.get(2, 0));
    const yaw = Math.atan2(matrix.get(2, 1), matrix.get(2, 2));
    const roll = Math.atan2(matrix.get(1, 0), matrix.get(0, 0));

    if (degrees) {
        const radToDeg = 180 / Math.PI;
        return [pitch * radToDeg, yaw * radToDeg, roll * radToDeg];
    }

    return [pitch, yaw, roll];
}

export function eulerToMatrix(pitch: number, yaw: number, roll: number, degrees: boolean = true): Matrix {

    if (degrees) {
        pitch *= Math.PI / 180;
        yaw *= Math.PI / 180;
        roll *= Math.PI / 180;
    }

    const cosPitch = Math.cos(pitch), sinPitch = Math.sin(pitch);
    const cosYaw = Math.cos(yaw), sinYaw = Math.sin(yaw);
    const cosRoll = Math.cos(roll), sinRoll = Math.sin(roll);

    const R_x = new Matrix([
        [1, 0, 0],
        [0, cosPitch, -sinPitch],
        [0, sinPitch, cosPitch],
    ]);

    const R_y = new Matrix([
        [cosYaw, 0, sinYaw],
        [0, 1, 0],
        [-sinYaw, 0, cosYaw],
    ]);

    const R_z = new Matrix([
        [cosRoll, -sinRoll, 0],
        [sinRoll, cosRoll, 0],
        [0, 0, 1],
    ]);

    // Final rotation matrix: R = Rz * Ry * Rx
    return R_z.mmul(R_y).mmul(R_x);
}

function pyrToVector(pitch: number, yaw: number, roll: number): number[] {
    // Convert spherical coordinates to Cartesian coordinates
    const x = Math.cos(pitch) * Math.sin(yaw);
    const y = Math.sin(pitch);
    const z = -Math.cos(pitch) * Math.cos(yaw);
    const vector = new Matrix([[x, y, z]]);

    // Apply roll rotation around the z-axis
    const [cos_r, sin_r] = [Math.cos(roll), Math.sin(roll)];
    const roll_matrix = new Matrix([
        [cos_r, -sin_r, 0],
        [sin_r, cos_r, 0],
        [0, 0, 1],
    ]);

    const rotated_vector = roll_matrix.mmul(vector.transpose()).transpose();
    return rotated_vector.to1DArray();
}

export function getHeadVector(
    tfMatrix: Matrix,
): number[] {

    // Extract the rotation part of the transformation matrix
    const rotationMatrix = new Matrix([
        [tfMatrix.get(0, 0), tfMatrix.get(0, 1), tfMatrix.get(0, 2)],
        [tfMatrix.get(1, 0), tfMatrix.get(1, 1), tfMatrix.get(1, 2)],
        [tfMatrix.get(2, 0), tfMatrix.get(2, 1), tfMatrix.get(2, 2)],
    ]);

    // Convert the matrix to euler angles and change the order/direction
    const [pitch, yaw, roll] = matrixToEuler(rotationMatrix, false);
    const [h_pitch, h_yaw, h_roll] = [-yaw, pitch, roll];

    // Construct a unit vector
    const vector = pyrToVector(h_pitch, h_yaw, h_roll);
    return vector;
}
