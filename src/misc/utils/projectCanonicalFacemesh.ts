// PROJECT FROM CANONICAL FACEMESH TO PIXEL -----------
import canonicalObj from './face_model_with_iris.obj?raw';
import type {Point} from "../../types.ts";
import type {Coordinate3D} from "./irisUnit.ts";

// Get canonical vertices --------
export const CANONICAL_VERTICES = parseCanonicalVertices(canonicalObj);

function parseCanonicalVertices(text: string): Coordinate3D[] {
    const vertices: Coordinate3D[] = [];
    for (const line of text.split('\n')) {
        if (line.startsWith('v ')) {
            const [, x, y, z] = line.trim().split(/\s+/);
            vertices.push({ x: +x, y: +y, z: +z });
        }
    }
    return vertices;
}

// Get location of canonical landmark -----
function projectCanonicalLandmark(
    v: Coordinate3D,
    transformationMatrix: number[],
): Coordinate3D {
    const m = transformationMatrix;
    const camX = m[0]*v.x + m[4]*v.y + m[8]*v.z  + m[12];
    const camY = m[1]*v.x + m[5]*v.y + m[9]*v.z  + m[13];
    const camZ = m[2]*v.x + m[6]*v.y + m[10]*v.z + m[14];
    const depth = -camZ;
    return {x: camX, y: camY, z: depth};
}

export function projectCanonicalLandmark3D(
    landmarkIndex: number,
    transformationMatrix: number[],
    fx: number,
): Point {
    const canonicalVertex = CANONICAL_VERTICES[landmarkIndex];
    const landmark3D = projectCanonicalLandmark(canonicalVertex, transformationMatrix);
    // Centered pixel matching the convention used in projectCanonicalToCanvas:
    // x positive right, y positive UP (hence -camY).
    return {
        x:  (landmark3D.x / landmark3D.z) * fx,
        y: (-landmark3D.y / landmark3D.z) * fx,
    }
}

export function projectCanonicalToCanvas(
    landmarkIndex: number,
    transformationMatrix: number[],
    fx: number,
    frameWidth: number,
    frameHeight: number,
): Point{
    const canonicalVertex = CANONICAL_VERTICES[landmarkIndex];
    const landmark3D = projectCanonicalLandmark(canonicalVertex, transformationMatrix);
    // Centered pixel matching the convention used in projectCanonicalToCanvas:
    // x positive right, y positive UP (hence -camY).
    // Same projection as computeHeadAxesDebug / rectSourceQuad — directly to
    // canvas coords, +y-down, y-flipped inside the projection.
    return {
        x: fx * (landmark3D.x / landmark3D.z) + frameWidth / 2,
        y: fx * (-landmark3D.y / landmark3D.z) + frameHeight / 2,
    };
}

export function getEyeballCenterFromCanonical(
    irisIndex: number,
    transformationMatrix: number[],
    pupilDepth: number,
): Coordinate3D {
    const R = 1.175;
    const iris = CANONICAL_VERTICES[irisIndex];
    const m = transformationMatrix;

    // 1. Transform canonical iris vertex to MP camera space.
    const irisCamX = m[0]*iris.x + m[4]*iris.y + m[8]*iris.z  + m[12];
    const irisCamY = m[1]*iris.x + m[5]*iris.y + m[9]*iris.z  + m[13];
    const irisCamZ = m[2]*iris.x + m[6]*iris.y + m[10]*iris.z + m[14];

    // 2. Head-forward axis in MP camera space = rotation column 2 (model +Z,
    //    points out of the face). Already unit length since it's a rotation column.
    const fwdX = m[8];
    const fwdY = m[9];
    const fwdZ = m[10];

    // 3. Walk back into the head by R along -forward.
    const eyeCamX = irisCamX - fwdX * R;
    const eyeCamY = irisCamY - fwdY * R;
    const eyeCamZ = irisCamZ - fwdZ * R;

    // 4. Convert to your +Z-away convention. MP has -Z forward, so depth = -camZ.
    //    Also +Y is up in both, so y is unchanged.
    const mpEyeDepth = -eyeCamZ;

    // 5. Rescale uniformly along the camera ray so the eyeball center sits at the
    //    depth implied by your iris-diameter measurement. The pupil is at pupilDepth;
    //    the eyeball center is R behind it along head-forward, which projects to a
    //    depth difference of R * |fwdZ| along the camera z-axis.
    const targetDepth = pupilDepth + R * Math.abs(fwdZ);
    const scale = targetDepth / mpEyeDepth;

    return {
        x: eyeCamX * scale,
        y: eyeCamY * scale,
        z: targetDepth,
    };
}


// Compute head axes for display - just used for a sense check -------
export function computeHeadAxesDisplay(
    transformationMatrix: ArrayLike<number>,
    frameWidth: number,
    frameHeight: number,
    fx: number,
    axisLengthCm = 4,
): { origin: Point; xAxis: Point; yAxis: Point; zAxis: Point } {
    const m = transformationMatrix;

    // Origin of the head frame in camera space = translation column.
    const ox = m[12];
    const oy = m[13];
    const oz = m[14];

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
