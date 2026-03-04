import { FaceLandmarker, DrawingUtils, type NormalizedLandmark } from "@mediapipe/tasks-vision";
import type {BiometricsResult} from "./pipeline/types.ts";
// MediaPipe Face Mesh Indices for rPPG signal
// Optimized ROIs based on GRGB rPPG research
export const FACE_ROIS = {
    // Center forehead (Avoids hair/eyebrows)
    forehead: [107, 66, 69, 109, 10, 338, 299, 296, 336, 9],

    // Left Cheek (Subject's Left - Indices > 200)
    leftCheek: [347, 348, 329, 355, 429, 279, 358, 423, 425, 280],

    // Right Cheek (Subject's Right - Indices < 200)
    rightCheek: [118, 119, 100, 126, 209, 49, 129, 203, 205, 50]
};

// Helper to get points from indices if you need it elsewhere
// (Assuming you pass the full landmarks array)
export function getPoints(indices: number[], landmarks: any[]) {
    return indices.map(i => landmarks[i]);
}
export function drawMesh(gaze_result: BiometricsResult, canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const drawingUtils = new DrawingUtils(ctx);
    const landmarks = gaze_result.faceLandmarker.faceLandmarks[0];
    const facialTransformationMatrix = gaze_result.faceLandmarker.facialTransformationMatrixes[0].data

    // 1. Clear Canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!landmarks || landmarks.length === 0) return;

    // 2. Draw Standard Mesh
    if (drawingUtils) {
        // Tesselation (Subtle)
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C040", lineWidth: 0.5 });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#FF3030" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#30FF30" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: "#FF3030", lineWidth: 2 });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: "#30FF30", lineWidth: 2 });
    }

    // 3. Draw Gaze Vectors
    if (facialTransformationMatrix) {
        // A. ORIGINAL GEOMETRIC RAYS (Cyan/Green) - Good for debugging raw geometry
        drawGazeVector(ctx, landmarks, facialTransformationMatrix, canvas.width, canvas.height);
        drawGazeVector3D(ctx, landmarks, facialTransformationMatrix, canvas.width, canvas.height);

        // B. NEW PHYSICALLY BASED RAY (Yellow) - Best for stability
        // We check if blendshapes exist (usually in result.faceBlendshapes[0])
        const blendshapesRoot = (gaze_result as any).faceBlendshapes;

        // Just pass the first face object directly
        if (blendshapesRoot && blendshapesRoot.length > 0) {
            drawPhysicallyBasedGaze(
                ctx,
                landmarks,
                facialTransformationMatrix,
                blendshapesRoot[0], // Pass the first face's simple object
                canvas.width,
                canvas.height,
                30
            );
        }
    }

    // 4. Draw Custom ROIs
    if (FACE_ROIS.forehead) drawPath(ctx, landmarks, FACE_ROIS.forehead, 'cyan', 2);
    if (FACE_ROIS.leftCheek) drawPath(ctx, landmarks, FACE_ROIS.leftCheek, 'yellow', 2);
    if (FACE_ROIS.rightCheek) drawPath(ctx, landmarks, FACE_ROIS.rightCheek, 'yellow', 2);
}

// =============================================================================
// NEW: PHYSICALLY BASED GAZE (Head Matrix + Scaled Blendshapes)
// =============================================================================

function drawPhysicallyBasedGaze(
    ctx: CanvasRenderingContext2D,
    landmarks: NormalizedLandmark[],
    matrixData: number[],
    blendshapes: any, // Expecting { "browDownLeft": 0.5, ... }
    width: number,
    height: number,
    maxRotationDegrees: number = 30,
    length: number = 20
) {
    if (!matrixData || matrixData.length < 16) return;

    // 1. Camera Intrinsics
    const fx = width;
    const fy = width;
    const cx = width / 2;
    const cy = height / 2;

    // 2. Extract Head Rotation (Quaternion)
    const m = matrixData;
    const trace = m[0] + m[5] + m[10];
    let Q_head = new SimpleQuaternion();

    if (trace > 0) {
        const s = 0.5 / Math.sqrt(trace + 1.0);
        Q_head.w = 0.25 / s;
        Q_head.x = (m[6] - m[9]) * s;
        Q_head.y = (m[8] - m[2]) * s;
        Q_head.z = (m[1] - m[4]) * s;
    } else {
        if (m[0] > m[5] && m[0] > m[10]) {
            const s = 2.0 * Math.sqrt(1.0 + m[0] - m[5] - m[10]);
            Q_head.w = (m[6] - m[9]) / s;
            Q_head.x = 0.25 * s;
            Q_head.y = (m[1] + m[4]) / s;
            Q_head.z = (m[2] + m[8]) / s;
        } else if (m[5] > m[10]) {
            const s = 2.0 * Math.sqrt(1.0 + m[5] - m[0] - m[10]);
            Q_head.w = (m[8] - m[2]) / s;
            Q_head.x = (m[1] + m[4]) / s;
            Q_head.y = 0.25 * s;
            Q_head.z = (m[6] + m[9]) / s;
        } else {
            const s = 2.0 * Math.sqrt(1.0 + m[10] - m[0] - m[5]);
            Q_head.w = (m[1] - m[4]) / s;
            Q_head.x = (m[2] + m[8]) / s;
            Q_head.y = (m[6] + m[9]) / s;
            Q_head.z = 0.25 * s;
        }
    }

    // 3. Process Each Eye
    const sides = ['left', 'right'] as const;
    const EYE_INDICES = { left: 473, right: 468 };
    const toRad = Math.PI / 180;
    const maxRad = maxRotationDegrees * toRad;

    // --- FIX: UPDATED GET SCORE HELPER ---
    // Direct property access instead of .find()
    const getScore = (name: string) => {
        // Try accessing directly: blendshapes["browDownLeft"]
        return blendshapes[name] !== undefined ? blendshapes[name] : 0;
    };

    sides.forEach(side => {
        const isLeft = side === 'left';
        const suffix = isLeft ? 'Left' : 'Right';

        // A. Calculate Eye Angles
        const lookDown = getScore(`eyeLookDown${suffix}`);
        const lookUp   = getScore(`eyeLookUp${suffix}`);
        const lookIn   = getScore(`eyeLookIn${suffix}`);
        const lookOut  = getScore(`eyeLookOut${suffix}`);

        // Debug Log (Uncomment if needed to verify data flow)
        // if (isLeft) console.log(`Left Eye: Down=${lookDown.toFixed(2)}, In=${lookIn.toFixed(2)}`);

        // Pitch (X-axis)
        const pitchAngle = (lookDown - lookUp) * maxRad;

        // Yaw (Y-axis)
        let yawAngle = 0;
        if (isLeft) {
            yawAngle = (lookOut - lookIn) * maxRad;
        } else {
            yawAngle = (lookIn - lookOut) * maxRad;
        }

        // B. Create Eye Quaternion
        const Q_pitch = new SimpleQuaternion().setFromAxisAngle({x:1, y:0, z:0}, pitchAngle);
        const Q_yaw   = new SimpleQuaternion().setFromAxisAngle({x:0, y:1, z:0}, yawAngle);
        const Q_eye   = Q_yaw.multiply(Q_pitch);

        // C. Combine Head + Eye
        const Q_total = Q_head.multiply(Q_eye);

        // D. Get Direction
        // If ray points backwards, change z to -1
        const forward = { x: 0, y: 0, z: 1 };
        const gazeDir = Q_total.applyToVector(forward);

        // E. Origin (Hybrid)
        const irisLm = landmarks[EYE_INDICES[side]];
        const irisPxX = irisLm.x * width;
        const irisPxY = irisLm.y * height;
        const headDepthZ = m[14];

        const origin = {
            x: (irisPxX - cx) * headDepthZ / fx,
            y: (irisPxY - cy) * headDepthZ / fy,
            z: headDepthZ
        };

        // F. Draw
        const tip = {
            x: origin.x + gazeDir.x * length,
            y: origin.y + gazeDir.y * length,
            z: origin.z + gazeDir.z * length
        };
        const end2D = {
            x: (tip.x * fx / tip.z) + cx,
            y: (tip.y * fy / tip.z) + cy
        };

        ctx.beginPath();
        ctx.moveTo(irisPxX, irisPxY);
        ctx.lineTo(end2D.x, end2D.y);
        ctx.strokeStyle = "#FFFF00"; // YELLOW
        ctx.lineWidth = 4;
        ctx.stroke();
    });
}

// --- Minimal Quaternion Helper Class ---
class SimpleQuaternion {
    x: number; y: number; z: number; w: number;
    constructor(x=0, y=0, z=0, w=1) { this.x=x; this.y=y; this.z=z; this.w=w; }

    setFromAxisAngle(axis: {x:number, y:number, z:number}, angle: number) {
        const halfAngle = angle / 2;
        const s = Math.sin(halfAngle);
        this.x = axis.x * s;
        this.y = axis.y * s;
        this.z = axis.z * s;
        this.w = Math.cos(halfAngle);
        return this;
    }

    multiply(q: SimpleQuaternion) {
        const x = this.x, y = this.y, z = this.z, w = this.w;
        const qx = q.x, qy = q.y, qz = q.z, qw = q.w;
        return new SimpleQuaternion(
            x * qw + w * qx + y * qz - z * qy,
            y * qw + w * qy + z * qx - x * qz,
            z * qw + w * qz + x * qy - y * qx,
            w * qw - x * qx - y * qy - z * qz
        );
    }

    applyToVector(v: {x:number, y:number, z:number}) {
        const x = v.x, y = v.y, z = v.z;
        const qx = this.x, qy = this.y, qz = this.z, qw = this.w;
        const ix = qw * x + qy * z - qz * y;
        const iy = qw * y + qz * x - qx * z;
        const iz = qw * z + qx * y - qy * x;
        const iw = -qx * x - qy * y - qz * z;
        return {
            x: ix * qw + iw * -qx + iy * -qz - iz * -qy,
            y: iy * qw + iw * -qy + iz * -qx - ix * -qz,
            z: iz * qw + iw * -qz + ix * -qy - iy * -qx
        };
    }
}

// =============================================================================
// EXISTING FUNCTIONS (Kept for reference/comparison)
// =============================================================================

function drawGazeVector(
    ctx: CanvasRenderingContext2D,
    landmarks: NormalizedLandmark[],
    faceRt: any,
    width: number,
    height: number,
    length: number = 200
) {
    const EYE_METRICS = {
        right: { pupil: 468, inner: 133, outer: 33 },
        left:  { pupil: 473, inner: 362, outer: 263 }
    };

    let matrixData = Array.isArray(faceRt) ? faceRt : (faceRt?.data || null);
    if (!matrixData || matrixData.length < 16) return;

    const headX = matrixData[8];
    const headY = matrixData[9];
    const headZ = matrixData[10];

    const sides = ['left', 'right'] as const;
    sides.forEach(side => {
        const indices = EYE_METRICS[side];
        const lmPupil = landmarks[indices.pupil];
        const lmInner = landmarks[indices.inner];
        const lmOuter = landmarks[indices.outer];

        const toPixel = (lm: NormalizedLandmark) => ({
            x: lm.x * width,
            y: lm.y * height,
            z: lm.z * width
        });

        const pPupil = toPixel(lmPupil);
        const pInner = toPixel(lmInner);
        const pOuter = toPixel(lmOuter);

        const midX = (pInner.x + pOuter.x) / 2;
        const midY = (pInner.y + pOuter.y) / 2;
        const midZ = (pInner.z + pOuter.z) / 2;

        const eyeWidthPx = Math.hypot(
            pOuter.x - pInner.x,
            pOuter.y - pInner.y,
            pOuter.z - pInner.z
        );

        const depthFactor = 0.5;

        const centerX = midX - (headX * eyeWidthPx * depthFactor);
        const centerY = midY - (headY * eyeWidthPx * depthFactor);
        const centerZ = midZ - (headZ * eyeWidthPx * depthFactor);

        let vecX = pPupil.x - centerX;
        let vecY = pPupil.y - centerY;
        let vecZ = pPupil.z - centerZ;

        const mag = Math.hypot(vecX, vecY, vecZ);
        vecX /= mag;
        vecY /= mag;

        const startX = pPupil.x;
        const startY = pPupil.y;
        const endX = startX + (vecX * length);
        const endY = startY + (vecY * length);

        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.strokeStyle = "#00FFFF"; // Cyan
        ctx.lineWidth = 3;
        ctx.stroke();
    });
}

function drawGazeVector3D(
    ctx: CanvasRenderingContext2D,
    landmarks: NormalizedLandmark[],
    faceRt: any,
    width: number,
    height: number,
    length: number = 15
) {
    const matrixData = Array.isArray(faceRt) ? faceRt : (faceRt?.data || null);
    if (!matrixData || matrixData.length < 16) return;

    const fovY = 60 * (Math.PI / 180);
    const fy = height / (2 * Math.tan(fovY / 2));
    const fx = fy;
    const cx = width / 2;
    const cy = height / 2;

    const headForward = { x: matrixData[8], y: matrixData[9], z: matrixData[10] };

    const EYE_METRICS = {
        right: { pupil: 468, inner: 133, outer: 33 },
        left:  { pupil: 473, inner: 362, outer: 263 }
    };
    const eyeRadius = 1.2;

    (['left', 'right'] as const).forEach(side => {
        const indices = EYE_METRICS[side];
        const lmInner = landmarks[indices.inner];
        const lmOuter = landmarks[indices.outer];
        const lmPupil = landmarks[indices.pupil];

        const headDepth = matrixData[14] || 30;

        const midU = (lmInner.x + lmOuter.x) / 2;
        const midV = (lmInner.y + lmOuter.y) / 2;

        const surfaceX = (midU * width - cx) * headDepth / fx;
        const surfaceY = (midV * height - cy) * headDepth / fy;
        const surfaceZ = headDepth;

        const corX = surfaceX - (headForward.x * eyeRadius);
        const corY = surfaceY - (headForward.y * eyeRadius);
        const corZ = surfaceZ - (headForward.z * eyeRadius);

        const pupilU = lmPupil.x * width;
        const pupilV = lmPupil.y * height;

        const rx = (pupilU - cx) / fx;
        const ry = (pupilV - cy) / fy;
        const rz = 1.0;
        const rMag = Math.sqrt(rx*rx + ry*ry + rz*rz);
        const rayDir = { x: rx/rMag, y: ry/rMag, z: rz/rMag };

        const OC = { x: -corX, y: -corY, z: -corZ };
        const b = 2 * (rayDir.x * OC.x + rayDir.y * OC.y + rayDir.z * OC.z);
        const c = (OC.x**2 + OC.y**2 + OC.z**2) - (eyeRadius**2);
        const delta = b*b - 4*c;

        let t = 0;
        if (delta >= 0) {
            t = (-b - Math.sqrt(delta)) / 2;
        } else {
            t = -b / 2;
        }

        const pupil3D = {
            x: t * rayDir.x,
            y: t * rayDir.y,
            z: t * rayDir.z
        };

        const vec3D = {
            x: pupil3D.x - corX,
            y: pupil3D.y - corY,
            z: pupil3D.z - corZ
        };

        const vecMag = Math.sqrt(vec3D.x**2 + vec3D.y**2 + vec3D.z**2);
        const gazeDir = { x: vec3D.x/vecMag, y: vec3D.y/vecMag, z: vec3D.z/vecMag };

        const end3D = {
            x: pupil3D.x + gazeDir.x * length,
            y: pupil3D.y + gazeDir.y * length,
            z: pupil3D.z + gazeDir.z * length
        };

        const start2D = {
            x: (pupil3D.x * fx / pupil3D.z) + cx,
            y: (pupil3D.y * fy / pupil3D.z) + cy
        };

        const end2D = {
            x: (end3D.x * fx / end3D.z) + cx,
            y: (end3D.y * fy / end3D.z) + cy
        };

        ctx.beginPath();
        ctx.moveTo(start2D.x, start2D.y);
        ctx.lineTo(end2D.x, end2D.y);
        ctx.strokeStyle = "#00FF00"; // Green
        ctx.lineWidth = 3;
        ctx.stroke();
    });
}

function drawPath(ctx: CanvasRenderingContext2D, landmarks: NormalizedLandmark[], indices: number[], color: string, lineWidth: number) {
    const width = ctx.canvas.width;
    const height = ctx.canvas.height;
    ctx.beginPath();
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = color;

    const firstPoint = landmarks[indices[0]];
    if (!firstPoint) return;

    ctx.moveTo(firstPoint.x * width, firstPoint.y * height);
    for (let i = 1; i < indices.length; i++) {
        const p = landmarks[indices[i]];
        if (p) ctx.lineTo(p.x * width, p.y * height);
    }
    ctx.closePath();
    ctx.stroke();

    ctx.save();
    ctx.globalAlpha = 0.2;
    ctx.fillStyle = color;
    ctx.fill();
    ctx.restore();
}
