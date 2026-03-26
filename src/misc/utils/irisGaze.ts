import type { FaceLandmarkerResult } from "@mediapipe/tasks-vision";
// Simple geometric gaze estimation from iris landmarks.
// measures where the iris centre sits within the eye opening vertically and horizontally
// Returns normalised [-1, +1] for each axis. No trig, no head rotation.


// Landmark indices -----
// Iris cardinal points:
//   Right eye: 469(left) 470(top) 471(right) 472(bottom), centre 468
//   Left eye:  474(left) 475(top) 476(right) 477(bottom), centre 473
// Eye corners:
//   Right eye: 33 (outer), 133 (inner)
//   Left eye:  362 (outer), 263 (inner)
// Upper/lower eyelid mid-points (for vertical range):
//   Right eye: 159 (upper), 145 (lower)
//   Left eye:  386 (upper), 374 (lower)

export interface GazeResult {
    // Normalised gaze: x [-1,+1] (left to right in image space), y [-1,+1] (up to down) */
    x: number;
    y: number;
    // Iris H/V aspect ratio (1.0 = circle/straight, <1 = looking sideways) */
    aspectRatio: number;
    // Raw iris diameters in px for debugging
    avgHPx: number;
    avgVPx: number;
}

export function estimateIrisGaze(
    result: FaceLandmarkerResult,
    frameWidth: number,
    frameHeight: number,
): GazeResult | null {
    const lm = result.faceLandmarks?.[0];
    if (!lm || lm.length < 478) return null;

    // ── Measure iris position within eye socket for each eye ─────────
    const rightX = eyeGazeX(lm, 468, 33, 133, frameWidth);
    const leftX  = eyeGazeX(lm, 473, 362, 263, frameWidth);

    const rightY = eyeGazeY(lm, 468, 159, 145, frameHeight);
    const leftY  = eyeGazeY(lm, 473, 386, 374, frameHeight);

    // Average both eyes
    const x = (rightX + leftX) / 2;
    const y = (rightY + leftY) / 2;

    // Aspect ratio (for info / future use)
    const rH = dist(lm[469], lm[471], frameWidth, frameHeight);
    const rV = dist(lm[470], lm[472], frameWidth, frameHeight);
    const lH = dist(lm[474], lm[476], frameWidth, frameHeight);
    const lV = dist(lm[475], lm[477], frameWidth, frameHeight);
    const avgH = (rH + lH) / 2;
    const avgV = (rV + lV) / 2;
    const aspectRatio = avgV > 0.1 ? avgH / avgV : 1;

    return { x, y, aspectRatio, avgHPx: avgH, avgVPx: avgV };
}

// Helpers ------

// Iris centre position normalised within eye corners (horizontal).
function eyeGazeX(
    lm: { x: number; y: number }[],
    irisCentre: number,
    outerCorner: number,
    innerCorner: number,
    w: number,
): number {
    const iris = lm[irisCentre].x * w;
    const outer = lm[outerCorner].x * w;
    const inner = lm[innerCorner].x * w;
    const left = Math.min(outer, inner);
    const right = Math.max(outer, inner);
    const range = right - left;
    if (range < 1) return 0;
    return ((iris - left) / range) * 2 - 1;
}

// Iris centre position normalised within eyelids (vertical).
function eyeGazeY(
    lm: { x: number; y: number }[],
    irisCentre: number,
    upperLid: number,
    lowerLid: number,
    h: number,
): number {
    const iris = lm[irisCentre].y * h;
    const upper = lm[upperLid].y * h;
    const lower = lm[lowerLid].y * h;
    const range = lower - upper;
    if (range < 1) return 0;
    return ((iris - upper) / range) * 2 - 1;
}

// Pixel distance between two normalised landmarks.
function dist(
    a: { x: number; y: number },
    b: { x: number; y: number },
    w: number,
    h: number,
): number {
    const dx = (a.x - b.x) * w;
    const dy = (a.y - b.y) * h;
    return Math.sqrt(dx * dx + dy * dy);
}
