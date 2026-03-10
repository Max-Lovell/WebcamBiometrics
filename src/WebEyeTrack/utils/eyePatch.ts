// ============================================================================
// Eye Patch Extraction and Homography
// ============================================================================

import type { Point } from "../../types.ts";

// ── Homography (Gaussian elimination, flat output) ──────────────────────────
// Homography solved via Gaussian elimination (8-param DLT, H[8]=1).
// No external dependencies — replaces the previous SVD/ml-matrix approach.
// For 4 well-conditioned point correspondences the results are identical.
// Compute 3×3 homography mapping src[i] → dst[i] for 4 point pairs.
// Returns flat row-major [h0..h8] with h8=1, or null if degenerate.
export function computeHomography(src: Point[], dst: Point[]): number[] | null {
    if (src.length !== 4 || dst.length !== 4) {
        throw new Error("Need exactly 4 source and 4 destination points");
    }

    const A: number[][] = [];
    const b: number[] = [];
    for (let i = 0; i < 4; i++) {
        const { x: sx, y: sy } = src[i];
        const { x: dx, y: dy } = dst[i];
        A.push([sx, sy, 1, 0, 0, 0, -sx * dx, -sy * dx]);
        b.push(dx);
        A.push([0, 0, 0, sx, sy, 1, -sx * dy, -sy * dy]);
        b.push(dy);
    }

    const n = 8;
    const M = A.map((row, i) => [...row, b[i]]);

    for (let col = 0; col < n; col++) {
        // Partial pivoting
        let maxRow = col;
        let maxVal = Math.abs(M[col][col]);
        for (let row = col + 1; row < n; row++) {
            if (Math.abs(M[row][col]) > maxVal) {
                maxVal = Math.abs(M[row][col]);
                maxRow = row;
            }
        }
        [M[col], M[maxRow]] = [M[maxRow], M[col]];

        const pivot = M[col][col];
        if (Math.abs(pivot) < 1e-12) return null;

        for (let j = col; j <= n; j++) M[col][j] /= pivot;
        for (let row = 0; row < n; row++) {
            if (row === col) continue;
            const factor = M[row][col];
            for (let j = col; j <= n; j++) M[row][j] -= factor * M[col][j];
        }
    }

    const h = M.map(row => row[n]);
    return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
}

// ── Homography helpers (flat 9-element format) ──────────────────────────────

// Apply a flat homography to a point.
export function applyHomography(H: number[], pt: Point): [number, number] {
    const { x, y } = pt;
    const denom = H[6] * x + H[7] * y + H[8];
    return [
        (H[0] * x + H[1] * y + H[2]) / denom,
        (H[3] * x + H[4] * y + H[5]) / denom,
    ];
}

// Invert a flat row-major 3×3 matrix. Returns null if singular.
export function invertHomography(m: number[]): number[] | null {
    const [a, b, c, d, e, f, g, h, i] = m;
    const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (Math.abs(det) < 1e-12) return null;
    const inv = 1 / det;
    return [
        (e * i - f * h) * inv, (c * h - b * i) * inv, (b * f - c * e) * inv,
        (f * g - d * i) * inv, (a * i - c * g) * inv, (c * d - a * f) * inv,
        (d * h - e * g) * inv, (b * g - a * h) * inv, (a * e - b * d) * inv,
    ];
}

// ── Eye Quad Computation (pure geometry, no pixels) ─────────────────────────
// single quad that goes straight from video frame to final 512×128 eye patch — no intermediate 512×512 image needed.
// Returns null if the homography is degenerate.
export function computeEyeQuad(
    faceLandmarks: Point[],
    facePaddingCoefs: [number, number] = [0.4, 0.2],
    faceCropSize: number = 512,
): Point[] | null {
    // Anchor landmarks: eyebrows to chin, nose centre for stability
    const center = faceLandmarks[4];

    let srcPts: Point[] = [
        faceLandmarks[103],  // leftTop
        faceLandmarks[150],  // leftBottom
        faceLandmarks[379],  // rightBottom
        faceLandmarks[332],  // rightTop
    ];

    // Radial padding: push each corner away from centre
    srcPts = srcPts.map(({ x, y }) => ({
        x: x + (x - center.x) * facePaddingCoefs[0],
        y: y + (y - center.y) * facePaddingCoefs[1],
    }));

    const dstPts: Point[] = [
        { x: 0, y: 0 },
        { x: 0, y: faceCropSize },
        { x: faceCropSize, y: faceCropSize },
        { x: faceCropSize, y: 0 },
    ];

    // H maps screen → 512×512 square
    const H = computeHomography(srcPts, dstPts);
    if (!H) return null;

    // Find eye strip bounds in the 512×512 space
    const topEyes = applyHomography(H, faceLandmarks[151]);
    const bottomEyes = applyHomography(H, faceLandmarks[195]);

    let cropY = topEyes[1];
    let cropBottom = bottomEyes[1];
    if (cropBottom <= cropY) cropBottom = cropY + 1;
    if (cropBottom > faceCropSize) cropBottom = faceCropSize;

    // The eye strip rectangle in 512×512 space:
    //   top-left:     (0, cropY)
    //   top-right:    (512, cropY)
    //   bottom-right: (512, cropBottom)
    //   bottom-left:  (0, cropBottom)
    //
    // Map these back to screen space via H⁻¹ to get the direct quad.
    const Hinv = invertHomography(H);
    if (!Hinv) return null;

    const cropCorners: Point[] = [
        { x: 0, y: cropY },
        { x: faceCropSize, y: cropY },
        { x: faceCropSize, y: cropBottom },
        { x: 0, y: cropBottom },
    ];

    return cropCorners.map(pt => {
        const [sx, sy] = applyHomography(Hinv, pt);
        return { x: sx, y: sy };
    });
}

// ── Image Warping (CPU, backward mapping) ───────────────────────────────────
// Warp source ImageData to a target rectangle using a flat homography.
// Uses backward mapping: iterates destination pixels, looks up source via H⁻¹.
export function warpImageData(
    srcImage: ImageData,
    H: number[],
    outWidth: number,
    outHeight: number
): ImageData | null {
    const Hinv = invertHomography(H);
    if (!Hinv) return null;

    const output = new ImageData(outWidth, outHeight);
    const src = srcImage.data;
    const dst = output.data;
    const srcW = srcImage.width;
    const srcH = srcImage.height;

    for (let y = 0; y < outHeight; y++) {
        for (let x = 0; x < outWidth; x++) {
            const denom = Hinv[6] * x + Hinv[7] * y + Hinv[8];
            const srcX = (Hinv[0] * x + Hinv[1] * y + Hinv[2]) / denom;
            const srcY = (Hinv[3] * x + Hinv[4] * y + Hinv[5]) / denom;

            const ix = Math.floor(srcX);
            const iy = Math.floor(srcY);

            if (ix < 0 || iy < 0 || ix >= srcW || iy >= srcH) continue;

            const srcIdx = (iy * srcW + ix) * 4;
            const dstIdx = (y * outWidth + x) * 4;
            dst[dstIdx]     = src[srcIdx];       // R
            dst[dstIdx + 1] = src[srcIdx + 1];   // G
            dst[dstIdx + 2] = src[srcIdx + 2];   // B
            dst[dstIdx + 3] = src[srcIdx + 3];   // A
        }
    }

    return output;
}

// ── Crop & Resize Utilities ─────────────────────────────────────────────────
export function cropImageData(
    source: ImageData,
    x: number,
    y: number,
    width: number,
    height: number
): ImageData {
    const output = new ImageData(width, height);
    const src = source.data;
    const dst = output.data;
    const srcWidth = source.width;

    for (let j = 0; j < height; j++) {
        for (let i = 0; i < width; i++) {
            const srcIdx = ((y + j) * srcWidth + (x + i)) * 4;
            const dstIdx = (j * width + i) * 4;
            dst[dstIdx]     = src[srcIdx];
            dst[dstIdx + 1] = src[srcIdx + 1];
            dst[dstIdx + 2] = src[srcIdx + 2];
            dst[dstIdx + 3] = src[srcIdx + 3];
        }
    }

    return output;
}

// Bilinear interpolation resize — matches OpenCV cv2.resize() INTER_LINEAR.
export function resizeImageData(
    source: ImageData,
    outWidth: number,
    outHeight: number
): ImageData {
    const output = new ImageData(outWidth, outHeight);
    const src = source.data;
    const dst = output.data;
    const { width: srcWidth, height: srcHeight } = source;

    const xRatio = (srcWidth - 1) / outWidth;
    const yRatio = (srcHeight - 1) / outHeight;

    for (let y = 0; y < outHeight; y++) {
        for (let x = 0; x < outWidth; x++) {
            const xDiff = Math.floor(xRatio * x);
            const yDiff = Math.floor(yRatio * y);
            const xWeight = xRatio * x - xDiff;
            const yWeight = yRatio * y - yDiff;

            const index = (y * outWidth + x) * 4;
            const aIdx = (yDiff * srcWidth + xDiff) * 4;
            const bIdx = (yDiff * srcWidth + (xDiff + 1)) * 4;
            const cIdx = ((yDiff + 1) * srcWidth + xDiff) * 4;
            const dIdx = ((yDiff + 1) * srcWidth + (xDiff + 1)) * 4;

            for (let i = 0; i < 3; i++) {
                dst[index + i] =
                    src[aIdx + i] * (1 - xWeight) * (1 - yWeight) +
                    src[bIdx + i] * xWeight * (1 - yWeight) +
                    src[cIdx + i] * yWeight * (1 - xWeight) +
                    src[dIdx + i] * xWeight * yWeight;
            }
            dst[index + 3] = 255;
        }
    }
    return output;
}

// ── Eye Patch Extraction (CPU convenience wrapper) ──────────────────────────
// Full CPU pipeline: computes eye quad then warps via CPU pixel loop.
// This is the drop-in replacement for the original obtainEyePatch.
// For the GPU path, call computeEyeQuad() then warpGPU() from eyePatchWarp.ts.
export function obtainEyePatch(
    frame: ImageData,
    faceLandmarks: Point[],
    facePaddingCoefs: [number, number] = [0.4, 0.2],
    faceCropSize: number = 512,
    dstImgSize: [number, number] = [512, 128]
): ImageData {
    const quad = computeEyeQuad(faceLandmarks, facePaddingCoefs, faceCropSize);
    if (!quad) {
        console.warn("[eyePatch] Degenerate eye quad — returning blank patch");
        return new ImageData(dstImgSize[0], dstImgSize[1]);
    }

    // Use warpImageData with the direct quad → 512×128 homography
    const H = computeHomography(quad, [
        { x: 0, y: 0 },
        { x: dstImgSize[0] - 1, y: 0 },
        { x: dstImgSize[0] - 1, y: dstImgSize[1] - 1 },
        { x: 0, y: dstImgSize[1] - 1 },
    ]);
    if (!H) {
        console.warn("[eyePatch] Degenerate warp homography — returning blank patch");
        return new ImageData(dstImgSize[0], dstImgSize[1]);
    }

    const result = warpImageData(frame, H, dstImgSize[0], dstImgSize[1]);
    if (!result) {
        console.warn("[eyePatch] Warp inversion failed — returning blank patch");
        return new ImageData(dstImgSize[0], dstImgSize[1]);
    }

    return result;
}

// ── Alt Eye Patch methods ───────────────────────────────────
// Bounding box from more stable landmarks
function midpoint(a: Point, b: Point): Point {
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

export function altEyePatch(
    landmarks: Point[],
    imageData: ImageData
): ImageData | null {
    const left = landmarks[372];
    const right = landmarks[143];
    const top = landmarks[9];
    const bottom = midpoint(landmarks[229], landmarks[449]);
    const center = midpoint(landmarks[133], landmarks[362]);

    const dx = right.x - left.x;
    const dy = right.y - left.y;
    const angle = Math.atan2(dy, dx);
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);

    const halfW = Math.sqrt(dx * dx + dy * dy) / 2;
    const topDist = Math.sqrt((top.x - center.x) ** 2 + (top.y - center.y) ** 2);
    const botDist = Math.sqrt((bottom.x - center.x) ** 2 + (bottom.y - center.y) ** 2);

    const srcPts: Point[] = [
        { x: center.x + halfW * cos - topDist * sin, y: center.y + halfW * sin + topDist * cos },
        { x: center.x - halfW * cos - topDist * sin, y: center.y - halfW * sin + topDist * cos },
        { x: center.x - halfW * cos + botDist * sin, y: center.y - halfW * sin - botDist * cos },
        { x: center.x + halfW * cos + botDist * sin, y: center.y + halfW * sin - botDist * cos },
    ];

    const outW = 512;
    const outH = 128;
    const dstPts: Point[] = [
        { x: 0, y: 0 },
        { x: outW, y: 0 },
        { x: outW, y: outH },
        { x: 0, y: outH },
    ];

    const H = computeHomography(srcPts, dstPts);
    if (!H) return null;
    return warpImageData(imageData, H, outW, outH);
}

// A better method to use in future by retraining - just extracts a box (sunglasses) pinned to the face transform.
// Can be used to extract directly using tf.image.transform. I've got this working manually on a separate repo, can't remember if below works 100% though.
export function rectSourceQuad(
    faceMatrix: ArrayLike<number>,
    frameWidth: number,
    frameHeight: number,
    config: {
        rectHalfW?: number;
        rectHalfH?: number;
        rectOffsetY?: number;
        rectOffsetZ?: number;
        vfovDeg?: number;
    } = {},
): [number, number][] | null {
    const hw = config.rectHalfW ?? 6;
    const hh = config.rectHalfH ?? 2;
    const oy = config.rectOffsetY ?? 3;
    const oz = config.rectOffsetZ ?? 4;
    const vfovDeg = config.vfovDeg ?? 63;

    // Camera intrinsics from vertical FOV
    const cx = frameWidth / 2;
    const cy = frameHeight / 2;
    const fy = cy / Math.tan((vfovDeg / 2) * Math.PI / 180);
    const fx = fy;

    // 4 corners in face-local space (homogeneous coords)
    const localCorners: [number, number, number, number][] = [
        [-hw,  hh + oy, oz, 1],  // top-left
        [ hw,  hh + oy, oz, 1],  // top-right
        [ hw, -hh + oy, oz, 1],  // bottom-right
        [-hw, -hh + oy, oz, 1],  // bottom-left
    ];

    // Transform to camera space and project
    const screenPoints: [number, number][] = [];
    for (const corner of localCorners) {
        // Column-major 4×4 multiply
        const cam: [number, number, number, number] = [
            faceMatrix[0]*corner[0] + faceMatrix[4]*corner[1] + faceMatrix[8]*corner[2]  + faceMatrix[12]*corner[3],
            faceMatrix[1]*corner[0] + faceMatrix[5]*corner[1] + faceMatrix[9]*corner[2]  + faceMatrix[13]*corner[3],
            faceMatrix[2]*corner[0] + faceMatrix[6]*corner[1] + faceMatrix[10]*corner[2] + faceMatrix[14]*corner[3],
            faceMatrix[3]*corner[0] + faceMatrix[7]*corner[1] + faceMatrix[11]*corner[2] + faceMatrix[15]*corner[3],
        ];

        // Pinhole projection (camera looks down -Z)
        const depth = -cam[2];
        if (depth <= 0.001) return null;
        screenPoints.push([
            fx * (cam[0] / depth) + cx,
            fy * (-cam[1] / depth) + cy,
        ]);
    }

    return screenPoints;
}
