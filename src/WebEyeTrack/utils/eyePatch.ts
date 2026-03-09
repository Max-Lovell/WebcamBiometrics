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

// ── Eye Patch Extraction Pipeline ───────────────────────────────────────────

// Extracts, dewarps, and resizes the eye region from a face image.
// Pipeline: face landmarks → homography warp to square → crop eye strip
//           → bilinear resize to CNN input size.
export function obtainEyePatch(
    frame: ImageData,
    faceLandmarks: Point[],
    facePaddingCoefs: [number, number] = [0.4, 0.2],
    faceCropSize: number = 512,
    dstImgSize: [number, number] = [512, 128]
): ImageData {
    // Anchor landmarks: eyebrows to chin, nose centre for stability
    const center = faceLandmarks[4];
    const leftTop = faceLandmarks[103];
    const leftBottom = faceLandmarks[150];
    const rightTop = faceLandmarks[332];
    const rightBottom = faceLandmarks[379];

    // Apply radial padding around centre
    let srcPts: Point[] = [leftTop, leftBottom, rightBottom, rightTop];
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

    // Compute homography (face in video → flat, normalised crop)
    const H = computeHomography(srcPts, dstPts);
    if (!H) {
        console.warn("[eyePatch] Degenerate homography — returning blank patch");
        return new ImageData(dstImgSize[0], dstImgSize[1]);
    }

    const warped = warpImageData(frame, H, faceCropSize, faceCropSize);
    if (!warped) {
        console.warn("[eyePatch] Warp inversion failed — returning blank patch");
        return new ImageData(dstImgSize[0], dstImgSize[1]);
    }

    // Crop eye strip using warped landmark positions
    const topEyes = applyHomography(H, faceLandmarks[151]);
    const bottomEyes = applyHomography(H, faceLandmarks[195]);

    const cropY = Math.round(topEyes[1]);
    let cropHeight = Math.round(bottomEyes[1] - topEyes[1]);

    if (cropHeight <= 0) {
        console.warn(
            `[eyePatch] Invalid crop height: ${cropHeight} (warp matrix flip). Defaulting to 1px.`
        );
        cropHeight = 1;
    }

    if (cropY + cropHeight > warped.height) {
        cropHeight = Math.max(1, warped.height - cropY);
    }

    const eyeStrip = cropImageData(warped, 0, cropY, warped.width, cropHeight);

    // Resize to CNN input dimensions (512×128)
    return resizeImageData(eyeStrip, dstImgSize[0], dstImgSize[1]);
}

// ── Alt Eye Patch (oriented bounding box) ───────────────────────────────────

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
