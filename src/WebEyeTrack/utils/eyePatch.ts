// ============================================================================
// Eye Patch Extraction and Homography
// ============================================================================
import {inverse, Matrix} from "ml-matrix";
import type {Point} from "../../types.ts";
import { safeSVD } from './safeSVD.ts';

// Estimates a 3x3 homography matrix from 4 point correspondences.
export function computeHomography(src: Point[], dst: Point[]): number[][] {
    // maps points from face in video to the flat, normalized crop
    // src = original 4 tilted corners, dst = 512*512 square
    if (src.length !== 4 || dst.length !== 4) {
        throw new Error("Need exactly 4 source and 4 destination points");
    }

    const A: number[][] = [];

    // Map points from one square to another using linear equations to solve H
    // Equations in the form Ah=0, h=vector of the 9 values of H, solve for 0
    for (let i = 0; i < 4; i++) {
        const {x, y} = src[i];
        const {x: u, y: v} = dst[i];
        // for every x,y point in original, two rows.
        // Formula derived from cross-product rule of vectors
        A.push([-x, -y, -1, 0, 0, 0, x * u, y * u, u]);
        A.push([0, 0, 0, -x, -y, -1, x * v, y * v, v]);
    }

    const A_mat = new Matrix(A);
    // Singular Value Decomposition solution
    // H corresponds to the right singular vector associated with the smallest singular value ("null space") of
    const svd = safeSVD(A_mat);

    // Last column of V (right-singular vectors) is the solution to Ah=0
    // const h = svd.V.getColumn(svd.V.columns - 1);
    const V = svd.rightSingularVectors;
    // Extract Right Singular Vector and reshape back to 3*3 matrix
    const h = V.getColumn(V.columns - 1);

    return [h.slice(0, 3), h.slice(3, 6), h.slice(6, 9)];
}

// Apply a homography matrix to a point.
export function applyHomography(H: number[][], pt: Point): number[] {
    const {x, y} = pt;
    const denom = H[2][0] * x + H[2][1] * y + H[2][2];
    const xPrime = (H[0][0] * x + H[0][1] * y + H[0][2]) / denom;
    const yPrime = (H[1][0] * x + H[1][1] * y + H[1][2]) / denom;
    return [xPrime, yPrime];
}

// Applies homography to warp a source ImageData to a target rectangle. Uses backward mapping (iterates destination pixels, looks up source).
export function warpImageData(
    srcImage: ImageData,
    H: number[][],
    outWidth: number,
    outHeight: number
): ImageData {
    // Invert the homography for backward mapping H^-1
    const Hinv = inverse(new Matrix(H)).to2DArray();

    const output = new ImageData(outWidth, outHeight);
    const src = srcImage.data;
    const dst = output.data;

    const srcW = srcImage.width;
    const srcH = srcImage.height;


    for (let y = 0; y < outHeight; y++) {
        for (let x = 0; x < outWidth; x++) {
            // Map (x, y) in destination → (x', y') in source
            // For every pixel in destination, find pixel in source image it corresponds to
            const denom = Hinv[2][0] * x + Hinv[2][1] * y + Hinv[2][2];
            const srcX = (Hinv[0][0] * x + Hinv[0][1] * y + Hinv[0][2]) / denom;
            const srcY = (Hinv[1][0] * x + Hinv[1][1] * y + Hinv[1][2]) / denom;

            const ix = Math.floor(srcX);
            const iy = Math.floor(srcY);

            // Bounds check
            if (ix < 0 || iy < 0 || ix >= srcW || iy >= srcH) {
                continue; // leave pixel transparent
            }

            const srcIdx = (iy * srcW + ix) * 4;
            const dstIdx = (y * outWidth + x) * 4;
            // Copy pixel colour over
            dst[dstIdx] = src[srcIdx];       // R
            dst[dstIdx + 1] = src[srcIdx + 1]; // G
            dst[dstIdx + 2] = src[srcIdx + 2]; // B
            dst[dstIdx + 3] = src[srcIdx + 3]; // A
        }
    }

    return output;
}

export function cropImageData(
    source: ImageData,
    x: number,
    y: number,
    width: number,
    height: number
): ImageData {
    // console.log('Cropping image to: ', {width, height}) // Note this is protected from error now
    const output = new ImageData(width, height);
    const src = source.data;
    const dst = output.data;
    const srcWidth = source.width;

    for (let j = 0; j < height; j++) {
        for (let i = 0; i < width; i++) {
            const srcIdx = ((y + j) * srcWidth + (x + i)) * 4;
            const dstIdx = (j * width + i) * 4;

            dst[dstIdx] = src[srcIdx];       // R
            dst[dstIdx + 1] = src[srcIdx + 1]; // G
            dst[dstIdx + 2] = src[srcIdx + 2]; // B
            dst[dstIdx + 3] = src[srcIdx + 3]; // A
        }
    }

    return output;
}

// Resizes an ImageData using bilinear interpolation - matches OpenCV's cv2.resize() default behavior (INTER_LINEAR).
export function resizeImageData(
    source: ImageData,
    outWidth: number,
    outHeight: number
): ImageData {
    const output = new ImageData(outWidth, outHeight);
    const src = source.data;
    const dst = output.data;
    const { width: srcWidth, height: srcHeight } = source;

    // -1 to align with pixel centers for bilinear calc
    const xRatio = (srcWidth - 1) / outWidth;
    const yRatio = (srcHeight - 1) / outHeight;

    for (let y = 0; y < outHeight; y++) {
        for (let x = 0; x < outWidth; x++) {
            const xDiff = Math.floor(xRatio * x);
            const yDiff = Math.floor(yRatio * y);
            const xWeight = xRatio * x - xDiff;
            const yWeight = yRatio * y - yDiff;

            const index = (y * outWidth + x) * 4;
            // 4 neighbors: a=TL, b=TR, c=BL, d=BR
            const aIdx = (yDiff * srcWidth + xDiff) * 4;
            const bIdx = (yDiff * srcWidth + (xDiff + 1)) * 4;
            const cIdx = ((yDiff + 1) * srcWidth + xDiff) * 4;
            const dIdx = ((yDiff + 1) * srcWidth + (xDiff + 1)) * 4;

            for (let i = 0; i < 3; i++) { // RGB only
                const a = src[aIdx + i];
                const b = src[bIdx + i];
                const c = src[cIdx + i];
                const d = src[dIdx + i];

                dst[index + i] =
                    a * (1 - xWeight) * (1 - yWeight) +
                    b * xWeight * (1 - yWeight) +
                    c * yWeight * (1 - xWeight) +
                    d * xWeight * yWeight;
            }
            dst[index + 3] = 255;
        }
    }
    return output;
}

// Extracts, dewarps, and resizes the eye region from a face image.
// Pipeline: face landmarks → homography warp to square → crop eye strip → bilinear resize to CNN input size.
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
    srcPts = srcPts.map(({x, y}) => {
        const dx = x - center.x;
        const dy = y - center.y;
        return {
            x: x + dx * facePaddingCoefs[0],
            y: y + dy * facePaddingCoefs[1]
        };
    });

    const dstPts: Point[] = [ // 4 corners of a perfect square
        {x: 0, y: 0},
        {x: 0, y: faceCropSize},
        {x: faceCropSize, y: faceCropSize},
        {x: faceCropSize, y: 0},
    ];

    // Compute homography matrix
    // maps points from entire face in video to the flat, normalized crop
    const H = computeHomography(srcPts, dstPts);
    const warped = warpImageData(frame, H, faceCropSize, faceCropSize);

    // Crop eye strip using warped landmark positions
    const warpedLandmarks = faceLandmarks.map((pt) => applyHomography(H, pt));
    const topEyes = warpedLandmarks[151];
    const bottomEyes = warpedLandmarks[195];

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

// TODO: Testing alternative approach to EyePatch
function midpoint(point1: Point, point2: Point): Point {
    return {
        x: (point1.x+point2.x)/2,
        y: (point1.y+point2.y)/2
    }
}

export function altEyePatch(landmarks: Point[], imageData: ImageData) {
    const left = landmarks[372]; //143
    const right = landmarks[143];
    const top = landmarks[9];
    const bottom = midpoint(landmarks[229], landmarks[449]);
    const center = midpoint(landmarks[133], landmarks[362]);

    // Build 4 corners of the rotated bounding box
    // Use the eye-line angle to rotate top/bottom points
    const dx = right.x - left.x;
    const dy = right.y - left.y;
    const angle = Math.atan2(dy, dx);
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);

    const halfW = ((dx ** 2 + dy ** 2) ** 0.5) / 2;
    const topDist = ((top.x - center.x) ** 2 + (top.y - center.y) ** 2) ** 0.5;
    const botDist = ((bottom.x - center.x) ** 2 + (bottom.y - center.y) ** 2) ** 0.5;

    // 4 corners of the oriented box around center
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
    return warpImageData(imageData, H, outW, outH);
}
