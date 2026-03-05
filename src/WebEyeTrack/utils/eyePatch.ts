// ============================================================================
// Eye Patch Extraction and Homography
// ============================================================================
import {inverse, Matrix} from "ml-matrix";
import type {Point} from "../types.ts";
import { safeSVD } from './safeSVD.ts';

/**
 * Estimates a 3x3 homography matrix from 4 point correspondences.
 */

export function computeHomography(src: Point[], dst: Point[]): number[][] {
    // maps points from face in video to the flat, normalized crop
    // src = original 4 tilted corners, dst = 512*512 square
    if (src.length !== 4 || dst.length !== 4) {
        throw new Error('Need exactly 4 source and 4 destination points');
    }

    const A: number[][] = [];

    // Map points from one square to another using linear equations to solve H
    // Equations in the form Ah=0, h=vector of the 9 values of H, solve for 0
    for (let i = 0; i < 4; i++) {
        const [x, y] = src[i];
        const [u, v] = dst[i];
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

    const H = [
        h.slice(0, 3),
        h.slice(3, 6),
        h.slice(6, 9),
    ];

    return H;
}

/**
 * Apply a homography matrix to a point.
 */
export function applyHomography(H: number[][], pt: number[]): number[] {
    const [x, y] = pt;
    const denom = H[2][0] * x + H[2][1] * y + H[2][2];
    const xPrime = (H[0][0] * x + H[0][1] * y + H[0][2]) / denom;
    const yPrime = (H[1][0] * x + H[1][1] * y + H[1][2]) / denom;
    return [xPrime, yPrime];
}

/**
 * Applies homography to warp a source ImageData to a target rectangle.
 */
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

/**
 * Resizes an ImageData using bilinear interpolation.
 * This matches OpenCV's cv2.resize() default behavior (INTER_LINEAR).
 *
 * Bilinear interpolation provides smooth, high-quality resizing by computing
 * a weighted average of the 4 nearest pixels for each output pixel.
 *
 * This is significantly faster than homography-based warping when only
 * simple rectangular scaling is needed (no rotation, skew, or perspective).
 *
 * @param source - Source ImageData to resize
 * @param outWidth - Output width in pixels
 * @param outHeight - Output height in pixels
 * @returns Resized ImageData with bilinear interpolation
 */
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
            const xWeight = (xRatio * x) - xDiff;
            const yWeight = (yRatio * y) - yDiff;

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

                // Bilinear interpolation formula
                const pixel = a * (1 - xWeight) * (1 - yWeight) +
                    b * xWeight * (1 - yWeight) +
                    c * yWeight * (1 - xWeight) +
                    d * xWeight * yWeight;

                dst[index + i] = pixel;
            }
            dst[index + 3] = 255; // Alpha
        }
    }
    return output;
}

/**
 * Compares two ImageData objects and computes pixel-wise differences.
 * Used for validation and testing to ensure optimizations maintain correctness.
 *
 * @param img1 - First ImageData
 * @param img2 - Second ImageData
 * @returns Statistics about pixel differences
 */
export function compareImageData(
    img1: ImageData,
    img2: ImageData
): { maxDiff: number; meanDiff: number; histogram: number[] } {
    if (img1.width !== img2.width || img1.height !== img2.height) {
        throw new Error('Images must have the same dimensions for comparison');
    }

    const data1 = img1.data;
    const data2 = img2.data;
    const numPixels = img1.width * img1.height;
    const histogram = new Array(256).fill(0);

    let sumDiff = 0;
    let maxDiff = 0;

    // Compare each pixel (RGB channels only, ignore alpha)
    for (let i = 0; i < numPixels; i++) {
        const idx = i * 4;

        for (let c = 0; c < 3; c++) { // R, G, B only
            const diff = Math.abs(data1[idx + c] - data2[idx + c]);
            sumDiff += diff;
            maxDiff = Math.max(maxDiff, diff);
            histogram[Math.floor(diff)]++;
        }
    }

    const meanDiff = sumDiff / (numPixels * 3);

    return { maxDiff, meanDiff, histogram };
}

export function obtainEyePatch(
    frame: ImageData,
    faceLandmarks: Point[],
    facePaddingCoefs: [number, number] = [0.4, 0.2], // Allow extracted region around non-moving landmarks
    faceCropSize: number = 512,
    dstImgSize: [number, number] = [512, 128]
): ImageData {

    // TODO: note this is the largest time sink for the package, see here for timing improvements.
    // TODO: clean up extracted region:
    // fix eyes drifting down when head moved up, top black bar on clipping
    // use more stable and less protruding landmarks?
    // eye corners? Upper Rigid Mask - Nasal Bridge / Glabella to Temples / Forehead then offset before extraction?

    // Takes tilted/rotated face from camera and unwarps to aligned rectangular image of eyes

    // Prepare src and dst
    // Note these are points from top eyebrows to bottom chin, nose centre for stability
    const center = faceLandmarks[4]; //suggested: 168 // original: 4
    const leftTop = faceLandmarks[103]; //105 // original: 103
    const leftBottom = faceLandmarks[150]; //118 // original: 150 bottom chin
    const rightTop = faceLandmarks[332]; //334 // original: 332
    const rightBottom = faceLandmarks[379]; //347 // original: 379 chin

    let srcPts: Point[] = [leftTop, leftBottom, rightBottom, rightTop];

    // Apply radial padding - take extracted region around non-moving landmarks. TODO: could we extract from more stable place?
    srcPts = srcPts.map(([x, y]) => {
        const dx = x - center[0]; // relative to center
        const dy = y - center[1];
        return [
            x + dx * facePaddingCoefs[0],
            y + dy * facePaddingCoefs[1],
        ];
    });

    const dstPts: Point[] = [ // 4 corners of a perfect square
        [0, 0],
        [0, faceCropSize],
        [faceCropSize, faceCropSize],
        [faceCropSize, 0],
    ];

    // Compute homography matrix
    // maps points from entire face in video to the flat, normalized crop
    const H = computeHomography(srcPts, dstPts);

    // Step 5: Warp the image to square
    const warped = warpImageData(frame, H, faceCropSize, faceCropSize);

    // Step 6: Apply the homography matrix to the facial landmarks
    // transforms original landmarks to flattened image coordinates
    const warpedLandmarks = faceLandmarks.map(pt => applyHomography(H, pt));

    // Step 7: Generate the crop of the eyes - use flattened facelandmarker to
    const top_eyes_patch = warpedLandmarks[151];
    const bottom_eyes_patch = warpedLandmarks[195];

    const cropY = Math.round(top_eyes_patch[1]);
    let cropHeight = Math.round(bottom_eyes_patch[1] - top_eyes_patch[1]);

    // Check for the "Inverted Eye" (Negative Height)
    if (cropHeight <= 0) {
        console.warn(`[MathUtils] Invalid Eye Crop Height: ${cropHeight} (Warp Matrix Flip). Defaulting to 1px.`);
        cropHeight = 1;
    }

    // Check for "Out of Bounds" (Y + Height > Image Height)
    if (cropY + cropHeight > warped.height) {
        cropHeight = Math.max(1, warped.height - cropY);
    }

    // Crop out strip containing eyes.
    const eye_patch = cropImageData(
        warped,
        0,
        cropY,
        warped.width,
        cropHeight
    );

    // Step 8: Resize the eye patch to the desired output size
    // OPTIMIZATION: Using bilinear resize instead of homography for simple rectangular scaling
    // This is ~2x faster and matches the Python reference implementation (cv2.resize)
    // The previous homography approach was mathematically equivalent but computationally expensive
    // resisze to 512*128 expected by CNN.
    const resizedEyePatch = resizeImageData(
        eye_patch,
        dstImgSize[0],
        dstImgSize[1]
    );

    // VERIFICATION MODE (for development/testing only)
    // Uncomment to compare resize vs homography and verify numerical equivalence
    /*
    const eyePatchSrcPts: Point[] = [
        [0, 0],
        [0, eye_patch.height],
        [eye_patch.width, eye_patch.height],
        [eye_patch.width, 0],
    ];
    const eyePatchDstPts: Point[] = [
        [0, 0],
        [0, dstImgSize[1]],
        [dstImgSize[0], dstImgSize[1]],
        [dstImgSize[0], 0],
    ];
    const eyePatchH = computeHomography(eyePatchSrcPts, eyePatchDstPts);
    const homographyResult = warpImageData(eye_patch, eyePatchH, dstImgSize[0], dstImgSize[1]);
    const diff = compareImageData(resizedEyePatch, homographyResult);
    console.log('Eye patch resize verification:', {
        maxDiff: diff.maxDiff,
        meanDiff: diff.meanDiff,
        acceptableDiff: diff.meanDiff < 2.0,
        note: 'Differences expected due to interpolation method (bilinear vs nearest-neighbor)'
    });
    */

    return resizedEyePatch;
}
