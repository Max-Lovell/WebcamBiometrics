/**
 * Plane-Orthogonal-to-Skin (POS) Algorithm
 * Based on: Wang et al. "Algorithmic Principles of Remote PPG"
 * IEEE Transactions on Biomedical Engineering, 2017
 * See: https://pure.tue.nl/ws/files/31563684/TBME_00467_2016_R1_preprint.pdf
 */

/**
 * Core POS algorithm on a window of RGB samples
 *
 * Steps from paper:
 * Temporal normalization: Cn = C / mean(C)
 * Projection: S = P · Cn where P = [[0, 1, -1], [-2, 1, 1]]
 * Tuning: h = S1 + (σ(S1)/σ(S2)) × S2
 *
 * @param r - Red channel values
 * @param g - Green channel values
 * @param b - Blue channel values
 * @returns Pulse signal value (mean-centered for overlap-adding)
 */

import { mean, std } from '../utils/math.ts';

// TODO: just pass the object into this function, and then loop through to cut down on duplication.
export function calculatePOS(r: Float32Array, g: Float32Array, b: Float32Array): Float32Array {
    const l = r.length;

    // Temporal normalization (Cn = C / mean(C))
    const rMean = mean(r);
    const gMean = mean(g);
    const bMean = mean(b);

    const rNorm = new Float32Array(l);
    const gNorm = new Float32Array(l);
    const bNorm = new Float32Array(l);

    for (let i = 0; i < l; i++) {
        rNorm[i] = r[i] / rMean;
        gNorm[i] = g[i] / gMean;
        bNorm[i] = b[i] / bMean;
    }

    // Projection
    const S1 = new Float32Array(l);
    const S2 = new Float32Array(l);

    for (let i = 0; i < l; i++) {
        S1[i] = gNorm[i] - bNorm[i]; // S1 = G - B, i.e. [0,1,-1] * RGB
        S2[i] = (-2 * rNorm[i]) + gNorm[i] + bNorm[i]; // S2 = -2R + G + B, e.g. (-2,1,1)*RGB
    }

    // Tuning
    const stdS1 = std(S1);
    const stdS2 = std(S2);
    const alpha = stdS2 > 0 ? stdS1 / stdS2 : 0;

    const h = new Float32Array(l);
    for (let i = 0; i < l; i++) {
        h[i] = S1[i] + alpha * S2[i];
    }

    // Return h array for overlap adding in main script TODO: consider this being part of a class containing POS array.
    const hMean = mean(h);
    for (let i = 0; i < l; i++) {
        h[i] -= hMean;
    }
    return h;
}
