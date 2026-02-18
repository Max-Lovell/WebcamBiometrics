/**
 * Plane-Orthogonal-to-Skin (POS) Algorithm
 * Based on: Wang et al. "Algorithmic Principles of Remote PPG"
 * IEEE Transactions on Biomedical Engineering, 2017
 * See: https://pure.tue.nl/ws/files/31563684/TBME_00467_2016_R1_preprint.pdf
 */

import { mean, std } from '../utils/math.ts';

// ─── Types ──────────────────────────────────────────────────────────────────

/** Three-channel RGB signal of equal length. Used across the pipeline. */
export interface RGBSignal {
    r: Float32Array;
    g: Float32Array;
    b: Float32Array;
}

/** RGB signal with associated timestamps (e.g., after interpolation). */
export interface TimedRGBSignal extends RGBSignal {
    times: Float64Array;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Channel keys for iteration — avoids duplicating per-channel logic. */
const RGB_KEYS = ['r', 'g', 'b'] as const;

/**
 * Temporal normalization: divide each channel by its mean.
 * Returns a new RGBSignal — does not mutate the input.
 *
 * Cn = C / mean(C) for each channel C ∈ {R, G, B}
 */
function temporalNormalize(rgb: RGBSignal): RGBSignal {
    const length = rgb.r.length;
    const result: Partial<Record<'r' | 'g' | 'b', Float32Array>> = {};

    for (const key of RGB_KEYS) {
        const channel = rgb[key];
        const mu = mean(channel);
        const norm = new Float32Array(length);
        for (let i = 0; i < length; i++) {
            norm[i] = channel[i] / mu;
        }
        result[key] = norm;
    }

    return result as RGBSignal;
}

// ─── Core Algorithm ─────────────────────────────────────────────────────────

/**
 * Core POS algorithm on a window of RGB samples.
 *
 * Steps from paper:
 *   1. Temporal normalization: Cn = C / mean(C)
 *   2. Projection: S = P · Cn where P = [[0, 1, -1], [-2, 1, 1]]
 *   3. Tuning: h = S1 + (σ(S1)/σ(S2)) × S2
 *   4. Mean-center h for overlap-adding
 *
 * @param rgb - RGB signal window (all channels must be the same length)
 * @returns Pulse signal array (mean-centered for overlap-adding)
 */
export function calculatePOS(rgb: RGBSignal): Float32Array {
    const l = rgb.r.length;

    // 1. Temporal normalization
    const { r: rN, g: gN, b: bN } = temporalNormalize(rgb);

    // 2. Projection onto skin-orthogonal plane
    const S1 = new Float32Array(l);
    const S2 = new Float32Array(l);

    for (let i = 0; i < l; i++) {
        S1[i] = gN[i] - bN[i];                    // [0, 1, -1] · RGB
        S2[i] = -2 * rN[i] + gN[i] + bN[i];       // [-2, 1, 1] · RGB
    }

    // 3. Tuning: combine projections weighted by their standard deviations
    const alpha = std(S2) > 0 ? std(S1) / std(S2) : 0;

    const h = new Float32Array(l);
    for (let i = 0; i < l; i++) {
        h[i] = S1[i] + alpha * S2[i];
    }

    // 4. Mean-center for overlap-add
    const hMean = mean(h);
    for (let i = 0; i < l; i++) {
        h[i] -= hMean;
    }

    return h;
}
