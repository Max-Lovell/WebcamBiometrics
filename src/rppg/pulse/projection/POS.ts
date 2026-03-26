/**
 * POS (Plane-Orthogonal-to-Skin) Method
 *
 * Based on: Wang et al. "Algorithmic Principles of Remote PPG"
 * IEEE Transactions on Biomedical Engineering, 2017
 * See 'Algorithm 1' in: https://pure.tue.nl/ws/files/31563684/TBME_00467_2016_R1_preprint.pdf
 */

import type { WindowedPulseMethod, RGBSignal } from './types';
import { mean, std } from '../../utils/math';

export class POS implements WindowedPulseMethod {
    readonly name = 'POS';
    readonly windowSize: number;
    readonly needsTemporalNormalization = false; // Must be false for tuning step

    // sampleRate in Hz, window size = e.g. 1.6*(1/sampleRate) TODO: allow to use BPM?
    constructor(sampleRate: number, windowMultiplier: number = 1.6) {
        this.windowSize = Math.ceil(sampleRate * windowMultiplier);
    }

    process(rgb: RGBSignal): Float32Array {
        const l = rgb.r.length;

        // Temporal normalization: Cn = C / mean(C)
        const rMu = mean(rgb.r);
        const gMu = mean(rgb.g);
        const bMu = mean(rgb.b);

        // Projection onto skin-orthogonal plane S=P·Cn where P=[[0, 1, -1], [-2, 1, 1]]
        const S1 = new Float32Array(l);
        const S2 = new Float32Array(l);

        for (let i = 0; i < l; i++) {
            const rN = rgb.r[i] / rMu;
            const gN = rgb.g[i] / gMu;
            const bN = rgb.b[i] / bMu;
            S1[i] = gN - bN;                    // [0, 1, -1] · [R, G, B]
            S2[i] = -2 * rN + gN + bN;          // [-2, 1, 1] · [R, G, B]
        }

        // Tuning: combine projections weighted by standard deviation ratio
        const stdS2 = std(S2);
        const alpha = stdS2 > 0 ? std(S1) / stdS2 : 0;

        const h = new Float32Array(l);
        for (let i = 0; i < l; i++) {
            h[i] = S1[i] + alpha * S2[i]; // h = S1 + (σ(S1)/σ(S2)) × S2
        }

        // Mean-center for overlap-add
        const hMu = mean(h);
        for (let i = 0; i < l; i++) {
            h[i] -= hMu;
        }

        return h; // Return full H array
    }
}
