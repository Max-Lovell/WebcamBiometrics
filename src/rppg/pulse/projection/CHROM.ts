/**
 * CHROM (Chrominance-Based) Method
 * Based on: De Haan & Jeanne, "Robust Pulse Rate from Chrominance-Based rPPG"
 * IEEE Transactions on Biomedical Engineering, 2013
 */

import type { WindowedPulseMethod, RGBSignal} from './types';
import { mean, std } from '../../utils/math';

export class CHROM implements WindowedPulseMethod {
    readonly name = 'CHROM';
    readonly windowSize: number;
    readonly needsTemporalNormalization = false;

    constructor(sampleRate: number, windowMultiplier: number = 1.6) {
        this.windowSize = Math.ceil(sampleRate * windowMultiplier);
    }

    process(rgb: RGBSignal): Float32Array {
        const l = rgb.r.length;

        // Temporal normalization - Cn = C / mean(C)
        const rMu = mean(rgb.r);
        const gMu = mean(rgb.g);
        const bMu = mean(rgb.b);

        // Chrominance projection
        const Xs = new Float32Array(l); // Xs = 3Rn - 2Gn (motion-sensitive axis)
        const Ys = new Float32Array(l); // Ys = 1.5Rn + Gn - 1.5Bn (pulse-sensitive axis)

        for (let i = 0; i < l; i++) {
            const rN = rgb.r[i] / rMu;
            const gN = rgb.g[i] / gMu;
            const bN = rgb.b[i] / bMu;
            Xs[i] = 3 * rN - 2 * gN;              // [3, -2, 0] · [R, G, B]
            Ys[i] = 1.5 * rN + gN - 1.5 * bN;     // [1.5, 1, -1.5] · [R, G, B]
        }

        // Tuning: subtract scaled motion component -
        const stdYs = std(Ys);
        const alpha = stdYs > 0 ? std(Xs) / stdYs : 0;

        const h = new Float32Array(l);
        for (let i = 0; i < l; i++) {
            h[i] = Xs[i] - alpha * Ys[i]; // h = Xs - (σ(Xs)/σ(Ys)) × Ys
        }

        // Mean-center for overlap-add
        const hMu = mean(h);
        for (let i = 0; i < l; i++) {
            h[i] -= hMu;
        }

        return h;
    }
}
