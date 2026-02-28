/**
 * PBV (Plane-orthogonal-to-Blood-Volume) Method
 *
 * Reference: de Haan & van Leest, "Improved motion robustness of remote-PPG
 * by using the blood volume pulse signature", Physiological Measurement, 2014
 */

import type { WindowedPulseMethod, RGBSignal } from './types.ts';
import { mean } from '../../utils/math.ts';

// Default PBV signature — empirically derived relative pulsatile amplitudes.
// These values are from the original paper's skin optics model.
const DEFAULT_PBV_SIGNATURE = { r: 0.169, g: 0.601, b: 0.230 };

export class PBV implements WindowedPulseMethod {
    readonly name = 'PBV';
    readonly windowSize: number;
    readonly needsTemporalNormalization = false;

    private readonly pbvR: number;
    private readonly pbvG: number;
    private readonly pbvB: number;

    constructor(
        sampleRate: number,
        windowMultiplier: number = 1.6,
        signature: { r: number; g: number; b: number } = DEFAULT_PBV_SIGNATURE
    ) {
        this.windowSize = Math.ceil(sampleRate * windowMultiplier);
        // Normalise signature to unit length
        const norm = Math.sqrt(signature.r ** 2 + signature.g ** 2 + signature.b ** 2);
        this.pbvR = signature.r / norm;
        this.pbvG = signature.g / norm;
        this.pbvB = signature.b / norm;
    }

    process(rgb: RGBSignal): Float32Array {
        const l = rgb.r.length;

        // Temporal normalization
        const rMu = mean(rgb.r);
        const gMu = mean(rgb.g);
        const bMu = mean(rgb.b);

        // Normalised channels
        const rN = new Float32Array(l);
        const gN = new Float32Array(l);
        const bN = new Float32Array(l);

        for (let i = 0; i < l; i++) {
            rN[i] = rgb.r[i] / rMu;
            gN[i] = rgb.g[i] / gMu;
            bN[i] = rgb.b[i] / bMu;
        }

        // PBV projection: project normalised RGB onto the blood volume pulse signature
        // This gives us the pulse component directly
        const h = new Float32Array(l);
        for (let i = 0; i < l; i++) {
            h[i] = this.pbvR * rN[i] + this.pbvG * gN[i] + this.pbvB * bN[i];
        }

        // Mean-center for overlap-add
        const hMu = mean(h);
        for (let i = 0; i < l; i++) {
            h[i] -= hMu;
        }

        return h;
    }
}
