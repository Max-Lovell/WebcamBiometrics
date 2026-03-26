/**
 * GRGB (Green-Red + Green-Blue Ratio) Method
 *
 * Reference: de Haan & Jeanne, "Robust Pulse Rate from Chrominance-Based rPPG"
 * (discussed as a baseline comparison method)
 */

import type { WindowedPulseMethod, RGBSignal } from './types';
import { mean } from '../../utils/math';

export class GRGB implements WindowedPulseMethod {
    readonly name = 'GRGB';
    readonly windowSize: number;
    readonly needsTemporalNormalization = false;

    constructor(sampleRate: number, windowMultiplier: number = 1.6) {
        this.windowSize = Math.ceil(sampleRate * windowMultiplier);
    }

    process(rgb: RGBSignal): Float32Array {
        const l = rgb.r.length;
        const h = new Float32Array(l);

        for (let i = 0; i < l; i++) {
            // Guard against zero division (shouldn't occur with real pixel data)
            const r = rgb.r[i] || 1e-6;
            const b = rgb.b[i] || 1e-6;
            h[i] = rgb.g[i] / r + rgb.g[i] / b;
        }

        // Mean-center for overlap-add
        const hMu = mean(h);
        for (let i = 0; i < l; i++) {
            h[i] -= hMu;
        }

        return h;
    }
}
