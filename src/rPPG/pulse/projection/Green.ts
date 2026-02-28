/**
 * Green Channel Method
 *
 * The simplest rPPG approach — just the green channel.
 * Useful as a baseline for benchmarking more sophisticated methods.
 *
 * Reference: Verkruysse et al. "Remote plethysmographic imaging using
 * ambient light", Optics Express, 2008
 */

import type { WindowedPulseMethod, RGBSignal } from './types.ts';
import { mean } from '../../utils/math.ts';

export class Green implements WindowedPulseMethod {
    readonly name = 'Green';
    readonly windowSize: number;
    readonly needsTemporalNormalization = false;

    constructor(sampleRate: number, windowMultiplier: number = 1.6) {
        this.windowSize = Math.ceil(sampleRate * windowMultiplier);
    }

    process(rgb: RGBSignal): Float32Array {
        const l = rgb.g.length;
        const h = new Float32Array(l);
        const mu = mean(rgb.g);

        for (let i = 0; i < l; i++) {
            h[i] = rgb.g[i] - mu;
        }

        return h;
    }
}
