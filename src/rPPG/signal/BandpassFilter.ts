/**
 * Bandpass Filter Module
 *
 * Implements a bandpass filter using cascaded biquad (2nd-order IIR) sections.
 * Designed for real-time rPPG signal processing at camera frame rates (~30fps).
 *
 * Architecture:
 *   BiquadFilter  — Single 2nd-order IIR section (the building block)
 *   BandpassFilter — Chains a high-pass + low-pass BiquadFilter
 *
 * Why Biquad IIR?
 *   At low sample rates (~30fps), FIR filters need many taps to achieve
 *   sharp cutoffs at low frequencies (0.7 Hz). Biquad IIR achieves this with
 *   just 5 coefficients and 4 values of state, making it ideal for real-time use.
 *
 * Why Butterworth?
 *   Maximally flat passband — no amplitude ripple within the heart rate range.
 *   We care about preserving relative amplitudes for FFT peak detection,
 *   not about phase linearity (which Butterworth doesn't guarantee, but
 *   we don't need it for frequency-domain BPM estimation).
 *
 * Reference: Robert Bristow-Johnson's Audio EQ Cookbook
 * https://www.w3.org/2011/audio/audio-eq-cookbook.html
 */

import { bpmToHz } from '../utils/math.ts';

// ─── Types ───────────────────────────────────────────────────────────────────

/** The 5 biquad coefficients: 3 feedforward (b) + 2 feedback (a) */
interface BiquadCoefficients {
    b0: number;
    b1: number;
    b2: number;
    a1: number;
    a2: number;
}

export type FilterType = 'lowpass' | 'highpass';


// ─── Coefficient Computation ─────────────────────────────────────────────────

/**
 * Compute biquad filter coefficients for a Butterworth filter.
 *
 * Uses the bilinear transform to convert an analog prototype filter to digital.
 * The bilinear transform maps the entire analog frequency axis (0 to ∞) onto
 * the digital frequency axis (0 to fs/2) using: s = (2/T) * (z-1)/(z+1)
 *
 * This introduces "frequency warping" — analog and digital frequencies don't
 * map linearly. We pre-warp the cutoff so the digital filter cuts off at
 * exactly the right frequency:
 *   ω_analog = (2/T) * tan(π * f_cutoff / f_sample)
 *
 * @param cutoffHz  - Cutoff frequency in Hz (e.g., 0.7 for high-pass, 4.0 for low-pass)
 * @param sampleRate - Sample rate in Hz (e.g., 30 for a 30fps camera)
 * @param type - 'lowpass' or 'highpass'
 * @returns BiquadCoefficients normalised so a0 = 1 (divided through)
 */
export function computeBiquadCoefficients(
    cutoffHz: number,
    sampleRate: number,
    type: FilterType
): BiquadCoefficients {
    // Pre-warp the cutoff frequency for the bilinear transform.
    // Without this, the actual digital cutoff would drift from the intended value,
    // especially as cutoff approaches Nyquist (fs/2).
    const omega = 2 * Math.PI * cutoffHz / sampleRate;  // Digital angular frequency
    const sinOmega = Math.sin(omega);
    const cosOmega = Math.cos(omega);

    // Q factor = 1/√2 ≈ 0.7071 for Butterworth (maximally flat).
    // Higher Q = sharper resonance at cutoff (Chebyshev-like).
    // Lower Q = gentler rolloff. Butterworth is the sweet spot — no ripple.
    const Q = Math.SQRT1_2; // 1/√2
    const alpha = sinOmega / (2 * Q);

    let b0: number, b1: number, b2: number, a0: number, a1: number, a2: number;

    if (type === 'lowpass') {
        // Butterworth low-pass from the Audio EQ Cookbook:
        //   H(s) = 1 / (s² + s/Q + 1)   (analog prototype)
        // After bilinear transform:
        b0 = (1 - cosOmega) / 2;
        b1 = 1 - cosOmega;            // = 2 * b0
        b2 = (1 - cosOmega) / 2;      // = b0
        a0 = 1 + alpha;
        a1 = -2 * cosOmega;
        a2 = 1 - alpha;
    } else {
        // Butterworth high-pass:
        //   H(s) = s² / (s² + s/Q + 1)   (analog prototype)
        b0 = (1 + cosOmega) / 2;
        b1 = -(1 + cosOmega);          // = -2 * b0
        b2 = (1 + cosOmega) / 2;       // = b0
        a0 = 1 + alpha;
        a1 = -2 * cosOmega;
        a2 = 1 - alpha;
    }

    // Normalise so a0 = 1 (standard Direct Form I convention).
    // This means we only need to store 5 values, not 6.
    return {
        b0: b0 / a0,
        b1: b1 / a0,
        b2: b2 / a0,
        a1: a1 / a0,
        a2: a2 / a0,
    };
}


// ─── BiquadFilter Class ─────────────────────────────────────────────────────

/**
 * A single biquad (2nd-order IIR) filter section.
 *
 * Implements the Direct Form I difference equation:
 *   y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
 *
 * State consists of just 4 numbers:
 *   x[n-1], x[n-2]  — previous two input samples
 *   y[n-1], y[n-2]  — previous two output samples
 *
 * Direct Form I is used over Direct Form II (transposed) because it's simpler
 * to understand and at 2nd-order the numerical differences are negligible
 * (DF2 matters more for higher-order or fixed-point implementations).
 */

export class BiquadFilter {
    // NOTE 2 of these are combined to create a bandpass filter
    private readonly coefficients: BiquadCoefficients;

    // Filter state (previous input and output samples)
    private x1: number = 0;  // x[n-1]
    private x2: number = 0;  // x[n-2]
    private y1: number = 0;  // y[n-1]
    private y2: number = 0;  // y[n-2]

    constructor(cutoffHz: number, sampleRate: number, type: FilterType) {
        this.coefficients = computeBiquadCoefficients(cutoffHz, sampleRate, type);
    }

    /**
     * Process a single sample through the filter.
     *
     * This is designed to be called once per frame — feed it each new POS H value
     * as it arrives and it returns the filtered value immediately. No buffering needed.
     *
     * @param x - Input sample (e.g., a POS H value)
     * @returns Filtered output sample
     */
    process(x: number): number {
        const { b0, b1, b2, a1, a2 } = this.coefficients;

        // The core difference equation:
        // Feedforward (b terms): weighted sum of current and past inputs
        // Feedback (a terms): weighted sum of past outputs (this is what makes it IIR)
        const y = b0 * x + b1 * this.x1 + b2 * this.x2
            - a1 * this.y1 - a2 * this.y2;

        // Shift state for next call
        this.x2 = this.x1;
        this.x1 = x;
        this.y2 = this.y1;
        this.y1 = y;

        return y;
    }

    /**
     * Process an entire buffer of samples.
     * Useful for batch processing (e.g., filtering an already-collected signal
     * for visualisation), but the filter state carries across calls either way.
     *
     * @param input - Array of input samples
     * @returns New Float32Array of filtered samples
     */
    processBuffer(input: Float32Array): Float32Array {
        const output = new Float32Array(input.length);
        for (let i = 0; i < input.length; i++) {
            output[i] = this.process(input[i]);
        }
        return output;
    }

    /**
     * Reset filter state to zero.
     * Call this when tracking is lost, scene changes, or any discontinuity
     * in the input signal — otherwise the filter's feedback terms will
     * "remember" the old signal and cause a transient (ringing artifact).
     */
    reset(): void {
        this.x1 = 0;
        this.x2 = 0;
        this.y1 = 0;
        this.y2 = 0;
    }
}


// ─── BandpassFilter Class ────────────────────────────────────────────────────

/**
 * Bandpass filter implemented as a cascaded high-pass → low-pass chain.
 *
 * Why cascade rather than a single bandpass biquad?
 *   - Independent control of each cutoff
 *   - Each section is a standard Butterworth (maximally flat)
 *   - A single biquad bandpass has a resonant peak that distorts amplitudes
 *   - Easier to reason about and debug
 *
 * Signal flow:
 *   input → [high-pass @ lowCutHz] → [low-pass @ highCutHz] → output
 *
 * The high-pass removes slow drift (breathing, lighting changes, movement).
 * The low-pass removes high-frequency noise (sensor noise, quantization).
 * What remains is the heart rate band.
 */
export class BandpassFilter {
    private readonly highPass: BiquadFilter;
    private readonly lowPass: BiquadFilter;

    /**
     * @param lowCutHz  - High-pass cutoff in Hz. Frequencies below this are removed.
     *                    Default 0.7 Hz = 42 BPM (covers bradycardia)
     * @param highCutHz - Low-pass cutoff in Hz. Frequencies above this are removed.
     *                    Default 4.0 Hz = 240 BPM (covers extreme tachycardia/exercise)
     * @param sampleRate - Sample rate in Hz (your camera FPS after interpolation)
     */
    constructor(
        lowCutHz: number = 0.7,
        highCutHz: number = 4.0,
        sampleRate: number = 30
    ) {
        this.highPass = new BiquadFilter(lowCutHz, sampleRate, 'highpass');
        this.lowPass = new BiquadFilter(highCutHz, sampleRate, 'lowpass');
    }

    /**
     * Create a BandpassFilter with cutoffs specified in BPM.
     *
     * Usage:
     *   const filter = BandpassFilter.fromBPM(42, 240, 30);
     *   // equivalent to: new BandpassFilter(0.7, 4.0, 30)
     *
     * @param lowCutBPM  - High-pass cutoff in BPM (heart rates below this are removed)
     * @param highCutBPM - Low-pass cutoff in BPM (heart rates above this are removed)
     * @param sampleRate - Sample rate in Hz
     */
    static fromBPM(lowCutBPM: number, highCutBPM: number, sampleRate: number = 30): BandpassFilter {
        return new BandpassFilter(bpmToHz(lowCutBPM), bpmToHz(highCutBPM), sampleRate);
    }

    /** Process a single sample through both filter stages */
    process(sample: number): number {
        return this.lowPass.process(this.highPass.process(sample));
    }

    /** Process an entire buffer through both filter stages */
    processBuffer(input: Float32Array): Float32Array {
        // We run high-pass first on the full buffer, then low-pass on the result.
        // This is equivalent to processing sample-by-sample through both,
        // because each filter maintains its own internal state.
        const afterHighPass = this.highPass.processBuffer(input);
        return this.lowPass.processBuffer(afterHighPass);
    }

    /** Reset both filter stages */
    reset(): void {
        this.highPass.reset();
        this.lowPass.reset();
    }
}
