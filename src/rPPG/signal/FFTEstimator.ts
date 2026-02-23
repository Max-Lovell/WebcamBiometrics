/**
 * FFT Estimator — Tier 3b
 * Batch FFT spectral analysis for heart rate estimation.
 *
 * More precise than peak detection (sub-bin parabolic interpolation),
 * better noise averaging over the signal window, but slower to respond
 * to HR changes and susceptible to harmonic confusion.
 *
 * Rate limiting:
 *   FFT doesn't need to run every frame — the signal only changes
 *   meaningfully over ~0.5s. The `update()` method tracks frames and
 *   only re-estimates every `estimateInterval` frames, returning the
 *   cached result in between.
 *
 * Batch filtering:
 *   Owns a pre-allocated work array and creates a fresh BandpassFilter
 *   per estimation (clean state, no transient bleed from previous runs).
 *   The raw fused signal from PulseProcessor is filtered before FFT analysis.
 *
 * Usage:
 *   const fft = new FFTEstimator(pulseProcessor.signalLength, { sampleRate: 30 });
 *
 *   // Per frame — handles rate limiting internally:
 *   const result = fft.update(pulseProcessor.getFusedSignal());
 */

import { computeSpectrum, findDominantFrequency, hanningWindow } from './FFT';
import { BandpassFilter } from './BandpassFilter';
import type { PipelineConfig } from '../types';
import { DEFAULT_PIPELINE_CONFIG } from '../types';

// ─── Types ──────────────────────────────────────────────────────────────────

export interface FFTEstimatorConfig extends PipelineConfig {
    /**
     * Re-estimate every N frames.
     * At 30fps, 15 frames ≈ 0.5s between FFT runs.
     * Default: 15
     */
    estimateInterval: number;
}

export const DEFAULT_FFT_CONFIG: FFTEstimatorConfig = {
    ...DEFAULT_PIPELINE_CONFIG,
    estimateInterval: 15,
};

export interface FFTEstimate {
    /** Estimated BPM (with sub-bin parabolic interpolation) */
    bpm: number;
    /**
     * Confidence: ratio of peak magnitude to total in-band magnitude.
     * Higher = more energy concentrated in one frequency = cleaner signal.
     */
    confidence: number;
    /** Precise frequency in Hz */
    frequencyHz: number;
    /** Magnitude at the dominant peak */
    peakMagnitude: number;
    /** Discriminant for fusion logic */
    method: 'fft';
}

// ─── FFTEstimator Class ─────────────────────────────────────────────────────

export class FFTEstimator {
    private readonly config: FFTEstimatorConfig;

    /** Cached Hanning window (only depends on signal length, computed once) */
    private readonly window: Float32Array;

    /** Pre-allocated work array for batch-filtered signal */
    private readonly filteredWork: Float32Array;

    /** Frame counter for rate limiting */
    private frameCount: number = 0;

    /** Most recent estimate (returned between re-estimations) */
    private lastEstimate: FFTEstimate | null = null;

    /**
     * @param signalLength - Length of the signal buffer from PulseProcessor.
     *   Use pulseProcessor.signalLength.
     * @param config - Estimation parameters
     */
    constructor(signalLength: number, config?: Partial<FFTEstimatorConfig>) {
        this.config = { ...DEFAULT_FFT_CONFIG, ...config };
        this.window = hanningWindow(signalLength);
        this.filteredWork = new Float32Array(signalLength);
    }

    // ─── Public API ─────────────────────────────────────────────────────

    /**
     * Per-frame update with internal rate limiting.
     *
     * Call every frame with the raw fused signal (or null if not ready).
     * Only actually runs bandpass + FFT every `estimateInterval` frames.
     * Returns the latest estimate (which may be from a previous run).
     *
     * @param rawSignal - Raw fused signal from PulseProcessor.getFusedSignal(), or null
     * @returns Latest FFT estimate, or null if no estimate has been made yet
     */
    update(rawSignal: Float32Array | null): FFTEstimate | null {
        this.frameCount++;

        if (rawSignal && this.frameCount >= this.config.estimateInterval) {
            this.frameCount = 0;
            this.estimate(rawSignal);
        }

        return this.lastEstimate;
    }

    /**
     * Force an immediate estimation, bypassing rate limiting.
     * Batch-filters the signal internally before running FFT.
     *
     * @param rawSignal - Raw fused signal in chronological order.
     *   Length must match the signalLength passed to the constructor.
     * @returns FFTEstimate or null if no valid peak found
     */
    estimate(rawSignal: Float32Array): FFTEstimate | null {
        // Batch filter with a fresh bandpass (clean state, no transient bleed)
        const filtered = this.batchFilter(rawSignal);

        const spectrum = computeSpectrum(filtered, this.config.sampleRate, this.window);
        const peak = findDominantFrequency(spectrum, this.config);

        if (peak) {
            this.lastEstimate = {
                bpm: peak.frequencyBPM,
                confidence: peak.snr,
                frequencyHz: peak.frequencyHz,
                peakMagnitude: peak.peakMagnitude,
                method: 'fft',
            };
        }

        return this.lastEstimate;
    }

    /** Get the most recent estimate without running FFT. */
    getLatest(): FFTEstimate | null {
        return this.lastEstimate;
    }

    /** Whether enough frames have passed to warrant re-estimation */
    shouldEstimate(): boolean {
        return this.frameCount >= this.config.estimateInterval;
    }

    /** Reset state (clears cached estimate and frame counter) */
    reset(): void {
        this.frameCount = 0;
        this.lastEstimate = null;
    }

    // ─── Internals ──────────────────────────────────────────────────────

    /**
     * Batch-filter the raw signal through a fresh bandpass filter.
     * Uses a new filter instance each time to avoid transient bleed
     * from previous estimations.
     */
    private batchFilter(rawSignal: Float32Array): Float32Array {
        const filter = BandpassFilter.fromBPM(
            this.config.minBPM,
            this.config.maxBPM,
            this.config.sampleRate
        );

        for (let i = 0; i < rawSignal.length; i++) {
            this.filteredWork[i] = filter.process(rawSignal[i]);
        }

        return this.filteredWork;
    }
}
