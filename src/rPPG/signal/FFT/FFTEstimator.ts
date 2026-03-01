/**
 * FFT Estimator
 * Batch FFT spectral analysis for heart rate estimation - calls FFT.ts
 */

import { computeSpectrum, findDominantFrequency, hanningWindow } from './FFT.ts';
import { BandpassFilter } from '../BandpassFilter.ts';
import type { PipelineConfig } from '../../types.ts';
import { DEFAULT_PIPELINE_CONFIG } from '../../types.ts';

// ─── Types ──────────────────────────────────────────────────────────────────

export interface FFTEstimatorConfig extends PipelineConfig {
    estimateInterval: number; // Re-estimate every N frames - default 15
}

export const DEFAULT_FFT_CONFIG: FFTEstimatorConfig = {
    ...DEFAULT_PIPELINE_CONFIG,
    estimateInterval: 15,
};

export interface FFTEstimate {
    bpm: number; // Estimated BPM (with sub-bin parabolic interpolation)
    confidence: number; // ratio of peak magnitude to total in-band magnitude. Higher = more energy concentrated in one frequency = cleaner signal.
    frequencyHz: number; // Precise frequency in Hz
    peakMagnitude: number; // Magnitude at the dominant peak
    method: 'fft'; // Discriminant for fusion logic
}

// ─── FFTEstimator Class ─────────────────────────────────────────────────────

export class FFTEstimator {
    private readonly config: FFTEstimatorConfig;
    private readonly window: Float32Array; // Cached Hanning window (only depends on signal length, computed once)
    private readonly filteredWork: Float32Array; // Pre-allocated work array for batch-filtered signal
    private frameCount: number = 0; // Frame counter for rate limiting
    private lastEstimate: FFTEstimate | null = null; // Most recent estimate (returned between re-estimations)
    constructor(signalLength: number, config?: Partial<FFTEstimatorConfig>) {
        this.config = { ...DEFAULT_FFT_CONFIG, ...config };
        this.window = hanningWindow(signalLength);
        this.filteredWork = new Float32Array(signalLength); // TODO: assumes PulseProcessor.signalLength and signalWindowSeconds are static...
    }

    // ─── Public API ─────────────────────────────────────────────────────
    // Per-frame update with internal rate limiting. Call every frame with the raw fused signal (or null if not ready).
    update(rawSignal: Float32Array | null): FFTEstimate | null {
        this.frameCount++;
        // Only actually runs bandpass + FFT every `estimateInterval` frames.
        if (rawSignal && this.frameCount >= this.config.estimateInterval) {
            this.frameCount = 0;
            this.estimate(rawSignal);
        }

        return this.lastEstimate; // Returns the latest estimate (which may be from a previous run).
    }

    // Force an immediate estimation, bypassing rate limiting. Batch-filters the signal internally before running FFT.
    estimate(rawSignal: Float32Array): FFTEstimate | null {
        // TODO: Fresh filter is slightly wasteful. Could run filter forward, reverse output, re-run, reverse back (filtfilt-style)
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

    // Get the most recent estimate without running FFT.
    getLatest(): FFTEstimate | null {
        return this.lastEstimate;
    }

    // Whether enough frames have passed to warrant re-estimation
    shouldEstimate(): boolean {
        return this.frameCount >= this.config.estimateInterval;
    }

    // Reset state (clears cached estimate and frame counter)
    reset(): void {
        this.frameCount = 0;
        this.lastEstimate = null;
    }

    // ─── Internals ──────────────────────────────────────────────────────
    // Batch-filter the raw signal through a fresh bandpass filter.
    private batchFilter(rawSignal: Float32Array): Float32Array {
        // Uses a new filter instance each time to avoid transient bleed from previous estimations.
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
