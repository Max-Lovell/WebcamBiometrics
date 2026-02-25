/**
 * Peak Estimator
 * Streaming peak detection on fused POS samples.
 *
 * Detects local maxima in the pulse waveform, records inter-peak intervals,
 * and estimates BPM from the median interval. Operates one-sample-at-a-time
 * so it can run inline with the frame loop — no buffering or batch processing.
 *
 * Detection algorithm:
 *   - Three-sample sliding window: checks if sample[t-1] > sample[t-2] AND sample[t-1] > sample[t]
 *   - Adaptive envelope: decays per sample, jumps on large amplitudes
 *   - Amplitude threshold: peak must exceed fraction of envelope
 *   - Refractory period: minimum samples between peaks (derived from maxBPM)
 *
 * BPM estimation:
 *   - Maintains a ring buffer of recent inter-peak intervals
 *   - BPM = 60 × sampleRate / median(intervals)
 *   - Confidence = fraction of intervals within 20% of median
 */

import { median } from '../utils/math.ts';
import type { PipelineConfig } from '../types.ts';
import { DEFAULT_PIPELINE_CONFIG } from '../types.ts';

// ─── Types ──────────────────────────────────────────────────────────────────

export interface PeakEstimatorConfig extends PipelineConfig {
    amplitudeThreshold: number; // Fraction of adaptive envelope a peak must exceed. (0–1). Default: 0.3
    envelopeDecay: number; // Envelope decay per sample. Default: 0.998
    maxIntervals: number; // Max inter-peak intervals to keep for median. Default: 8
    minIntervals: number; // Min intervals before producing an estimate. Default: 3
}

export const DEFAULT_PEAK_CONFIG: PeakEstimatorConfig = {
    ...DEFAULT_PIPELINE_CONFIG,
    envelopeDecay: 0.998,
    amplitudeThreshold: 0.2,
    maxIntervals: 8,
    minIntervals: 2,
};

export interface PeakResult {
    bpm: number;
    confidence: number; // Fraction of intervals within 20% of median
    intervalCount: number;
    method: 'peak';
}

// ─── PeakEstimator Class ────────────────────────────────────────────────────

export class PeakEstimator {
    private readonly config: PeakEstimatorConfig;

    // Three-sample sliding window
    private prevPrev: number = 0;   // sample at t-2
    private prev: number = 0;       // sample at t-1 (candidate peak)
    private sampleCount: number = 0;
    private samplesSinceLast: number = 0;

    // Adaptive amplitude envelope
    private envelope: number = 0;

    // Refractory period: minimum samples between peaks (from maxBPM)
    private readonly minDistance: number;

    // Inter-peak interval ring buffer
    private readonly intervals: Float32Array;
    private intervalWrite: number = 0;
    private intervalCount: number = 0;
    private firstPeakSeen: boolean = false; // Skip first interval (not a real IBI)

    // Cached result
    private cachedResult: PeakResult | null = null; // TODO: why did I cache results again??
    peakDetectedThisFrame: boolean = false;

    constructor(config?: Partial<PeakEstimatorConfig>) {
        this.config = { ...DEFAULT_PEAK_CONFIG, ...config };
        // At maxBPM=200 and 30fps: 60/200 * 30 = 9 samples minimum between peaks
        this.minDistance = Math.floor((60 / this.config.maxBPM) * this.config.sampleRate);
        this.intervals = new Float32Array(this.config.maxIntervals);
    }

    // ─── Public API ─────────────────────────────────────────────────────
    // Called once per frame on POS sample. Returns updated PeakResult if enough intervals have been collected, else null.
    pushSample(sample: number): PeakResult | null {
        // Maintains 3 sample sliding window (t-2, t-1, t)
        this.sampleCount++;
        this.samplesSinceLast++;
        this.peakDetectedThisFrame = false;

        // Update adaptive envelope: decay, then jump if new sample is larger
        // If no peak detected for a while, decay faster
        if (this.samplesSinceLast > this.config.sampleRate * 2) {
            this.envelope *= 0.98; // No peak for 2 seconds — something's wrong, decay fast
        } else {
            this.envelope *= this.config.envelopeDecay;
        }
        const absSample = Math.abs(sample);
        if (absSample > this.envelope) {
            this.envelope = absSample;
        }

        // Check if prev (t-1) was a local maximum — need at least 3 samples
        if (this.sampleCount >= 3) {
            const threshold = this.envelope * this.config.amplitudeThreshold;

            if (
                this.prev > this.prevPrev &&              // rising edge
                this.prev > sample &&                      // falling edge (prev is the peak)
                this.prev > threshold &&                   // above adaptive threshold
                this.samplesSinceLast >= this.minDistance   // refractory period
            ) {
                this.onPeakDetected();
            }
        }

        // Shift window forward
        this.prevPrev = this.prev;
        this.prev = sample;

        return this.cachedResult;
    }

    // Most recent estimate, or null if not enough data yet.
    getLatest(): PeakResult | null {
        return this.cachedResult;
    }

    // Whether enough intervals have been collected for a valid estimate.
    isReady(): boolean {
        return this.intervalCount >= this.config.minIntervals;
    }

    // Reset all state. Call when face tracking is lost or scene changes.
    reset(): void {
        this.prevPrev = 0;
        this.prev = 0;
        this.sampleCount = 0;
        this.samplesSinceLast = 0;
        this.envelope = 0;
        this.intervals.fill(0);
        this.intervalWrite = 0;
        this.intervalCount = 0;
        this.firstPeakSeen = false;
        this.cachedResult = null;
    }

    // ─── Internals ──────────────────────────────────────────────────────

    private onPeakDetected(): void {
        const interval = this.samplesSinceLast;
        this.samplesSinceLast = 0;
        this.peakDetectedThisFrame = true;

        // Skip the first peak — interval from stream start isn't a real IBI
        if (!this.firstPeakSeen) {
            this.firstPeakSeen = true;
            return;
        }

        // Record interval in ring buffer
        this.intervals[this.intervalWrite] = interval;
        this.intervalWrite = (this.intervalWrite + 1) % this.config.maxIntervals;
        if (this.intervalCount < this.config.maxIntervals) {
            this.intervalCount++;
        }

        // Update cached estimate if we have enough intervals
        if (this.intervalCount >= this.config.minIntervals) {
            this.cachedResult = this.computeEstimate();
        }
    }

    private computeEstimate(): PeakResult {
        const n = this.intervalCount;
        const vals: number[] = new Array(n);

        for (let i = 0; i < n; i++) {
            const idx = (this.intervalWrite - n + i + this.config.maxIntervals)
                % this.config.maxIntervals;
            vals[i] = this.intervals[idx];
        }

        const medianInterval = median(vals);
        const bpm = (60 * this.config.sampleRate) / medianInterval;
        const clampedBPM = Math.max(this.config.minBPM, Math.min(this.config.maxBPM, bpm));

        // Confidence: fraction of intervals within 20% of median
        const tolerance = medianInterval * 0.2;
        let consistent = 0;
        for (const val of vals) {
            if (Math.abs(val - medianInterval) <= tolerance) consistent++;
        }

        return {
            bpm: clampedBPM,
            confidence: consistent / n,
            intervalCount: n,
            method: 'peak',
        };
    }
}
