/**
 * Peak Estimator
 * Streaming peak detection on fused POS samples.
 *
 * Detects local maxima in the pulse waveform, records inter-peak intervals
 * in real time (milliseconds), and estimates BPM from the median interval.
 * Operates one-sample-at-a-time so it can run inline with the frame loop —
 * no buffering or batch processing.
 *
 * Detection algorithm:
 *   - Three-sample sliding window: checks if sample[t-1] > sample[t-2] AND sample[t-1] > sample[t]
 *   - Adaptive envelope: time-based decay (frame-rate independent)
 *   - Amplitude threshold: peak must exceed fraction of envelope
 *   - Refractory period: minimum time between peaks (derived from maxBPM)
 *
 * BPM estimation:
 *   - Maintains a ring buffer of recent inter-peak intervals (in ms)
 *   - BPM = 60000 / median(intervals)
 *   - Confidence = fraction of intervals within 20% of median
 */

import { median } from '../utils/math.ts';
import type { PipelineConfig } from '../types.ts';
import { DEFAULT_PIPELINE_CONFIG } from '../types.ts';

// ─── Types ──────────────────────────────────────────────────────────────────

export interface PeakEstimatorConfig extends PipelineConfig {
    amplitudeThreshold: number; // Fraction of adaptive envelope a peak must exceed (amplitudeThreshold × envelope). Default: 0.2
    envelopeDecayRate: number;  // Envelope half-life in milliseconds. ~500ms means envelope halves every 500ms regardless of frame rate.
    maxIntervals: number;       // Max inter-peak intervals to keep for median. Default: 8
    minIntervals: number;       // Min intervals before producing an estimate. Default: 2
    envelopeFastDecayAfterMs: number; // If no peak detected for this long (ms), decay envelope faster. Default: 2000
}

export const DEFAULT_PEAK_CONFIG: PeakEstimatorConfig = {
    ...DEFAULT_PIPELINE_CONFIG,
    amplitudeThreshold: 0.2,
    envelopeDecayRate: 500, // Half-life in ms — envelope halves every ~500ms
    maxIntervals: 8,
    minIntervals: 2,
    envelopeFastDecayAfterMs: 2000, // No peak for 2s switch to fast decay
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

    // Three-sample sliding window (values and timestamps)
    private prevPrevSample: number = 0; // sample at t-2
    private prevSample: number = 0;     // sample at t-1 (candidate peak)
    private prevTime: number = 0;       // timestamp at t-1 (candidate peak timestamp)
    private sampleCount: number = 0;

    // Adaptive amplitude envelope (time-based decay)
    private envelope: number = 0;
    private lastPeakTimeMs: number = 0;     // Timestamp of most recent detected peak
    private lastSampleTimeMs: number = 0;   // Timestamp of most recent sample (for dt)

    // Refractory period: minimum time between peaks (from maxBPM), in ms
    private readonly minIntervalMs: number;

    // Inter-peak interval ring buffer (in milliseconds)
    private readonly intervals: Float64Array;
    private intervalWrite: number = 0;
    private intervalCount: number = 0;
    private firstPeakSeen: boolean = false; // Skip first interval (not a real IBI)

    // Cached result
    private cachedResult: PeakResult | null = null; // for returning last result
    peakDetectedThisFrame: boolean = false;

    // Precomputed decay constant: ln(2) / halfLife
    private readonly decayLambda: number;
    private readonly fastDecayLambda: number;

    constructor(config?: Partial<PeakEstimatorConfig>) {
        this.config = { ...DEFAULT_PEAK_CONFIG, ...config };
        // Refractory period in ms: at maxBPM=200, minInterval = 300ms
        this.minIntervalMs = 60000 / this.config.maxBPM;
        this.intervals = new Float64Array(this.config.maxIntervals); // Store timestamps
        // Exponential decay: envelope *= e^(-lambda * dt). Half-life formula: lambda = ln(2) / halfLifeMs
        this.decayLambda = Math.LN2 / this.config.envelopeDecayRate;
        this.fastDecayLambda = this.decayLambda * 4; // Fast decay: 4× faster when no peak detected for a while
    }

    // ─── Public API ─────────────────────────────────────────────────────
    // Called once per frame on POS sample. Returns updated PeakResult if enough intervals have been collected, else null.
    pushSample(sample: number, timeMs: number): PeakResult | null {
        // Maintains 3 sample sliding window (t-2, t-1, t)
        this.sampleCount++;
        this.peakDetectedThisFrame = false;

        // Time-based envelope decay (frame-rate independent)
        if (this.lastSampleTimeMs > 0) {
            const dt = timeMs - this.lastSampleTimeMs;
            const timeSinceLastPeak = timeMs - this.lastPeakTimeMs;
            // Use fast decay if no peak detected for a while
            const lambda = (this.lastPeakTimeMs > 0 && timeSinceLastPeak > this.config.envelopeFastDecayAfterMs)
                ? this.fastDecayLambda
                : this.decayLambda;

            this.envelope *= Math.exp(-lambda * dt);
        }

        const absSample = Math.abs(sample);
        if (absSample > this.envelope) {
            this.envelope = absSample;
        }

        // Check if prev (t-1) was a local maximum — need at least 3 samples
        if (this.sampleCount >= 3) {
            const threshold = this.envelope * this.config.amplitudeThreshold;
            const timeSinceLastPeak = this.prevTime - this.lastPeakTimeMs;

            if (
                this.prevSample > this.prevPrevSample &&       // rising edge
                this.prevSample > sample &&                     // falling edge (prev is the peak)
                this.prevSample > threshold &&                  // above adaptive threshold
                (this.lastPeakTimeMs === 0 || timeSinceLastPeak >= this.minIntervalMs) // refractory period
            ) {
                this.onPeakDetected(this.prevTime);
            }
        }

        // Shift window forward
        this.prevPrevSample = this.prevSample;
        this.prevSample = sample;
        this.prevTime = timeMs;
        this.lastSampleTimeMs = timeMs;

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
        this.prevPrevSample = 0;
        this.prevSample = 0;
        this.prevTime = 0;
        this.sampleCount = 0;
        this.envelope = 0;
        this.lastPeakTimeMs = 0;
        this.lastSampleTimeMs = 0;
        this.intervals.fill(0);
        this.intervalWrite = 0;
        this.intervalCount = 0;
        this.firstPeakSeen = false;
        this.cachedResult = null;
        this.peakDetectedThisFrame = false;
    }

    // ─── Internals ──────────────────────────────────────────────────────

    private onPeakDetected(peakTimeMs: number): void {
        this.peakDetectedThisFrame = true;

        // Skip the first peak — no previous peak to compute interval from
        if (!this.firstPeakSeen) {
            this.firstPeakSeen = true;
            this.lastPeakTimeMs = peakTimeMs;
            return;
        }

        // Compute interval in milliseconds
        const intervalMs = peakTimeMs - this.lastPeakTimeMs;
        this.lastPeakTimeMs = peakTimeMs;

        // Record interval in ring buffer
        this.intervals[this.intervalWrite] = intervalMs;
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
        // Unroll ring buffer
        const n = this.intervalCount;
        const vals: number[] = new Array(n);

        for (let i = 0; i < n; i++) {
            const idx = (this.intervalWrite - n + i + this.config.maxIntervals)
                % this.config.maxIntervals;
            vals[i] = this.intervals[idx];
        }

        // Median more robust to outliers than mean
        const medianIntervalMs = median(vals);
        const bpm = 60000 / medianIntervalMs;

        // Clamp to configured range
        const clampedBPM = Math.max(this.config.minBPM, Math.min(this.config.maxBPM, bpm));

        // Confidence: fraction of intervals within 20% of median
        const tolerance = medianIntervalMs * 0.2;
        let consistent = 0;
        for (const val of vals) {
            if (Math.abs(val - medianIntervalMs) <= tolerance) consistent++;
        }

        return {
            bpm: clampedBPM,
            confidence: consistent / n,
            intervalCount: n,
            method: 'peak',
        };
    }
}
