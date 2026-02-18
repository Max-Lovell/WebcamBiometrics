/**
 * Time-Domain Peak Detector
 *
 * Estimates heart rate by counting peaks in the bandpass-filtered POS signal,
 * rather than using frequency-domain (FFT) analysis.
 *
 * Pipeline position:
 *   POS → Bandpass → **PeakDetector** → BPM estimate
 *                  ↘ Window → FFT → BPM estimate  (existing path)
 *
 * Why have both?
 *   - FFT is more robust to noise but can lock onto harmonics/subharmonics
 *   - Peak counting is immune to harmonic confusion (it counts actual pulses)
 *   - Peak counting responds faster (no need to fill a full FFT window)
 *   - Cross-checking both catches errors: if FFT says 40 but peaks say 78,
 *     the FFT grabbed a subharmonic
 *
 * Limitations:
 *   - More sensitive to noise (a motion artifact = a false peak)
 *   - Needs a clean bandpass-filtered signal to work well
 *   - Less precise than FFT + parabolic interpolation for exact BPM
 *
 * This module is designed to process the *already bandpass-filtered* signal.
 * Do NOT feed it raw POS output — it needs the bandpass to remove noise.
 */

import { median } from '../utils/math.ts';

// ─── Types ──────────────────────────────────────────────────────────────────

export interface PeakDetectorResult {
    /** Estimated BPM from peak counting */
    bpm: number;

    /** Number of peaks found in the analysis window */
    peakCount: number;

    /** Indices of detected peaks within the input buffer */
    peakIndices: number[];

    /** Confidence: ratio of consistent inter-peak intervals to total intervals.
     *  1.0 = all intervals are similar (strong periodic signal).
     *  0.0 = intervals are all over the place (noise). */
    confidence: number;
}

export interface PeakDetectorConfig {
    /** Sample rate in Hz (your interpolated FPS) */
    sampleRate: number;

    /** Minimum BPM to detect. Peaks closer together than this rate are allowed,
     *  but the final BPM estimate won't go below this. Also used to set the
     *  minimum peak distance. Default: 40 */
    minBPM: number;

    /** Maximum BPM to detect. Sets the minimum time between peaks.
     *  Default: 200 */
    maxBPM: number;

    /** Amplitude threshold as a fraction of the signal's max amplitude.
     *  Peaks below this fraction of the maximum are ignored.
     *  0.0 = accept all local maxima, 1.0 = only the absolute maximum.
     *  Default: 0.3 */
    amplitudeThreshold: number;
}

const DEFAULT_CONFIG: PeakDetectorConfig = {
    sampleRate: 30,
    minBPM: 40,
    maxBPM: 200,
    amplitudeThreshold: 0.3,
};

// ─── Peak Detection ─────────────────────────────────────────────────────────

/**
 * Find peaks (local maxima) in a signal buffer.
 *
 * A peak is a sample that is greater than both its neighbours AND above
 * the amplitude threshold. Peaks that are too close together (based on maxBPM)
 * are resolved by keeping the taller one.
 *
 * @param signal - Bandpass-filtered signal buffer
 * @param config - Detection parameters
 * @returns Array of peak indices into the signal buffer
 */
export function findPeaks(
    signal: Float32Array,
    config: PeakDetectorConfig = DEFAULT_CONFIG
): number[] {
    const { sampleRate, maxBPM, amplitudeThreshold } = config;

    // Minimum samples between peaks based on maximum expected heart rate
    // At 200 BPM and 30 fps: 60/200 * 30 = 9 samples minimum between beats
    const minPeakDistance = Math.floor((60 / maxBPM) * sampleRate);

    // Find the max absolute amplitude for thresholding
    let maxAmp = 0;
    for (let i = 0; i < signal.length; i++) {
        const abs = Math.abs(signal[i]);
        if (abs > maxAmp) maxAmp = abs;
    }
    const threshold = maxAmp * amplitudeThreshold;

    // Step 1: Find all local maxima above threshold
    const candidates: number[] = [];
    for (let i = 1; i < signal.length - 1; i++) {
        if (
            signal[i] > signal[i - 1] &&
            signal[i] > signal[i + 1] &&
            signal[i] > threshold
        ) {
            candidates.push(i);
        }
    }

    // Step 2: Enforce minimum peak distance (keep the taller peak)
    if (candidates.length <= 1) return candidates;

    const peaks: number[] = [candidates[0]];
    for (let i = 1; i < candidates.length; i++) {
        const lastPeak = peaks[peaks.length - 1];
        const distance = candidates[i] - lastPeak;

        if (distance >= minPeakDistance) {
            // Far enough apart — accept this peak
            peaks.push(candidates[i]);
        } else {
            // Too close — keep the taller one
            if (signal[candidates[i]] > signal[lastPeak]) {
                peaks[peaks.length - 1] = candidates[i];
            }
        }
    }

    return peaks;
}

/**
 * Estimate BPM from a bandpass-filtered signal buffer using peak counting.
 *
 * The core idea: find peaks, measure the intervals between them, and
 * convert the average interval to BPM.
 *
 * Using median interval (not mean) makes this robust to occasional
 * false peaks or missed beats.
 *
 * @param signal - Bandpass-filtered POS signal
 * @param config - Detection parameters
 * @returns Detection result with BPM, confidence, and peak locations
 */
export function detectBPMFromPeaks(
    signal: Float32Array,
    config: Partial<PeakDetectorConfig> = {}
): PeakDetectorResult | null {
    const cfg = { ...DEFAULT_CONFIG, ...config };
    const peaks = findPeaks(signal, cfg);

    // Need at least 2 peaks to measure an interval
    if (peaks.length < 2) {
        return null;
    }

    // Calculate inter-peak intervals (in samples)
    const intervals: number[] = [];
    for (let i = 1; i < peaks.length; i++) {
        intervals.push(peaks[i] - peaks[i - 1]);
    }

    // Use median interval for robustness
    const medianInterval = median(intervals);

    // Convert interval in samples to BPM
    // interval_seconds = interval_samples / sampleRate
    // beats_per_second = 1 / interval_seconds
    // BPM = beats_per_second * 60
    const bpm = (60 * cfg.sampleRate) / medianInterval;

    // Clamp to valid range
    const clampedBPM = Math.max(cfg.minBPM, Math.min(cfg.maxBPM, bpm));

    // Calculate confidence based on interval consistency
    // If all intervals are similar, the signal is strongly periodic
    const confidence = intervalConsistency(intervals, medianInterval);

    return {
        bpm: clampedBPM,
        peakCount: peaks.length,
        peakIndices: peaks,
        confidence,
    };
}

// ─── Streaming Peak Detector ────────────────────────────────────────────────

/**
 * Streaming wrapper that accumulates bandpass-filtered samples and
 * periodically runs peak detection on the buffer.
 *
 * Usage:
 *   const detector = new StreamingPeakDetector({ sampleRate: 30 });
 *
 *   // In your frame loop, after bandpass filtering:
 *   detector.push(filteredSample);
 *   const result = detector.estimate();
 *   if (result) {
 *       console.log(`Peak-based BPM: ${result.bpm}`);
 *   }
 */
export class StreamingPeakDetector {
    private readonly config: PeakDetectorConfig;
    private readonly bufferSize: number;
    private readonly buffer: Float32Array;
    private writeIndex: number = 0;
    private sampleCount: number = 0;

    /**
     * @param config - Detection parameters
     * @param windowSeconds - How many seconds of signal to analyse. Default 10.
     *   Longer = more peaks to average = more stable, but slower to respond.
     *   At 30fps and 10s, this is a 300-sample buffer.
     */
    constructor(config: Partial<PeakDetectorConfig> = {}, windowSeconds: number = 10) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.bufferSize = Math.ceil(this.config.sampleRate * windowSeconds);
        this.buffer = new Float32Array(this.bufferSize);
    }

    /** Push a single bandpass-filtered sample */
    push(sample: number): void {
        this.buffer[this.writeIndex] = sample;
        this.writeIndex = (this.writeIndex + 1) % this.bufferSize;
        this.sampleCount++;
    }

    /** Push multiple samples at once */
    pushBuffer(samples: Float32Array): void {
        for (let i = 0; i < samples.length; i++) {
            this.push(samples[i]);
        }
    }

    /**
     * Run peak detection on the current buffer contents.
     *
     * Returns null if the buffer isn't full enough (need at least 3 seconds
     * of data to reliably detect peaks at resting heart rates).
     */
    estimate(): PeakDetectorResult | null {
        const minSamples = Math.ceil(this.config.sampleRate * 3);
        if (this.sampleCount < minSamples) return null;

        // Get the signal in chronological order
        const length = Math.min(this.sampleCount, this.bufferSize);
        const signal = new Float32Array(length);

        if (this.sampleCount >= this.bufferSize) {
            // Buffer is full — unwrap from the circular write position
            for (let i = 0; i < length; i++) {
                signal[i] = this.buffer[(this.writeIndex + i) % this.bufferSize];
            }
        } else {
            // Buffer not yet full — just copy what we have
            signal.set(this.buffer.subarray(0, length));
        }

        return detectBPMFromPeaks(signal, this.config);
    }

    /** Reset the buffer (e.g., when face tracking is lost) */
    reset(): void {
        this.buffer.fill(0);
        this.writeIndex = 0;
        this.sampleCount = 0;
    }

    /** Whether enough data has been collected for an estimate */
    isReady(): boolean {
        return this.sampleCount >= Math.ceil(this.config.sampleRate * 3);
    }
}

// ─── Utilities ──────────────────────────────────────────────────────────────
/**
 * Measure how consistent a set of inter-peak intervals are.
 *
 * Returns a value from 0 to 1:
 *   1.0 = all intervals are identical (perfect periodicity)
 *   0.0 = intervals vary wildly (noise)
 *
 * Uses the coefficient of variation (std/mean), mapped to 0-1.
 * A CV below 0.1 is very consistent; above 0.5 is essentially random.
 */
function intervalConsistency(intervals: number[], medianInterval: number): number {
    if (intervals.length < 2) return 0;

    // Count how many intervals are within 20% of the median
    const tolerance = medianInterval * 0.2;
    let consistent = 0;
    for (const interval of intervals) {
        if (Math.abs(interval - medianInterval) <= tolerance) {
            consistent++;
        }
    }

    return consistent / intervals.length;
}
