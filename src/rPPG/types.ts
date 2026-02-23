export interface RGB {
    r: number;
    g: number;
    b: number;
}

/**
 * Shared Pipeline Configuration Types
 *
 * Core parameters that flow through the entire rPPG signal processing pipeline.
 * Defining these centrally ensures consistent BPM ranges and sample rates
 * across BandpassFilter, FFT, PeakDetector, and HeartRateEstimator.
 */

// ─── Core Pipeline Config ───────────────────────────────────────────────────

/**
 * Parameters shared across all signal processing modules.
 *
 * Set these once at the top level (HeartRateEstimator) and they propagate
 * to all sub-modules, eliminating the risk of mismatched BPM ranges
 * or sample rates between filter, FFT, and peak detector.
 */
export interface PipelineConfig {
    /** Sample rate in Hz (camera FPS, or target FPS after interpolation). */
    sampleRate: number;

    /** Minimum detectable heart rate in BPM.
     *  Sets the high-pass filter cutoff and lower bound for FFT/peak search.
     *  42 BPM covers bradycardia; raise to 50-60 for exercise contexts. */
    minBPM: number;

    /** Maximum detectable heart rate in BPM.
     *  Sets the low-pass filter cutoff and upper bound for FFT/peak search.
     *  240 BPM covers extreme tachycardia; lower to 180-200 for resting contexts. */
    maxBPM: number;
}

export const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
    sampleRate: 30,
    minBPM: 50,
    maxBPM: 100,
};
