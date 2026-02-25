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
// Parameters shared across all signal processing modules. TODO: extend with more options
export interface PipelineConfig {
    sampleRate: number; // Sample rate in Hz (camera FPS, or target FPS after interpolation).
    minBPM: number; // Minimum detectable heart rate in BPM/ high-pass filter cutoff - 42 BPM covers bradycardia; raise to 50-60 for exercise contexts.
    maxBPM: number; // Sets the low-pass filter cutoff - 240 would be quite high, 180-200 more reasonable
}

export const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
    sampleRate: 30,
    minBPM: 40,
    maxBPM: 200,
};
