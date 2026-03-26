/**
 * rPPG — Remote Photoplethysmography Package
 *
 * Webcam-based heart rate estimation using the POS algorithm.
 *
 * Quick start (most users):
 *   import { HeartRateMonitor } from 'rppg';
 *   const monitor = new HeartRateMonitor();
 *   const result = monitor.processFrame(frame, landmarks, time);
 *
 * Tiered API:
 *   Tier 1 — ROIExtractor:    video frame → per-region RGB averages
 *   Tier 2 — PulseProcessor:  RGB stream → filtered pulse signal
 *   Tier 3a — PeakEstimator:  pulse → BPM via time-domain peak counting
 *   Tier 3b — FFTEstimator:   pulse → BPM via frequency-domain analysis
 *   Tier 4 — HeartRateMonitor: all of the above in one call
 */

// ─── Tier 4: Convenience orchestrator (most users start here) ───────────────

export { HeartRateMonitor } from './HeartRateMonitor';
export type {
    HeartRateMonitorConfig,
    HeartRateResult,
    RegionDetail,
} from './HeartRateMonitor';

// ─── Tier 1: ROI Extraction ─────────────────────────────────────────────────

export { ROIExtractor, FACE_ROIS } from './signal/ROIExtractor';
export type {
    RegionResult,
    LandmarkerROIs,
} from './signal/ROIExtractor';

// ─── Tier 2: Pulse Signal Processing ────────────────────────────────────────

export { PulseProcessor } from './pulse/PulseProcessor';
export type {
    PulseProcessorConfig,
    PulseFrame,
} from './pulse/PulseProcessor';

// ─── Tier 3: Heart Rate Estimation ──────────────────────────────────────────
export { FFTEstimator } from './signal/FFT/FFTEstimator';
export type {
    FFTEstimatorConfig,
    FFTEstimate,
} from './signal/FFT/FFTEstimator';

// ─── Signal Processing Primitives (advanced users) ──────────────────────────

export { BandpassFilter, BiquadFilter } from './signal/BandpassFilter';
export { fft, computeSpectrum, findDominantFrequency, hanningWindow } from './signal/FFT/FFT';
export type { SpectrumResult, DominantFrequencyResult } from './signal/FFT/FFT';

// ─── Smoothing ──────────────────────────────────────────────────────────────

export { MedianSmoother, EMASmoother, CombinedSmoother } from './signal/smoothing/TemporalSmoothing';
export type { BPMSmoother } from './signal/smoothing/TemporalSmoothing';

// ─── Utilities ──────────────────────────────────────────────────────────────

export { FloatRingBuffer, Float64RingBuffer } from './FloatRingBuffer';
export { mean, std, median, bpmToHz, hzToBpm } from './utils/math';
export type { PipelineConfig } from './types';
export { DEFAULT_PIPELINE_CONFIG } from './types';
