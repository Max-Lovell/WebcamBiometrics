/**
 * Heart Rate Monitor — Tier 4 Orchestrator
 * Once per frame: ROIExtractor (Tier 1) → PulseProcessor (Tier 2) → Estimators (Tier 3)
 *
 * This class owns no signal processing state — it composes the lower tiers
 * and handles fusion logic (cross-validating peak vs FFT estimates).
 *
 * Usage (simple):
 *   const monitor = new HeartRateMonitor();
 *   const result = monitor.processFrame(frame, landmarks, time);
 *
 * Usage (access internals):
 *   monitor.roi    — ROIExtractor
 *   monitor.pulse  — PulseProcessor
 *   monitor.peak   — PeakEstimator (if method includes peak)
 *   monitor.fft    — FFTEstimator  (if method includes FFT)
 */

import type { FaceLandmarkerResult } from '@mediapipe/tasks-vision';

import { ROIExtractor } from './signal/ROIExtractor.ts';
import type { LandmarkerROIs } from './signal/ROIExtractor.ts';
import type { VideoFrameData, Point } from "../types.ts";

import { PulseProcessor } from './signal/PulseProcessor';

import { PeakEstimator } from './signal/PeakEstimator';
import type { PeakResult } from './signal/PeakEstimator';

import { FFTEstimator } from './signal/FFTEstimator';
import type { FFTEstimate } from './signal/FFTEstimator';

import { MedianSmoother, EMASmoother, CombinedSmoother } from './signal/TemporalSmoothing';
import type { BPMSmoother } from './signal/TemporalSmoothing';

import type { PipelineConfig, RGB } from './types';
import { DEFAULT_PIPELINE_CONFIG } from './types';
import { BandpassFilter } from "./signal/BandpassFilter.ts";

// ─── Types ──────────────────────────────────────────────────────────────────

export type EstimationMethod = 'peak' | 'fft' | 'fused';
export type SmoothingStrategy = 'median' | 'ema' | 'combined' | 'none';

export interface HeartRateMonitorConfig extends PipelineConfig {
    method: EstimationMethod;
    // PulseProcessor
    posWindowMultiplier: number;
    signalWindowSeconds: number;
    // PeakEstimator
    peakAmplitudeThreshold: number;
    peakEnvelopeDecay: number;
    peakMaxIntervals: number;
    peakMinIntervals: number;
    // FFTEstimator
    fftInterval: number;
    // Smoothing
    smoothing: SmoothingStrategy;
    smootherWindow: number;
    smootherAlpha: number;
}

export const DEFAULT_MONITOR_CONFIG: HeartRateMonitorConfig = {
    ...DEFAULT_PIPELINE_CONFIG,
    method: 'fused',
    posWindowMultiplier: 1.6,
    signalWindowSeconds: 15,
    peakAmplitudeThreshold: 0.3,
    peakEnvelopeDecay: 0.998,
    peakMaxIntervals: 8,
    peakMinIntervals: 3,
    fftInterval: 15,
    smoothing: 'median',
    smootherWindow: 5,
    smootherAlpha: 0.3,
};

// Per-region data (for visualization)
export interface RegionDetail {
    polygon: Point[];
    rgb: RGB | null;
    pulse: number | null;
}

// Full result from processFrame()
export interface HeartRateResult {
    timestamp: number;
    bpm: number | null;
    confidence: number;
    fusedSample: number | null;
    filteredSample: number | null;
    raw: {
        peak?: PeakResult;
        fft?: FFTEstimate;
    };
    regions: Record<string, RegionDetail>;
}

// ─── HeartRateMonitor Class ─────────────────────────────────────────────────

export class HeartRateMonitor {
    private readonly config: HeartRateMonitorConfig;

    // Sub-components (public for advanced access)
    readonly roi: ROIExtractor;
    readonly pulse: PulseProcessor;
    private readonly bandpass: BandpassFilter;
    readonly peak: PeakEstimator | null;
    readonly fft: FFTEstimator | null;
    private readonly smoother: BPMSmoother;

    // State
    private lastBPM: number | null = null;
    private lastConfidence: number = 0;

    constructor(config?: Partial<HeartRateMonitorConfig>, rois?: LandmarkerROIs) {
        this.config = { ...DEFAULT_MONITOR_CONFIG, ...config };
        const cfg = this.config;

        this.roi = new ROIExtractor(rois);

        this.pulse = new PulseProcessor(this.roi.regionNames, {
            sampleRate: cfg.sampleRate,
            minBPM: cfg.minBPM,
            maxBPM: cfg.maxBPM,
            posWindowMultiplier: cfg.posWindowMultiplier,
            signalWindowSeconds: cfg.signalWindowSeconds,
        });

        this.bandpass = BandpassFilter.fromPipelineConfig(cfg);

        this.peak = (cfg.method === 'peak' || cfg.method === 'fused')
            ? new PeakEstimator({
                sampleRate: cfg.sampleRate,
                minBPM: cfg.minBPM,
                maxBPM: cfg.maxBPM,
                amplitudeThreshold: cfg.peakAmplitudeThreshold,
                envelopeDecay: cfg.peakEnvelopeDecay,
                maxIntervals: cfg.peakMaxIntervals,
                minIntervals: cfg.peakMinIntervals,
            })
            : null;

        this.fft = (cfg.method === 'fft' || cfg.method === 'fused')
            ? new FFTEstimator(this.pulse.signalLength, {
                sampleRate: cfg.sampleRate,
                minBPM: cfg.minBPM,
                maxBPM: cfg.maxBPM,
                estimateInterval: cfg.fftInterval,
            })
            : null;

        this.smoother = createSmoother(cfg.smoothing, cfg.smootherWindow, cfg.smootherAlpha);
    }

    // ─── Public API ─────────────────────────────────────────────────────

    processFrame(
        frame: VideoFrameData,
        landmarks: FaceLandmarkerResult,
        time: number
    ): HeartRateResult {
        // Extract RGB average from each region
        const roiResult = this.roi.extract(frame, landmarks);
        // Extract POS signal from each RGB Average
        const pulseFrame = this.pulse.pushFrame(roiResult, time);
        // TODO: Bandpass filter the POS signal

        // Estimate BPM
        const raw: HeartRateResult['raw'] = {};
        // Peak estimation: feed raw fused POS sample
        let filteredSample: number | null = null;
        if (pulseFrame.fusedSample !== null && this.peak) {
            filteredSample = this.bandpass.process(pulseFrame.fusedSample);
            const peakResult = this.peak.pushSample(filteredSample);
            if (peakResult) raw.peak = peakResult;
        }

        // FFT estimation: pass raw fused signal (FFTEstimator handles rate limiting + filtering)
        if (this.fft) {
            const fftResult = this.fft.update(this.pulse.getFusedSignal());
            if (fftResult) raw.fft = fftResult;
        }

        // Fusion + Smoothing
        const fusedBPM = this.fuseBPM(raw.peak ?? null, raw.fft ?? null);
        if (fusedBPM !== null) {
            this.lastBPM = this.smoother.update(fusedBPM);
            this.lastConfidence = this.getConfidence(raw.peak ?? null, raw.fft ?? null);
        }

        // Build result
        const regions: Record<string, RegionDetail> = {};
        for (const [name, roiRegion] of Object.entries(roiResult)) {
            regions[name] = {
                polygon: roiRegion.polygon,
                rgb: roiRegion.rgb,
                pulse: pulseFrame.regionPulses[name] ?? null,
            };
        }

        return {
            timestamp: time,
            bpm: this.lastBPM,
            confidence: this.lastConfidence,
            fusedSample: pulseFrame.fusedSample,
            filteredSample,
            raw,
            regions,
        };
    }

    reset(): void {
        this.pulse.reset();
        this.peak?.reset();
        this.fft?.reset();
        this.smoother.reset();
        this.lastBPM = null;
        this.lastConfidence = 0;
    }

    // ─── Fusion Logic ───────────────────────────────────────────────────

    /**
     * Combine peak and FFT estimates into a single BPM value.
     *
     * Strategy depends on the configured method:
     *   - 'peak': use peak estimate only
     *   - 'fft': use FFT estimate only
     *   - 'fused': cross-validate and pick the best
     *
     * The fusion logic for 'fused' mode:
     *   1. If both agree (within 10%), trust FFT (more precise)
     *   2. If FFT ≈ peak/2, FFT grabbed a subharmonic → trust peak
     *   3. If FFT ≈ peak×2, FFT grabbed a harmonic → trust peak
     *   4. Otherwise, trust whichever has higher confidence
     *
     * Why this works:
     *   The pulse waveform has a sharp systolic peak and dicrotic notch,
     *   producing a strong 2nd harmonic. FFT can lock onto this harmonic
     *   (reporting 2× true HR) or its subharmonic. Peak counting is immune
     *   because it counts actual pulses. So when they disagree by a factor
     *   of 2, peak detection is almost certainly right.
     */
    private fuseBPM(peak: PeakResult | null, fft: FFTEstimate | null): number | null {
        const method = this.config.method;

        if (method === 'peak') return peak?.bpm ?? null;
        if (method === 'fft') return fft?.bpm ?? null;

        // Fused mode
        if (!peak && !fft) return null;
        if (!peak) return fft!.bpm;
        if (!fft) return peak.bpm;

        // Both available — cross-validate
        const ratio = Math.abs(peak.bpm - fft.bpm) / fft.bpm;

        // Agreement (within 10%): trust FFT for precision
        if (ratio < 0.10) return fft.bpm;

        // FFT ≈ half of peak → FFT grabbed a subharmonic
        const halfRatio = Math.abs(fft.bpm - peak.bpm / 2) / peak.bpm;
        if (halfRatio < 0.10) return peak.bpm;

        // FFT ≈ double peak → FFT grabbed a harmonic
        const doubleRatio = Math.abs(fft.bpm - peak.bpm * 2) / peak.bpm;
        if (doubleRatio < 0.10) return peak.bpm;

        // Disagreement with no harmonic relationship: trust higher confidence
        return peak.confidence > fft.confidence ? peak.bpm : fft.bpm;
    }

    private getConfidence(peak: PeakResult | null, fft: FFTEstimate | null): number {
        const method = this.config.method;

        if (method === 'peak') return peak?.confidence ?? 0;
        if (method === 'fft') return fft?.confidence ?? 0;

        // Fused: return confidence from whichever we'd trust
        if (!peak && !fft) return 0;
        if (!peak) return fft!.confidence;
        if (!fft) return peak.confidence;

        // If they agree, use FFT's SNR (more informative than interval consistency)
        const ratio = Math.abs(peak.bpm - fft.bpm) / fft.bpm;
        if (ratio < 0.10) return fft.confidence;

        // Otherwise, confidence from whichever we trusted
        return peak.confidence > fft.confidence ? peak.confidence : fft.confidence;
    }
}

// ─── Smoother Factory ───────────────────────────────────────────────────────

function createSmoother(
    strategy: SmoothingStrategy,
    windowSize: number,
    alpha: number
): BPMSmoother {
    switch (strategy) {
        case 'median':
            return new MedianSmoother(windowSize);
        case 'ema':
            return new EMASmoother(alpha);
        case 'combined':
            return new CombinedSmoother(windowSize, alpha);
        case 'none':
            return {
                update: (bpm: number) => bpm,
                reset: () => {},
                isReady: () => true,
            };
    }
}
