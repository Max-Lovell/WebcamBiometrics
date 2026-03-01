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

import { PulseProcessor } from './pulse/PulseProcessor.ts';

import { PeakEstimator } from './signal/PeakEstimator';
import type { PeakResult } from './signal/PeakEstimator';

import { FFTEstimator } from './signal/FFT/FFTEstimator.ts';
import type { FFTEstimate } from './signal/FFT/FFTEstimator.ts';

import { createSmoother } from './signal/smoothing/TemporalSmoothing.ts';
import type { BPMSmoother, SmoothingStrategy } from './signal/smoothing/TemporalSmoothing.ts';
// import { fuseBPM, getFusionConfidence } from './signal/smoothing/BPMFusion';
import type { EstimationMethod } from './signal/smoothing/BPMFusion';

import type { PipelineConfig, RGB } from './types';
import { DEFAULT_PIPELINE_CONFIG } from './types';
import { BandpassFilter } from "./signal/BandpassFilter.ts";

import type { WindowedPulseMethod } from './pulse/projection/types';
import { createMethod } from './pulse/registry';

// import {CHROM} from "./pulse/projection/CHROM.ts";

// ─── Types ──────────────────────────────────────────────────────────────────
export interface HeartRateMonitorConfig extends PipelineConfig {
    method: EstimationMethod;
    // Projection method — name from registry (e.g., 'POS', 'CHROM'), or pass instance to constructor
    projectionMethods: string[],
    // PulseProcessor
    rgbWindowMultiplier: number;
    signalWindowSeconds: number;
    interpolate: boolean;
    // PeakEstimator
    peakMinBPM: number; // Different to bandpass filter BPM - should be lower.
    peakMaxBPM: number;
    peakAmplitudeThreshold: number;
    peakEnvelopeDecayRate: number;  // Half-life in ms (replaces per-sample envelopeDecay)
    peakMaxIntervals: number;
    peakMinIntervals: number;
    // FFTEstimator
    fftInterval: number;
    // Smoothing
    smoothing: SmoothingStrategy;
    smootherWindow: number;
    smootherAlpha: number;
}

export const DEFAULT_MONITOR_CONFIG: HeartRateMonitorConfig = { // TODO: this needs to be sorted out a bit - either names or nesting
    ...DEFAULT_PIPELINE_CONFIG,
    method: 'fused',
    projectionMethods: ['POS'],
    rgbWindowMultiplier: 1.6,
    signalWindowSeconds: 15,
    interpolate: true,
    peakMinBPM: 50,
    peakMaxBPM: 150,
    peakAmplitudeThreshold: 0.2,
    peakEnvelopeDecayRate: 500,  // 500ms half-life
    peakMaxIntervals: 8,
    peakMinIntervals: 2,
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
    methodPulses: Record<string, number | null>;
    peakDetected: boolean;
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

    constructor(config?: Partial<HeartRateMonitorConfig>, rois?: LandmarkerROIs, methodInstances?: WindowedPulseMethod[]) {
        this.config = { ...DEFAULT_MONITOR_CONFIG, ...config };
        const cfg = this.config;

        this.roi = new ROIExtractor(rois);
        // Build or use provided projection method
        const methods = methodInstances ?? cfg.projectionMethods.map(name =>
            createMethod(name, {
                sampleRate: cfg.sampleRate,
                windowMultiplier: cfg.rgbWindowMultiplier,
            })
        );

        this.pulse = new PulseProcessor(this.roi.regionNames, {
            sampleRate: cfg.sampleRate,
            posWindowMultiplier: cfg.rgbWindowMultiplier,
            signalWindowSeconds: cfg.signalWindowSeconds,
            interpolate: cfg.interpolate,
        }, methods);

        this.bandpass = BandpassFilter.fromPipelineConfig(cfg);

        this.peak = (cfg.method === 'peak' || cfg.method === 'fused')
            ? new PeakEstimator({
                sampleRate: cfg.sampleRate,
                minBPM: cfg.peakMinBPM, // Pass separate minBPM
                maxBPM: cfg.peakMaxBPM, // Pass separate maxBPM
                amplitudeThreshold: cfg.peakAmplitudeThreshold,
                envelopeDecayRate: cfg.peakEnvelopeDecayRate,
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
        // Extract POS signal from each RGB Average.
        // When interpolation is on, pulseFrame.fusedSamples may contain 0..N grid-aligned samples.
        const pulseFrame = this.pulse.pushFrame(roiResult, time);

        // Process all fused samples through bandpass and peak detector
        // Usually 1 iteration. On the first interpolated frame: 0. Occasionally 2 if camera was late.
        const raw: HeartRateResult['raw'] = {};
        let filteredSample: number | null = null;
        let latestFusedSample: number | null = null;
        let peakDetected = false;

        for (const { value, time: sampleTime } of pulseFrame.fusedSamples) {
            latestFusedSample = value;
            // Bandpass filter — IIR assumes uniform dt, which interpolation guarantees
            filteredSample = this.bandpass.process(value);
            // Peak estimation — sees every grid sample with correct timestamp
            if (this.peak) {
                const peakResult = this.peak.pushSample(filteredSample, sampleTime);
                if (peakResult) raw.peak = peakResult;
                // A peak detected on any grid sample this frame counts
                if (this.peak.peakDetectedThisFrame) peakDetected = true;
            }
        }

        // FFT estimation - pass raw fused signal (FFTEstimator handles rate limiting + filtering)
        if (this.fft) {
            const fftResult = this.fft.update(this.pulse.getFusedSignal());
            if (fftResult) {
                raw.fft = fftResult;
                this.lastBPM = fftResult.bpm;
            } // Replace if new result available
        }

        // Fusion + Smoothing - TODO: look into this, might be a nice step later.
        // const fusedBPM = fuseBPM(this.config.method, raw.peak ?? null, raw.fft ?? null);
        // if (fusedBPM !== null) {
        //     this.lastBPM = this.smoother.update(fusedBPM);
        //     this.lastConfidence = getFusionConfidence(this.config.method, raw.peak ?? null, raw.fft ?? null);
        // }

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
            fusedSample: latestFusedSample,
            filteredSample,
            raw,
            regions,
            peakDetected,
            methodPulses: pulseFrame.methodPulses,
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
}
