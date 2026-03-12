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

import {PulseProcessor, type PulseProcessorConfig} from './pulse/PulseProcessor.ts';

import {PeakEstimator, type PeakEstimatorConfig} from './signal/PeakEstimator';
import type { PeakResult } from './signal/PeakEstimator';

import {FFTEstimator, type FFTEstimatorConfig} from './signal/FFT/FFTEstimator.ts';
import type { FFTEstimate } from './signal/FFT/FFTEstimator.ts';

import { createSmoother } from './signal/smoothing/TemporalSmoothing.ts';
import type { BPMSmoother, SmoothingStrategy } from './signal/smoothing/TemporalSmoothing.ts';
// import { fuseBPM, getFusionConfidence } from './signal/smoothing/BPMFusion';

import type { PipelineConfig, RGB } from './types';
import { DEFAULT_PIPELINE_CONFIG } from './types';
import { BandpassFilter } from "./signal/BandpassFilter.ts";

import type { WindowedPulseMethod } from './pulse/projection/types';
import { createMethod } from './pulse/registry';

// import {CHROM} from "./pulse/projection/CHROM.ts";

// ─── Types ──────────────────────────────────────────────────────────────────
export interface SmoothingConfig {
    strategy: SmoothingStrategy;
    window: number;
    alpha: number;
}

export interface HeartRateMonitorConfig {
    pipeline: PipelineConfig;
    // Projection methods — names from registry (e.g., 'POS', 'CHROM') or pass WindowedPulseMethod instances directly
    projectionMethods: string[];
    maxConsecutiveMisses: number;
    pulse: Partial<PulseProcessorConfig>;
    // Both present = fused mode (cross-validate peak vs FFT). TODO: remove fused mode
    peak?: Partial<PeakEstimatorConfig>;
    fft?: Partial<FFTEstimatorConfig>;
    smoothing: SmoothingConfig;
}

export const DEFAULT_MONITOR_CONFIG: HeartRateMonitorConfig = {
    pipeline: { ...DEFAULT_PIPELINE_CONFIG },
    projectionMethods: ['POS'],
    maxConsecutiveMisses: 30,
    pulse: { // TODO: rename to BPV?
        posWindowMultiplier: 1.6,
        signalWindowSeconds: 15,
        maxConsecutiveMisses: 3,
        interpolate: true,
    },
    peak: {
        minBPM: 45,
        maxBPM: 170,
        amplitudeThreshold: 0.3,
        envelopeDecayRate: 1000,
        maxIntervals: 8,
        minIntervals: 2,
        envelopeFastDecayAfterMs: 2000, // No peak for Xs switch to fast decay
    },
    fft: {
        estimateInterval: 15,
    },
    smoothing: {
        strategy: 'median',
        window: 5,
        alpha: 0.3,
    },
};

// Per-region data (for visualization)
export interface RegionDetail { // TODO: export raw RGB as 1d array for PCA approaches
    polygon: Point[];
    rgb: RGB | null;
    pulse: number | null;
}

// Full result from processFrame()
export interface HeartRateResult {
    timestamp: number;
    status: 'pending' | 'ready'; //'pending' until the signal buffer has filled and an estimator has produced output.
    bpm: number | null; // BPM estimate — null while status is 'pending'.
    confidence: number; // Confidence in the estimate, 0 while pending.
    signal: { // Waveform data for real-time plotting
        raw: number | null; // Fused BPV sample (pre-bandpass) — null if no sample this frame
        filtered: number | null; // Post-bandpass filtered sample — null if no sample this frame
        peakDetected: boolean; // Whether a pulse peak was detected this frame
    };
    estimators: { // Raw estimator outputs — only present for enabled estimators
        peak?: PeakResult;
        fft?: FFTEstimate;
    };
    regions: Record<string, RegionDetail>; // Per-region geometry + signal for face overlay rendering
    methods: Record<string, number | null>; // Per-method pulse values (for multi-projection visualisation, e.g. POS vs CHROM)
}

// Deep-merge user config over defaults. 1 level deep - nested objects spread not recursed
function mergeConfig(
    defaults: HeartRateMonitorConfig,
    overrides?: Partial<HeartRateMonitorConfig>,
): HeartRateMonitorConfig {
    if (!overrides) return { ...defaults };

    return {
        pipeline: { ...defaults.pipeline, ...overrides.pipeline },
        projectionMethods: overrides.projectionMethods ?? defaults.projectionMethods,
        maxConsecutiveMisses: overrides.maxConsecutiveMisses ?? defaults.maxConsecutiveMisses ?? 0,
        pulse: { ...defaults.pulse, ...overrides.pulse },
        // Estimators: explicit undefined disables; omitted key inherits default
        peak: 'peak' in overrides
            ? (overrides.peak ? { ...defaults.peak, ...overrides.peak } : undefined)
            : defaults.peak,
        fft: 'fft' in overrides
            ? (overrides.fft ? { ...defaults.fft, ...overrides.fft } : undefined)
            : defaults.fft,
        smoothing: { ...defaults.smoothing, ...overrides.smoothing },
    };
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
    private consecutiveEmpty = 0;

    constructor(config?: Partial<HeartRateMonitorConfig>, rois?: LandmarkerROIs, methodInstances?: WindowedPulseMethod[]) {
        this.config = mergeConfig(DEFAULT_MONITOR_CONFIG, config);
        const cfg = this.config;
        const { pipeline } = cfg;

        this.roi = new ROIExtractor(rois);
        // Build or use provided projection method
        const methods = methodInstances ?? cfg.projectionMethods.map(name =>
            createMethod(name, {
                sampleRate: pipeline.sampleRate,
                windowMultiplier: cfg.pulse.posWindowMultiplier ?? 1.6,
            })
        );

        this.pulse = new PulseProcessor(this.roi.regionNames, {
            ...pipeline,
            ...cfg.pulse,
        }, methods);

        this.bandpass = BandpassFilter.fromPipelineConfig(pipeline);

        this.peak = cfg.peak
            ? new PeakEstimator({ sampleRate: pipeline.sampleRate, ...cfg.peak })
            : null;

        this.fft = cfg.fft
            ? new FFTEstimator(this.pulse.signalLength, { ...pipeline, ...cfg.fft })
            : null;

        this.smoother = createSmoother( // Unused at this point.
            cfg.smoothing.strategy,
            cfg.smoothing.window,
            cfg.smoothing.alpha,
        );
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
        const estimators: HeartRateResult['estimators'] = {};
        let filteredSample: number | null = null;
        let latestRawSample: number | null = null;
        let peakDetected = false;

        if (pulseFrame.fusedSamples.length === 0) {
            this.consecutiveEmpty++;
            if (this.consecutiveEmpty === this.config.maxConsecutiveMisses) {
                this.bandpass.reset();
                this.peak?.reset(); // peak detector state also goes stale
            } else if (this.consecutiveEmpty > this.config.pipeline.sampleRate * 2) {
                this.reset(); // full reset — signal is too old
            }
        } else {
            this.consecutiveEmpty = 0;
        }

        for (const { value, time: sampleTime } of pulseFrame.fusedSamples) {
            latestRawSample = value;

            // Bandpass filter — IIR assumes uniform dt, which interpolation guarantees
            filteredSample = this.bandpass.process(value);

            // Peak estimator — sees every grid sample with correct timestamp
            if (this.peak) {
                const peakResult = this.peak.pushSample(filteredSample, sampleTime);
                if (peakResult) estimators.peak = peakResult;
                if (this.peak.peakDetectedThisFrame) peakDetected = true;
            }
        }

        // FFT estimation - pass raw fused signal (FFTEstimator handles rate limiting + filtering)
        if (this.fft) {
            const fftResult = this.fft.update(this.pulse.getFusedSignal());
            if (fftResult) estimators.fft = fftResult;
        }

        this.resolveBPM(estimators);

        const regions: Record<string, RegionDetail> = {};
        for (const [name, roiRegion] of Object.entries(roiResult)) {
            regions[name] = {
                polygon: roiRegion.polygon,
                rgb: roiRegion.rgb,
                pulse: pulseFrame.regionPulses[name] ?? null,
            };
        }
        const hasEstimate = this.lastBPM !== null;

        // Fusion + Smoothing - TODO: look into this, might be a nice step later.
        // const fusedBPM = fuseBPM(this.config.method, raw.peak ?? null, raw.fft ?? null);
        // if (fusedBPM !== null) {
        //     this.lastBPM = this.smoother.update(fusedBPM);
        //     this.lastConfidence = getFusionConfidence(this.config.method, raw.peak ?? null, raw.fft ?? null);
        // }

        return {
            timestamp: time,
            status: hasEstimate ? 'ready' : 'pending',
            bpm: this.lastBPM,
            confidence: this.lastConfidence,
            signal: {
                raw: latestRawSample,
                filtered: filteredSample,
                peakDetected,
            },
            estimators,
            regions,
            methods: pulseFrame.methodPulses,
        };
    }

    reset(): void {
        this.bandpass.reset();
        this.pulse.reset();
        this.peak?.reset();
        this.fft?.reset();
        this.smoother.reset();
        this.lastBPM = null;
        this.lastConfidence = 0;
    }

    private resolveBPM(estimators: HeartRateResult['estimators']): void {
        // FFT takes priority — more stable over time
        if (estimators.fft) {
            this.lastBPM = this.smoother.update(estimators.fft.bpm);
            this.lastConfidence = estimators.fft.confidence ?? 0;
            return;
        }

        if (estimators.peak) {
            this.lastBPM = this.smoother.update(estimators.peak.bpm);
            this.lastConfidence = estimators.peak.confidence;
            return;
        }
    }
}
