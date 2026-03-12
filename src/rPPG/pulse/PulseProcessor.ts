/**
 * Pulse Processor
 * Converts RGB samples into a fused pulse signal for heart rate estimation.
 * Per-region RGBs stored in ring buffers, unrolled once full, run through POS.
 * Region H arrays are combined then overlap-added into a single fused buffer for analysis.
 * Estimation-agnostic — downstream consumers decide how to analyse the signal.
 */

import { RGBBuffer } from './RGBBuffer.ts';
import type { WindowedPulseMethod } from './projection/types.ts';
import {POS} from './projection/POS.ts';
import {DEFAULT_PIPELINE_CONFIG, type RGB, type PipelineConfig} from '../types.ts';

// ─── Types ──────────────────────────────────────────────────────────────────

export interface PulseProcessorConfig extends PipelineConfig {
    // Note the speed/accuracy tradeoff on both of these...
    // N frames POS algorithm operates on
    posWindowMultiplier: number; // From the POS paper: l = fps × 1.6 ≈ 32 frames at 20fps.
    // Seconds of POS output to buffer for later analysis - where overlap adding occurs
    signalWindowSeconds: number; // TODO: this primarily controls the FFT length - the length should really be requested by the FFTEstimator?
    // Max consecutive missing frames before a region is excluded from fusion.
    // Below this threshold, the last valid RGB is held to keep buffers aligned.
    maxConsecutiveMisses: number;
    // Whether to interpolate RGB onto a uniform grid before POS processing.
    interpolate: boolean; // Toggle interpolation for bandpass filter and FFT.
}

export const DEFAULT_PULSE_CONFIG: PulseProcessorConfig = {
    ...DEFAULT_PIPELINE_CONFIG,
    posWindowMultiplier: 1.6, // From the POS paper: l = fps × 1.6 ≈ 32 frames at 20fps.
    signalWindowSeconds: 10,
    maxConsecutiveMisses: 3,
    interpolate: true,
};

// A single fused sample with its associated (possibly synthetic grid) timestamp
export interface FusedSample {
    value: number;
    time: number;
}

export interface PulseFrame {
    // Fused POS samples from this frame. When interpolation is off: 0 or 1 entry.
    // When on: 0+ entries (one per grid point that fell between previous and current real frame).
    fusedSamples: FusedSample[];
    // Per-region raw POS H values (before region fusion) - for per region visualisation.
    // Reflects the latest processed state (last grid point if interpolating).
    regionPulses: Record<string, number | null>;
    methodPulses: Record<string, number | null>;  // TODO: these should probably be nested inside the regions? Work on the return type structure once everything else is done.
    // Whether the signal buffer has enough data for BPM estimation
    signalReady: boolean;
}

// ─── PulseProcessor Class ───────────────────────────────────────────────────
export class PulseProcessor {
    private readonly config: PulseProcessorConfig;
    private readonly methods: WindowedPulseMethod[];
    private readonly buffer: RGBBuffer;

    // Derived constants
    private readonly signalCapacity: number;

    // Fused overlap-add buffer — H arrays averaged across regions and are overlap-added here
    private readonly fusedSignalBuffer: Float32Array;
    private fusedWriteIndex: number = 0;
    private fusedReady: boolean = false;

    // Work arrays
    // Fuse the overlap added hArrays from each region
    private readonly hArrayWork: Float32Array; // Temp storage for accumulating H array average across regions
    private readonly fusedWork: Float32Array; // For unwrapping fusedSignalBuffer into chronological order (used by getFusedSignal)

    // Note Partial<> makes every property optional.
    constructor(regionNames: string[] = ['region'], config?: Partial<PulseProcessorConfig>, methods?: WindowedPulseMethod[]) {
        this.config = { ...DEFAULT_PULSE_CONFIG, ...config }; // Config with sensible defaults if not used.
        const fps = this.config.sampleRate;

        // Use provided method or default to POS
        this.methods = methods ?? [new POS(fps, this.config.posWindowMultiplier)];
        // Validate all methods share the same window size TODO: consider changing this to separate windows I guess?
        const windowSize = this.methods[0].windowSize;
        for (const m of this.methods) {
            if (m.windowSize !== windowSize) {
                throw new Error(
                    `All methods must share the same windowSize. ` +
                    `'${this.methods[0].name}' has ${windowSize}, ` +
                    `'${m.name}' has ${m.windowSize}.`
                );
            }
        }
        // Create RGB buffer with the methods window size
        this.buffer = new RGBBuffer(regionNames, {
            sampleRate: fps,
            windowSize,
            interpolate: this.config.interpolate,
            maxConsecutiveMisses: this.config.maxConsecutiveMisses,
        });

        this.signalCapacity = Math.ceil(fps * this.config.signalWindowSeconds);
        // Fused POS buffer and H array work array
        this.fusedSignalBuffer = new Float32Array(this.signalCapacity); // Stores overlap added POS samples
        this.hArrayWork = new Float32Array(windowSize); // H array length <= rgbCapacity
        this.fusedWork = new Float32Array(this.signalCapacity); // Unroll workspace for getFusedSignal()
    }

    // ─── Public API ────────────────────────────────────────────────────
    // Process a single camera frame through the full pulse pipeline. Call once per real frame.
    pushFrame(rgbPerRegion: Record<string, { rgb: RGB | null }>, time: number): PulseFrame {
        const ticks = this.buffer.pushFrame(rgbPerRegion, time);
        const fusedSamples: FusedSample[] = [];
        let regionPulses: Record<string, number | null> = {};
        let methodPulses: Record<string, number | null> = {};

        for (const tick of ticks) {
            const tickRegionPulses: Record<string, number | null> = {};
            const tickMethodPulses: Record<string, number | null> = {};

            // Per-method region-fused H arrays (one per method that produced output)
            const methodHArrays: Float32Array[] = [];

            for (const method of this.methods) {
                const regionHArrays: Float32Array[] = [];

                for (const [name, window] of Object.entries(tick.regionWindows)) {
                    if (!window) {
                        // Only set null if not already set by another method
                        if (!(name in tickRegionPulses)) {
                            tickRegionPulses[name] = null;
                        }
                        continue;
                    }

                    const h = method.process(window);

                    // Region pulse: last value from the first method that produces one
                    // (keeps existing behaviour — one value per region for display)
                    if (!(name in tickRegionPulses) || tickRegionPulses[name] === null) {
                        tickRegionPulses[name] = h[h.length - 1];
                    }

                    regionHArrays.push(h);
                }

                // Fuse regions for this method
                if (regionHArrays.length > 0) {
                    const methodH = this.fuseRegions(regionHArrays);
                    tickMethodPulses[method.name] = methodH[methodH.length - 1];
                    methodHArrays.push(methodH);
                } else {
                    tickMethodPulses[method.name] = null;
                }
            }

            regionPulses = tickRegionPulses;
            methodPulses = tickMethodPulses;

            // Fuse across methods and overlap-add
            if (methodHArrays.length > 0) {
                const fusedValue = this.fuseAndOverlapAdd(methodHArrays);
                this.advanceFusedIndex();
                fusedSamples.push({ value: fusedValue, time: tick.time });
            }
        }

        // Fill in nulls for any regions not covered
        for (const name of this.buffer.regionNames) {
            if (!(name in regionPulses)) {
                regionPulses[name] = null;
            }
        }

        return {
            fusedSamples,
            regionPulses,
            methodPulses,
            signalReady: this.isSignalReady(),
        };
    }

    // Average an array of same-length H arrays element-wise into hArrayWork
    private fuseRegions(hArrays: Float32Array[]): Float32Array {
        const hLen = hArrays[0].length;
        const result = new Float32Array(hLen);
        for (const h of hArrays) {
            for (let i = 0; i < hLen; i++) {
                result[i] += h[i];
            }
        }
        for (let i = 0; i < hLen; i++) {
            result[i] /= hArrays.length;
        }
        return result;
    }

    // Get the full fused signal in chronological order (for FFT analysis).
    // Returns null if the buffer hasn't filled once yet.
    getFusedSignal(output?: Float32Array): Float32Array | null {
        if (!this.fusedReady) return null;

        const n = this.signalCapacity;
        const out = output ?? this.fusedWork;

        // Unroll fused buffer into chronological order
        for (let i = 0; i < n; i++) {
            out[i] = this.fusedSignalBuffer[(this.fusedWriteIndex + i) % n];
        }

        return out;
    }

    // Whether the fused signal buffer has filled at least once
    isSignalReady(): boolean {
        return this.fusedReady;
    }

    // Length of the signal buffers
    get signalLength(): number {
        return this.signalCapacity;
    }

    // Sample rate from config
    get sampleRate(): number {
        return this.config.sampleRate;
    }

    // Whether interpolation is enabled
    get interpolating(): boolean {
        return this.config.interpolate;
    }

    // Reset all state - Call when face tracking lost or user request
    reset(): void {
        this.buffer.reset();
        this.fusedSignalBuffer.fill(0);
        this.fusedWriteIndex = 0;
        this.fusedReady = false;
    }

    // ─── Overlap-Add Logic ──────────────────────────────────────────────
    // Average H arrays element-wise and overlap-add into the fused signal buffer
    // and return the most recent fused sample (for stream filtering)
    private fuseAndOverlapAdd(
        hArrays: Float32Array[],
        exportCurrentSample: boolean = true // True = POS value for this frame, False = fully overlap added sample from -l frames
    ): number {
        const hLen = hArrays[0].length;

        // Element-wise average into work array
        this.hArrayWork.fill(0);
        for (const h of hArrays) {
            for (let i = 0; i < hLen; i++) {
                this.hArrayWork[i] += h[i];
            }
        }
        for (let i = 0; i < hLen; i++) {
            this.hArrayWork[i] /= hArrays.length;
        }

        // Overlap-add into fused buffer
        this.fusedSignalBuffer[this.fusedWriteIndex] = 0;
        const windowStart = (
            this.fusedWriteIndex - hLen + 1 + this.signalCapacity
        ) % this.signalCapacity;
        for (let i = 0; i < hLen; i++) {
            const idx = (windowStart + i) % this.signalCapacity;
            this.fusedSignalBuffer[idx] += this.hArrayWork[i];
        }

        // export this frame's POS value, or overlap added value
        return exportCurrentSample ? this.hArrayWork[hLen - 1] : this.fusedSignalBuffer[windowStart];
    }

    // Advance the fused POS write index, wrapping and marking ready on first fill.
    private advanceFusedIndex(): void {
        this.fusedWriteIndex++;
        if (this.fusedWriteIndex >= this.signalCapacity) {
            this.fusedWriteIndex = 0;
            this.fusedReady = true;
        }
    }
}
