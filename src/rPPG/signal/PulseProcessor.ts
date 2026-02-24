/**
 * Pulse Processor
 * Converts RGB samples into a fused pulse signal for heart rate estimation.
 * Per-region RGBs stored in ring buffers, unrolled once full, run through POS.
 * Region H arrays are averaged then overlap-added into a single fused buffer for analysis.
 * Estimation-agnostic — downstream consumers decide how to analyse the signal.
 * Owns all signal-processing state:
 *   - Per-region RGB ring buffers (short window for POS)
 *   - Fused POS overlap-add buffer (long window for analysis)
 *   - Pre-allocated work arrays (no per-frame allocation)
 *
 *   TODO: Linear interpolation for Bandpass filter and FFT method only
 *    Idea: Maintain reference to fused overlap added buffer start point and
 *      optionally interpolate/predict RGB onto a unified FPS grid
 *      so doesn't estimate to different timepoints each frame and cause smear.
 *      But can't linearly interpolate to future points
 */

import {Float64RingBuffer, FloatRingBuffer} from '../FloatRingBuffer';
import {calculatePOS} from './POS';
import {DEFAULT_PIPELINE_CONFIG, type RGB, type PipelineConfig} from '../types.ts';

// ─── Types ──────────────────────────────────────────────────────────────────

export interface PulseProcessorConfig extends PipelineConfig {
    // Note the speed/accuracy tradeoff on both of these...
    // N frames POS algorithm operates on
    posWindowMultiplier: number; // From the POS paper: l = fps × 1.6 ≈ 32 frames at 20fps.
    // Seconds of POS output to buffer for later analysis - where overlap adding occurs
    signalWindowSeconds: number;
    // Max consecutive missing frames before a region is excluded from fusion.
    // Below this threshold, the last valid RGB is held to keep buffers aligned.
    maxConsecutiveMisses: number;
}

export const DEFAULT_PULSE_CONFIG: PulseProcessorConfig = {
    ...DEFAULT_PIPELINE_CONFIG,
    posWindowMultiplier: 1.6, // From the POS paper: l = fps × 1.6 ≈ 32 frames at 20fps.
    signalWindowSeconds: 15,
    maxConsecutiveMisses: 3,
};


export interface PulseFrame {
    // Result returned by pushFrame() once per frame
    // Latest fused POS sample from overlap-add across regions (for streaming peak detection)
    fusedSample: number | null;
    // Per-region raw POS H values (before region fusion) - for per region visualisation
    regionPulses: Record<string, number | null>;
    // Whether the signal buffer has enough data for BPM estimation */
    signalReady: boolean;
}

// ─── Per-Region State ───────────────────────────────────────────────────────
// Internal state for a single region's RGB ring buffers.
// Handles missing frames via hold-last-value with a dropout counter —
// if a region misses too many consecutive frames, it's excluded from fusion.
class RegionState {
    // Note time is handled separately as not a per-region stat. See FloatRingBuffer class

    // Raw RGB ring buffers that POS operates on each frame
    readonly rBuffer: FloatRingBuffer;
    readonly gBuffer: FloatRingBuffer;
    readonly bBuffer: FloatRingBuffer;

    // Pre-allocated work arrays for putting the ring buffers in chronological order
    readonly unrollR: Float32Array;
    readonly unrollG: Float32Array;
    readonly unrollB: Float32Array;
    readonly unrollTimes: Float64Array;

    // Hold-last-value for missing frames
    private lastR: number = 0;
    private lastG: number = 0;
    private lastB: number = 0;
    private hasReceived: boolean = false; // Whether we've ever received a valid RGB sample

    // Dropout tracking — consecutive frames with no RGB data
    private _consecutiveMisses: number = 0;

    constructor(rgbCapacity: number) {
        this.rBuffer = new FloatRingBuffer(rgbCapacity);
        this.gBuffer = new FloatRingBuffer(rgbCapacity);
        this.bBuffer = new FloatRingBuffer(rgbCapacity);

        this.unrollR = new Float32Array(rgbCapacity);
        this.unrollG = new Float32Array(rgbCapacity);
        this.unrollB = new Float32Array(rgbCapacity);
        this.unrollTimes = new Float64Array(rgbCapacity);
    }

    pushRGB(r: number, g: number, b: number): void {
        this.rBuffer.push(r);
        this.gBuffer.push(g);
        this.bBuffer.push(b);
        this.lastR = r;
        this.lastG = g;
        this.lastB = b;
        this.hasReceived = true;
        this._consecutiveMisses = 0;
    }

    // Push the last known RGB value to keep buffers aligned with timestamps. Returns false if never received a valid sample (nothing to hold).
    pushHeld(): boolean {
        if (!this.hasReceived) return false;
        this.rBuffer.push(this.lastR);
        this.gBuffer.push(this.lastG);
        this.bBuffer.push(this.lastB);
        this._consecutiveMisses++;
        return true;
    }

    get consecutiveMisses(): number {
        return this._consecutiveMisses;
    }

    get rgbReady(): boolean {
        return this.rBuffer.isFull;
    }

    // Unroll RGB ring buffers into chronological order using pre-allocated work arrays for interpolation/pos calculation
    unrollRGB(times: Float64RingBuffer): number {
        const n = this.rBuffer.count;
        this.rBuffer.copyOrdered(this.unrollR);
        this.gBuffer.copyOrdered(this.unrollG);
        this.bBuffer.copyOrdered(this.unrollB);
        times.copyOrdered(this.unrollTimes);
        return n;
    }

    reset(): void {
        this.rBuffer.reset();
        this.gBuffer.reset();
        this.bBuffer.reset();
        this.lastR = 0;
        this.lastG = 0;
        this.lastB = 0;
        this.hasReceived = false;
        this._consecutiveMisses = 0;
    }
}

// ─── PulseProcessor Class ───────────────────────────────────────────────────
export class PulseProcessor {
    private readonly config: PulseProcessorConfig;

    // Per-region state (RGB ring buffers + work arrays)
    private readonly regions: Record<string, RegionState> = {};
    // Shared timestamp ring buffer — per-region alignment handled by hold-last-value
    private readonly timeBuffer: Float64RingBuffer;

    // Derived constants
    private readonly rgbCapacity: number;
    private readonly signalCapacity: number;

    // Fuse the overlap added hArrays from each region
    private readonly hArrayWork: Float32Array; // Temp storage for accumulating H array average across regions

    // Fused POS overlap-add buffer — H arrays averaged across regions and are overlap-added into this
    private readonly fusedPosBuffer: Float32Array;
    private fusedPosIndex: number = 0;
    private fusedPosReady: boolean = false;

    // For unwrapping fusedPosBuffer into chronological order (used by getFusedSignal)
    private readonly fusedWork: Float32Array;


    // Note Partial<> makes every property optional.
    constructor(regionNames: string[] = ['region'], config?: Partial<PulseProcessorConfig>) {
        this.config = { ...DEFAULT_PULSE_CONFIG, ...config }; // Config with sensible defaults if not used.

        const fps = this.config.sampleRate;
        this.rgbCapacity = Math.ceil(fps * this.config.posWindowMultiplier);
        this.signalCapacity = Math.ceil(fps * this.config.signalWindowSeconds);

        // Initialize per-region state
        for (const name of regionNames) {
            this.regions[name] = new RegionState(this.rgbCapacity)
        }

        // Shared timestamp buffer
        this.timeBuffer = new Float64RingBuffer(this.rgbCapacity);

        // Fused POS buffer and H array work array
        this.fusedPosBuffer = new Float32Array(this.signalCapacity); // Stores overlap added POS samples
        this.hArrayWork = new Float32Array(this.rgbCapacity); // H array length <= rgbCapacity
        this.fusedWork = new Float32Array(this.signalCapacity); // Unroll workspace for getFusedSignal()
    }

    // ─── Public API ────────────────────────────────────────────────────
    // Process a single frame through the full pulse pipeline. Call once per frame.
    pushFrame(rgbPerRegion: Record<string, { rgb: RGB | null }>, time: number): PulseFrame {
        this.timeBuffer.push(time);

        // Per-region: ingest RGB, run POS, collect H arrays
        const { regionPulses, hArrays } = this.processRegions(rgbPerRegion);

        // Fuse H arrays → overlap-add → advance write head
        let fusedSample: number | null = null;
        if (hArrays.length > 0) {
            fusedSample = this.fuseAndOverlapAdd(hArrays);
            this.advanceFusedIndex();
        }

        return {
            fusedSample,
            regionPulses,
            signalReady: this.isSignalReady(),
        };
    }

    // Get the full fused signal in chronological order (for FFT analysis).
    // Returns null if the buffer hasn't filled once yet.
    getFusedSignal(output?: Float32Array): Float32Array | null {
        if (!this.fusedPosReady) return null;

        const n = this.signalCapacity;
        const out = output ?? this.fusedWork;

        // Unroll fused buffer into chronological order
        for (let i = 0; i < n; i++) {
            out[i] = this.fusedPosBuffer[(this.fusedPosIndex + i) % n];
        }

        return out;
    }

    // Whether the fused signal buffer has filled at least once
    isSignalReady(): boolean {
        return this.fusedPosReady;
    }

    // Length of the signal buffers
    get signalLength(): number {
        return this.signalCapacity;
    }

    // Sample rate from config
    get sampleRate(): number {
        return this.config.sampleRate;
    }

    // Reset all state - Call when face tracking lost or user request
    reset(): void {
        this.timeBuffer.reset();
        for (const state of Object.values(this.regions)) {
            state.reset();
        }
        this.fusedPosBuffer.fill(0);
        this.fusedPosIndex = 0;
        this.fusedPosReady = false;
    }

    // ─── Pipeline Steps (called by pushFrame) ────────────────────────

    // Ingest RGB for each region, hold-last-value on dropout, run POS when ready.
    private processRegions(rgbPerRegion: Record<string, { rgb: RGB | null }>): {
        regionPulses: Record<string, number | null>;
        hArrays: Float32Array[];
    } {
        const regionPulses: Record<string, number | null> = {};
        const hArrays: Float32Array[] = [];

        for (const [name, state] of Object.entries(this.regions)) {
            const rgb = rgbPerRegion[name]?.rgb ?? null;

            if (rgb) {
                state.pushRGB(rgb.r, rgb.g, rgb.b);
            } else if (state.consecutiveMisses < this.config.maxConsecutiveMisses) {
                if (!state.pushHeld()) {
                    regionPulses[name] = null;
                    continue; // Never received data — nothing to hold
                }
            } else {
                regionPulses[name] = null;
                continue; // Sustained dropout — exclude from fusion
            }

            if (state.rgbReady) {
                const hArray = this.processRegionPOS(state);
                regionPulses[name] = hArray[hArray.length - 1];
                hArrays.push(hArray);
            } else {
                regionPulses[name] = null;
            }
        }

        return { regionPulses, hArrays };
    }

    // Average H arrays element-wise and overlap-add into the fused signal buffer and return the most recent fused sample (for stream filtering)
    private fuseAndOverlapAdd(
        hArrays: Float32Array[],
        exportCurrentSample: boolean = true // True = POS value for this frame, False = fully overlap added sample from -l frames
    ): number {
        const hLen = hArrays[0].length;

        // Element-wise average into work array
        this.hArrayWork.fill(0);
        for (const h of hArrays) {
            // console.log('Region H Length: ', h.length, this.rgbCapacity)
            for (let i = 0; i < hLen; i++) {
                this.hArrayWork[i] += h[i];
            }
        }
        for (let i = 0; i < hLen; i++) {
            this.hArrayWork[i] /= hArrays.length;
        }

        // Overlap-add into fused buffer
        const windowStart = (
            this.fusedPosIndex - hLen + 1 + this.signalCapacity
        ) % this.signalCapacity;
        for (let i = 0; i < hLen; i++) {
            const idx = (windowStart + i) % this.signalCapacity;
            this.fusedPosBuffer[idx] += this.hArrayWork[i];
        }

        // export this frame's POS value, or overlap added value
        return exportCurrentSample ? this.hArrayWork[hLen - 1] : this.fusedPosBuffer[windowStart];
    }

    // Advance the fused POS write index, wrapping and marking ready on first fill.
    private advanceFusedIndex(): void {
        this.fusedPosIndex++;
        if (this.fusedPosIndex >= this.signalCapacity) {
            this.fusedPosIndex = 0;
            this.fusedPosReady = true;
        }
    }

    // ─── Internals ──────────────────────────────────────────────────────
    // Process one region's RGB buffer through POS. Returns the H array for fusion.
    private processRegionPOS(state: RegionState): Float32Array {
        // Unroll circular RGB buffer into chronological order
        state.unrollRGB(this.timeBuffer);

        // Run POS directly on raw samples — POS doesn't require uniform spacing
        return calculatePOS({
            r: state.unrollR,
            g: state.unrollG,
            b: state.unrollB,
        });
    }
}
