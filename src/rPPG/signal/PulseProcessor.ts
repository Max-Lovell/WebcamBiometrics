/**
 * Pulse Processor
 * Converts RGB samples into bandpass-filtered pulse signal for heart rate estimation.
 * Per-region RGBs stored in ring buffers, unrolled once full, interpolated, run through POS.
 * Region H arrays are averaged then overlap-added into a single fused buffer for analysis.
 * Filtered signal either streamed per frame for peak estimator, or batch filtered for FFT.
 * Owns all signal-processing state:
 *   - Per-region RGB ring buffers (short window for POS)
 *   - Fused POS overlap-add buffer (long window for analysis)
 *   - Streaming bandpass filter (persistent, one sample per frame)
 *   - Filtered signal ring buffer (for display + peak estimator)
 *   - Pre-allocated work arrays (no per-frame allocation)
 */

import {Float64RingBuffer, FloatRingBuffer} from '../FloatRingBuffer';
import {BandpassFilter} from './BandpassFilter';
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
    // Latest bandpass-filtered pulse value fused across regions - for displaying waveform
    pulse: number | null;
    // Per-region raw POS H values (before region fusion or bandpass filtering) - for per region visualisation
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

    // Streaming filter
    // TODO: batch vs streaming probably useless? But need to make sure the H array is overlap added for all regions.
    // Persistent bandpass filter for streaming option - uses 1 fused sample per frame. Reset when signal lost
    private readonly streamFilter: BandpassFilter;
    // Buffer of filtered samples read by peak estimator for real-time waveform display.
    private readonly filteredBuffer: FloatRingBuffer;

    // Derived constants
    private readonly rgbCapacity: number;
    private readonly signalCapacity: number;

    // Pre-allocated work arrays
    // Interpolated RGB — exactly rgbCapacity samples on a fixed grid
    private readonly interpR: Float32Array;
    private readonly interpG: Float32Array;
    private readonly interpB: Float32Array;
    // Fused POS overlap-add buffer — H arrays averaged across regions before overlap-adding here
    private readonly fusedPosBuffer: Float32Array;
    private fusedPosIndex: number = 0;
    private fusedPosReady: boolean = false;
    // Used by getBatchFilteredSignal() chronological fused raw signal
    private readonly fusedWork: Float32Array;
    // Used by getBatchFilteredSignal() for batch-filtered output
    private readonly batchFilteredWork: Float32Array;
    // Temp storage for accumulating H array average across regions
    private readonly hArrayWork: Float32Array;
    // Latest stream-filtered pulse value (null until enough data)
    private _latestPulse: number | null = null;

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

        // Streaming bandpass filter (persistent across frames)
        this.streamFilter = BandpassFilter.fromBPM(
            this.config.minBPM,
            this.config.maxBPM,
            this.config.sampleRate
        );

        // Filtered signal ring buffer — same capacity as POS signal buffer
        this.filteredBuffer = new FloatRingBuffer(this.signalCapacity);

        // Interpolation work arrays — exactly rgbCapacity, matching the fixed output grid
        this.interpR = new Float32Array(this.rgbCapacity);
        this.interpG = new Float32Array(this.rgbCapacity);
        this.interpB = new Float32Array(this.rgbCapacity);

        // Batch filtering work arrays
        this.fusedWork = new Float32Array(this.signalCapacity);
        this.batchFilteredWork = new Float32Array(this.signalCapacity);

        // Fused POS buffer and H array work array
        this.fusedPosBuffer = new Float32Array(this.signalCapacity);
        this.hArrayWork = new Float32Array(this.rgbCapacity); // H array length <= rgbCapacity
    }

    // ─── Public API ────────────────────────────────────────────────────
    // Process a single frame through the full pulse pipeline. Call once per frame.
    pushFrame(rgbPerRegion: Record<string, { rgb: RGB | null }>, time: number): PulseFrame {
        this.timeBuffer.push(time);

        // Per-region: ingest RGB, run POS, collect H arrays
        const { regionPulses, hArrays } = this.processRegions(rgbPerRegion);

        // Fuse H arrays → overlap-add → advance write head → stream filter
        if (hArrays.length > 0) {
            const fusedPulse = this.fuseAndOverlapAdd(hArrays);
            this.advanceFusedIndex();
            this.streamFilterSample(fusedPulse);
        }

        return {
            pulse: this._latestPulse,
            regionPulses,
            signalReady: this.isSignalReady(),
        };
    }

    // Filtered signal for peak-based BPM estimation. Samples are bandpass filtered as they arrive.
    // output - Optional pre-allocated array (must be >= signalLength)
    // returns filtered samples in chronological order, or null if not ready
    getStreamFilteredSignal(output?: Float32Array): Float32Array | null {
        if (!this.filteredBuffer.isFull) return null;

        const out = output ?? new Float32Array(this.signalCapacity);
        this.filteredBuffer.copyOrdered(out);
        return out;
    }

    // Get a batch-filtered signal for FFT-based BPM estimation.
    // Note filters from scratch each time. call infrequently (e.g., every ~0.5s).
    getBatchFilteredSignal(): Float32Array | null {
        if (!this.fusedPosReady) return null;

        const n = this.signalCapacity;

        // Unroll fused buffer into chronological order
        for (let i = 0; i < n; i++) {
            const idx = (this.fusedPosIndex + i) % n;
            this.fusedWork[i] = this.fusedPosBuffer[idx];
        }

        // Fresh bandpass filter (clean state, no transient bleed)
        const batchFilter = BandpassFilter.fromBPM(
            this.config.minBPM,
            this.config.maxBPM,
            this.config.sampleRate
        );

        for (let i = 0; i < n; i++) {
            this.batchFilteredWork[i] = batchFilter.process(this.fusedWork[i]);
        }

        return this.batchFilteredWork;
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
        // reset streaming bandpass filter incase face lost so doesn't remember old signal TODO: do we want it to remember maybe?
        this.streamFilter.reset();
        this.filteredBuffer.reset();
        this._latestPulse = null;
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
    private fuseAndOverlapAdd(hArrays: Float32Array[]): number {
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
        const windowStart = (
            this.fusedPosIndex - hLen + 1 + this.signalCapacity
        ) % this.signalCapacity;
        for (let i = 0; i < hLen; i++) {
            const idx = (windowStart + i) % this.signalCapacity;
            this.fusedPosBuffer[idx] += this.hArrayWork[i];
        }

        return this.hArrayWork[hLen - 1];
    }

    // Advance the fused POS write index, wrapping and marking ready on first fill.
    private advanceFusedIndex(): void {
        this.fusedPosIndex++;
        if (this.fusedPosIndex >= this.signalCapacity) {
            this.fusedPosIndex = 0;
            this.fusedPosReady = true;
        }
    }

    // Feed a single fused pulse sample through the streaming bandpass filter
    private streamFilterSample(sample: number): void {
        const filtered = this.streamFilter.process(sample);
        this.filteredBuffer.push(filtered);
        this._latestPulse = filtered;
    }

    // ─── Internals ──────────────────────────────────────────────────────
    // Process one region's RGB buffer through POS. Returns the H array for fusion.
    private processRegionPOS(state: RegionState): Float32Array {
        // Unroll circular RGB buffer into chronological order
        state.unrollRGB(this.timeBuffer);

        // Interpolate onto a fixed grid: rgbCapacity samples at sampleRate, anchored to most recent frame
        this.interpolateRGB(
            state.unrollR, state.unrollG, state.unrollB, state.unrollTimes,
            this.rgbCapacity
        );

        // Run POS on the uniformly-spaced signal — always exactly rgbCapacity samples
        return calculatePOS({
            r: this.interpR,
            g: this.interpG,
            b: this.interpB,
        });
    }

    // Interpolate raw RGB samples onto fixed FPS grid for POS window length of frames
    // Starts at most recent frames timestamp and gets rgbCapacity samples at 1/sampleRate spacing
    // Note this even spacing is only a requirement for FFT
    private interpolateRGB(
        r: Float32Array, g: Float32Array, b: Float32Array,
        times: Float64Array,
        count: number
    ): void {
        // TODO: just note this interpolates to a different gird anchored on recent frame each time - could smear signal
        //  Would be better to rely on a global grid
        if (count < 2) {
            // Not enough data to interpolate — fill with the single value or zeros
            const val = count === 1;
            for (let i = 0; i < this.rgbCapacity; i++) {
                this.interpR[i] = val ? r[0] : 0;
                this.interpG[i] = val ? g[0] : 0;
                this.interpB[i] = val ? b[0] : 0;
            }
            return;
        }

        const n = this.rgbCapacity;
        const dt = 1000 / this.config.sampleRate; // ms between grid samples
        const endTime = times[count - 1];          // anchor: most recent frame
        // console.log('Actual FPS: ', Math.floor(times.length / (endTime-times[0]) * 1000));
        // Walk backwards through source data to interpolate onto the grid
        let sourceIdx = count - 2; // start just before the last source sample

        for (let i = n - 1; i >= 0; i--) {
            const targetTime = endTime - (n - 1 - i) * dt;

            // Advance source index backwards to bracket targetTime
            while (sourceIdx > 0 && times[sourceIdx] > targetTime) {
                sourceIdx--;
            }

            if (targetTime <= times[0]) {
                // Before earliest sample — clamp to first value
                this.interpR[i] = r[0];
                this.interpG[i] = g[0];
                this.interpB[i] = b[0];
            } else if (targetTime >= times[count - 1]) {
                // After latest sample — clamp to last value
                this.interpR[i] = r[count - 1];
                this.interpG[i] = g[count - 1];
                this.interpB[i] = b[count - 1];
            } else {
                // Linear interpolation between sourceIdx and sourceIdx+1
                const t0 = times[sourceIdx];
                const t1 = times[sourceIdx + 1];
                const alpha = (targetTime - t0) / (t1 - t0);

                this.interpR[i] = r[sourceIdx] + alpha * (r[sourceIdx + 1] - r[sourceIdx]);
                this.interpG[i] = g[sourceIdx] + alpha * (g[sourceIdx + 1] - g[sourceIdx]);
                this.interpB[i] = b[sourceIdx] + alpha * (b[sourceIdx + 1] - b[sourceIdx]);
            }
        }
    }
}
