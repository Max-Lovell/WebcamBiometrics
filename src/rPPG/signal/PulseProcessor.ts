/**
 * Pulse Processor
 * Converts RGB sample into bandpass-filtered pulse signal for heart rate estimation.
 * Per-region RGBs stored in ring buffers, unroll once full, interpolate, run POS, overlap add
 * Filtered signal either stream per frame for peak estimator, or batch filtered signal for FFT
 * Owns all signal-processing state:
 *   - Per-region RGB ring buffers (short window for POS)
 *   - Per-region POS overlap-add buffers (long window for analysis)
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
}

export const DEFAULT_PULSE_CONFIG: PulseProcessorConfig = {
    ...DEFAULT_PIPELINE_CONFIG,
    posWindowMultiplier: 1.6, // From the POS paper: l = fps × 1.6 ≈ 32 frames at 20fps.
    signalWindowSeconds: 15,
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
// Internal state for single region's signal RGB and POS overlap-added values for MRC averaging later
class RegionState {
    // Note time is handled separately as not a per-region stat. See FloatRingBuffer class

    // Raw RGB ring buffers that POS operates on each frame
    readonly rBuffer: FloatRingBuffer;
    readonly gBuffer: FloatRingBuffer;
    readonly bBuffer: FloatRingBuffer;
    // POS overlap-add buffer - raw unfiltered pulse signal. Note not a ring buffer
    readonly posBuffer: Float32Array;
    posIndex: number = 0;
    posReady: boolean = false;

    // Pre-allocated work arrays for putting the ring buffers in chronological order
    readonly unrollR: Float32Array;
    readonly unrollG: Float32Array;
    readonly unrollB: Float32Array;
    readonly unrollTimes: Float64Array;

    constructor(
        rgbCapacity: number,    // POS window size (e.g., 1.6s = 48 at 30fps)
        signalCapacity: number  // Long signal buffer (e.g., 450 at 30fps × 15s)
    ) {
        this.rBuffer = new FloatRingBuffer(rgbCapacity);
        this.gBuffer = new FloatRingBuffer(rgbCapacity);
        this.bBuffer = new FloatRingBuffer(rgbCapacity);
        // NOTE not a ring buffer, too fiddly with random location access.
        this.posBuffer = new Float32Array(signalCapacity);

        this.unrollR = new Float32Array(rgbCapacity);
        this.unrollG = new Float32Array(rgbCapacity);
        this.unrollB = new Float32Array(rgbCapacity);
        this.unrollTimes = new Float64Array(rgbCapacity);
    }

    pushRGB(r: number, g: number, b: number): void {
        this.rBuffer.push(r);
        this.gBuffer.push(g);
        this.bBuffer.push(b);
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

    // Overlap-add a POS H array into the long-term signal buffer.
    overlapAdd(hArray: Float32Array, windowStart: number): void {
        const n = this.posBuffer.length;
        for (let i = 0; i < hArray.length; i++) {
            const idx = (windowStart + i) % n;
            this.posBuffer[idx] += hArray[i];
        }
    }

    advancePOS(): void {
        // Remember not a ring buffer here, so index reset is fine.
        this.posIndex++;
        if (this.posIndex >= this.posBuffer.length) {
            this.posIndex = 0;
            this.posReady = true;
        }
    }

    reset(): void {
        this.rBuffer.reset();
        this.gBuffer.reset();
        this.bBuffer.reset();
        this.posBuffer.fill(0);
        this.posIndex = 0;
        this.posReady = false;
    }
}

// ─── PulseProcessor Class ───────────────────────────────────────────────────
export class PulseProcessor {
    private readonly config: PulseProcessorConfig;

    // Per-region state (RGB buffers, POS buffers, work arrays)
    private readonly regions: Record<string, RegionState> = {}; // Or could use Map<string, RegionState> = new Map();
    // Shared timestamp ring buffer added to per frame not per region
    // TODO: switch to per-region timestamp incase a region doesn't have rgb data for a frame, so can be correctly interpolated.
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
    // Interpolated - sized with headroom for FPS variation TODO: no need to allow for FPS variation.
    private readonly interpR: Float32Array;
    private readonly interpG: Float32Array;
    private readonly interpB: Float32Array;
    private readonly interpTimes: Float64Array;
    // Used by getBatchFilteredSignal() chronological fused raw signal
    private readonly fusedWork: Float32Array;
    // Used by getBatchFilteredSignal() for batch-filtered output
    private readonly batchFilteredWork: Float32Array;
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
            this.regions[name] = new RegionState(this.rgbCapacity, this.signalCapacity)
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

        // Interpolation work arrays
        const interpMax = this.rgbCapacity * 2;
        this.interpR = new Float32Array(interpMax);
        this.interpG = new Float32Array(interpMax);
        this.interpB = new Float32Array(interpMax);
        this.interpTimes = new Float64Array(interpMax);

        // Batch filtering work arrays
        this.fusedWork = new Float32Array(this.signalCapacity);
        this.batchFilteredWork = new Float32Array(this.signalCapacity);
    }

    // Public API --------
    // POS estimate, and fuse per region results for each frame
    // Note rgbPerRegion type is to easily accept the output from the ROI RGB averager code elsewhere.
    pushFrame(rgbPerRegion: Record<string, { rgb: RGB | null }>, time: number): PulseFrame {
        // TODO: Big issue here - what happens if a region doesn't contain pixels (e.g. off screen)? RGB buffers etc then become misaligned
        //  Each region could have own time buffer and POS index - but then how to fuse them later?
        //  Or just store sentinel null/Nan/grey data and don't add it to fused estimate?
        //  Maybe do region frame timestamps and interpolate to the same times between regions (using start time) on H arrays can be averaged and combined?
        const regionPulses: Record<string, number | null> = {};
        let anyRegionProduced = false; // Store if we got valid data from any of the regions. Note no face detected is handled elsewhere

        // Store timestamp (shared across regions)
        this.timeBuffer.push(time);

        // Process each region
        for (const [name, state] of Object.entries(this.regions)) {
            const rgb = rgbPerRegion[name]?.rgb ?? null;

            if (!rgb) { // TODO: Note this leaves gaps in RGB misaligning it with timestamp data...
                // TODO: state.pushRGB(NaN, NaN, NaN);??
                regionPulses[name] = null;
                continue;
            }

            // Push RGB to regional buffer
            state.pushRGB(rgb.r, rgb.g, rgb.b);

            // If POS window full, start processing
            if (state.rgbReady) {
                regionPulses[name] = this.processRegionPOS(state);
                anyRegionProduced = true;
            } else {
                regionPulses[name] = null;
            }
        }

        // Advance POS buffer index if any region produced data
        if (anyRegionProduced) {
            for (const state of Object.values(this.regions)){
                state.advancePOS();
            }
        }

        // Fuse regions → stream filter → store filtered sample
        if (anyRegionProduced) {
            const validPulses = Object.values(regionPulses).filter(
                (v): v is number => v !== null
            );

            if (validPulses.length > 0) {
                // Average raw POS values across valid regions
                const fusedRaw = validPulses.reduce((s, v) => s + v, 0) / validPulses.length;

                // Feed through persistent bandpass → one clean sample out
                const filtered = this.streamFilter.process(fusedRaw);
                this.filteredBuffer.push(filtered);
                this._latestPulse = filtered;
            }
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
        if (!this.isSignalReady()) return null;

        const n = this.signalCapacity;
        const regionStates = Object.values(this.regions);
        const numRegions = regionStates.length;

        // All regions share the same posIndex (advanced together)
        const start = regionStates[0].posIndex;

        // Fuse: average all regions' POS buffers, chronological
        for (let i = 0; i < n; i++) {
            const idx = (start + i) % n;
            let sum = 0;
            for (const state of regionStates) {
                sum += state.posBuffer[idx];
            }
            this.fusedWork[i] = sum / numRegions;
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

    // Whether the signal buffer has filled at least once
    isSignalReady(): boolean {
        for (const state of Object.values(this.regions)) {
            if (state.posReady) return true;
        }
        return false;
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
        // reset streaming bandpass filter incase face lost so doesn't remember old signal TODO: do we want it to remember maybe?
        this.streamFilter.reset();
        this.filteredBuffer.reset();
        this._latestPulse = null;
    }

    // ─── Internals ──────────────────────────────────────────────────────
    // Process one region's RGB buffer through the POS algorithm.
    private processRegionPOS(state: RegionState): number {
        // Unroll circular RGB buffer into chronological order
        const n = state.unrollRGB(this.timeBuffer);
        // Interpolate to even spacing at target FPS
        const interpLen = this.interpolateRGB(
            state.unrollR, state.unrollG, state.unrollB, state.unrollTimes,
            n,
            this.config.sampleRate
        );
        //Run POS to extract pulse signal window
        const hArray = calculatePOS({
            r: this.interpR.subarray(0, interpLen),
            g: this.interpG.subarray(0, interpLen),
            b: this.interpB.subarray(0, interpLen),
        });

        // Overlap-add into long-term buffer
        // TODO: Double check this is getting the right index?
        const windowStart = (
            state.posIndex - hArray.length + 1 + this.signalCapacity
        ) % this.signalCapacity;
        state.overlapAdd(hArray, windowStart);

        // TODO: make it so average before overlap adding on H fused array as well.
        //  In future for MRC can keep per-region buffers and average them based on signal quality in each frame with proper fused overlap add.
        return hArray[hArray.length - 1];
    }

    // Linear interpolation of RGB channels to evenly spaced target sample rate.
      // Writes into pre-allocated interpR/G/B/Times arrays and returns number of interpolated samples
    private interpolateRGB(
        r: Float32Array, g: Float32Array, b: Float32Array,
        times: Float64Array,
        count: number,
        targetFps: number
    ): number {
        // If not enough points to interpolate between
        if (count < 2) {
            if (count === 1) {
                this.interpR[0] = r[0];
                this.interpG[0] = g[0];
                this.interpB[0] = b[0];
                this.interpTimes[0] = times[0];
            }
            return count;
        }

        const startTime = times[0];
        const endTime = times[count - 1];
        const duration = endTime - startTime;

        const targetSamples = Math.min(
            Math.ceil(duration * targetFps / 1000) + 1,
            this.interpR.length
        );
        const dt = duration / (targetSamples - 1);

        let sourceIdx = 0;

        for (let i = 0; i < targetSamples; i++) {
            const targetTime = startTime + i * dt;
            this.interpTimes[i] = targetTime;

            while (sourceIdx < count - 1 && times[sourceIdx + 1] < targetTime) {
                sourceIdx++;
            }

            if (sourceIdx >= count - 1) {
                this.interpR[i] = r[count - 1];
                this.interpG[i] = g[count - 1];
                this.interpB[i] = b[count - 1];
            } else {
                const t0 = times[sourceIdx];
                const t1 = times[sourceIdx + 1];
                const alpha = (targetTime - t0) / (t1 - t0);

                this.interpR[i] = r[sourceIdx] + alpha * (r[sourceIdx + 1] - r[sourceIdx]);
                this.interpG[i] = g[sourceIdx] + alpha * (g[sourceIdx + 1] - g[sourceIdx]);
                this.interpB[i] = b[sourceIdx] + alpha * (b[sourceIdx + 1] - b[sourceIdx]);
            }
        }

        return targetSamples;
    }
}
