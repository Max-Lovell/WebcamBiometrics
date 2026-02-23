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
// Internal state for a single region's RGB ring buffers
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
    }
}

// ─── PulseProcessor Class ───────────────────────────────────────────────────
export class PulseProcessor {
    private readonly config: PulseProcessorConfig;

    // Per-region state (RGB ring buffers + work arrays)
    private readonly regions: Record<string, RegionState> = {};
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
    // Fused POS overlap-add buffer — H arrays averaged across regions before overlap-adding
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

        // Interpolation work arrays
        const interpMax = this.rgbCapacity * 2;
        this.interpR = new Float32Array(interpMax);
        this.interpG = new Float32Array(interpMax);
        this.interpB = new Float32Array(interpMax);
        this.interpTimes = new Float64Array(interpMax);

        // Batch filtering work arrays
        this.fusedWork = new Float32Array(this.signalCapacity);
        this.batchFilteredWork = new Float32Array(this.signalCapacity);

        // Fused POS buffer and H array work array
        this.fusedPosBuffer = new Float32Array(this.signalCapacity);
        this.hArrayWork = new Float32Array(this.rgbCapacity); // H array length <= rgbCapacity
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

        // Process each region, collecting H arrays for fusion
        const hArrays: Float32Array[] = [];

        for (const [name, state] of Object.entries(this.regions)) {
            const rgb = rgbPerRegion[name]?.rgb ?? null;

            if (!rgb) {
                // TODO: Note this leaves gaps in RGB misaligning it with timestamp data, probably just persisting previously assigned value
                //    maybe assign NaN, then ignore NaNs in interpolation function, and NaN in region H array, and factor that into averaging for fused estimate?
                //    state.pushRGB(NaN, NaN, NaN);??
                regionPulses[name] = null;
                continue;
            }

            // Push RGB to regional buffer
            state.pushRGB(rgb.r, rgb.g, rgb.b);

            // If POS window full, start processing
            if (state.rgbReady) {
                const hArray = this.processRegionPOS(state);
                regionPulses[name] = hArray[hArray.length - 1]; // store most recent sample
                hArrays.push(hArray);
                anyRegionProduced = true;
            } else {
                regionPulses[name] = null;
            }
        }

        // Average H arrays across valid regions and overlap-add into fused buffer
        if (hArrays.length > 0) {
            // All H arrays should be the same length (same rgbCapacity, same interpolation target)
            const hLen = hArrays[0].length;

            // Average element-wise into work array
            this.hArrayWork.fill(0);
            for (const h of hArrays) {
                for (let i = 0; i < hLen; i++) {
                    this.hArrayWork[i] += h[i];
                }
            }
            for (let i = 0; i < hLen; i++) {
                this.hArrayWork[i] /= hArrays.length;
            }

            // Overlap-add averaged H into fused buffer
            const windowStart = (
                this.fusedPosIndex - hLen + 1 + this.signalCapacity
            ) % this.signalCapacity;
            const fusedH = this.hArrayWork.subarray(0, hLen);
            for (let i = 0; i < hLen; i++) {
                const idx = (windowStart + i) % this.signalCapacity;
                this.fusedPosBuffer[idx] += fusedH[i];
            }
        }

        // Advance fused POS index if any region produced data
        if (anyRegionProduced) {
            this.fusedPosIndex++;
            if (this.fusedPosIndex >= this.signalCapacity) {
                this.fusedPosIndex = 0;
                this.fusedPosReady = true;
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

    // ─── Internals ──────────────────────────────────────────────────────
    // Process one region's RGB buffer through POS. Returns the H array for fusion.
    private processRegionPOS(state: RegionState): Float32Array {
        // Unroll circular RGB buffer into chronological order
        const n = state.unrollRGB(this.timeBuffer);
        // Interpolate to even spacing at target FPS
        // TODO: NOTE start/end time different each from so are interpolated to slightly differently grid each time
        //  Possibly affecting the H values coming out with noise
        const interpLen = this.interpolateRGB(
            state.unrollR, state.unrollG, state.unrollB, state.unrollTimes,
            n,
            this.config.sampleRate
        );
        // Run POS to extract pulse signal window
        // TODO: note this allocates a new Float32Array each time, might want to put into work buffer. Doesn't really matter though.
        //   Also just consider moving POS algorithm here?
        return calculatePOS({
            r: this.interpR.subarray(0, interpLen),
            g: this.interpG.subarray(0, interpLen),
            b: this.interpB.subarray(0, interpLen),
        });
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

        // TODO: need to make sure we don't accidentally add more values than the POS buffer is expecting - should only output l values? but might have fast frames too...
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
