/**
 * Pulse Processor — Tier 2 of the rPPG pipeline
 *
 * Converts a stream of per-region RGB samples into a clean, bandpass-filtered
 * pulse signal suitable for heart rate estimation.
 *
 * Pipeline (per frame):
 *   1. Store RGB sample for each region in a ring buffer
 *   2. Once the RGB buffer is full, for each region:
 *      a. Unroll the circular buffer into chronological order
 *      b. Interpolate to even sample spacing (camera FPS varies)
 *      c. Run POS algorithm to extract pulse signal
 *      d. Overlap-add the POS window into the long-term signal buffer
 *   3. Fuse per-region POS values by averaging
 *   4. Stream the fused sample through a persistent bandpass filter
 *   5. Store the filtered sample in a ring buffer
 *
 * Two ways to read the filtered signal:
 *
 *   getStreamFilteredSignal()  — ring buffer of per-frame filtered samples.
 *     Each sample was filtered as it arrived through the persistent streaming
 *     bandpass filter. Updated every frame. Use for PeakEstimator and
 *     real-time waveform display.
 *
 *   getBatchFilteredSignal()   — re-filters the raw POS buffer from scratch.
 *     Uses a fresh bandpass filter with clean state, so it captures all
 *     overlap-add corrections that the streaming filter missed (because
 *     overlap-add modifies past samples retroactively). More accurate,
 *     but more expensive. Call infrequently. Use for FFTEstimator.
 *
 * Why two filtered outputs?
 *
 *   POS overlap-add modifies past samples. When a new POS window is processed,
 *   its H values are *added* to positions in the long buffer that overlap with
 *   previous windows. The streaming filter has already processed and emitted
 *   those positions — it can't go back. For display and peak counting this
 *   doesn't matter. For FFT frequency analysis, it's worth re-filtering.
 *
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
import type {PipelineConfig} from './types';
import {DEFAULT_PIPELINE_CONFIG} from './types';

// ─── Types ──────────────────────────────────────────────────────────────────

export interface PulseProcessorConfig extends PipelineConfig {
    /**
     * POS window size as a multiplier of 1/sampleRate.
     * From the POS paper: l = fps × 1.6 ≈ 32 frames at 20fps.
     * This defines how many RGB samples the POS algorithm operates on
     * at a time. Larger = more temporal context, smaller = faster response.
     * Default: 1.6
     */
    posWindowMultiplier: number;

    /**
     * Seconds of pulse signal to buffer for downstream analysis.
     * This is the long buffer that FFT and peak detectors consume.
     * Longer = better FFT frequency resolution, but slower to fill and
     * less responsive to heart rate changes.
     * Default: 15
     */
    signalWindowSeconds: number;
}

export const DEFAULT_PULSE_CONFIG: PulseProcessorConfig = {
    ...DEFAULT_PIPELINE_CONFIG,
    posWindowMultiplier: 1.6,
    signalWindowSeconds: 15,
};

/** Per-region RGB average (same as ROIExtractor's RegionRGB) */
export interface RegionRGB {
    r: number;
    g: number;
    b: number;
}

/** Result returned by pushFrame() on every frame */
export interface PulseFrame {
    /**
     * Latest bandpass-filtered pulse value (fused across all valid regions).
     * Null if not enough data has accumulated yet.
     * Use this for real-time waveform display.
     */
    pulse: number | null;

    /**
     * Per-region raw POS H values (before region fusion or bandpass filtering).
     * Useful for per-region visualization or quality monitoring.
     * Null for regions that had no RGB data this frame or whose RGB buffer
     * isn't full yet.
     */
    regionPulses: Record<string, number | null>;

    /** Whether the signal buffer has enough data for BPM estimation */
    signalReady: boolean;
}

// ─── Per-Region State ───────────────────────────────────────────────────────

/**
 * Internal state for a single face region's signal accumulation.
 *
 * Each region independently accumulates RGB samples and builds its own
 * POS overlap-add buffer. These are kept separate so we can fuse them
 * later with equal weighting (Multi-Site averaging).
 */
class RegionState {
    // ── Short-term RGB ring buffers ──
    // Hold the most recent ~48 frames (at 30fps × 1.6) of raw RGB values.
    // POS operates on this window each frame.
    readonly rBuffer: FloatRingBuffer;
    readonly gBuffer: FloatRingBuffer;
    readonly bBuffer: FloatRingBuffer;

    // ── Long-term POS overlap-add buffer ──
    // Each POS window produces an H array that gets summed into this buffer.
    // Where windows overlap, values accumulate — reinforcing the true pulse
    // and averaging out noise. This is the raw (unfiltered) pulse signal.
    readonly posBuffer: Float32Array;
    posIndex: number = 0;
    posReady: boolean = false;

    // ── Pre-allocated work arrays ──
    // Used by unrollRGB() — allocated once, reused every frame.
    readonly unrollR: Float32Array;
    readonly unrollG: Float32Array;
    readonly unrollB: Float32Array;
    readonly unrollTimes: Float64Array;

    constructor(
        rgbCapacity: number,    // POS window size (e.g., 48 at 30fps)
        signalCapacity: number  // Long signal buffer (e.g., 450 at 30fps × 15s)
    ) {
        this.rBuffer = new FloatRingBuffer(rgbCapacity);
        this.gBuffer = new FloatRingBuffer(rgbCapacity);
        this.bBuffer = new FloatRingBuffer(rgbCapacity);
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

    /**
     * Unroll RGB ring buffers into chronological order using pre-allocated
     * work arrays. Returns the number of valid samples.
     */
    unrollRGB(times: Float64RingBuffer): number {
        const n = this.rBuffer.count;
        this.rBuffer.copyOrdered(this.unrollR);
        this.gBuffer.copyOrdered(this.unrollG);
        this.bBuffer.copyOrdered(this.unrollB);
        times.copyOrdered(this.unrollTimes);
        return n;
    }

    /**
     * Overlap-add a POS H array into the long-term signal buffer.
     */
    overlapAdd(hArray: Float32Array, windowStart: number): void {
        const n = this.posBuffer.length;
        for (let i = 0; i < hArray.length; i++) {
            const idx = (windowStart + i) % n;
            this.posBuffer[idx] += hArray[i];
        }
    }

    advancePOS(): void {
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

    /** Per-region state (RGB buffers, POS buffers, work arrays) */
    private readonly regions: Map<string, RegionState> = new Map();

    /**
     * Shared timestamp ring buffer.
     * Pushed once per frame (not per region) since all regions
     * share the same frame timestamp.
     */
    private readonly timeBuffer: Float64RingBuffer;

    // ── Streaming filter path ──

    /**
     * Persistent bandpass filter for the streaming path.
     *
     * Fed one fused sample per frame. Because it's IIR, its internal
     * state carries across frames — this is correct for streaming use
     * since each new sample follows the previous one naturally in time.
     *
     * Reset when tracking is lost (signal discontinuity would cause ringing).
     */
    private readonly streamFilter: BandpassFilter;

    /**
     * Ring buffer of stream-filtered samples.
     * Read by PeakEstimator and used for real-time waveform display.
     * One sample pushed per frame.
     */
    private readonly filteredBuffer: FloatRingBuffer;

    // ── Derived constants ──

    private readonly rgbCapacity: number;
    private readonly signalCapacity: number;

    // ── Pre-allocated work arrays ──

    /** Interpolation output (sized with headroom for FPS variation) */
    private readonly interpR: Float32Array;
    private readonly interpG: Float32Array;
    private readonly interpB: Float32Array;
    private readonly interpTimes: Float64Array;

    /** For getBatchFilteredSignal() — fused raw signal, chronological */
    private readonly fusedWork: Float32Array;

    /** For getBatchFilteredSignal() — batch-filtered output */
    private readonly batchFilteredWork: Float32Array;

    /** Latest stream-filtered pulse value (null until enough data) */
    private _latestPulse: number | null = null;

    constructor(config?: Partial<PulseProcessorConfig>, regionNames?: string[]) {
        this.config = { ...DEFAULT_PULSE_CONFIG, ...config };

        const fps = this.config.sampleRate;
        this.rgbCapacity = Math.ceil(fps * this.config.posWindowMultiplier);
        this.signalCapacity = Math.ceil(fps * this.config.signalWindowSeconds);

        // Initialize per-region state
        const names = regionNames ?? ['forehead', 'leftCheek', 'rightCheek'];
        for (const name of names) {
            this.regions.set(name, new RegionState(this.rgbCapacity, this.signalCapacity));
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

    // ─── Public API ─────────────────────────────────────────────────────

    /**
     * Feed one frame's worth of RGB data from all regions.
     *
     * Call this once per frame with the output from ROIExtractor.
     * Null regions (occluded, missing face) are skipped.
     *
     * @param rgbPerRegion - Per-region RGB averages. Null for missing regions.
     * @param time - Frame timestamp in ms (DOMHighResTimeStamp)
     * @returns Pulse data for this frame
     */
    pushFrame(
        rgbPerRegion: Record<string, RegionRGB | null>,
        time: number
    ): PulseFrame {
        const regionPulses: Record<string, number | null> = {};
        let anyRegionProduced = false;

        // Store timestamp (shared across regions)
        this.timeBuffer.push(time);

        // Process each region
        for (const [name, state] of this.regions) {
            const rgb = rgbPerRegion[name];

            if (!rgb) {
                regionPulses[name] = null;
                continue;
            }

            // 1. Push RGB into this region's short-term buffer
            state.pushRGB(rgb.r, rgb.g, rgb.b);

            // 2. If POS window is full, process it
            if (state.rgbReady) {
                regionPulses[name] = this.processRegionPOS(state);
                anyRegionProduced = true;
            } else {
                regionPulses[name] = null;
            }
        }

        // 3. Advance POS buffer index if any region produced data
        if (anyRegionProduced) {
            for (const state of this.regions.values()) {
                state.advancePOS();
            }
        }

        // 4. Fuse regions → stream filter → store filtered sample
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

    /**
     * Get the stream-filtered signal for peak-based BPM estimation
     * and real-time waveform display.
     *
     * Each sample was bandpass-filtered as it arrived, one per frame.
     * The waveform is clean enough for peak detection and display.
     *
     * @param output - Optional pre-allocated array (must be >= signalLength).
     *   Pass one in to avoid per-call allocation.
     * @returns Filtered samples in chronological order, or null if not ready
     */
    getStreamFilteredSignal(output?: Float32Array): Float32Array | null {
        if (!this.filteredBuffer.isFull) return null;

        const out = output ?? new Float32Array(this.signalCapacity);
        this.filteredBuffer.copyOrdered(out);
        return out;
    }

    /**
     * Get a batch-filtered signal for FFT-based BPM estimation.
     *
     * Re-filters the raw fused POS signal from scratch using a fresh
     * bandpass filter with clean state. Captures all overlap-add
     * corrections that the streaming filter missed.
     *
     * More accurate than getStreamFilteredSignal() but more expensive.
     * Call infrequently (e.g., every ~0.5s).
     *
     * Returns a reference to an internal work array — it will be
     * overwritten on the next call. Copy if you need to keep it.
     *
     * @returns Filtered samples in chronological order, or null if not ready
     */
    getBatchFilteredSignal(): Float32Array | null {
        if (!this.isSignalReady()) return null;

        const n = this.signalCapacity;
        const regionStates = [...this.regions.values()];
        const numRegions = regionStates.length;

        // All regions share the same posIndex (advanced together)
        const start = regionStates[0].posIndex;

        // 1. Fuse: average all regions' POS buffers, chronological
        for (let i = 0; i < n; i++) {
            const idx = (start + i) % n;
            let sum = 0;
            for (const state of regionStates) {
                sum += state.posBuffer[idx];
            }
            this.fusedWork[i] = sum / numRegions;
        }

        // 2. Fresh bandpass filter (clean state, no transient bleed)
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

    /** Whether the signal buffer has filled at least once */
    isSignalReady(): boolean {
        for (const state of this.regions.values()) {
            if (state.posReady) return true;
        }
        return false;
    }

    /** Length of the signal buffers */
    get signalLength(): number {
        return this.signalCapacity;
    }

    /** Sample rate from config */
    get sampleRate(): number {
        return this.config.sampleRate;
    }

    /**
     * Reset all state.
     *
     * Call when face tracking is lost, scene changes, or user requests reset.
     *
     * Critically resets the streaming bandpass filter — without this,
     * the filter's feedback terms would "remember" the old signal and
     * cause ringing artifacts when tracking resumes.
     */
    reset(): void {
        this.timeBuffer.reset();
        for (const state of this.regions.values()) {
            state.reset();
        }
        this.streamFilter.reset();
        this.filteredBuffer.reset();
        this._latestPulse = null;
    }

    // ─── Internals ──────────────────────────────────────────────────────

    /**
     * Process one region's RGB buffer through the POS algorithm.
     *
     *   1. Unroll circular RGB buffer into chronological order
     *   2. Interpolate to even spacing at target FPS
     *   3. Run POS to extract pulse signal window
     *   4. Overlap-add into long-term buffer
     *
     * @returns Latest POS H value for this region
     */
    private processRegionPOS(state: RegionState): number {
        const n = state.unrollRGB(this.timeBuffer);

        const interpLen = this.interpolateRGB(
            state.unrollR, state.unrollG, state.unrollB, state.unrollTimes,
            n,
            this.config.sampleRate
        );

        const hArray = calculatePOS({
            r: this.interpR.subarray(0, interpLen),
            g: this.interpG.subarray(0, interpLen),
            b: this.interpB.subarray(0, interpLen),
        });

        const windowStart = (
            state.posIndex - hArray.length + 1 + this.signalCapacity
        ) % this.signalCapacity;
        state.overlapAdd(hArray, windowStart);

        return hArray[hArray.length - 1];
    }

    /**
     * Linear interpolation of RGB channels to a target sample rate.
     *
     * Camera frames arrive at uneven intervals. POS and FFT assume
     * evenly-spaced samples, so we resample to exact intervals at
     * the configured target FPS.
     *
     * Writes into pre-allocated interpR/G/B/Times arrays.
     *
     * @returns Number of interpolated samples produced
     */
    private interpolateRGB(
        r: Float32Array, g: Float32Array, b: Float32Array,
        times: Float64Array,
        count: number,
        targetFps: number
    ): number {
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
