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
 *   - Optional linear interpolation onto a uniform grid (for bandpass/FFT correctness)
 */

import {Float64RingBuffer, FloatRingBuffer} from '../FloatRingBuffer.ts';
import type { WindowedPulseMethod } from './projection/types.ts';
import {POS} from './projection/POS.ts';
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
    // Whether to interpolate RGB onto a uniform grid before POS processing.
    interpolate: boolean; // Toggle interpolation for bandpass filter and FFT.
}

export const DEFAULT_PULSE_CONFIG: PulseProcessorConfig = {
    ...DEFAULT_PIPELINE_CONFIG,
    posWindowMultiplier: 1.6, // From the POS paper: l = fps × 1.6 ≈ 32 frames at 20fps.
    signalWindowSeconds: 15,
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
    // Whether the signal buffer has enough data for BPM estimation
    signalReady: boolean;
}

// ─── Interpolation State ────────────────────────────────────────────────────
// Per-region 2-frame buffer for linear interpolation onto a uniform time grid.
// Stores the two most recent real-frame RGB values and their timestamps.
// On each real frame, the caller can lerp RGB for grid points in (prevTime, curTime).
class InterpolationState {
    // Previous and current real-frame samples
    private prevR: number = 0;
    private prevG: number = 0;
    private prevB: number = 0;
    private prevTime: number = 0; // Note timestamps are always aligned between regions and so could be managed globally

    private curR: number = 0;
    private curG: number = 0;
    private curB: number = 0;
    private curTime: number = 0;

    // Count number of interpolated frames
    private _frameCount: number = 0; // Mostly used to make sure we have at least 2 frames before trying to interpolate

    // Push a new real-frame RGB sample. Returns true if we now have ≥2 frames (can interpolate).
    push(r: number, g: number, b: number, time: number): boolean {
        this.prevR = this.curR;
        this.prevG = this.curG;
        this.prevB = this.curB;
        this.prevTime = this.curTime;

        this.curR = r;
        this.curG = g;
        this.curB = b;
        this.curTime = time;

        this._frameCount++;
        return this._frameCount >= 2;
    }

    // Linearly interpolate RGB at a given time between prevTime and curTime.
    // Caller must ensure prevTime < time <= curTime.
    lerp(time: number): RGB {
        const span = this.curTime - this.prevTime;
        // Guard against zero span (duplicate timestamps)
        const t = span > 0 ? (time - this.prevTime) / span : 1;
        return {
            r: this.prevR + t * (this.curR - this.prevR),
            g: this.prevG + t * (this.curG - this.prevG),
            b: this.prevB + t * (this.curB - this.prevB),
        };
    }

    get hasTwoFrames(): boolean {
        return this._frameCount >= 2;
    }

    reset(): void {
        this.prevR = 0; this.prevG = 0; this.prevB = 0; this.prevTime = 0;
        this.curR = 0; this.curG = 0; this.curB = 0; this.curTime = 0;
        this._frameCount = 0;
    }
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

    // Dropout tracking — consecutive real frames with no RGB data from camera
    private _consecutiveMisses: number = 0; // If interpolating, tracks dropped frames in region to hold last value or exclude

    // Per-region interpolation state (only used when interpolation is enabled)
    readonly interpolation: InterpolationState; // 2-frame sliding window of real camera frame RGBs and times for interpolation

    constructor(rgbCapacity: number) {
        this.rBuffer = new FloatRingBuffer(rgbCapacity);
        this.gBuffer = new FloatRingBuffer(rgbCapacity);
        this.bBuffer = new FloatRingBuffer(rgbCapacity);

        this.unrollR = new Float32Array(rgbCapacity);
        this.unrollG = new Float32Array(rgbCapacity);
        this.unrollB = new Float32Array(rgbCapacity);
        this.unrollTimes = new Float64Array(rgbCapacity);

        this.interpolation = new InterpolationState();
    }

    // Push RGB into ring buffers
    // Note when interpolating we don't call PushHoldRGB to avoid resetting miss counter as could interpolate multiple points per actual frame.
    pushRGB(r: number, g: number, b: number): void {
        this.rBuffer.push(r);
        this.gBuffer.push(g);
        this.bBuffer.push(b);
    }

    // Push RGB into ring buffers and reset miss counter - Used by the direct (non-interpolated) path.
    pushHoldRGB(r: number, g: number, b: number): void {
        this.pushRGB(r, g, b);
        this.lastR = r;
        this.lastG = g;
        this.lastB = b;
        this.hasReceived = true;
        this._consecutiveMisses = 0;
    }

    // Push last known RGB value to keep buffers aligned with timestamps if not interpolating
    pushHeld(): boolean {
        if (!this.hasReceived) return false; // return false if no valid sample as nothing to hold
        this.pushRGB(this.lastR, this.lastG, this.lastB);
        this._consecutiveMisses++;
        return true;
    }

    // Record a missed frame for interpolation - push last held frame into buffer
    holdForInterpolation(time: number): boolean {
        if (!this.hasReceived) return false; // Return false if never received data.
        this.interpolation.push(this.lastR, this.lastG, this.lastB, time);
        this._consecutiveMisses++;
        return true;
    }

    // Record current actual frame for later interpolation
    // Push new RGB into interpolation buffer (not ring buffers which is per grid point in pushRGB())
    receiveForInterpolation(r: number, g: number, b: number, time: number): void {
        this.interpolation.push(r, g, b, time);
        // Update last-known values for hold-last-value
        this.lastR = r;
        this.lastG = g;
        this.lastB = b;
        this.hasReceived = true;
        this._consecutiveMisses = 0;
    }

    get consecutiveMisses(): number {
        return this._consecutiveMisses;
    }

    get canHold(): boolean {
        return this.hasReceived;
    }

    get rgbReady(): boolean {
        return this.rBuffer.isFull;
    }

    // Unroll RGB ring buffers into chronological order using pre-allocated work arrays
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
        this.interpolation.reset();
    }
}

// ─── PulseProcessor Class ───────────────────────────────────────────────────
export class PulseProcessor {
    private readonly config: PulseProcessorConfig;
    private readonly method: WindowedPulseMethod;

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

    // ─── Interpolation grid state ───────────────────────────────────────
    // Global grid epoch (ms) — set from the first real frame's timestamp.
    private gridEpoch: number = -1; // Stores the start of the grid. Grid ticks at epoch + n * gridIntervalMs.
    private readonly gridIntervalMs: number; // Grid interval in ms (derived from sampleRate)
    // Timestamp of the previous real camera frame (for computing grid range)
    private lastFrameTime: number = -1; // Note slightly redundant - could just use region InterpolationStates for this?

    // Note Partial<> makes every property optional.
    constructor(regionNames: string[] = ['region'], config?: Partial<PulseProcessorConfig>, method?: WindowedPulseMethod) {
        this.config = { ...DEFAULT_PULSE_CONFIG, ...config }; // Config with sensible defaults if not used.

        const fps = this.config.sampleRate;
        // Use provided method or default to POS
        this.method = method ?? new POS(fps, this.config.posWindowMultiplier);
        this.rgbCapacity = this.method.windowSize;
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

        // Grid interval: e.g. at 20fps → 50ms between grid points
        this.gridIntervalMs = 1000 / fps;
    }

    // ─── Public API ────────────────────────────────────────────────────
    // Process a single camera frame through the full pulse pipeline. Call once per real frame.
    pushFrame(rgbPerRegion: Record<string, { rgb: RGB | null }>, time: number): PulseFrame {
        if (this.config.interpolate) {
            // When interpolation is enabled, may produce 0 or more fused samples per call
            // (0 on first frame, typically 1, occasionally 2 if camera was late).
            return this.pushFrameInterpolated(rgbPerRegion, time);
        } else {
            return this.pushFrameDirect(rgbPerRegion, time);
        }
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

    // Whether interpolation is enabled
    get interpolating(): boolean {
        return this.config.interpolate;
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
        this.gridEpoch = -1;
        this.lastFrameTime = -1;
    }

    // ─── Direct (non-interpolated) path ─────────────────────────────────
    // From one real frame to one POS sample
    private pushFrameDirect(rgbPerRegion: Record<string, { rgb: RGB | null }>, time: number): PulseFrame {
        this.timeBuffer.push(time);
        // Per-region: ingest RGB, run POS, collect H arrays
        const { regionPulses, hArrays } = this.rgb2pos(rgbPerRegion);

        // Combine regions and overlap add
        const fusedSamples: FusedSample[] = [];
        if (hArrays.length > 0) {
            const fusedValue = this.fuseAndOverlapAdd(hArrays);
            this.advanceFusedIndex(); // advance write head
            fusedSamples.push({ value: fusedValue, time });
        }

        return {
            fusedSamples,
            regionPulses,
            signalReady: this.isSignalReady(),
        };
    }

    // ─── Interpolated path ──────────────────────────────────────────────
    // Create per-region interpolation buffers and return grid-aligned samples.
    private pushFrameInterpolated(
        rgbPerRegion: Record<string, { rgb: RGB | null }>,
        time: number
    ): PulseFrame {
        const fusedSamples: FusedSample[] = [];
        let regionPulses: Record<string, number | null> = {};
        // For each region, get RGB (or hold-last-value) and push to region's InterpolationState (2-frame buffer).
        // Feed real-frame RGB into each region's InterpolationState
        const excludedRegions = new Set<string>();
        // If region has no RGB, hold the last value into the interpolation buffer.
        for (const [name, state] of Object.entries(this.regions)) {
            const rgb = rgbPerRegion[name]?.rgb ?? null;
            if (rgb) {
                state.receiveForInterpolation(rgb.r, rgb.g, rgb.b, time);
            } else if (state.consecutiveMisses < this.config.maxConsecutiveMisses) {
                if (!state.holdForInterpolation(time)) {
                    excludedRegions.add(name); // Never received data
                }
            } else {
                excludedRegions.add(name); // Sustained dropout
            }
        }

        // Establish grid epoch on first frame and return no output, need two frames to interpolate
        if (this.gridEpoch < 0) {
            this.gridEpoch = time;
            this.lastFrameTime = time;
            for (const name of Object.keys(this.regions)) {
                regionPulses[name] = null;
            }
            return { fusedSamples: [], regionPulses, signalReady: false };
        }

        // On subsequent frames, compute grid points between this and previous frame
        // then lerp RGB per region, push to ring buffers, compute POS, fuse, overlap-add.

        // Calc grid-aligned samples between last and current frame (epoch+n*interval)
        // floor+1 gives first grid index strictly after lastFrameTime. Max with 1 skips epoch (no interp data).
        const startIdx = Math.max(Math.floor((this.lastFrameTime - this.gridEpoch) / this.gridIntervalMs) + 1, 1);
        const endIdx = Math.floor((time - this.gridEpoch) / this.gridIntervalMs);

        for (let i = startIdx; i <= endIdx; i++) {
            const gridTime = this.gridEpoch + i * this.gridIntervalMs;

            // Build interpolated RGB per region for this grid point
            const gridRGB: Record<string, { rgb: RGB | null }> = {};

            for (const [name, state] of Object.entries(this.regions)) {
                if (excludedRegions.has(name) || !state.interpolation.hasTwoFrames) {
                    gridRGB[name] = { rgb: null };
                } else {
                    gridRGB[name] = { rgb: state.interpolation.lerp(gridTime) };
                }
            }

            // Run POS on interpolated sample, rgb2posInterpolated pushes directly to buffers
            this.timeBuffer.push(gridTime);
            const result = this.rgb2posInterpolated(gridRGB);
            regionPulses = result.regionPulses;

            if (result.hArrays.length > 0) {
                const fusedValue = this.fuseAndOverlapAdd(result.hArrays);
                this.advanceFusedIndex();
                fusedSamples.push({ value: fusedValue, time: gridTime });
            }
        }

        this.lastFrameTime = time;

        // Fill in any remaining region pulses
        for (const name of Object.keys(this.regions)) {
            if (!(name in regionPulses)) regionPulses[name] = null;
        }

        return {
            fusedSamples,
            regionPulses,
            signalReady: this.isSignalReady(),
        };
    }

    // ─── Pipeline Steps ─────────────────────────────────────────────────
    // Ingest RGB for each region (direct/non-interpolated path).
    // Handles hold-last-value and dropout tracking inline.
    private rgb2pos(rgbPerRegion: Record<string, { rgb: RGB | null }>): {
        regionPulses: Record<string, number | null>;
        hArrays: Float32Array[];
    } {
        const regionPulses: Record<string, number | null> = {};
        const hArrays: Float32Array[] = [];

        for (const [name, state] of Object.entries(this.regions)) {
            const rgb = rgbPerRegion[name]?.rgb ?? null;

            if (rgb) {
                state.pushHoldRGB(rgb.r, rgb.g, rgb.b);
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

    // Process interpolated RGB for each region (note dropouts handled during interploation)
    private rgb2posInterpolated(gridRGB: Record<string, { rgb: RGB | null }>): {
        regionPulses: Record<string, number | null>;
        hArrays: Float32Array[];
    } {
        const regionPulses: Record<string, number | null> = {};
        const hArrays: Float32Array[] = [];

        for (const [name, state] of Object.entries(this.regions)) {
            const rgb = gridRGB[name]?.rgb ?? null;

            if (rgb) {
                // Push interpolated RGB into ring buffers (no miss counter reset)
                state.pushRGB(rgb.r, rgb.g, rgb.b);
            } else { // Null RGB = region excluded for this grid point.
                regionPulses[name] = null;
                continue;
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
        return this.method.process({
            r: state.unrollR,
            g: state.unrollG,
            b: state.unrollB,
        });
    }
}
