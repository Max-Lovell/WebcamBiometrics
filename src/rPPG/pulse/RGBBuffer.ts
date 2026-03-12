/**
 * RGBBuffer — Per-region RGB collection with uniform resampling.
 * // TODO: maybe separate out the interpolator from here.
 */

import { FloatRingBuffer } from '../FloatRingBuffer.ts';
import type { RGB } from '../types.ts';

// ─── Types ──────────────────────────────────────────────────────────────────
export interface RGBBufferConfig {
    sampleRate: number; // Target sample rate in Hz (defines grid interval when interpolating)
    windowSize: number; // RGB window size in samples (= method's windowSize)
    interpolate: boolean; // Whether to interpolate onto a uniform grid
    maxConsecutiveMisses: number; // Max consecutive missing frames before excluding a region. Default: 3 // TODO: make optional?
}

// Chronologically-ordered RGB window for one region, ready for a method
export interface RGBWindow {
    r: Float32Array;
    g: Float32Array;
    b: Float32Array;
}

// A single grid tick emitted by pushFrame()
export interface BufferTick {
    time: number; // Grid-aligned timestamp (ms). When not interpolating, equals the real frame time.
    regionWindows: Record<string, RGBWindow | null>; // RGB windows per region. null if region isn't ready or is excluded.
}

// ─── Interpolation State ────────────────────────────────────────────────────
// Per-region 2-frame sliding window for linear interpolation - stores 2 most recent RGB values and timestamps.
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
// if a region misses too many consecutive frames, it's excluded from fusion
class RegionState {
    // Raw RGB ring buffers that the projection method operates on each frame
    readonly rBuffer: FloatRingBuffer;
    readonly gBuffer: FloatRingBuffer;
    readonly bBuffer: FloatRingBuffer;

    // Pre-allocated work arrays for putting ring buffers in chronological order
    readonly unrollR: Float32Array;
    readonly unrollG: Float32Array;
    readonly unrollB: Float32Array;

    // Hold-last-value for missing frames
    private lastR: number = 0;
    private lastG: number = 0;
    private lastB: number = 0;
    private hasReceived: boolean = false; // Whether we've ever received a valid RGB sample

    // Dropout tracking — consecutive real frames with no RGB data from camera
    private _consecutiveMisses: number = 0; // Tracks dropped frames to hold last value or exclude

    // Per-region interpolation state (only used when interpolation is enabled)
    readonly interpolation: InterpolationState; // 2-frame sliding window of real camera frame RGBs and times

    constructor(windowSize: number) {
        this.rBuffer = new FloatRingBuffer(windowSize);
        this.gBuffer = new FloatRingBuffer(windowSize);
        this.bBuffer = new FloatRingBuffer(windowSize);

        this.unrollR = new Float32Array(windowSize);
        this.unrollG = new Float32Array(windowSize);
        this.unrollB = new Float32Array(windowSize);

        this.interpolation = new InterpolationState();
    }

    // Push RGB into ring buffers
    // Note when interpolating we don't call pushHoldRGB to avoid resetting miss counter
    // as we could interpolate multiple points per actual frame.
    pushRGB(r: number, g: number, b: number): void {
        this.rBuffer.push(r);
        this.gBuffer.push(g);
        this.bBuffer.push(b);
    }

    // Push RGB into ring buffers and reset miss counter — used by the direct (non-interpolated) path.
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
        if (!this.hasReceived) return false; // nothing to hold
        this.pushRGB(this.lastR, this.lastG, this.lastB);
        this._consecutiveMisses++;
        return true;
    }

    // Record a missed frame for interpolation — push last held frame into buffer
    holdForInterpolation(time: number): boolean {
        if (!this.hasReceived) return false; // Never received data.
        this.interpolation.push(this.lastR, this.lastG, this.lastB, time);
        this._consecutiveMisses++;
        return true;
    }

    // Record current actual frame for later interpolation
    // Push new RGB into interpolation buffer (not ring buffers — that happens per grid point in pushRGB())
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

    get isReady(): boolean {
        return this.rBuffer.isFull;
    }

    // Unroll ring buffers into chronological order and return as RGBWindow
    unrollToWindow(): RGBWindow {
        this.rBuffer.copyOrdered(this.unrollR);
        this.gBuffer.copyOrdered(this.unrollG);
        this.bBuffer.copyOrdered(this.unrollB);
        return {
            r: this.unrollR,
            g: this.unrollG,
            b: this.unrollB,
        };
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

// ─── RGBBuffer Class ────────────────────────────────────────────────────────
export class RGBBuffer {
    private readonly config: RGBBufferConfig; // makes all fields required
    private readonly regions: Record<string, RegionState> = {};

    // Grid state (interpolation)
    private gridEpoch: number = -1; // Stores the start of the grid. Grid ticks at epoch + n * gridIntervalMs.
    private readonly gridIntervalMs: number; // Grid interval in ms (derived from sampleRate)
    private lastFrameTime: number = -1; // Timestamp of the previous real camera frame

    constructor(regionNames: string[], config: RGBBufferConfig) {
        this.config = config;

        for (const name of regionNames) {
            this.regions[name] = new RegionState(config.windowSize);
        }

        this.gridIntervalMs = 1000 / config.sampleRate; // Grid interval: e.g. at 20fps = 50ms between grid points
    }

    // ─── Public API ─────────────────────────────────────────────────────
    // Turn actual frame to array of interpolated value(s) ready for pulse processing. Note none returned on first frame.
    pushFrame(
        rgbPerRegion: Record<string, { rgb: RGB | null }>,
        time: number
    ): BufferTick[] {
        if (this.config.interpolate) {
            return this.pushFrameInterpolated(rgbPerRegion, time);
        } else {
            return this.pushFrameDirect(rgbPerRegion, time);
        }
    }

    // Region names buffer was configured with
    get regionNames(): string[] {
        return Object.keys(this.regions);
    }

    // Is interpolation is enabled
    get interpolating(): boolean {
        return this.config.interpolate;
    }

    // Reset all state (call when face tracking lost)
    reset(): void {
        for (const state of Object.values(this.regions)) {
            state.reset();
        }
        this.gridEpoch = -1;
        this.lastFrameTime = -1;
    }

    // ─── Direct (non-interpolated) Path ─────────────────────────────────
    // From one real frame, produce one tick with per-region RGB windows.
    private pushFrameDirect(
        rgbPerRegion: Record<string, { rgb: RGB | null }>,
        time: number
    ): BufferTick[] {
        const regionWindows: Record<string, RGBWindow | null> = {};
        const maxMisses = this.config.maxConsecutiveMisses;

        for (const [name, state] of Object.entries(this.regions)) {
            const rgb = rgbPerRegion[name]?.rgb ?? null;

            if (rgb) {
                state.pushHoldRGB(rgb.r, rgb.g, rgb.b);
            } else if (state.consecutiveMisses < maxMisses) {
                if (!state.pushHeld()) {
                    regionWindows[name] = null;
                    continue; // Never received data — nothing to hold
                }
            } else {
                regionWindows[name] = null;
                continue; // Sustained dropout — exclude from fusion
            }

            regionWindows[name] = state.isReady ? state.unrollToWindow() : null;
        }

        return [{ time, regionWindows }];
    }

    // ─── Interpolated Path ──────────────────────────────────────────────
    // Feed real-frame RGB into interpolation buffers, then compute grid points
    // between previous and current frame, lerping RGB at each grid tick.
    private pushFrameInterpolated(
        rgbPerRegion: Record<string, { rgb: RGB | null }>,
        time: number
    ): BufferTick[] {
        const maxMisses = this.config.maxConsecutiveMisses;
        const maxGapMs = this.config.maxConsecutiveMisses * this.gridIntervalMs;
        const excludedRegions = new Set<string>();

        // Feed real-frame RGB into each region's InterpolationState
        // If region has no RGB, hold the last value into the interpolation buffer.
        for (const [name, state] of Object.entries(this.regions)) {
            const rgb = rgbPerRegion[name]?.rgb ?? null;
            if (rgb) {
                state.receiveForInterpolation(rgb.r, rgb.g, rgb.b, time);
            } else if (state.consecutiveMisses < maxMisses) {
                if (!state.holdForInterpolation(time)) {
                    excludedRegions.add(name); // Never received data
                }
            } else {
                excludedRegions.add(name); // Sustained dropout
            }
        }

        // Establish grid epoch on first frame and return no output — need two frames to interpolate
        if (this.gridEpoch < 0) {
            this.gridEpoch = time;
            this.lastFrameTime = time;
            return [];
        }

        // Gap too large — re-anchor the grid instead of backfilling
        if (time - this.lastFrameTime > maxGapMs) {
            for (const state of Object.values(this.regions)) {
                state.interpolation.reset();
            }
            this.gridEpoch = time;
            this.lastFrameTime = time;
            return [];
        }

        // On subsequent frames, compute grid points between this and previous frame
        // then lerp RGB per region, push to ring buffers, and emit ticks.

        // Calc grid-aligned samples between last and current frame (epoch + n * interval)
        // floor+1 gives first grid index strictly after lastFrameTime. Max with 1 skips epoch (no interp data).
        const startIdx = Math.max(
            Math.floor((this.lastFrameTime - this.gridEpoch) / this.gridIntervalMs) + 1,
            1
        );
        const endIdx = Math.floor((time - this.gridEpoch) / this.gridIntervalMs);

        const ticks: BufferTick[] = [];

        for (let i = startIdx; i <= endIdx; i++) {
            const gridTime = this.gridEpoch + i * this.gridIntervalMs;
            const regionWindows: Record<string, RGBWindow | null> = {};

            for (const [name, state] of Object.entries(this.regions)) {
                if (excludedRegions.has(name) || !state.interpolation.hasTwoFrames) {
                    regionWindows[name] = null;
                    continue;
                }

                // Lerp RGB at this grid point and push into ring buffer
                const rgb = state.interpolation.lerp(gridTime);
                state.pushRGB(rgb.r, rgb.g, rgb.b);

                // Emit window if buffer is full
                regionWindows[name] = state.isReady ? state.unrollToWindow() : null;
            }

            ticks.push({ time: gridTime, regionWindows });
        }

        this.lastFrameTime = time;
        return ticks;
    }
}
