/**
 * FloatRingBuffer — Circular buffer for floating-point signal data.
 *
 * A ring buffer (also called circular buffer) is a fixed-size array that
 * wraps around when it reaches the end. Instead of shifting all elements
 * left when we add a new value (O(n) per insert), we just overwrite the
 * oldest value and advance a pointer (O(1) per insert).
 *
 * Visual example with capacity 5:
 *
 *   After pushing values 1, 2, 3:
 *   ┌───┬───┬───┬───┬───┐
 *   │ 1 │ 2 │ 3 │ 0 │ 0 │   writeIndex = 3, count = 3, isFull = false
 *   └───┴───┴───┴───┴───┘
 *     ▲           ▲
 *     oldest      writeIndex (next write goes here)
 *
 *   After pushing 4, 5, 6 (wraps around, overwrites 1):
 *   ┌───┬───┬───┬───┬───┐
 *   │ 6 │ 2 │ 3 │ 4 │ 5 │   writeIndex = 1, count = 6, isFull = true
 *   └───┴───┴───┴───┴───┘
 *         ▲
 *         writeIndex AND oldest (they're the same once full)
 *
 *   Chronological order: [2, 3, 4, 5, 6]  (starting from writeIndex, wrapping)
 *
 * Why use this?
 *   In rPPG we continuously accumulate signal samples (RGB values, POS output,
 *   filtered pulse values) but only care about the most recent N seconds.
 *   A ring buffer gives us O(1) insertion with no memory allocation after
 *   construction, which matters when we're processing 30 samples/second
 *   in a real-time loop.
 *
 * Used by:
 *   - PulseProcessor (RGB timestamps, POS overlap-add signal, fused signal)
 *   - PeakEstimator (accumulates filtered samples for peak counting)
 */
export class FloatRingBuffer {
    /**
     * The underlying fixed-size array. Once allocated, this never changes size.
     * Values are overwritten in-place as the buffer wraps around.
     */
    private readonly buffer: Float32Array;

    /**
     * Points to the NEXT position we'll write to.
     * After a write, this advances by 1 (modulo capacity).
     * When the buffer is full, writeIndex also points to the oldest value
     * (which is about to be overwritten).
     */
    private writeIndex: number = 0;

    /**
     * Total number of values ever pushed. This can exceed capacity —
     * we use it to distinguish "buffer not yet full" from "buffer full and wrapping".
     * The number of valid samples is min(count, capacity).
     */
    private _count: number = 0;

    /**
     * @param capacity - Maximum number of samples to store.
     *   Choose based on your analysis window:
     *   - For PeakEstimator at 30fps with 10s window: capacity = 300
     *   - For PulseProcessor at 30fps with 15s window: capacity = 450
     *   - For RGB samples at 30fps with 1.6s POS window: capacity = 48
     */
    constructor(capacity: number) {
        if (capacity < 1) {
            throw new Error(`FloatRingBuffer capacity must be >= 1, got ${capacity}`);
        }
        this.buffer = new Float32Array(capacity);
    }

    // ─── Properties ─────────────────────────────────────────────────────

    /** Maximum number of samples this buffer can hold */
    get capacity(): number {
        return this.buffer.length;
    }

    /**
     * Number of valid samples currently in the buffer.
     * Increases with each push() until it hits capacity, then stays there.
     */
    get count(): number {
        return Math.min(this._count, this.buffer.length);
    }

    /** Whether the buffer has wrapped around at least once */
    get isFull(): boolean {
        return this._count >= this.buffer.length;
    }

    // ─── Writing ────────────────────────────────────────────────────────

    /**
     * Add a single value to the buffer.
     *
     * O(1) — no allocation, no shifting. Just one array write and a
     * pointer increment.
     *
     * If the buffer is full, this overwrites the oldest value.
     *
     * @param value - The sample to store (e.g., a filtered pulse value)
     */
    push(value: number): void {
        this.buffer[this.writeIndex] = value;
        this.writeIndex = (this.writeIndex + 1) % this.buffer.length;
        this._count++;
    }

    /**
     * Add multiple values at once.
     *
     * Equivalent to calling push() in a loop, but slightly clearer
     * at call sites when you have a batch of samples.
     *
     * @param values - Array of samples to push (oldest first)
     */
    pushBatch(values: Float32Array | number[]): void {
        for (let i = 0; i < values.length; i++) {
            this.push(values[i]);
        }
    }

    // ─── Reading ────────────────────────────────────────────────────────

    /**
     * Copy the buffer contents into `output` in chronological order
     * (oldest sample first, newest sample last).
     *
     * This is the main way to read the buffer for analysis. You provide
     * a pre-allocated output array (to avoid per-call allocation), and
     * this method fills it with the samples in the correct order.
     *
     * Why copy rather than expose the internal array?
     *   The internal array isn't in chronological order once the buffer
     *   wraps — it has the newest samples at the start and oldest in the
     *   middle. Exposing it directly would force every consumer to know
     *   about the write index and do their own unwrapping.
     *
     * @param output - Pre-allocated array to copy into. Must be at least
     *   `count` elements long. Extra elements are left untouched.
     * @returns Number of valid samples copied (= this.count)
     *
     * @example
     *   const work = new Float32Array(buffer.capacity);
     *   const n = buffer.copyOrdered(work);
     *   const signal = work.subarray(0, n); // chronological signal
     */
    copyOrdered(output: Float32Array): number {
        const len = this.count;

        if (this.isFull) {
            // Buffer has wrapped — writeIndex points to the oldest sample.
            // We need to copy from writeIndex to end, then from 0 to writeIndex.
            //   buffer: [6, 7, 3, 4, 5]    writeIndex = 2
            //                ▲
            //   output: [3, 4, 5, 6, 7]    (chronological)
            // First chunk: writeIndex → end of array
            const firstChunkSize = this.buffer.length - this.writeIndex;
            output.set(
                this.buffer.subarray(this.writeIndex, this.writeIndex + firstChunkSize),
                0
            );
            // Second chunk: start of array → writeIndex
            output.set(
                this.buffer.subarray(0, this.writeIndex),
                firstChunkSize
            );
        } else {
            // Buffer hasn't wrapped yet — data starts at index 0
            output.set(this.buffer.subarray(0, len));
        }

        return len;
    }

    /**
     * Read a single value by logical index (0 = oldest, count-1 = newest).
     *
     * Useful for spot checks or when you need just one value without
     * copying the entire buffer. No bounds checking for performance —
     * caller is responsible for checking 0 <= i < count.
     *
     * @param i - Logical index (0 = oldest available sample)
     * @returns The sample value at that position
     */
    peek(i: number): number {
        const start = this.isFull ? this.writeIndex : 0;
        return this.buffer[(start + i) % this.buffer.length];
    }

    /**
     * Get the most recently pushed value.
     * Returns 0 if the buffer is empty (no push has been called).
     */
    peekLast(): number {
        if (this._count === 0) return 0;
        // writeIndex points to the NEXT write position, so the last
        // written value is one step back (with wrapping).
        const lastIdx = (this.writeIndex - 1 + this.buffer.length) % this.buffer.length;
        return this.buffer[lastIdx];
    }

    // ─── Lifecycle ──────────────────────────────────────────────────────

    /**
     * Clear the buffer and reset to empty state.
     *
     * Call this when:
     *   - Face tracking is lost (the signal is discontinuous)
     *   - Scene changes (lighting conditions invalidate old data)
     *   - User requests a reset
     *
     * The underlying Float32Array is zeroed but NOT reallocated —
     * no garbage collection pressure.
     */
    reset(): void {
        this.buffer.fill(0);
        this.writeIndex = 0;
        this._count = 0;
    }
}


// ─── Float64 variant ────────────────────────────────────────────────────────

/**
 * Same as FloatRingBuffer but backed by Float64Array.
 *
 * Used specifically for timestamps (DOMHighResTimeStamp is a double).
 * At 20+ FPS, Float32 timestamps become indistinguishable because
 * 32-bit floats only have ~7 decimal digits of precision, and
 * performance.now() returns values like 123456.789 where the integer
 * part alone consumes most of those digits.
 */
export class Float64RingBuffer {
    private readonly buffer: Float64Array;
    private writeIndex: number = 0;
    private _count: number = 0;

    constructor(capacity: number) {
        if (capacity < 1) {
            throw new Error(`Float64RingBuffer capacity must be >= 1, got ${capacity}`);
        }
        this.buffer = new Float64Array(capacity);
    }

    get capacity(): number { return this.buffer.length; }
    get count(): number { return Math.min(this._count, this.buffer.length); }
    get isFull(): boolean { return this._count >= this.buffer.length; }

    push(value: number): void {
        this.buffer[this.writeIndex] = value;
        this.writeIndex = (this.writeIndex + 1) % this.buffer.length;
        this._count++;
    }

    copyOrdered(output: Float64Array): number {
        const len = this.count;
        if (this.isFull) {
            const firstChunkSize = this.buffer.length - this.writeIndex;
            output.set(this.buffer.subarray(this.writeIndex, this.writeIndex + firstChunkSize), 0);
            output.set(this.buffer.subarray(0, this.writeIndex), firstChunkSize);
        } else {
            output.set(this.buffer.subarray(0, len));
        }
        return len;
    }

    peek(i: number): number {
        const start = this.isFull ? this.writeIndex : 0;
        return this.buffer[(start + i) % this.buffer.length];
    }

    peekLast(): number {
        if (this._count === 0) return 0;
        const lastIdx = (this.writeIndex - 1 + this.buffer.length) % this.buffer.length;
        return this.buffer[lastIdx];
    }

    reset(): void {
        this.buffer.fill(0);
        this.writeIndex = 0;
        this._count = 0;
    }
}
