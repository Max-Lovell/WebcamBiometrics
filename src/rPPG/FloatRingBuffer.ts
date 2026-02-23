/**
 * FloatRingBuffer — Circular buffer for floating-point signal data.
 * Wraps around when reaches the end - super quick with no memory allocation after construction
 */
export class FloatRingBuffer {
    private readonly buffer: Float32Array; // never changes size
    private writeIndex: number = 0; // Next position to write to

    // Total number of values ever pushed
    private _count: number = 0;

    constructor(capacity: number) {
        if (capacity < 1) {
            throw new Error(`FloatRingBuffer capacity must be >= 1, got ${capacity}`);
        }
        this.buffer = new Float32Array(capacity);
    }

    // ─── Properties ─────────────────────────────────────────────────────

    // Maximum number of samples buffer can hold
    get capacity(): number {
        return this.buffer.length;
    }

    // Number of valid samples currently in the buffer
    get count(): number {
        return Math.min(this._count, this.buffer.length);
    }

    // has wrapped around at least once
    get isFull(): boolean {
        return this._count >= this.buffer.length;
    }

    // ─── Writing ────────────────────────────────────────────────────────
    push(value: number): void {
        this.buffer[this.writeIndex] = value;
        this.writeIndex = (this.writeIndex + 1) % this.buffer.length;
        this._count++;
    }

    pushBatch(values: Float32Array | number[]): void {
        for (let i = 0; i < values.length; i++) {
            this.push(values[i]);
        }
    }


    // Copy ring buffer into chronological order to provided array.
    copyOrdered(output: Float32Array): number {
        const len = this.count;

        if (this.isFull) {
            // Buffer has wrapped — writeIndex points to the oldest sample.
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

    // Read value by index
    peek(i: number): number {
        const start = this.isFull ? this.writeIndex : 0;
        return this.buffer[(start + i) % this.buffer.length];
    }

    peekLast(): number {
        if (this._count === 0) return 0;
        // writeIndex points to the NEXT write position, so the last
        // written value is one step back (with wrapping).
        const lastIdx = (this.writeIndex - 1 + this.buffer.length) % this.buffer.length;
        return this.buffer[lastIdx];
    }

    // RESET - for when facetracking lost
    reset(): void {
        this.buffer.fill(0);
        this.writeIndex = 0;
        this._count = 0;
    }
}

// Same as FloatRingBuffer but backed by Float64Array.
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
