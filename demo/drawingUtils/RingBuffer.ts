/**
 * Lightweight typed ring buffer for demo drawing utilities.
 * Pass in the backing TypedArray at construction to control type (Float64Array, Int32Array, etc).
 */
export class RingBuffer<T extends Float64Array | Int32Array> {
    private buf: T;
    private head = 0;
    count = 0;

    constructor(buf: T) {
        this.buf = buf;
    }

    push(value: number): void {
        this.buf[this.head] = value;
        this.head = (this.head + 1) % this.buf.length;
        if (this.count < this.buf.length) this.count++;
    }

    /** i=0 is oldest, i=count-1 is newest */
    get(i: number): number {
        const start = (this.head - this.count + this.buf.length) % this.buf.length;
        return this.buf[(start + i) % this.buf.length];
    }

    /** Most recently pushed value */
    last(): number {
        const idx = (this.head - 1 + this.buf.length) % this.buf.length;
        return this.buf[idx];
    }

    get capacity(): number {
        return this.buf.length;
    }
}
