// Pulse signal graph — renders raw + filtered traces with peak markers

class RingBuffer {
    private buf: Float64Array | Int32Array;
    private head = 0;
    count = 0;

    constructor(size: number, int = false) {
        this.buf = int ? new Int32Array(size) : new Float64Array(size);
    }

    push(value: number): void {
        this.buf[this.head] = value;
        this.head = (this.head + 1) % this.buf.length;
        if (this.count < this.buf.length) this.count++;
    }

    // i=0 is oldest, i=count-1 is newest
    get(i: number): number {
        const start = (this.head - this.count + this.buf.length) % this.buf.length;
        return this.buf[(start + i) % this.buf.length];
    }

    get capacity(): number {
        return this.buf.length;
    }
}

export class PulseGraph {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private maxPoints: number;

    private raw: RingBuffer;
    private filtered: RingBuffer;
    private peaks: RingBuffer;
    private frameCounter = 0;

    constructor(canvas: HTMLCanvasElement, maxPoints = 300) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d')!;
        this.maxPoints = maxPoints;

        this.raw = new RingBuffer(maxPoints);
        this.filtered = new RingBuffer(maxPoints);
        this.peaks = new RingBuffer(64, true);
    }

    update(signal: number, filteredSignal: number | null, peakDetected: boolean): void {
        this.frameCounter++;
        if (peakDetected) this.peaks.push(this.frameCounter);
        this.raw.push(signal);
        this.filtered.push(filteredSignal ?? 0);
        this.draw();
    }

    private draw(): void {
        if (this.raw.count < 2) return;

        const w = this.canvas.width;
        const h = this.canvas.height;
        this.ctx.clearRect(0, 0, w, h);

        // Compute min/max across both buffers
        let min = Infinity, max = -Infinity;
        for (let i = 0; i < this.raw.count; i++) {
            const v = this.raw.get(i);
            if (v < min) min = v;
            if (v > max) max = v;
        }
        for (let i = 0; i < this.filtered.count; i++) {
            const v = this.filtered.get(i);
            if (v < min) min = v;
            if (v > max) max = v;
        }
        let range = max - min || 1;
        min -= range * 0.1;
        max += range * 0.1;
        range = max - min;

        this.drawTrace(this.raw, 'green', w, h, min, range);
        if (this.filtered.count >= 2) {
            this.drawTrace(this.filtered, 'red', w, h, min, range);
        }
        this.drawPeaks(w, h, min, range);
    }

    private drawTrace(
        buf: RingBuffer, color: string,
        w: number, h: number, min: number, range: number
    ): void {
        const ctx = this.ctx;
        const offset = this.maxPoints - buf.count;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < buf.count; i++) {
            const x = ((i + offset) / (this.maxPoints - 1)) * w;
            const y = h - ((buf.get(i) - min) / range) * h;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    private drawPeaks(w: number, h: number, min: number, range: number): void {
        if (this.filtered.count === 0) return;

        const ctx = this.ctx;
        const offset = this.maxPoints - this.filtered.count;
        ctx.fillStyle = '#ffff00';

        for (let i = 0; i < this.peaks.count; i++) {
            const age = this.frameCounter - this.peaks.get(i);
            if (age >= this.maxPoints) continue;

            const idx = this.filtered.count - 1 - age;
            if (idx < 0 || idx >= this.filtered.count) continue;

            const x = ((idx + offset) / (this.maxPoints - 1)) * w;
            const y = h - ((this.filtered.get(idx) - min) / range) * h;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fill();
        }
    }
}
