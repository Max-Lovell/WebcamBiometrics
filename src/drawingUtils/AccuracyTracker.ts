export interface AccuracyStats {
    dist: number;
    avg: number;
    stddev: number;
    min: number;
    max: number;
    count: number;
}

export class AccuracyTracker {
    private buf: Float64Array;
    private head = 0;
    private count = 0;
    private sum = 0;
    private target: [number, number] = [0, 0];
    active = false;

    constructor(private bufSize = 300) {
        this.buf = new Float64Array(bufSize);
    }

    start(target: [number, number] = [0, 0]): void {
        this.target = target;
        this.head = 0;
        this.count = 0;
        this.sum = 0;
        this.buf.fill(0);
        this.active = true;
    }

    stop(): AccuracyStats | null {
        this.active = false;
        if (this.count === 0) return null;
        return this.computeStats(this.lastDist());
    }

    sample(normPog: [number, number]): AccuracyStats | null {
        if (!this.active) return null;

        const dx = normPog[0] - this.target[0];
        const dy = normPog[1] - this.target[1];
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (this.count === this.bufSize) this.sum -= this.buf[this.head];
        this.buf[this.head] = dist;
        this.sum += dist;
        this.head = (this.head + 1) % this.bufSize;
        if (this.count < this.bufSize) this.count++;

        return this.computeStats(dist);
    }

    private lastDist(): number {
        const idx = (this.head - 1 + this.bufSize) % this.bufSize;
        return this.buf[idx];
    }

    private computeStats(dist: number): AccuracyStats {
        let min = Infinity, max = -Infinity;
        for (let i = 0; i < this.count; i++) {
            if (this.buf[i] < min) min = this.buf[i];
            if (this.buf[i] > max) max = this.buf[i];
        }
        const avg = this.sum / this.count;
        let varSum = 0;
        for (let i = 0; i < this.count; i++) varSum += (this.buf[i] - avg) ** 2;
        const stddev = Math.sqrt(varSum / this.count);

        return { dist, avg, stddev, min, max, count: this.count };
    }

    // Optional helper — creates a fixed-position overlay and returns an update function
    static createOverlay(): (stats: AccuracyStats) => void {
        const el = document.createElement('div');
        el.style.cssText = 'position:fixed;top:10px;left:10px;color:lime;font-family:monospace;font-size:14px;z-index:9999;background:rgba(0,0,0,0.7);padding:8px;';
        document.body.appendChild(el);
        return (s) => {
            el.innerText = `Dist: ${s.dist.toFixed(4)} | Avg: ${s.avg.toFixed(4)} | Jitter: ${s.stddev.toFixed(4)} | Range: ${s.min.toFixed(4)}-${s.max.toFixed(4)} | n=${s.count}`;
        };
    }
}
