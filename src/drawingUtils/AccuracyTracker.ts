import { RingBuffer } from './RingBuffer';

export interface AccuracyStats {
    dist: number;
    avg: number;
    stddev: number;
    min: number;
    max: number;
    count: number;
}

export class AccuracyTracker {
    private ring: RingBuffer<Float64Array>;
    private sum = 0;
    active = false;
    private target: [number, number] = [0, 0];

    constructor(private bufSize = 300) {
        this.ring = new RingBuffer(new Float64Array(bufSize));
    }

    start(target: [number, number] = [0, 0]): void {
        this.target = target;
        this.ring = new RingBuffer(new Float64Array(this.bufSize));
        this.sum = 0;
        this.active = true;
    }

    stop(): AccuracyStats | null {
        this.active = false;
        if (this.ring.count === 0) return null;
        return this.computeStats(this.ring.last());
    }

    sample(normPog: [number, number]): AccuracyStats | null {
        if (!this.active) return null;

        const dx = normPog[0] - this.target[0];
        const dy = normPog[1] - this.target[1];
        const dist = Math.sqrt(dx * dx + dy * dy);

        // When full, subtract the value about to be overwritten
        if (this.ring.count === this.bufSize) this.sum -= this.ring.get(0);
        this.ring.push(dist);
        this.sum += dist;

        return this.computeStats(dist);
    }

    private computeStats(dist: number): AccuracyStats {
        const n = this.ring.count;
        let min = Infinity, max = -Infinity;
        for (let i = 0; i < n; i++) {
            const v = this.ring.get(i);
            if (v < min) min = v;
            if (v > max) max = v;
        }
        const avg = this.sum / n;
        let varSum = 0;
        for (let i = 0; i < n; i++) varSum += (this.ring.get(i) - avg) ** 2;
        const stddev = Math.sqrt(varSum / n);

        return { dist, avg, stddev, min, max, count: n };
    }
}
