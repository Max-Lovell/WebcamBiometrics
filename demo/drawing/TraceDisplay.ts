interface TraceEntry {
    step: string;
    timestamp: number;
}

interface TraceStat {
    current: number;
    mean: number;
    min: number;
    max: number;
}

const DURATIONS = [
    { label: 'Worker', start: 'pipeline_start', end: 'pipeline_end' },
    { label: '  face', start: 'face_start', end: 'face_end' },
    { label: '  gaze', start: 'gaze_start', end: 'gaze_end' },
    { label: '  heart', start: 'heart_start', end: 'heart_end' },
    { label: 'Total', start: 'client_send', end: 'frame_received' },
] as const;

export class TraceDisplay {
    private el: HTMLElement;
    private bufs: Map<string, Float64Array> = new Map();
    private head = 0;
    private count = 0;

    constructor(el: HTMLElement, private bufSize = 60) {
        this.el = el
        for (const d of DURATIONS) {
            this.bufs.set(d.label, new Float64Array(bufSize));
        }
    }

    update(trace: TraceEntry[]): void {
        const step = (name: string) => trace.find(t => t.step === name)?.timestamp ?? null;

        for (const d of DURATIONS) {
            const a = step(d.start);
            const b = step(d.end);
            const buf = this.bufs.get(d.label)!;
            buf[this.head] = (a != null && b != null) ? b - a : -1;
        }

        this.head = (this.head + 1) % this.bufSize;
        if (this.count < this.bufSize) this.count++;

        this.render();
    }

    private stat(label: string): TraceStat | null {
        const buf = this.bufs.get(label)!;
        // find the most recent entry
        const idx = (this.head - 1 + this.bufSize) % this.bufSize;
        const current = buf[idx];
        if (current < 0) return null;

        let sum = 0, min = Infinity, max = -Infinity, n = 0;
        for (let i = 0; i < this.count; i++) {
            const v = buf[i];
            if (v < 0) continue;
            sum += v;
            if (v < min) min = v;
            if (v > max) max = v;
            n++;
        }
        if (n === 0) return null;
        return { current, mean: sum / n, min, max };
    }

    private render(): void {
        const lines: string[] = [];
        const pad = 8; // label column width

        lines.push(
            'Stage'.padEnd(pad) + 'now'.padStart(7) + 'avg'.padStart(7) + 'range'.padStart(13)
        );

        for (const d of DURATIONS) {
            const s = this.stat(d.label);
            if (!s) {
                lines.push(`${d.label.padEnd(pad)}${'—'.padStart(7)}`);
            } else {
                const now = s.current.toFixed(1).padStart(5) + 'ms';
                const avg = s.mean.toFixed(1).padStart(5) + 'ms';
                const range = `${s.min.toFixed(1)}-${s.max.toFixed(1)}ms`;
                lines.push(`${d.label.padEnd(pad)}${now.padStart(7)}${avg.padStart(7)}${range.padStart(13)}`);
            }
        }

        this.el.innerText = lines.join('\n');
    }
}
