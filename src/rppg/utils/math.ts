// ─── Basic Statistics ───────────────────────────────────────────────────────
// Arithmetic mean of a Float32Array */
export function mean(arr: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
        sum += arr[i];
    }
    return sum / arr.length;
}

// Population standard deviation of a Float32Array
export function std(arr: Float32Array): number {
    const mu = mean(arr);
    let sumSquaredDiff = 0;
    for (let i = 0; i < arr.length; i++) {
        const diff = arr[i] - mu;
        sumSquaredDiff += diff * diff;
    }
    return Math.sqrt(sumSquaredDiff / arr.length);
}

// Median of a number array (non-destructive — does not mutate input)..
export function median(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) {
        return (sorted[mid - 1] + sorted[mid]) / 2;
    }
    return sorted[mid];
}

// ─── Unit Conversions ───────────────────────────────────────────────────────
// Convert BPM to Hz. Useful for specifying filter cutoffs in heart-rate terms.
export function bpmToHz(bpm: number): number {
    return bpm / 60;
}

// Convert Hz to BPM. Useful for interpreting FFT frequency bins as heart rate.
export function hzToBpm(hz: number): number {
    return hz * 60;
}
