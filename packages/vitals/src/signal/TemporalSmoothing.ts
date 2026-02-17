/**
 * Temporal Smoothing Module
 *
 * Stabilises BPM estimates over time by filtering out transient outliers.
 * Sits after FFT-based BPM estimation in the pipeline:
 *
 *   POS → Bandpass → Window → FFT → Peak Detection → **Temporal Smoothing** → Display
 *
 * Provides three strategies:
 *   - MedianSmoother:   Rejects occasional outlier estimates (e.g., drops to 40 BPM).
 *   - EMASmoother:      Smooths jitter when estimates are noisy but not wildly wrong.
 *   - CombinedSmoother: Median → EMA chain for maximum stability.
 *
 * All implement the same BPMSmoother interface so they're interchangeable.
 */

// ─── Interface ──────────────────────────────────────────────────────────────

export interface BPMSmoother {
    /** Add a new raw BPM estimate and get back the smoothed value */
    update(bpm: number): number;

    /** Reset internal state (e.g., when tracking is lost) */
    reset(): void;

    /** Whether enough samples have been collected for a reliable smoothed estimate */
    isReady(): boolean;
}

// ─── Median Smoother ────────────────────────────────────────────────────────

/**
 * Median filter for BPM estimates.
 *
 * Why median over mean?
 *   If your last 5 BPM estimates are [78, 80, 42, 79, 81], the mean is 72 —
 *   pulled down by the outlier. The median is 79 — the outlier is simply ignored.
 *   This is exactly what you want for occasional drops to 40 BPM.
 *
 * Window size tradeoffs:
 *   - 3: Minimal latency, rejects single outliers
 *   - 5: Rejects up to 2 consecutive bad estimates (recommended starting point)
 *   - 7: Very stable but adds noticeable lag to real BPM changes
 *
 * @param windowSize - Number of recent estimates to consider (odd numbers work best)
 */
export class MedianSmoother implements BPMSmoother {
    private readonly windowSize: number;
    private readonly buffer: number[] = [];

    constructor(windowSize: number = 5) {
        if (windowSize < 3) {
            throw new Error('Median window size must be at least 3');
        }
        this.windowSize = windowSize;
    }

    update(bpm: number): number {
        this.buffer.push(bpm);

        // Keep only the most recent windowSize estimates
        if (this.buffer.length > this.windowSize) {
            this.buffer.shift();
        }

        return median(this.buffer);
    }

    reset(): void {
        this.buffer.length = 0;
    }

    isReady(): boolean {
        return this.buffer.length >= Math.ceil(this.windowSize / 2);
    }
}

// ─── EMA Smoother ───────────────────────────────────────────────────────────

/**
 * Exponential Moving Average smoother for BPM estimates.
 *
 *   smoothed = α * newEstimate + (1 - α) * previousSmoothed
 *
 * Alpha tradeoffs:
 *   - 0.1: Very smooth, slow to respond to real changes
 *   - 0.3: Good balance (recommended starting point)
 *   - 0.6: Responsive but less smoothing
 *
 * Unlike median, EMA is *pulled* by outliers rather than rejecting them.
 * Best for jittery-but-not-wildly-wrong estimates.
 *
 * @param alpha - Smoothing factor between 0 and 1 (higher = more responsive)
 */
export class EMASmoother implements BPMSmoother {
    private readonly alpha: number;
    private smoothed: number | null = null;

    constructor(alpha: number = 0.3) {
        if (alpha <= 0 || alpha > 1) {
            throw new Error('Alpha must be between 0 (exclusive) and 1 (inclusive)');
        }
        this.alpha = alpha;
    }

    update(bpm: number): number {
        if (this.smoothed === null) {
            this.smoothed = bpm;
        } else {
            this.smoothed = this.alpha * bpm + (1 - this.alpha) * this.smoothed;
        }
        return this.smoothed;
    }

    reset(): void {
        this.smoothed = null;
    }

    isReady(): boolean {
        return this.smoothed !== null;
    }
}

// ─── Combined Smoother ──────────────────────────────────────────────────────

/**
 * Median + EMA combo: raw BPM → Median (reject outliers) → EMA (smooth jitter)
 *
 * Probably overkill to start — try MedianSmoother alone first.
 */
export class CombinedSmoother implements BPMSmoother {
    private readonly median: MedianSmoother;
    private readonly ema: EMASmoother;

    constructor(medianWindow: number = 5, emaAlpha: number = 0.3) {
        this.median = new MedianSmoother(medianWindow);
        this.ema = new EMASmoother(emaAlpha);
    }

    update(bpm: number): number {
        const afterMedian = this.median.update(bpm);
        return this.ema.update(afterMedian);
    }

    reset(): void {
        this.median.reset();
        this.ema.reset();
    }

    isReady(): boolean {
        return this.median.isReady();
    }
}

// ─── Utility ────────────────────────────────────────────────────────────────

/** Compute the median of a number array (non-destructive) */
function median(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);

    if (sorted.length % 2 === 0) {
        return (sorted[mid - 1] + sorted[mid]) / 2;
    }
    return sorted[mid];
}