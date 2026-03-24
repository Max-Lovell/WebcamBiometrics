/**
 * Temporal Smoothing Module
 * Stabilises BPM estimates over time by filtering out transient outliers.
 */

import { median } from '../../utils/math';

// ─── Types ──────────────────────────────────────────────────────────────────
export type SmoothingStrategy = 'median' | 'ema' | 'combined' | 'none';

export interface BPMSmoother {
    update(bpm: number): number; // Add a new raw BPM estimate and get back the smoothed value
    reset(): void; // Reset internal state (e.g., when tracking is lost)
    isReady(): boolean; // Whether enough samples have been collected for a reliable smoothed estimate
}

// ─── Median Smoother ────────────────────────────────────────────────────────
// Median filter for BPM estimates.
export class MedianSmoother implements BPMSmoother {
    private readonly windowSize: number; // Should be range 3-7ish, accuracy/lag tradeoff
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
// Exponential Moving Average smoother for BPM estimates.
    // smoothed = α * newEstimate + (1 - α) * previousSmoothed
    // Unlike median, EMA is *pulled* by outliers rather than rejecting them. Best for jittery-but-not-wildly-wrong estimates.
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
// Median + EMA combo: raw BPM → Median (reject outliers) → EMA (smooth jitter)
// Probably overkill to start — try MedianSmoother alone first.
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

// ─── Smoother Factory ───────────────────────────────────────────────────────
// Create a BPMSmoother from a strategy name and parameters.
export function createSmoother(
    strategy: SmoothingStrategy,
    windowSize: number,
    alpha: number,
): BPMSmoother {
    switch (strategy) {
        case 'median':
            return new MedianSmoother(windowSize);
        case 'ema':
            return new EMASmoother(alpha);
        case 'combined':
            return new CombinedSmoother(windowSize, alpha);
        case 'none':
            return {
                update: (bpm: number) => bpm,
                reset: () => {},
                isReady: () => true,
            };
    }
}
