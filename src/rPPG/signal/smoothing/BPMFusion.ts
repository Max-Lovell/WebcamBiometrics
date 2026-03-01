/**
 * BPM Fusion Module
 *
 * Pure functions for cross-validating and selecting between estimators.
 * Sits between estimation and smoothing in the pipeline:
 *
 * The pulse waveform has a sharp systolic peak and dicrotic notch,
 * producing a strong 2nd harmonic. FFT can lock onto this harmonic
 * (reporting 2× true HR) or its subharmonic. Peak counting is immune
 * because it counts actual pulses. So when they disagree by a factor
 * of 2, peak detection is almost certainly right.
 */

import type { PeakResult } from '../PeakEstimator';
import type { FFTEstimate } from '../FFT/FFTEstimator.ts';

// ─── Types ──────────────────────────────────────────────────────────────────

export type EstimationMethod = 'peak' | 'fft' | 'fused';

// ─── Fusion Logic ───────────────────────────────────────────────────────────
// Combine peak and FFT estimates into a single BPM value. 'fused': cross-validate and pick the best
export function fuseBPM(
    method: EstimationMethod,
    peak: PeakResult | null,
    fft: FFTEstimate | null,
): number | null {
    // TODO: Consider:
    //  harmonic/subharmonic detection in fusion overlaps with the harmonic rejection in findDominantFrequency.
    //  Interval validation e.g. FFT says 70BPM and peak estimator measures a 400ms interval (150 BPM) = false peak.
    //  Add hysteresis or lockout to stop flip-flopping between estimates
    //  Bayesian prior on BPM not changing much for temporal continuity
    //  Adaptive refractory period so peak estimation is 70% of median interval or FFT estimate
    if (method === 'peak') return peak?.bpm ?? null;
    if (method === 'fft') return fft?.bpm ?? null;

    // Fused mode
    if (!peak && !fft) return null;
    if (!peak) return fft!.bpm;
    if (!fft) return peak.bpm;

    // Both available — cross-validate
    const ratio = Math.abs(peak.bpm - fft.bpm) / fft.bpm;

    // Agreement (within 10%): trust FFT for precision
    // TODO: 10% agreement threshold is fixed and symmetric. Could lower or use absolute tolerance (e.g +-5BPM)
    if (ratio < 0.10) return fft.bpm;

    // FFT~1/2 peak means FFT grabbed subharmonic
    const halfRatio = Math.abs(fft.bpm - peak.bpm / 2) / peak.bpm;
    if (halfRatio < 0.10) return peak.bpm;

    // FFT~2*peak means FFT grabbed harmonic
    const doubleRatio = Math.abs(fft.bpm - peak.bpm * 2) / peak.bpm;
    if (doubleRatio < 0.10) return peak.bpm;

    // Disagreement with no harmonic relationship: trust higher confidence
    // TODO: Note confidence metrics aren't same scale. Could normalizing or weighting, or bias to FFT
    return peak.confidence > fft.confidence ? peak.bpm : fft.bpm;
}

// Get the confidence value corresponding to the fused BPM decision.
// Mirrors the logic in fuseBPM so the returned confidence always corresponds to whichever estimator was actually trusted.
export function getFusionConfidence(
    method: EstimationMethod,
    peak: PeakResult | null,
    fft: FFTEstimate | null,
): number {
    if (method === 'peak') return peak?.confidence ?? 0;
    if (method === 'fft') return fft?.confidence ?? 0;

    // Fused: return confidence from whichever we'd trust
    if (!peak && !fft) return 0;
    if (!peak) return fft!.confidence;
    if (!fft) return peak.confidence;

    // If they agree, use FFT's SNR (more informative than interval consistency)
    const ratio = Math.abs(peak.bpm - fft.bpm) / fft.bpm;
    if (ratio < 0.10) return fft.confidence;

    // Otherwise, confidence from whichever we trusted
    return peak.confidence > fft.confidence ? peak.confidence : fft.confidence;
}
