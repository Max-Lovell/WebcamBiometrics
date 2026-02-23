/**
 * FFT & Spectrum Analysis Module
 *
 * Computes the frequency spectrum of a time-domain signal (the filtered POS waveform)
 * to identify the dominant heart rate frequency.
 *
 * Pipeline: signal → detrend → window → zero-pad → FFT → magnitude spectrum
 *
 * Architecture:
 *   hanningWindow(N)           — Generate window coefficients (reusable)
 *   fft(re, im)                — In-place radix-2 Cooley-Tukey FFT
 *   computeSpectrum(signal, fs) — High-level: window + FFT + magnitude + frequency axis
 *
 * Why implement our own FFT?
 *   - No dependency on Web Audio API or external libraries
 *   - Full control over windowing, zero-padding, and output format
 *   - Transparent and testable
 *   - At N=512 or 1024 (typical for rPPG), performance is trivial (~0.1ms)
 */

import { bpmToHz, hzToBpm } from '../utils/math.ts';
import {DEFAULT_PIPELINE_CONFIG, type PipelineConfig} from "../types.ts";


// ─── Windowing ───────────────────────────────────────────────────────────────

/**
 * Generate a Hanning (Hann) window of length N.
 *
 * The window is a raised cosine that tapers smoothly to zero at both ends:
 *   w[n] = 0.5 * (1 - cos(2π * n / (N - 1)))
 *
 * Why Hanning?
 *   - Good spectral leakage suppression (sidelobes ~31dB down)
 *   - Reasonable main lobe width (not too blurry)
 *   - The standard default choice for frequency analysis
 *   - Hamming is similar but doesn't reach zero at the edges,
 *     which matters less here but Hanning is more conventional for FFT analysis
 *
 * This should be computed once and cached, not per-frame.
 *
 * @param N - Window length (should match your signal length before zero-padding)
 * @returns Float32Array of window coefficients
 */
export function hanningWindow(N: number): Float32Array {
    const window = new Float32Array(N);
    for (let n = 0; n < N; n++) {
        window[n] = 0.5 * (1 - Math.cos((2 * Math.PI * n) / (N - 1)));
    }
    return window;
}


// ─── Utility ─────────────────────────────────────────────────────────────────

/**
 * Find the next power of 2 greater than or equal to n.
 *
 * The radix-2 FFT algorithm requires input length to be a power of 2.
 * We zero-pad the signal to this length.
 *
 * Zero-padding doesn't add new information (it can't invent frequencies),
 * but it interpolates between existing frequency bins, giving a smoother
 * spectrum that's easier to peak-pick. Think of it as increasing the
 * "display resolution" of the spectrum without improving the underlying
 * frequency resolution (which is determined by the original signal length).
 */
function nextPowerOf2(n: number): number {
    let p = 1;
    while (p < n) p <<= 1;
    return p;
}


// ─── FFT Core ────────────────────────────────────────────────────────────────

/**
 * In-place radix-2 Cooley-Tukey FFT.
 *
 * This is the classic divide-and-conquer FFT algorithm:
 * 1. Bit-reversal permutation — reorders elements so the recursive
 *    "divide" step can be done iteratively (bottom-up)
 * 2. Butterfly operations — combines pairs of elements with complex
 *    "twiddle factors" (roots of unity: e^(-j2πk/N))
 *
 * The algorithm works in log2(N) stages. At each stage, it combines
 * pairs of sub-transforms of increasing size, using the identity:
 *   DFT_N = combine(DFT_{N/2} of even indices, DFT_{N/2} of odd indices)
 *
 * Complexity: O(N log N) vs O(N²) for naive DFT
 *
 * @param re - Real parts (modified in-place). Length must be a power of 2.
 * @param im - Imaginary parts (modified in-place). Same length as re.
 */
export function fft(re: Float64Array, im: Float64Array): void {
    const N = re.length;

    // ── Step 1: Bit-reversal permutation ──
    // The iterative FFT needs inputs in bit-reversed order.
    // Example for N=8: index 1 (001) swaps with index 4 (100),
    // index 3 (011) swaps with index 6 (110), etc.
    //
    // Why? The recursive FFT would split [0,1,2,3,4,5,6,7] into
    // evens [0,2,4,6] and odds [1,3,5,7], then split again, etc.
    // The final "leaf" order is the bit-reversal of the original indices.
    // By pre-permuting, we can do the butterfly stages in sequential order.
    for (let i = 1, j = 0; i < N; i++) {
        let bit = N >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;

        if (i < j) {
            // Swap re[i] ↔ re[j] and im[i] ↔ im[j]
            [re[i], re[j]] = [re[j], re[i]];
            [im[i], im[j]] = [im[j], im[i]];
        }
    }

    // ── Step 2: Butterfly stages ──
    // Process log2(N) stages, with sub-transform size doubling each time:
    // size=2, 4, 8, ..., N
    for (let size = 2; size <= N; size *= 2) {
        const halfSize = size / 2;

        // Twiddle factor step: the angle increment for this stage
        // W_N^k = e^(-j * 2π * k / size)
        // We step through k = 0, 1, ..., halfSize-1
        const angleStep = (-2 * Math.PI) / size;

        // For each group of 'size' elements
        for (let i = 0; i < N; i += size) {
            // For each butterfly within the group
            for (let k = 0; k < halfSize; k++) {
                const angle = angleStep * k;
                const twiddleRe = Math.cos(angle); // Real part of twiddle factor
                const twiddleIm = Math.sin(angle); // Imaginary part of twiddle factor

                // Indices of the butterfly pair
                const evenIdx = i + k;
                const oddIdx = i + k + halfSize;

                // Complex multiply: twiddle * odd element
                // (a + jb)(c + jd) = (ac - bd) + j(ad + bc)
                const tRe = twiddleRe * re[oddIdx] - twiddleIm * im[oddIdx];
                const tIm = twiddleRe * im[oddIdx] + twiddleIm * re[oddIdx];

                // Butterfly: combine even and odd
                // even' = even + twiddle * odd
                // odd'  = even - twiddle * odd
                re[oddIdx] = re[evenIdx] - tRe;
                im[oddIdx] = im[evenIdx] - tIm;
                re[evenIdx] = re[evenIdx] + tRe;
                im[evenIdx] = im[evenIdx] + tIm;
            }
        }
    }
}


// ─── Spectrum Analysis ───────────────────────────────────────────────────────

/** Result of spectrum computation */
export interface SpectrumResult {
    /** Magnitude at each frequency bin */
    magnitudes: Float64Array;
    /** Frequency (Hz) corresponding to each bin */
    frequencies: Float64Array;
    /** Frequency resolution in Hz (bin spacing) */
    frequencyResolution: number;
    /** Length of the FFT (after zero-padding) */
    fftSize: number;
}

/**
 * Compute the frequency spectrum of a signal.
 *
 * Full pipeline:
 *   1. Detrend — subtract mean to remove DC offset (0 Hz component).
 *      Without this, the DC bin dominates and can mask the heart rate peak.
 *   2. Window — apply Hanning window to reduce spectral leakage.
 *   3. Zero-pad — extend to next power of 2 for FFT and smoother spectrum.
 *   4. FFT — transform to frequency domain.
 *   5. Magnitude — compute |X[k]| = sqrt(re² + im²) for each bin.
 *
 * Only the first N/2+1 bins are returned (positive frequencies).
 * The spectrum is symmetric for real input, so the upper half is redundant.
 *
 * @param signal - Time-domain signal (e.g., filtered POS H values)
 * @param sampleRate - Sample rate in Hz (e.g., 30 for 30fps camera)
 * @param window - Optional pre-computed Hanning window (pass this in to avoid
 *                 recomputing every frame — generate once with hanningWindow(N)
 *                 and cache it)
 * @returns SpectrumResult with magnitudes and corresponding frequencies
 */
export function computeSpectrum(
    signal: Float32Array,
    sampleRate: number,
    window?: Float32Array
): SpectrumResult {
    const N = signal.length;

    // 1. Generate or validate window
    //    If no window provided, create one. In production, cache this.
    const win = window ?? hanningWindow(N);

    // 2. Detrend (remove mean) and apply window in one pass
    //    Detrending removes the DC offset so the 0 Hz bin doesn't dominate.
    //    We combine both steps to avoid an extra loop.
    let mean = 0;
    for (let i = 0; i < N; i++) mean += signal[i];
    mean /= N;

    // 3. Zero-pad to next power of 2
    const fftSize = nextPowerOf2(N);
    const re = new Float64Array(fftSize); // Zero-initialized (padding is free)
    const im = new Float64Array(fftSize); // Imaginary part starts at zero (real input)

    for (let i = 0; i < N; i++) {
        re[i] = (signal[i] - mean) * win[i];
        // im[i] stays 0 — our input is real-valued
    }
    // Indices N..fftSize-1 are already zero (zero-padding)

    // 4. FFT
    fft(re, im);

    // 5. Compute magnitude spectrum (positive frequencies only)
    //    For a real-valued input of length N, the FFT output is symmetric:
    //    X[k] = conj(X[N-k]), so we only need bins 0 through N/2.
    const numBins = fftSize / 2 + 1;
    const magnitudes = new Float64Array(numBins);
    const frequencies = new Float64Array(numBins);

    // Frequency resolution: the spacing between adjacent bins
    // Each bin k represents frequency k * (sampleRate / fftSize)
    const freqResolution = sampleRate / fftSize;

    for (let k = 0; k < numBins; k++) {
        frequencies[k] = k * freqResolution;
        magnitudes[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
    }

    return {
        magnitudes,
        frequencies,
        frequencyResolution: freqResolution,
        fftSize,
    };
}


// ─── Peak Detection ──────────────────────────────────────────────────────────

/** Result of dominant frequency detection */
export interface PeakResult {
    /** Estimated frequency in Hz (with sub-bin interpolation) */
    frequencyHz: number;
    /** Estimated frequency in BPM */
    frequencyBPM: number;
    /** Magnitude at the peak (useful for confidence metric) */
    peakMagnitude: number;
    /** Signal-to-noise ratio: peak power vs total power in band.
     *  Higher = more confident. Typical threshold: > 0.15–0.25 */
    snr: number;
}

/**
 * Find the dominant frequency within a BPM range from a spectrum.
 *
 * Steps:
 *   1. Restrict search to bins within [minBPM, maxBPM] range
 *   2. Find the bin with highest magnitude
 *   3. Apply parabolic (quadratic) interpolation for sub-bin accuracy
 *   4. Compute a simple SNR confidence metric
 *
 * Parabolic interpolation:
 *   The true peak usually falls between two FFT bins. Given the magnitudes
 *   of the peak bin and its two neighbors, we fit a parabola and find its vertex.
 *   This gives sub-bin frequency accuracy essentially for free.
 *
 *   If the peak bin is at index k with magnitude α, and neighbors are β (k-1)
 *   and γ (k+1), the interpolated offset is:
 *     δ = 0.5 * (β - γ) / (β - 2α + γ)
 *   And the interpolated frequency is:
 *     f = (k + δ) * frequencyResolution
 *
 * @param spectrum - Output from computeSpectrum()
 * @param config - Partial<PipelineConfig>
 * @returns PeakResult with interpolated frequency and confidence, or null if no valid peak
 */
export function findDominantFrequency(
    spectrum: SpectrumResult,
    config: Partial<PipelineConfig> = {}
): PeakResult | null {
    const { minBPM, maxBPM } = { ...DEFAULT_PIPELINE_CONFIG, ...config };
    const { magnitudes, frequencies, frequencyResolution } = spectrum;

    const minHz = bpmToHz(minBPM);
    const maxHz = bpmToHz(maxBPM);

    // Find the bin range to search within
    // (These should align with the bandpass filter cutoffs)
    let startBin = 0;
    let endBin = magnitudes.length - 1;

    for (let k = 0; k < frequencies.length; k++) {
        if (frequencies[k] >= minHz) { startBin = k; break; }
    }
    for (let k = frequencies.length - 1; k >= 0; k--) {
        if (frequencies[k] <= maxHz) { endBin = k; break; }
    }

    if (startBin >= endBin) return null;

    // Find peak bin within the range
    let peakBin = startBin;
    let peakMag = magnitudes[startBin];

    for (let k = startBin + 1; k <= endBin; k++) {
        if (magnitudes[k] > peakMag) {
            peakMag = magnitudes[k];
            peakBin = k;
        }
    }

    if (peakMag === 0) return null;

    // ── Harmonic rejection ──
    // The pulse waveform is not a pure sine — it has a sharp systolic peak
    // and dicrotic notch, so the 2nd harmonic (2× true HR) can sometimes
    // exceed the fundamental in magnitude. If we find a peak at frequency f,
    // check if there's a plausible sub-harmonic at f/2. A real heartbeat at
    // f/2 will produce a harmonic at f, but not vice versa, so we prefer
    // the sub-harmonic when it has reasonable power.
    const subHarmonicHz = (peakBin * frequencyResolution) / 2;
    const subHarmonicMinHz = bpmToHz(minBPM);

    if (subHarmonicHz >= subHarmonicMinHz) {
        // Find the bin closest to f/2
        const subBinCenter = subHarmonicHz / frequencyResolution;
        // Search a small window around the expected sub-harmonic (±2 bins)
        // to account for the fundamental not sitting exactly at 2× a bin center
        const searchStart = Math.max(startBin, Math.floor(subBinCenter - 2));
        const searchEnd = Math.min(endBin, Math.ceil(subBinCenter + 2));

        let subPeakBin = searchStart;
        let subPeakMag = magnitudes[searchStart];
        for (let k = searchStart + 1; k <= searchEnd; k++) {
            if (magnitudes[k] > subPeakMag) {
                subPeakMag = magnitudes[k];
                subPeakBin = k;
            }
        }

        // Accept the sub-harmonic if it has at least 40% of the dominant peak's
        // magnitude. This threshold is intentionally generous — if a sub-harmonic
        // exists at all with meaningful power, it's almost certainly the true
        // fundamental. The 2nd harmonic of a real pulse is typically 30-60% of
        // the fundamental's power, so even a weakened fundamental should clear this.
        const SUB_HARMONIC_THRESHOLD = 0.4;
        if (subPeakMag >= peakMag * SUB_HARMONIC_THRESHOLD) {
            peakBin = subPeakBin;
            peakMag = subPeakMag;
        }
    }

    // Parabolic interpolation for sub-bin accuracy
    let interpolatedBin = peakBin;

    if (peakBin > startBin && peakBin < endBin) {
        const alpha = magnitudes[peakBin - 1]; // left neighbor
        const beta = magnitudes[peakBin];       // peak
        const gamma = magnitudes[peakBin + 1]; // right neighbor

        // Denominator of the parabolic interpolation formula
        const denom = alpha - 2 * beta + gamma;

        // Only interpolate if the parabola is concave (denom < 0)
        // If it's flat or convex, just use the bin center
        if (denom < 0) {
            const delta = 0.5 * (alpha - gamma) / denom;
            interpolatedBin = peakBin + delta;
        }
    }

    const frequencyHz = interpolatedBin * frequencyResolution;
    const frequencyBPM = hzToBpm(frequencyHz);

    // Simple SNR: ratio of peak magnitude to sum of all magnitudes in band.
    // A clean periodic signal concentrates energy in one bin → high ratio.
    // A noisy signal spreads energy → low ratio.
    // This is a rough but effective confidence metric.
    let totalMag = 0;
    for (let k = startBin; k <= endBin; k++) {
        totalMag += magnitudes[k];
    }

    const snr = totalMag > 0 ? peakMag / totalMag : 0;

    return {
        frequencyHz,
        frequencyBPM,
        peakMagnitude: peakMag,
        snr,
    };
}
