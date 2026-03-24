/**
 * FFT & Spectrum Analysis Module
 * generates magnitude spectrum from detrended zero-padded Hanning window
 */

import { bpmToHz, hzToBpm } from '../../utils/math';
import {DEFAULT_PIPELINE_CONFIG} from "../../types";
import type {FFTEstimatorConfig} from "./FFTEstimator";

// ─── Windowing ───────────────────────────────────────────────────────────────
// Generate a Hanning window of length N - computed once and cached
export function hanningWindow(N: number): Float32Array {
    const window = new Float32Array(N);
    // Raise window to cosine that tapers smoothly to zero at both ends:
    for (let n = 0; n < N; n++) {
        // w[n] = 0.5 * (1 - cos(2π * n / (N - 1)))
        window[n] = 0.5 * (1 - Math.cos((2 * Math.PI * n) / (N - 1)));
    }
    return window;
}

// ─── Utility ─────────────────────────────────────────────────────────────────
// Find the next power of 2 greater than or equal to n - zero-pad the signal to this length.
    // Gives smoother spectrum for peak picking; increases display resolution
function nextPowerOf2(n: number): number {
    let p = 1;
    while (p < n) p <<= 1;
    return p;
}

// ─── FFT Core ────────────────────────────────────────────────────────────────
// In-place radix-2 Cooley-Tukey FFT.
// re - Real parts (modified in-place). im - Imaginary parts (modified in-place). Lengths must be a power of 2.
export function fft(re: Float64Array, im: Float64Array): void {
    // combines pairs of sub-transforms of increasing size, using identity: DFT_N = combine(DFT_{N/2} of even indices, DFT_{N/2} of odd indices)
    const N = re.length;
    // Bit-reversal permutation
    // swap indices using bits to reorder elements so recursive divide/butterfly step is done iteratively/sequentially bottom-up
    for (let i = 1, j = 0; i < N; i++) {
        let bit = N >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;

        if (i < j) {
            // Swap re[i] <-> re[j] and im[i] <-> im[j]
            [re[i], re[j]] = [re[j], re[i]];
            [im[i], im[j]] = [im[j], im[i]];
        }
    }

    // Butterfly - combines pairs of elements with complex "twiddle factors" (roots of unity: e^(-j2πk/N))
    for (let size = 2; size <= N; size *= 2) {
        const halfSize = size / 2;

        // Twiddle factor step: the angle increment for this stage W_N^k = e^(-j * 2π * k / size)
        const angleStep = (-2 * Math.PI) / size; // step through k = 0, 1, ..., halfSize-1

        for (let i = 0; i < N; i += size) { // For each group of 'size' elements
            for (let k = 0; k < halfSize; k++) { // For each butterfly within the group
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
                // odd'  = even - twiddle * odd
                re[oddIdx] = re[evenIdx] - tRe;
                im[oddIdx] = im[evenIdx] - tIm;
                // even' = even + twiddle * odd
                re[evenIdx] = re[evenIdx] + tRe;
                im[evenIdx] = im[evenIdx] + tIm;
            }
        }
    }
}


// ─── Spectrum Analysis ───────────────────────────────────────────────────────
// Result of spectrum computation
export interface SpectrumResult {
    magnitudes: Float64Array; // Magnitude at each frequency bin
    frequencies: Float64Array; // Frequency (Hz) corresponding to each bin
    frequencyResolution: number; // Frequency resolution in Hz (bin spacing)
    fftSize: number; // Length of the FFT (after zero-padding)
}

// Compute the frequency spectrum of a signal
export function computeSpectrum(
    signal: Float32Array, // Time-domain signal (e.g., filtered POS H values)
    sampleRate: number, // Sample rate in Hz (e.g., 30 for 30fps camera)
    window?: Float32Array // Optional pre-computed Hanning window (pass this in to avoid
): SpectrumResult {
    const N = signal.length;
    // Generate or validate window - If no window provided, create one. Should be cached from FFT Estimator.
    const win = window ?? hanningWindow(N);

    // Detrend - subtract mean to remove DC offset (0Hz bin component) + apply hanning window in one pass
    // TODO: could improve with linear detrend (subtract best-fit line) before windowing
    let mean = 0;
    for (let i = 0; i < N; i++) mean += signal[i];
    mean /= N;

    // Zero-pad to next power of 2 Zero-pad for FFT and smoother spectrum.
    const fftSize = nextPowerOf2(N);
    const re = new Float64Array(fftSize); // Zero-initialized
    const im = new Float64Array(fftSize); // Imaginary part starts at zero (real input)

    for (let i = 0; i < N; i++) { // Note idx N..fftSize-1 already zero padded
        re[i] = (signal[i] - mean) * win[i]; // im[i] stays 0 - input is real-valued
    }

    // FFT - transform to frequency domain.
    fft(re, im);

    // 5. Compute magnitude spectrum (positive frequencies only)
    //    For a real-valued input of length N, the FFT output is symmetric:
    //    X[k] = conj(X[N-k]), so we only need bins 0 through N/2.
    // Magnitude — compute |X[k]| = sqrt(re² + im²) for each bin.
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

    // Only the first N/2+1 bins are returned (positive frequencies).
    // The spectrum is symmetric for real input, so the upper half is redundant.
    return { // SpectrumResult with magnitudes and corresponding frequencies
        magnitudes,
        frequencies,
        frequencyResolution: freqResolution,
        fftSize,
    };
}

// ─── Peak Detection ──────────────────────────────────────────────────────────
// Result of dominant frequency detection
export interface DominantFrequencyResult {
    frequencyHz: number; // Estimated frequency in Hz (with sub-bin interpolation)
    frequencyBPM: number; // Estimated frequency in BPM
    peakMagnitude: number; // Magnitude at the peak (useful for confidence metric)
    snr: number; // Signal-to-noise ratio: peak power vs total power in band. Higher = more confident. Typical threshold: > 0.15–0.25
}

// Find dominant frequency within a BPM range from a spectrum.
export function findDominantFrequency(
    spectrum: SpectrumResult, // Output from computeSpectrum()
    config: Partial<FFTEstimatorConfig> = {}
): DominantFrequencyResult | null {
    const { minBPM, maxBPM, harmonicRejection } = { ...DEFAULT_PIPELINE_CONFIG, ...config };
    const { magnitudes, frequencies, frequencyResolution } = spectrum;

    const minHz = bpmToHz(minBPM);
    const maxHz = bpmToHz(maxBPM);

    // Find the bin range [minBPM, maxBPM] to search in - should align with the bandpass filter cutoffs
    let startBin = 0;
    let endBin = magnitudes.length - 1;

    for (let k = 0; k < frequencies.length; k++) {
        if (frequencies[k] >= minHz) { startBin = k; break; }
    }
    for (let k = frequencies.length - 1; k >= 0; k--) {
        if (frequencies[k] <= maxHz) { endBin = k; break; }
    }

    if (startBin >= endBin) return null;

    // Find peak bin with highest magnitude in range
    let peakBin = startBin;
    let peakMag = magnitudes[startBin];

    for (let k = startBin + 1; k <= endBin; k++) {
        if (magnitudes[k] > peakMag) {
            peakMag = magnitudes[k];
            peakBin = k;
        }
    }

    if (peakMag === 0) return null;

    // Harmonic rejection - necessary?
    // waveform is not pure sine, has systolic peak and dicrotic notch, so 2nd harmonic (2×HR) can exceed fundamental.
    // If peak at f, check for sub-harmonic at f/2 - real heartbeat at f/2 produces harmonic at f, but not vice-versa,
    // so prefer sub-harmonic if has reasonable power.
    // TODO: Harmonic correction: if peak = 2* FFT, probably a diacrotic notch double counted too.
    //  or iterative approach — check f/2, then check that result's f/2, stop when leaves valid band
    if (harmonicRejection) {
        const subHarmonicHz = (peakBin * frequencyResolution) / 2;

        if (subHarmonicHz >= bpmToHz(minBPM)) {
            const subBinCenter = subHarmonicHz / frequencyResolution;
            // ±2 bins, but clamped strictly inside valid band
            const searchStart = Math.max(startBin, Math.floor(subBinCenter) - 2);
            const searchEnd = Math.min(endBin, Math.ceil(subBinCenter) + 2);

            if (searchStart <= searchEnd) {
                let subPeakBin = searchStart;
                let subPeakMag = magnitudes[searchStart];
                for (let k = searchStart + 1; k <= searchEnd; k++) {
                    if (magnitudes[k] > subPeakMag) {
                        subPeakMag = magnitudes[k];
                        subPeakBin = k;
                    }
                }
                const SUB_HARMONIC_THRESHOLD = 0.4;
                if (subPeakMag >= peakMag * SUB_HARMONIC_THRESHOLD) {
                    peakBin = subPeakBin;
                    peakMag = subPeakMag;
                }
            }
        }
    }

    // Parabolic (quadratic) interpolation for sub-bin accuracy. Necessary?
    // Detect peak between two FFT bins by finding vertex of fitted parabola
    // If peak bin at idx k with magnitude α, and neighbors β (k-1) and γ (k+1):
    // offset = δ = 0.5 * (β - γ) / (β - 2α + γ), and frequency f = (k + δ) * frequencyResolution
    let interpolatedBin = peakBin;

    if (peakBin > startBin && peakBin < endBin) {
        const alpha = magnitudes[peakBin - 1]; // left neighbor
        const beta = magnitudes[peakBin];       // peak
        const gamma = magnitudes[peakBin + 1]; // right neighbor

        // Denominator of the parabolic interpolation formula
        const denom = alpha - 2 * beta + gamma;

        // interpolate if parabola is concave (denom < 0) - if flat or convex use bin center
        if (denom < 0) {
            const delta = 0.5 * (alpha - gamma) / denom;
            interpolatedBin = peakBin + delta;
        }
    }

    const frequencyHz = interpolatedBin * frequencyResolution;
    const frequencyBPM = hzToBpm(frequencyHz);

    // Simple SNR confidence: ratio of peak magnitude to sum of all magnitudes in band.
    // clean periodic signal concentrates energy in one bin → high ratio, noisy signal spreads energy → low ratio.
    // TODO: change from magnitude-ratio to power-ratio?
    let totalMag = 0;
    for (let k = startBin; k <= endBin; k++) {
        totalMag += magnitudes[k];
    }

    const snr = totalMag > 0 ? peakMag / totalMag : 0;
    //  PeakResult with interpolated frequency and confidence, or null if no valid peak
    return {
        frequencyHz,
        frequencyBPM,
        peakMagnitude: peakMag,
        snr,
    };
}
