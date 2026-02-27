export interface RGBSignal {
    r: Float32Array;
    g: Float32Array;
    b: Float32Array;
}

// WindowedPulse Estimation Method Interfaces
    // Contract for pulse extraction algorithms - core signal extraction from an RGB window
    // Windowed methods (POS, CHROM, GRGB, LGI, etc.) take in RGB window and return pulse signal of same length.
    // TODO: Separate FullBufferPulseMethod interface for ICA/PCA-based approaches using full signal history.
export interface WindowedPulseMethod {
    readonly name: string; // Human-readable for logging e.g. CHROM or POS
    readonly windowSize: number; // N samples for RGB ring buffer size, Math.ceil(sampleRate * 1.6)
    // If true channels are pre-divided by their mean
    readonly needsTemporalNormalization: boolean; // Set to false most of the time (e.g. POS), only simpler methods like green channel use this.
    // Should be stateless and return Float32Array of length windowSize, mean-centered for overlap-add.
    // RGB channels: length=windowSize, chronologically ordered, guaranteed full by pipeline
    process(rgb: RGBSignal): Float32Array; // Extract a pulse signal from one window of RGB data.
}
