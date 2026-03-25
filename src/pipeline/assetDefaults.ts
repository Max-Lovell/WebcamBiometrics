/**
 * Default asset URLs and configuration resolution.
 *
 * all heavy binaries (WASM, models) fetched from public CDNs at runtime.
 * Users can override paths via BiometricsClientConfig.assets
 */

export interface AssetConfig {
    wasmBasePath?: string; // Base URL for the MediaPipe Vision WASM
    faceLandmarkerModelPath?: string; // URL to the face_landmarker.task model file.
    gazeModelPath?: string; // BlazeGaze model path
}

// Pinned versions
const __PACKAGE_VERSION__: string = '0.1.0';

// CDN defaults for published package
const CDN_DEFAULTS: Required<AssetConfig> = {
    wasmBasePath: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm`, // update version here
    // https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models
    faceLandmarkerModelPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    gazeModelPath: `https://cdn.jsdelivr.net/npm/webcam-biometrics@${__PACKAGE_VERSION__}/models/blazegaze/model.json`,
};

// Local dev defaults - Vite serves from public/ at root
const LOCAL_DEFAULTS: Required<AssetConfig> = {
    wasmBasePath: '/wasm',
    faceLandmarkerModelPath: '/wasm/face_landmarker.task',
    gazeModelPath: '/models/model.json',
};

// Resolver ---------
// Detect local dev server on main thread so location available
function isLocalDev(): boolean {
    try {
        const host = globalThis?.location?.hostname;
        return host === 'localhost' || host === '127.0.0.1';
    } catch {
        return false;
    }
}

// Merge user overrides with defaults. CDN URL when published, fallback to local on localhost
export function resolveAssets(overrides?: AssetConfig): Required<AssetConfig> {
    const defaults = isLocalDev() ? LOCAL_DEFAULTS : CDN_DEFAULTS;

    return {
        wasmBasePath:            overrides?.wasmBasePath            ?? defaults.wasmBasePath,
        faceLandmarkerModelPath: overrides?.faceLandmarkerModelPath ?? defaults.faceLandmarkerModelPath,
        gazeModelPath:           overrides?.gazeModelPath           ?? defaults.gazeModelPath,
    };
}

// Export CDN defaults for power users who want to inspect or reference them
export const ASSET_DEFAULTS = CDN_DEFAULTS;