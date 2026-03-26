/**
 * webcam-biometrics
 * Real-time webcam biometrics: face landmarks, gaze tracking, and heart rate estimation.
 *
 * This is the public entry point for the package.
 * Only types and classes that consumers should use are exported here.
 */

// ── Main client (the primary API surface) ───────────────────────────────────
export { BiometricsClient } from './pipeline/BiometricsClient';
export type { BiometricsClientConfig, TrackerConfig } from './pipeline/BiometricsClient';

// ── Asset configuration (for self-hosting / offline use) ────────────────────
export type { AssetConfig } from './pipeline/assetDefaults';
export { ASSET_DEFAULTS } from './pipeline/assetDefaults';

// ── Result types (what onResult returns) ────────────────────────────────────
export type {
    BiometricsResult,
    FaceContext,
    FrameMetadata,
    FrameTrace,
} from './pipeline/types';

// ── Sub-module result types ─────────────────────────────────────────────────
export type { WebEyeTrackResult } from './webeyetrack/types';
export type { HeartRateResult } from './rppg';

// ── Shared types ────────────────────────────────────────────────────────────
export type { Point, VideoFrameData } from './types';
