// Pipeline Core Types
// defines "contract" for the pipeline/blackboard architecture where each stage reads from/writes to shared blackboard (FrameContext)
// Can add new biometric (e.g. Affect) with new optional slot on FrameContext and register with addStage()
import type {FaceLandmarkerResult} from '@mediapipe/tasks-vision';
import type {HeartRateResult} from '../rPPG';
import type {WebEyeTrackResult} from "../WebEyeTrack/types";
import type {VideoFrameData} from "../types";

// Face landmarks + derived head pose, Written by FaceLandmarkerStage; read by GazeStage, HeartRateStage, etc. TODO: could put in /Core
export interface FaceContext { // TODO: consider trimming some of this output down for DX and using full FaceLandmarkerResult in the code only
    // see https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/web_js#handle_and_display_results
    // TODO: Could add normalised landmarks, head rotation, position, distance?
    faceLandmarkerResult: FaceLandmarkerResult; // contains face_landmarks, facial_transformation_matrixes, and optionally face_blendshapes
    detected: boolean; // at least one face detected this frame - derived from landmarks although could just check
}

export interface FrameTrace { step: string; timestamp: number }

export interface FrameMetadata { // TODO: name of this and FrameContext, FrameContext.frameData etc, slightly confusing?
    videoTime: number; // metadata.mediaTime (rVFC) OR performance.now() (fallback)
    systemTime: number; // Source: metadata.expectedDisplayTime (rVFC) OR performance.now() (fallback)
    frameId: number; // Source: metadata.presentedFrames (rVFC) OR incrementing counter (fallback)
    rawMetadata?: VideoFrameCallbackMetadata; // raw data for debugging (optional incase fallback)
    trace?: FrameTrace[];
}

// export interface miscSummaryResult {
//     // TBC, space for any extra general results? e.g. new distance calculation using 25% landmarker distance, 75% webeyetrack distance
//     distance: number;
// }

// The shared data object ("blackboard") that flows through the pipeline with optional output slots
// Use dependsOn to read from earlier stages first
// TODO: might need more separation between the blackboard and the result
export interface BiometricsResult {
    frameMetadata: FrameMetadata;
    face?: FaceContext;
    gaze?: WebEyeTrackResult;
    heart?: HeartRateResult;
    misc?: Record<string, any>;
    // debug?: {}; // For anything else.
    errors?: Record<string, string>;

}

export interface Blackboard extends BiometricsResult {
    // Stores the actual frame to be used by pipeline stages
    // Always populated by the pipeline before any stage runs.
    frame: VideoFrameData;
}

// A processing stage in the pipeline
// Each has a unique name and reads from any FrameContext stages it depends on first (i.e. basically just facelandmarker)
// Thin adapters that wrap actual FaceLandmarker code (which e.g. reads from ctx.frame and writes to ctx.face
export interface Stage {
    readonly name: string; // Unique ID, match with dependsOn.
    readonly dependsOn: string[]; // Names of stages to complete first - empty = run immediately
    initialize?(): Promise<void>; // For loading models/workers
    process(ctx: BiometricsResult): Promise<void> | void; // Process one frame - read from/write to ctx. sync or async.
    reset?(): void; // Reset internal state (e.g., between sessions, on face lost).
    dispose?(): void; // Clean up resources (models, buffers, etc.)
    // Out-of-band commands (calibration, config changes, etc.)
    handleCommand?(command: string, payload: unknown): unknown | Promise<unknown>;
}
