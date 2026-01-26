import type {NormalizedLandmark, Matrix, Classifications} from "@mediapipe/tasks-vision";
import type { RGBSample } from "@webcambiometrics/vitals";

// export type Point = [number, number];
export type Point = number[];

export enum TrackingStatus {
  FAILED = 0,
  SUCCESS = 1,
}

export interface GazeResult {

  // Inputs
  facialLandmarks: NormalizedLandmark[]; // [N, 5]
  faceRt: Matrix;
  faceBlendshapes: Classifications[]; // [N, 1]

  // Preprocessing
  eyePatch: ImageData; // [H, W, 3] - RGB image of the eye region
  headVector: Array<number>; // [3,] - Head vector in camera coordinates
  faceOrigin3D: Array<number>; // X, Y, Z

  // Face Reconstruction
  // metricFace: Matrix;
  metric_transform: Matrix; // [4, 4]

  // Gaze state (blinking)
  gazeState: 'open' | 'closed';

  // PoG (normalized screen coordinates)
  normPog: Array<number>; // [2,] - Normalized screen coordinates

  // Meta data
  durations: Record<string, number>; // seconds
  timestamp: number; // milliseconds

  //rPPG
  roiSignals?: {
    forehead: RGBSample;
    leftCheek: RGBSample;
    rightCheek: RGBSample;
  };
}

export interface TrackingContext {
  videoTime: number; // metadata.mediaTime (rVFC) OR performance.now() (fallback)
  systemTime: number; // Source: metadata.expectedDisplayTime (rVFC) OR performance.now() (fallback)
  frameId: number; // Source: metadata.presentedFrames (rVFC) OR incrementing counter (fallback)
  rawMetadata?: VideoFrameCallbackMetadata; // raw data for debugging (optional incase fallback)
}
