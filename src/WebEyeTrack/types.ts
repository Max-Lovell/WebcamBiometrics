export interface WebEyeTrackResult {
  // Preprocessing
  headVector: Array<number>; // [3,] - Head vector in camera coordinates
  faceOrigin3D: Array<number>; // X, Y, Z
  // Gaze state (blinking)
  gazeState: 'open' | 'closed';
  // PoG (normalized screen coordinates)
  normPog: Array<number>; // [2,] - Normalized screen coordinates
  // Meta data
  durations: Record<string, number>; // seconds
  timestamp: number; // milliseconds
  debug?: {}; // For anything else.
}


