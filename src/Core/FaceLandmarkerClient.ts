import type {FaceLandmarkerResult} from "@mediapipe/tasks-vision";
import {FaceLandmarker, FilesetResolver} from "@mediapipe/tasks-vision";

import type { VideoFrameData } from "../types";

// References
// https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/web_js#video
export default class FaceLandmarkerClient {
  private faceLandmarker: FaceLandmarker | null = null;

  constructor() {
  }

  async initialize() {
    const filesetResolver = await FilesetResolver.forVisionTasks("/wasm");
    try {
      // Facelandmarker is broken for vite (self.import() error) so below is a hack to get it working
      // Noted here: https://github.com/google-ai-edge/mediapipe/issues/5257
      // TODO: find safer fix than eval here??
      // Find new files at https://www.npmjs.com/package/@mediapipe/tasks-vision
      // Current version 10.32
        // wasm: https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm
      const response = await fetch(filesetResolver.wasmLoaderPath);
      // Use indirect eval to execute the script in the global scope. This is required for the library to find the ModuleFactory.
      (0, eval)(await response.text());
      // FIX: Cast to 'any' to bypass TS2790 strict check
      // delete wasmLoaderPath to trick FaceLandmarker.createFromOptions into thinking it doesn't need to load the script
      delete (filesetResolver as any).wasmLoaderPath;

      return this.faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: `/wasm/face_landmarker.task`, //https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
          delegate: "GPU",
        },
        // TODO: Make some of these optional and pass through config from constructor
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
        runningMode: "VIDEO",
        numFaces: 1,
        // minTrackingConfidence: 0.5 // note set this higher (.7) for a more stable solution, or decrease for smoother (.3)
      });
    } catch (e) {
      console.error("Failed to manually load MediaPipe WASM loader:", e);
    }
  }

  async processFrame(frame: VideoFrameData, timestamp: number): Promise<FaceLandmarkerResult | null> {
    if (!this.faceLandmarker) {
      console.error("FaceLandmarker is not loaded yet.");
      return null;
    }

    return this.faceLandmarker.detectForVideo(frame, timestamp);
  }

  dispose(): void {
    if (this.faceLandmarker) {
      this.faceLandmarker.close();
      this.faceLandmarker = null;
    }
  }
}
