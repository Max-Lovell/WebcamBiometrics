import type {FaceLandmarkerResult} from "@mediapipe/tasks-vision";
import {FaceLandmarker, FilesetResolver} from "@mediapipe/tasks-vision";

import type { VideoFrameData } from "../types";

// References
// https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/web_js#video
export default class FaceLandmarkerClient {
  private faceLandmarker: FaceLandmarker | null = null;

  constructor() {
  }

  async initialize(wasmPath: string, modelPath: string): Promise<FaceLandmarker|undefined> {
    // Note useModule parameter currently undocumented, see mediapipe/tasks/web/core/fileset_resolver.ts.template createFileset()
    // Conditionally calls vision_wasm_module_internal - ES module variant that will work with Vite.
    const filesetResolver = await FilesetResolver.forVisionTasks(wasmPath, true);
    try {
      // Find new files at https://www.npmjs.com/package/@mediapipe/tasks-vision
      return this.faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: modelPath,
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
      console.error("Failed to initialize FaceLandmarker:", e);
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
