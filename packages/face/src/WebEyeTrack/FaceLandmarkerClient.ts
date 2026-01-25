import type {FaceLandmarkerResult} from "@mediapipe/tasks-vision";
import {FaceLandmarker, FilesetResolver} from "@mediapipe/tasks-vision";

// References
// https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/web_js#video
export default class FaceLandmarkerClient {
  private faceLandmarker: FaceLandmarker | null = null;

  constructor() {
  }

  async initialize() {
    // TODO facelandmarker is broken for vite so below is a hack to get it working
    //https://github.com/google-ai-edge/mediapipe/issues/5257
    const filesetResolver = await FilesetResolver.forVisionTasks("/wasm");
    try {
      const response = await fetch(filesetResolver.wasmLoaderPath);
      // Use indirect eval to execute the script in the global scope.
      // This is required for the library to find the ModuleFactory.
      (0, eval)(await response.text());
      // FIX: Cast to 'any' to bypass TS2790 strict check
      // delete wasmLoaderPath to trick FaceLandmarker.createFromOptions into thinking it doesn't need to load the script
      delete (filesetResolver as any).wasmLoaderPath;

      return this.faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: `/wasm/face_landmarker.task`,
          delegate: "GPU",
        },
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
        runningMode: "VIDEO",
        numFaces: 1,
      });
    } catch (e) {
      console.error("Failed to manually load MediaPipe WASM loader:", e);
    }
  }

  async processFrame(frame: ImageData, timestamp: number): Promise<FaceLandmarkerResult | null> {
    if (!this.faceLandmarker) {
      console.error("FaceLandmarker is not loaded yet.");
      return null;
    }

    return this.faceLandmarker.detectForVideo(frame, timestamp);
  }
}
