import WebcamClient from "./WebcamClient";
import type {GazeResult, TrackingContext} from "./types";

interface TrackerConfig {
  maxPoints?: number;
  clickTTL?: number;
  modelPath?: string;
  maxCalibPoints?: number;  // New Dual-Buffer config
  maxClickPoints?: number;  // New Dual-Buffer config
}

// import WebEyeTrackWorker from "worker-loader?inline=no-fallback!./WebEyeTrackWorker.ts";
export default class WebEyeTrackProxy {
  private worker: Worker;
  private clickHandler: ((e: MouseEvent) => void) | null = null;
  private messageHandler: ((e: MessageEvent) => void) | null = null;
  private adaptResolve: (() => void) | null = null;
  private adaptReject: ((error: Error) => void) | null = null;

  public status: 'idle' | 'inference' | 'calib' = 'idle';

  constructor(webcamClient: WebcamClient, config: TrackerConfig = {}) {

    // Initialize the WebEyeTrackWorker - CHANGED to use modern syntax for automatic bundling with vite.
    this.worker = new Worker(new URL('./WebEyeTrackWorker.ts', import.meta.url), {
      type: 'module'
    });
    console.log('WebEyeTrackProxy worker initialized');

    this.messageHandler = (mess) => {
      // Switch state based on message type
      switch (mess.data.type) {
        case 'ready':
          console.log('[WebEyeTrackProxy] Worker is ready');

          // Start the webcam client and set up the frame callback
          void webcamClient.startWebcam(async (frame: ImageData, context: TrackingContext) => {
            // Send the frame to the worker for processing
            if (this.status === 'idle') {
              // extract the buffer to transfer ownership
              const buffer = frame.data.buffer; // Note: frame.data is an overview of raw memory in frame.data.buffer

              this.worker.postMessage({
                type: 'step',
                payload: { frame, context }
              }, [buffer]) // Transfer memory ownership from main thread to worker (auto-populates 'frame' object), rather than copy.
            }
          });
          break;

        case 'stepResult':
          // Handle gaze results
          const gazeResult: GazeResult = mess.data.result;
          this.onGazeResults(gazeResult);
          break;

        case 'statusUpdate':
          this.status = mess.data.status;
          break;

        case 'adaptComplete':
          // Handle adaptation completion
          if (mess.data.success) {
            console.log('[WebEyeTrackProxy] Adaptation completed successfully');
          } else {
            console.error('[WebEyeTrackProxy] Adaptation failed:', mess.data.error);
          }
          // Resolve promise if we stored it
          if (this.adaptResolve && this.adaptReject) {
            if (mess.data.success) {
              this.adaptResolve();
            } else {
              this.adaptReject(new Error(mess.data.error));
            }
            this.adaptResolve = null;
            this.adaptReject = null;
          }
          break;

        default:
          console.warn(`[WebEyeTrackProxy] Unknown message type: ${mess.data.type}`);
          break;
      }
    }

    this.worker.onmessage = this.messageHandler;

    // Initialize the worker
    this.worker.postMessage({
      type: 'init',
      payload: config
    })

    // Add mouse handler for re-calibration
    this.clickHandler = (e: MouseEvent) => {
      const normX = (e.clientX / window.innerWidth) - 0.5;
      const normY = (e.clientY / window.innerHeight) - 0.5;
      console.log(`[WebEyeTrackProxy] Click at (${normX}, ${normY})`);
      this.worker.postMessage({ type: 'click', payload: { x: normX, y: normY }});
    };

    window.addEventListener('click', this.clickHandler);
  }

  // Callback for gaze results
  onGazeResults: (gazeResult: GazeResult) => void = () => {
    console.warn('onGazeResults callback not set');
  }
}
