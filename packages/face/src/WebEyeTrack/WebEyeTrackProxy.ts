import WebcamClient from "./WebcamClient";
import type {GazeResult, TrackingContext} from "./types";

interface TrackerConfig {
  maxPoints?: number;
  clickTTL?: number;
  modelUrl?: string;
}

// import WebEyeTrackWorker from "worker-loader?inline=no-fallback!./WebEyeTrackWorker.ts";
export default class WebEyeTrackProxy {
  private worker: Worker;

  public status: 'idle' | 'inference' | 'calib' = 'idle';

  constructor(webcamClient: WebcamClient, config: TrackerConfig = {}) {

    // Initialize the WebEyeTrackWorker - CHANGED to use modern syntax for automatic bundling with vite.
    this.worker = new Worker(new URL('./WebEyeTrackWorker.ts', import.meta.url), {
      type: 'module'
    });
    console.log('WebEyeTrackProxy worker initialized');

    this.worker.onmessage = (mess) =>{
      // console.log(`[WebEyeTrackWorker] ${mess.data}`)
      // console.log('[WebEyeTrackProxy] received message', mess);

      // Switch state based on message type
      switch (mess.data.type) {
        case 'ready':
          console.log('[WebEyeTrackProxy] Worker is ready');

          // Start the webcam client and set up the frame callback
          webcamClient.startWebcam(async (frame: ImageData, context: TrackingContext) => {
            // Send the frame to the worker for processing
            if (this.status === 'idle') {
              // extract the buffer to transfer ownership
              const buffer = frame.data.buffer; // Transfer memory ownership rather than copy.

              this.worker.postMessage({
                type: 'step',
                payload: { frame, context }
              }, [buffer])
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

        default:
          console.warn(`[WebEyeTrackProxy] Unknown message type: ${mess.data.type}`);
          break;
      }
    }

    // Initialize the worker
    this.worker.postMessage({
      type: 'init',
      payload: config
    })

    // Add mouse handler for re-calibration
    window.addEventListener('click', (e: MouseEvent) => {
      // Convert px to normalized coordinates
      const normX = (e.clientX / window.innerWidth) - 0.5;
      const normY = (e.clientY / window.innerHeight) - 0.5;
      console.log(`[WebEyeTrackProxy] Click at (${normX}, ${normY})`);
      this.worker.postMessage({ type: 'click', payload: { x: normX, y: normY }});
    })
  }

  // Callback for gaze results
  onGazeResults: (gazeResult: GazeResult) => void = () => {
    console.warn('onGazeResults callback not set');
  }
}
