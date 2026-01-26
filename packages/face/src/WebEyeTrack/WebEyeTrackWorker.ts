// C.F. https://github.com/google-ai-edge/mediapipe/issues/5257

import WebEyeTrack from './WebEyeTrack';
import type { TrackingContext } from './types';
import { extractAverageRGB } from '@webcambiometrics/vitals'; // Import the decoupled logic
import { FACE_ROIS } from '../utils/roiUtils';

let tracker: WebEyeTrack;

type InitMessage = {
  type: 'init';
  payload: {
    maxPoints?: number;
    clickTTL?: number;
    modelUrl?: string;
  };
};

type StepMessage = {
  type: 'step';
  payload: {
    frame: ImageData;
    context: TrackingContext;
  };
};

type ClickMessage = {
  type: 'click';
  payload: {
    x: number;
    y: number;
  };
};

type WorkerMessage = InitMessage | StepMessage | ClickMessage;

// const ctx: Worker = self as any;
let status: 'idle' | 'inference' | 'calib' = 'idle';
// let lastTimestamp: number | null = null;

self.onmessage = async (e: MessageEvent) => {
  const data = e.data as WorkerMessage;

  switch (data.type) {
    case 'init':
      tracker = new WebEyeTrack(
          data.payload.maxPoints,
          data.payload.clickTTL,
          data.payload.modelUrl
      );
      await tracker.initialize();
      self.postMessage({ type: 'ready' });
      status = 'idle';
      break;

    case 'step':
      if (status === 'idle') {

        status = 'inference';
        self.postMessage({ type: 'statusUpdate', status: status});
        const { frame, context } = data.payload;

        const gazeResult = await tracker.step(frame, context.videoTime);
        console.log('gazeResult', gazeResult);
        // add rPPG
        let vitalsResult = null;
        if (gazeResult.facialLandmarks && gazeResult.facialLandmarks.length > 0) {
          // Helper to convert normalized landmarks to pixels
          // (We might need to expose a helper for this or do it inline)
          const w = frame.width;
          const h = frame.height;
          const getPoints = (indices: number[]) => indices.map(i => ({
            x: gazeResult.facialLandmarks[i].x * w,
            y: gazeResult.facialLandmarks[i].y * h
          }));

          vitalsResult = {
            forehead: extractAverageRGB(frame, getPoints(FACE_ROIS.forehead), context.videoTime),
            leftCheek: extractAverageRGB(frame, getPoints(FACE_ROIS.leftCheek), context.videoTime),
            rightCheek: extractAverageRGB(frame, getPoints(FACE_ROIS.rightCheek), context.videoTime),
          };
        }
        // Attach context to result so main thread can log
        (gazeResult as any).context = context;
        const finalResult = {
          ...gazeResult,
          roiSignals: vitalsResult
        };
        self.postMessage({ type: 'stepResult', result: finalResult });

        status = 'idle';
        self.postMessage({ type: 'statusUpdate', status: status});
      }
      break;

    case 'click':
      // Handle click event for re-calibration
      status = 'calib';
      self.postMessage({ type: 'statusUpdate', status: status});

      tracker.handleClick(data.payload.x, data.payload.y);

      status = 'idle';
      self.postMessage({ type: 'statusUpdate', status: status});
      break;

    default:
      console.warn(`[WebEyeTrackWorker] Unknown message data: ${data}`);
      break;
  }
};

export {}; // for TS module mode
