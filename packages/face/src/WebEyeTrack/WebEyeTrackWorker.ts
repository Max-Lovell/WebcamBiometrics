// C.F. https://github.com/google-ai-edge/mediapipe/issues/5257

import WebEyeTrack from './WebEyeTrack';
// import type { TrackingContext } from './types';
import { FACE_ROIS } from './utils/roiUtils';
import { HeartRateEstimator } from '@webcambiometrics/vitals';

let tracker: WebEyeTrack;

// Instantiate heart rate estimator in global scope
const foreheadEstimator = new HeartRateEstimator(10);


let status: 'idle' | 'inference' | 'calib' = 'idle';

self.onmessage = async (e: MessageEvent) => {
  const { type, payload } = e.data;

  switch (type) {
    case 'init':
      tracker = new WebEyeTrack(
          payload?.maxPoints,        // backward compat
          payload?.clickTTL,
          payload?.maxCalibPoints,   // new dual-buffer config
          payload?.maxClickPoints    // new dual-buffer config
      );
      await tracker.initialize(payload?.modelPath);
      self.postMessage({ type: 'ready' });
      status = 'idle';
      break;

    case 'step':
      if (status === 'idle') {

        status = 'inference';
        self.postMessage({ type: 'statusUpdate', status: status});
        const { frame, context } = payload;
        try {
          const gazeResult = await tracker.step(frame, context.videoTime);
          // console.log('gazeResult', gazeResult);
          // add rPPG
          let vitalsResult = null;
          if (gazeResult.facialLandmarks && gazeResult.facialLandmarks.length > 0) {
            // Convert ROI indices to Pixel Coordinates
            // gazeResult.facialLandmarks are normalized (0.0 - 1.0)
            const foreheadPoints = FACE_ROIS.forehead.map((index: number) => {
              const landmark = gazeResult.facialLandmarks[index];
              return {
                x: landmark.x * frame.width,  // Scale to pixels
                y: landmark.y * frame.height
              };
            });

            // 3. Process the frame using the persistent estimator
            vitalsResult = foreheadEstimator.processFrame(
                frame,
                foreheadPoints,
                context.videoTime
            );
          }
          // Attach context to result so main thread can log
          const finalResult = {
            ...gazeResult,
            // Attach context so main thread can log it
            context: context,
            // Default to 0s if no face detected or buffer not full
            vitals: {
              bpm: vitalsResult ? vitalsResult.bpm : 0,
              wave: vitalsResult ? vitalsResult.signal : 0,
              confidence: vitalsResult ? vitalsResult.confidence : 0
            }
          };
          self.postMessage({ type: 'stepResult', result: finalResult });
        } catch (err) {
          console.error(err);
        } finally {
          // Must manage memory of video frames
          if (frame && typeof frame.close === 'function') {
            frame.close();
          }
        }
        status = 'idle';
        self.postMessage({ type: 'statusUpdate', status: status});
      } else {
        // Race edge case handling if the worker receives frame while busy
        if (payload.frame && typeof payload.frame.close === 'function') {
          payload.frame.close();
        }
      }
      break;

    case 'click':
      // Handle click event for re-calibration
      status = 'calib';
      self.postMessage({ type: 'statusUpdate', status: status});

      tracker.handleClick(payload.x, payload.y);

      status = 'idle';
      self.postMessage({ type: 'statusUpdate', status: status});
      break;

    case 'adapt':
      // Handle manual calibration MAML adaptation
      status = 'calib';
      self.postMessage({ type: 'statusUpdate', status: status});

      try {
        tracker.adapt(
            payload.eyePatches,
            payload.headVectors,
            payload.faceOrigins3D,
            payload.normPogs,
            payload.stepsInner,
            payload.innerLR,
            payload.ptType
        );
        self.postMessage({ type: 'adaptComplete', success: true });
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Adaptation failed';
        self.postMessage({ type: 'adaptComplete', success: false, error: errorMessage });
      }

      status = 'idle';
      self.postMessage({ type: 'statusUpdate', status: status});
      break;

    case 'clearCalibration':
      // Clear calibration buffer for re-calibration
      if (tracker) {
        tracker.clearCalibrationBuffer();
      }
      break;

    case 'clearClickstream':
      // Clear clickstream buffer while preserving calibration
      if (tracker) {
        tracker.clearClickstreamPoints();
      }
      break;

    case 'resetAllBuffers':
      // Reset both calibration and clickstream buffers
      if (tracker) {
        tracker.resetAllBuffers();
      }
      break;

    case 'dispose':
      // Clean up tracker resources before worker termination
      if (tracker) {
        tracker.dispose();
      }
      break;

    default:
      console.warn(`[WebEyeTrackWorker] Unknown message data: Type: ${type}, Payload: ${payload}`);
      break;
  }
};

export {}; // for TS module mode
