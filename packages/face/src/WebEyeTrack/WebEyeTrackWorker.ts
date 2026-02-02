// C.F. https://github.com/google-ai-edge/mediapipe/issues/5257

import WebEyeTrack from './WebEyeTrack';
// import type { TrackingContext } from './types';

let tracker: WebEyeTrack;

// Instantiate heart rate estimator in global scope


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
          // TODO: move from "Stop-and-Wait" protocol to run BlazeGaze in parallel if have previous face mesh waiting (worth it?)
            // TODO: Make next frame start processing immediately using single-slot buffer where next frame is overridden with most recently received one whilst processing
          const gazeResult = await tracker.step(frame, context.videoTime);
          // console.log('gazeResult', gazeResult);
          // add rPPG

          // Attach context to result so main thread can log
          const finalResult = {
            ...gazeResult,
            // Attach context so main thread can log it
            context: context,
            // Default to 0s if no face detected or buffer not full
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
