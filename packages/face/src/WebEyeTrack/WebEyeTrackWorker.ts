// C.F. https://github.com/google-ai-edge/mediapipe/issues/5257

import WebEyeTrack from './WebEyeTrack';
// import type { TrackingContext } from './types';
import FaceLandmarkerClient from '../Core/FaceLandmarkerClient';
import type { BiometricsResult } from "./types.ts";

let faceLandmarker: FaceLandmarkerClient;
let tracker: WebEyeTrack;

// Instantiate heart rate estimator in global scope

let status: 'initializing' | 'idle' | 'inference' | 'calib' = 'initializing';

self.onmessage = async (e: MessageEvent) => {
  const { type, payload } = e.data;

  switch (type) {
    case 'init':
      faceLandmarker = new FaceLandmarkerClient();
      await faceLandmarker.initialize();
      tracker = new WebEyeTrack(
          payload?.maxPoints,        // backward compat
          payload?.clickTTL,
          payload?.maxCalibPoints,   // new dual-buffer config
          payload?.maxClickPoints    // new dual-buffer config
      );
      await tracker.initialize(payload?.modelPath);
      status = 'idle';
      self.postMessage({ type: 'ready' });
      break;

    case 'step':
      if (status === 'initializing') {
        if (payload.frame) payload.frame.close();
        // Tell proxy we aren't ready so it unlocks
        self.postMessage({type: 'stepError', error: 'Worker not initialized'});
        return;
      } else if (status !== 'idle') {
        if (payload.frame && typeof payload.frame.close === 'function') {
          payload.frame.close();
        }
        return;
      }

      status = 'inference';
      const { frame, context } = payload;
      context.trace.push({ step: 'worker_start', timestamp: performance.now() });
      try {
        // TODO: move from "Stop-and-Wait" protocol to run BlazeGaze in parallel if have previous face mesh waiting (worth it?)
          // TODO: Make next frame start processing immediately using single-slot buffer where next frame is overridden with most recently received one whilst processing
        context.trace.push({ step: 'facelandmarker_start', timestamp: performance.now() });
        const faceResult = await faceLandmarker.processFrame(frame, context.videoTime);
        context.trace.push({ step: 'facelandmarker_end', timestamp: performance.now() });
        const isFaceDetected = faceResult && faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0;

        // TODO: should really handle no face detected here in Worker and not WebEyeTrack tracker...
        context.trace.push({ step: 'webeyetrack_start', timestamp: performance.now() });
        const gazeResult = await tracker.step(frame, context.videoTime, faceResult);
        context.trace.push({ step: 'webeyetrack_end', timestamp: performance.now() });

        // Attach context to result so main thread can log
        context.trace.push({ step: 'worker_end', timestamp: performance.now() });

        let summary;

        if (isFaceDetected) {
          const facialTransformationMatrix = faceResult.facialTransformationMatrixes[0].data;
          summary = {
            faceDetected: true,
            // TODO distance: a weighted split works best for now but figure out what is going wrong in other models
            //  Note: mediapipe = center of head, WET = Nose bridge. MP=stable but heavy filtering, WET=unstable but jittery
            //  MP uses 1.2cm iris, wet uses 15cm eye distance (might be slightly off just for me?)
            //  Solution: bring MP estimate forward and up, stabilise and perform better estimate for WET faceWidth.
            distance: ((facialTransformationMatrix[14] * -1) * .25) + (gazeResult.faceOrigin3D[2] * .75),
            headRotation: [facialTransformationMatrix[2], facialTransformationMatrix[6], facialTransformationMatrix[10]],
            headPosition: [facialTransformationMatrix[12], facialTransformationMatrix[13]*-1, facialTransformationMatrix[14]*-1],
          };
        } else {
          // Fallback summary so the UI doesn't crash
          summary = {
            faceDetected: false,
            distance: 0, // Or maintain the last known distance if you cache it
            headRotation: [0, 0, 0],
            headPosition: [0, 0, 0],
          };
        }

        const finalResult: BiometricsResult = {
          faceLandmarker: faceResult,
          webEyeTrack: gazeResult,
          context: context,
          summary: summary
        };

        self.postMessage({ type: 'stepResult', result: finalResult });
      } catch (err) {
        console.error(err);
        self.postMessage({ type: 'stepError', error: String(err) });
      } finally {
        // Must manage memory of video frames
        if (frame && typeof frame.close === 'function') {
          frame.close();
        }
      }
      status = 'idle';
      break;

    case 'click':
      // Handle click event for re-calibration
      // status = 'calib'; // TODO: consider blocking processing for calibration again in future?
      try {
        tracker.handleClick(payload.x, payload.y);
      } catch (err) {
        console.error(err);
      } finally {
        // status = 'idle'; // Reset to idle when handled.
      }
      break;

    case 'adapt':
      // Handle manual calibration MAML adaptation
      status = 'calib';

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
