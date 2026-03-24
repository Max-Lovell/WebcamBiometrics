/**
 * SIMPLE WORKER
 * Prototype to test out a more straightforward worker to potentially use. Uses the client but not the worker/pipeline files
*/
import FaceLandmarkerClient from '../Core/FaceLandmarkerClient';
import WebEyeTrack from '../WebEyeTrack/WebEyeTrack';
import { HeartRateMonitor } from '../rPPG';
import type { BiometricsResult } from './types';

let face: FaceLandmarkerClient;
let gaze: WebEyeTrack;
let heart: HeartRateMonitor;
let status: 'initializing' | 'idle' | 'inference' | 'calib' = 'initializing';

self.onmessage = async (e: MessageEvent) => {
    const { type, payload } = e.data;

    switch (type) {
        case 'init': {
            face = new FaceLandmarkerClient();
            gaze = new WebEyeTrack(
                payload?.maxPoints,
                payload?.clickTTL,
                payload?.maxCalibPoints,
                payload?.maxClickPoints,
            );
            heart = new HeartRateMonitor(payload?.heartRateConfig);

            await face.initialize();
            await gaze.initialize(payload?.modelPath);

            status = 'idle';
            self.postMessage({ type: 'ready' });
            break;
        }

        case 'step': {
            if (status !== 'idle') {
                payload.frame?.close?.();
                if (status === 'initializing') {
                    self.postMessage({ type: 'stepError', error: 'Not initialized' });
                }
                return;
            }

            status = 'inference';
            const { frame, context } = payload;

            try {
                // TODO: consider running faceLandmarker whilst previous frame's gaze and heart are already running.
                // Face first — both gaze and heart depend on it
                context.trace!.push({ step: `facelandmarker_start`, timestamp: performance.now() });
                const faceResult = await face.processFrame(frame, context.videoTime);
                context.trace!.push({ step: `facelandmarker_end`, timestamp: performance.now() });
                const detected = !!(faceResult?.faceLandmarks?.length);

                // Gaze and heart in parallel - TODO: stop gaze blocking thread
                context.trace!.push({ step: `gaze_start`, timestamp: performance.now() });
                const gazePromise = gaze.step(frame, context.videoTime, faceResult);
                context.trace!.push({ step: `heart_start`, timestamp: performance.now() });
                const heartResult = detected
                    ? heart.processFrame(frame, faceResult, context.videoTime)
                    : undefined;
                context.trace!.push({ step: `heart_end`, timestamp: performance.now() });
                const gazeResult = await gazePromise;
                context.trace!.push({ step: `gaze_end`, timestamp: performance.now() });

                const result: BiometricsResult = {
                    frameMetadata: context,
                    face: faceResult ? { faceLandmarkerResult: faceResult, detected } : undefined,
                    gaze: gazeResult,
                    heart: heartResult,
                };

                self.postMessage({ type: 'stepResult', result });
            } catch (err) {
                console.error(err);
                self.postMessage({ type: 'stepError', error: String(err) });
            } finally {
                try { frame?.close?.(); } catch {}
                status = 'idle';
            }
            break;
        }

        case 'command': {
            const { id, stage, command, args } = payload;
            try {
                if (stage !== 'gaze') {
                    self.postMessage({ type: 'commandError', id, error: `Unknown stage "${stage}"` });
                    return;
                }
                let result: unknown;
                switch (command) {
                    // Handle the gaze calibration
                    case 'click': result = gaze.handleClick(args.x, args.y); break;
                    case 'adapt': result = gaze.adapt(
                        args.eyePatches, args.headVectors,
                        args.faceOrigins3D, args.normPogs,
                        args.stepsInner, args.innerLR, args.ptType,
                    ); break;
                    case 'clearCalibration': result = gaze.clearCalibrationBuffer(); break;
                    case 'clearClickstream': result = gaze.clearClickstreamPoints(); break;
                    case 'resetAllBuffers': result = gaze.resetAllBuffers(); break;
                    default: throw new Error(`Unknown command "${command}"`);
                }
                self.postMessage({ type: 'commandResult', id, result });
            } catch (err) {
                self.postMessage({ type: 'commandError', id, error: String(err) });
            }
            break;
        }

        case 'dispose': {
            gaze?.dispose();
            face?.dispose();
            break;
        }
    }
};
