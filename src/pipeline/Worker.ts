/**
 * Pipeline Worker
 * Recieves postMessages from Client, and calls functions in Pipeline, sends result back to client
 * receives frame from webcam via Client using postMessage as 'step'
 * then calls `pipeline.processFrame(frame, metadata)`and posts the result back.
 * A generic message router - doesn't know what stages exist or what they do.
 */

import { Pipeline } from './Pipeline';
import type { BiometricsResult } from './types';

let pipeline: Pipeline;
let status: 'initializing' | 'idle' | 'inference' = 'initializing';

self.onmessage = async (e: MessageEvent) => {
    const { type, payload } = e.data;

    switch (type) {
        case 'init': {
            const { createDefaultPipeline } = await import('./defaultPipeline');
            pipeline = await createDefaultPipeline(payload);
            status = 'idle';
            self.postMessage({ type: 'ready' });
            break;
        }

        case 'step': {
            if (status !== 'idle') {
                payload.frame?.close?.(); // Note that Client will close this itself, but left here to allow decoupling.
                if (status === 'initializing') {
                    self.postMessage({ type: 'stepError', error: 'Not initialized' });
                }
                return;
            }
            status = 'inference';
            try {
                const result: BiometricsResult = await pipeline.processFrame(payload.frame, payload.context);
                self.postMessage({ type: 'stepResult', result });
            } catch (err) {
                console.error(err);
                self.postMessage({ type: 'stepError', error: String(err) });
            } finally {
                status = 'idle';
            }
            break;
        }

        case 'command': {
            const { id, stage, command, args } = payload;
            try {
                const target = pipeline.getStage(stage);
                if (!target?.handleCommand) {
                    self.postMessage({
                        type: 'commandError',
                        id,
                        error: `Stage "${stage}" not found or does not accept commands`,
                    });
                    return;
                }
                const result = await target.handleCommand(command, args); // TODO: consider typed commands - return here is untyped
                self.postMessage({ type: 'commandResult', id, result });
            } catch (err) {
                self.postMessage({ type: 'commandError', id, error: String(err) });
            }
            break;
        }

        case 'dispose': {
            pipeline?.dispose();
            self.close()
            break;
        }
    }
};
