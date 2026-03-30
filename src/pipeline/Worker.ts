/**
 * Pipeline Worker
 * Recieves postMessages from Client, and calls functions in Pipeline, sends result back to client
 * receives frame from webcam via Client using postMessage as 'step'
 * then calls `pipeline.processFrame(frame, metadata)`and posts the result back.
 * A generic message router - doesn't know what stages exist or what they do.
 */

import { Pipeline } from './Pipeline';
import { createDefaultPipeline } from './defaultPipeline';
import type { BiometricsResult } from './types';

let pipeline: Pipeline;
let status: 'initializing' | 'idle' | 'inference' = 'initializing';

// Chrome on iOS can't console log from the worker.so this passes back messages to client
function log(...args: unknown[]) {
    self.postMessage({ type: 'log', message: args.map(String).join(' ') });
}

// BUG FIX LOAD MEDIAPIPE WASM ------- TODO: PR THIS

// ── Polyfills for MediaPipe Web Worker compatibility ────────────────
// These must run before any @mediapipe/tasks-vision code is imported.

// Fix Bug 1: MediaPipe accesses `document` in workers for iOS/Mac detection.
// Provide a Proxy stub that returns OffscreenCanvas for createElement("canvas")
// and returns false for all "in" checks (e.g. "ontouchend" in document).
// Polyfill minimal document for MediaPipe's broken iOS detection in workers
// MediaPipe checks "ontouchend" in document to detect iOS - this prevents the crash
if (typeof document === 'undefined') {
    const handler: ProxyHandler<any> = {
        get(_target, prop) {
            if (prop === 'createElement') {
                return (tag: string) => {
                    if (tag === 'canvas') return new OffscreenCanvas(1, 1);
                    return {};
                };
            }
            return undefined;
        },
        has() { return false; },
    };
    (self as any).document = new Proxy({}, handler);
}

// Fix Bug 2: MediaPipe uses self.import() to dynamically load WASM modules.
// Vite's worker transform doesn't expose this. Bridge it to native import().
if (typeof (self as any).import !== 'function') {
    (self as any).import = async (url: string) => {
        const module = await import(/* @vite-ignore */ url);
        if (module.default) {
            (self as any).ModuleFactory = module.default;
        }
        return module;
    };
}

// Fix Bug 3: vision_wasm_module_internal.js (lines 8369-8374) has a strict-mode scoping bug
// where custom_dbg is declared inside an if-block but referenced outside it.
(self as any).custom_dbg = (...args: unknown[]) => {
    const msg = String(args[0]);
    // Suppress MediaPipe's C++ INFO/WARNING lines that get misrouted to stderr
    if (msg.startsWith('INFO:') || msg.startsWith('W0') || msg.startsWith('I0')) {
        return;
    }
    console.warn(...args);
};

self.onmessage = async (e: MessageEvent) => {
    const { type, payload } = e.data;

    switch (type) {
        case 'init': {
            const { pipeline: pipelineConfig, assets } = payload;
            pipeline = await createDefaultPipeline(pipelineConfig, assets);
            status = 'idle';
            self.postMessage({ type: 'ready' });
            break;
        }

        case 'step': {
            // log('[Worker] step received, status:', status, 'frame:', typeof payload.frame, payload.frame?.width, payload.frame?.height);
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
                if (result.errors) {
                    log('PIPELINE ERRORS:', JSON.stringify(result.errors));
                }
                self.postMessage({ type: 'stepResult', result });
            } catch (err) {
                console.error(err);
                log('[Worker] step error:', err);
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
            self.close();
            break;
        }
    }
};
