/**
 * Pipeline BiometricsClient
 * The public API - this is the entry point to the code for developers.
 * Owns the webcam and worker lifecycle, creates video feed, processes each frame, returns the result via callback.
 * Webcam -> Client -postMessage> Worker -> Pipeline -> Stages -postMessage> Client -> onResult callback
 * Handles communication back and forth over the worker boundary -
 * Worker, pipeline, and stages are inside a worker thread and communicate over that boundary via postMessages
 */

import WebcamClient from '../Core/WebcamClient';
import type { WebcamStatus } from '../Core/WebcamClient';
import type { BiometricsResult, FrameMetadata } from './types';
import type { VideoFrameData } from '../types';
import type { HeartRateMonitorConfig } from "../rPPG";

// ─── Config ─────────────────────────────────────────────────────────────────
export interface TrackerConfig {
    maxPoints?: number;
    clickTTL?: number;
    modelPath?: string;
    maxCalibPoints?: number;  // New Dual-Buffer config
    maxClickPoints?: number;  // New Dual-Buffer config
}

export interface BiometricsClientConfig {
    // Config forwarded to the pipeline factory inside the worker
    // TODO: Add config from modules.
    pipeline?: { // TODO: wire this up properly
        // TODO: add facelandmarker config
        gaze?: TrackerConfig | false;
        heart?: HeartRateMonitorConfig | false;
    };
}

// ─── Client ─────────────────────────────────────────────────────────────────
export class BiometricsClient {
    private worker: Worker;
    private webcam: WebcamClient;
    private messageHandler: ((e: MessageEvent) => void) | null = null;
    private inputHandler: ((e: PointerEvent) => void) | null = null;
    private _disposed = false;

    // Worker readiness — resolves when worker posts 'ready'
    private readyPromise: Promise<void>;
    private readyResolve!: () => void;

    // Backpressure — true while worker is processing a frame
    private workerBusy = false;

    // Pending command promises (for request/response pattern)
    private commandId = 0;
    private pendingCommands = new Map<number, {
        resolve: (value: unknown) => void;
        reject: (error: Error) => void;
    }>();

    // ── Public callbacks ────────────────────────────────────────────────
    // Called with every processed frame result
    onResult: (result: BiometricsResult) => void = () => {};
    // Called when webcam status changes (active, inactive, waiting, error)
    onWebcamStatus: (status: WebcamStatus, message?: string) => void = () => {};
    // Called when a frame is dropped due to backpressure. Useful for diagnostics.
    onFrameDropped: () => void = () => {};

    // ── Constructor ─────────────────────────────────────────────────────
    constructor(videoElementId: string, config?: BiometricsClientConfig) {
        // Set up ready promise before anything that might trigger messages (e.g. load worker and models first, then start webcam)
        this.readyPromise = new Promise((resolve) => {
            this.readyResolve = resolve;
        });

        // Create webcam (doesn't start capture yet)
        this.webcam = new WebcamClient(videoElementId);
        this.webcam.onStatusChange = (status, msg) => this.onWebcamStatus(status, msg);

        // Spin up worker
        this.worker = new Worker(
            // TODO: Note swap workers out here. e.g. new URL('./SimpleWorker.ts' or 'Worker.ts'
            new URL('./Worker.ts', import.meta.url),
            { type: 'module' },
        );

        // Wire up message handler
        this.messageHandler = (e: MessageEvent) => this.handleWorkerMessage(e); // For receiving messages back
        this.worker.onmessage = this.messageHandler;

        // Send init — worker will build pipeline from config and post 'ready'.
        // Do this before start() so we can load everything up immediately. note start() awaits readyPromise.
        this.worker.postMessage({
            type: 'init',
            payload: config?.pipeline ?? {},
        });

        // Pointer listener for gaze click-to-calibrate TODO: messy coupling here - but has to live on main thread...
        if (config?.pipeline?.gaze !== false) { // TODO: lives on window, consider scoping to specific area?
            this.inputHandler = (e: PointerEvent) => {
                if (!e.isPrimary) return;

                const viewWidth = document.documentElement.clientWidth || window.innerWidth;
                const viewHeight = document.documentElement.clientHeight || window.innerHeight;

                const normX = Math.max(-0.5, Math.min(0.5, (e.clientX / viewWidth) - 0.5));
                const normY = Math.max(-0.5, Math.min(0.5, (e.clientY / viewHeight) - 0.5));

                this.sendCommand('gaze', 'click', { x: normX, y: normY });
            };
            window.addEventListener('pointerdown', this.inputHandler);
        }
    }

    // ── Lifecycle ───────────────────────────────────────────────────────
    // Start capturing and processing frame - resolves once the worker is initialized and the webcam is streaming.
    async start(): Promise<void> {
        if (this._disposed) throw new Error('Client is disposed');
        // Wait for worker to finish loading models
        await this.readyPromise;
        // Start webcam — each frame is sent to the worker
        await this.webcam.startWebcam(
            (frame: VideoFrameData, metadata: FrameMetadata) => this.onFrame(frame, metadata),
        );
    }

    // Stop capture. Can be resumed with start().
    stop(): void {
        this.webcam.stopWebcam();
    }

    // Permanently tear down webcam + worker. Not resumable.
    dispose(): void {
        if (this._disposed) return;

        // Remove pointer listener
        if (this.inputHandler) {
            window.removeEventListener('pointerdown', this.inputHandler);
            this.inputHandler = null;
        }

        this.stop();
        this.webcam.dispose();

        // Tell worker to dispose pipeline, then terminate
        this.worker.postMessage({ type: 'dispose' });
        // Delay call so message can be delivered as terminate is immediate - could acknowledge message received back as well...
        // setTimeout(() => this.worker.terminate(), 100); // Handled with self.close() in worker now.

        if (this.messageHandler) {
            this.worker.onmessage = null;
            this.messageHandler = null;
        }

        // Reject any pending commands
        for (const [, pending] of this.pendingCommands) {
            pending.reject(new Error('Client disposed'));
        }
        this.pendingCommands.clear();

        this._disposed = true;
    }

    get isDisposed(): boolean {
        return this._disposed;
    }

    // ── Frame handling ──────────────────────────────────────────────────
    // Called by WebcamClient on every frame.
    // Implements backpressure: if the worker is still processing the previous frame, this frame is dropped (closed) and onFrameDropped is called.
    // TODO: Store most recent frame ready to be used.
    private async onFrame(frame: VideoFrameData, metadata: FrameMetadata): Promise<void> {
        if (this._disposed || this.workerBusy) {
            // Drop frame — close to prevent GPU memory leak TODO: allow most recent frame to be held
            if (frame && typeof (frame as any).close === 'function') {
                (frame as any).close();
            }
            this.onFrameDropped();
            return;
        }

        this.workerBusy = true;
        metadata.trace ??= []; // nullish coalesce
        metadata.trace.push({ step: 'client_send', timestamp: performance.now() });

        // Build transfer list for zero-copy - allows the frame to be passed into the worker properly
        const transferList: Transferable[] = [];
        if (frame instanceof VideoFrame || frame instanceof ImageBitmap) {
            transferList.push(frame);
        } else if (frame instanceof ImageData) {
            transferList.push(frame.data.buffer);
        }

        this.worker.postMessage(
            { type: 'step', payload: { frame, context: metadata } },
            transferList,
        );
    }

    // ── Worker message handling ─────────────────────────────────────────
    // Recieve message back
    private handleWorkerMessage(e: MessageEvent): void {
        // Note in profiling there is a minor GC caused by collection of these - the less data crossing the threshold the better
        // Using Transferable objects or SharedArrayBuffer with Cross-Origin-Isolation headers could help... (not worth it)
        const { type } = e.data;

        switch (type) {
            case 'ready':
                this.readyResolve(); // Mark as initialised
                break;

            case 'stepResult':
                this.workerBusy = false;
                this.onResult(e.data.result);
                break;

            case 'stepError':
                this.workerBusy = false;
                console.warn('[BiometricsClient] Worker step error:', e.data.error);
                break;

            case 'commandResult': {
                // Match to commandId passed in sendCommand
                const pending = this.pendingCommands.get(e.data.id);
                if (pending) {
                    pending.resolve(e.data.result);
                    this.pendingCommands.delete(e.data.id);
                }
                break;
            }

            case 'commandError': {
                const pending = this.pendingCommands.get(e.data.id);
                if (pending) {
                    pending.reject(new Error(e.data.error));
                    this.pendingCommands.delete(e.data.id);
                }
                break;
            }

            default:
                console.warn(`[BiometricsClient] Unknown worker message: ${type}`);
                break;
        }
    }

    // ── Command channel ─────────────────────────────────────────────────
    // Send a command to a specific stage in the worker. Returns a promise that resolves with the command result. Async.
    // TODO: Consider typed overload or generic sendCommand<T>(stage, command, args): Promise<T>,
    //  or expose named methods on the client like clearCalibration() that wrap sendCommand internally.
    async sendCommand(stage: string, command: string, args?: unknown): Promise<unknown> {
        if (this._disposed) return Promise.reject(new Error('Client is disposed'));
        await this.readyPromise; // queue commands until ready if not initialised yet
        // Track commands so they can be matched up - allows for async calls.
        const id = ++this.commandId;

        return new Promise((resolve, reject) => {
            this.pendingCommands.set(id, { resolve, reject });
            this.worker.postMessage({
                type: 'command',
                payload: { id, stage, command, args },
            });
        });
    }
}
