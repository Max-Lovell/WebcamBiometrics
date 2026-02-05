import type {TrackingContext, VideoFrameData} from '../WebEyeTrack/types.ts'; // Note move these up?

export type WebcamStatus = 'active' | 'inactive' | 'waiting' | 'error';

export default class WebcamClient {
    private videoElement: HTMLVideoElement;
    private stream?: MediaStream;
    private frameCallback?: (frame: VideoFrameData, context: TrackingContext) => Promise<void>;

    // State
    private isRunning: boolean = false;  // Flag to kill the loop
    private shouldBeRunning: boolean = false;

    // Cleanup
    private animationFrameId: number | null = null;
    private videoFrameId: number | null = null;
    private _disposed: boolean = false;
    private abortController: AbortController | null = null;
    private deviceChangeListener: () => void;

    // External
    public onStatusChange?: (status: WebcamStatus, msg?: string) => void;

    constructor(videoElementId: string) {
        const videoElement = document.getElementById(videoElementId) as HTMLVideoElement;
        if (!videoElement) {
            throw new Error(`Video element with id '${videoElementId}' not found`);
        }
        this.videoElement = videoElement;

        // note `this` is instance of class WebcamClient, but not inside function passed to event listener
            // .bind allows e.g. this.shouldBeRunning to be referenced in event handler.
            // assigned to variable this.deviceChangeListener to be removed later
        this.deviceChangeListener = this.handleDeviceChange.bind(this);

        navigator.mediaDevices.addEventListener('devicechange', this.deviceChangeListener);
    }

    async startWebcam(frameCallback?: (frame: VideoFrameData, context: TrackingContext) => Promise<void>): Promise<void> {
        this.shouldBeRunning = true;
        if (frameCallback) this.frameCallback = frameCallback;

        await this.attemptStreamStart();
    }

    private async attemptStreamStart(): Promise<void> {
        if (!this.shouldBeRunning || this.isRunning) return;

        try {
            // Guard: If we already have a stream, don't get another
            if (this.stream) return;

            const constraints: MediaStreamConstraints = {
                video: { // Note the higher constraints are good for extracting vitals later, but slower for eyetracking
                    // width: { ideal: 1920 },
                    // height: { ideal: 1080 },
                    // frameRate: { ideal: 60 },
                    width: { ideal: 1920 },
                    height: { ideal: 1080 },
                    frameRate: { ideal: 60 },
                    facingMode: "user"
                },
                audio: false
            };

            this.notifyStatus('active', "Starting camera...");
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);

            // Handle stream death (e.g. unplug/lid close)
            const track = this.stream.getVideoTracks()[0];
            track.onended = () => {
                console.warn("Camera stream ended unexpectedly.");
                this.handleStreamLoss();
            };

            this.videoElement.srcObject = this.stream;

            // Wait for video to be ready
            await new Promise<void>((resolve) => {
                if (this.videoElement.readyState >= HTMLMediaElement.HAVE_METADATA) return resolve();
                this.videoElement.onloadedmetadata = () => resolve();
            });

            await this.videoElement.play();
            this.isRunning = true;
            this.notifyStatus('active', "Tracking");

            // Start Processing Loop
            if ('MediaStreamTrackProcessor' in window && 'VideoFrame' in window) {
                void this._startStreamProcessor();
            } else {
                this._processFrames();
            }

        } catch (error) {
            this.stream = undefined; // Ensure clean state

            // Handle "Camera Not Found"
            if (error instanceof DOMException && (error.name === "NotFoundError" || error.name === "NotReadableError")) {
                console.warn("Camera not found. Entering waiting mode.");
                this.notifyStatus('waiting', "Camera disconnected. Waiting for device...");
            } else { // Consider other handlers
                console.error("Critical webcam error:", error);
                this.notifyStatus('error', `Error: ${error}`);
                this.shouldBeRunning = false; // Give up on critical errors
                throw error;
            }
        }
    }

    private handleStreamLoss() {
        this.stopProcessingLoop(); // Kill the loop/worker
        this.stream = undefined;   // Clear the stream object
        this.notifyStatus('waiting', "Camera disconnected. Waiting for device...");
        // Don't set shouldBeRunning = false so will try reconnect
    }

    private handleDeviceChange() {
        // Check if the camera should be running but isn't and restart
        if (this.shouldBeRunning && !this.isRunning) {
            console.log("Device change detected. Attempting recovery...");
            // Add small delay for drivers to load
            setTimeout(() => this.attemptStreamStart(), 1000);
        }
    }

    stopWebcam(): void {
        // Note this keeps instance alive but places it inactive to be potentially resumed. Use dispose() to kill and clear webcam.
        this.shouldBeRunning = false;
        this.notifyStatus('inactive');
        this.stopProcessingLoop();

        // Fully kill stream
        if (this.stream) {
            this.stream.getTracks().forEach(track => {
                track.onended = null; // Clear handler to avoid loop
                track.stop();
            });
            this.stream = undefined;
        }

        if (this.videoElement) {
            this.videoElement.srcObject = null;
        }
    }

    private stopProcessingLoop() {
        this.isRunning = false;
        if (this.abortController) {
            this.abortController.abort();
            this.abortController = null;
        }
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        if (this.videoFrameId !== null) {
            this.videoElement.cancelVideoFrameCallback(this.videoFrameId);
            this.videoFrameId = null;
        }
    }

    private notifyStatus(status: WebcamStatus, msg?: string) {
        if (this.onStatusChange) this.onStatusChange(status, msg);
    }

    private async _startStreamProcessor() {
        console.log("Using MediaStreamTrackProcessor (High Performance)");

        const track = this.stream?.getVideoTracks()[0];
        if (!track) return;

        // @ts-ignore - 'webcodecs' types might be missing in some setups
        const processor = new MediaStreamTrackProcessor({ track });
        const reader = processor.readable.getReader();

        this.abortController = new AbortController();
        const signal = this.abortController.signal;

        try {
            while (!signal.aborted) {
                const result = await reader.read();
                if (result.done) break;

                const frame = result.value as VideoFrame;

                if (this.frameCallback) {
                    const context: TrackingContext = {
                        videoTime: frame.timestamp / 1000, // micro to milli
                        systemTime: performance.now(),
                        frameId: 0, // Not available in this API
                        trace: [{ step: 'start_webcam', timestamp: performance.now() }],
                    };

                    // Await the callback so we don't close the frame before the user is done with it.
                    await this.frameCallback(frame, context);
                }

                // Release GPU memory
                frame.close();
            }
        } catch (e) {
            if (!signal.aborted) console.error("Stream processor error:", e);
        } finally {
            reader.releaseLock();
        }
    }

    private _processFrames(): void {
        const process = async (now: number, metadata?: VideoFrameCallbackMetadata) => {
            if (!this.isRunning) return;

            if (this.frameCallback) {
                const context: TrackingContext = {
                    videoTime: (metadata?.mediaTime || this.videoElement.currentTime) * 1000 || 0.0001,
                    systemTime: now,
                    frameId: metadata?.presentedFrames || 0,
                    trace: [{ step: 'start_process_frames', timestamp: performance.now() }],
                };
                // createImageBitmap is async but much faster than getImageData as it stays on GPU often
                    // previous approach: const imageData = this.convertVideoFrameToImageData(this.videoElement);
                const bitmap = await createImageBitmap(this.videoElement);
                try {
                    await this.frameCallback(bitmap, context);
                } finally {
                    bitmap.close();
                }
            }

            // Re-schedule
            if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
                this.videoFrameId = this.videoElement.requestVideoFrameCallback(process);
            } else {
                this.animationFrameId = requestAnimationFrame(() => process(performance.now()));
            }
        };

        // Initial trigger
        if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
            console.log("Using requestVideoFrameCallback (Standard)");
            this.videoFrameId = this.videoElement.requestVideoFrameCallback(process);
        } else {
            console.log("Using requestAnimationFrame (Legacy)");
            this.animationFrameId = requestAnimationFrame(() => process(performance.now()));
        }
    }

    pauseProcessing(): void {
        this.frameCallback = undefined;
    }

    dispose(): void {
        if (this._disposed) return;

        this.stopWebcam(); // Stop logic and streams
        // Unbind the global event listener - consider this.deviceChangeListener = undefined;
        navigator.mediaDevices.removeEventListener('devicechange', this.deviceChangeListener);

        // Break reference to external UI components
        this.frameCallback = undefined;
        this.onStatusChange = undefined;

        // consider this.videoElement = null; with ts-ignore to release the DOM element reference for GC
        this._disposed = true;
    }

    get isDisposed(): boolean {
        return this._disposed;
    }
}
