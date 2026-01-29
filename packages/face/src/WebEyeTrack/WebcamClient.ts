import type { TrackingContext } from './types';
export type VideoFrameData = VideoFrame | ImageData;

export default class WebcamClient {
    private videoElement: HTMLVideoElement;
    private stream?: MediaStream;
    private frameCallback?: (frame: VideoFrameData, context: TrackingContext) => Promise<void>;
    private isRunning: boolean = false; // Flag to kill the loop
    private animationFrameId: number | null = null;
    private videoFrameId: number | null = null;
    private _disposed: boolean = false;
    private cachedCanvas: HTMLCanvasElement | null = null;
    private cachedContext: CanvasRenderingContext2D | null = null;
    private abortController: AbortController | null = null;

    constructor(videoElementId: string) {
        const videoElement = document.getElementById(videoElementId) as HTMLVideoElement;
        if (!videoElement) {
            throw new Error(`Video element with id '${videoElementId}' not found`);
        }
        this.videoElement = videoElement;
    }

    async startWebcam(frameCallback?: (frame: VideoFrameData, context: TrackingContext) => Promise<void>): Promise<void> {
        // Guard against double run of stream
        if (this.isRunning || this.stream) {
            console.warn("Webcam is already running or starting.");
            return;
        }

        try {
            const constraints: MediaStreamConstraints = {
                video: { // Note the higher constraints are good for extracting vitals later, but slower for eyetracking
                    // width: { ideal: 1920 },
                    // height: { ideal: 1080 },
                    // frameRate: { ideal: 60 },
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30 },
                    facingMode: "user"
                },
                audio: false
            };

            // Request webcam access
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.videoElement.srcObject = this.stream;
            this.isRunning = true;

            // Set the callback if provided
            if (frameCallback) {
                this.frameCallback = frameCallback;
            }

            // Start video playback - Promise guarantees video has height and width when it starts
            await new Promise<void>((resolve) => {
                if (this.videoElement.readyState >= HTMLMediaElement.HAVE_METADATA) return resolve();
                this.videoElement.onloadedmetadata = () => resolve();
            });
            this.videoElement.play().catch(e => console.warn("Autoplay blocked:", e));

            if ('MediaStreamTrackProcessor' in window && 'VideoFrame' in window) {
                void this._startStreamProcessor();
            } else {
                this._processFrames();
            }
        } catch (error) {
            // Reset state on failure so the user can try again
            this.isRunning = false;
            this.stream = undefined;
            console.error("Error accessing the webcam:", error);
            throw error; // Re-throw so the UI knows it failed
        }
    }

    stopWebcam(): void {
        this.isRunning = false; // Stop the rvfc/raf loop

        if (this.abortController) {
            this.abortController.abort();
            this.abortController = null;
        }

        // Cancel pending frame
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }

        if (this.videoFrameId !== null) {
            this.videoElement.cancelVideoFrameCallback(this.videoFrameId);
            this.videoFrameId = null;
        }

        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = undefined;
        }

        this.videoElement.srcObject = null;
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
        const process = (now: number, metadata?: VideoFrameCallbackMetadata) => {
            if (!this.isRunning) return;

            // Extract pixel data using the fallback canvas method
            const imageData = this.convertVideoFrameToImageData(this.videoElement);

            const context: TrackingContext = {
                videoTime: (metadata?.mediaTime || this.videoElement.currentTime) * 1000 || 0.0001,
                systemTime: now,
                frameId: metadata?.presentedFrames || 0,
            };

            if (this.frameCallback) void this.frameCallback(imageData, context);

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

    /**
     * Converts video frame to ImageData using a cached canvas for performance.
     * Canvas is created once and reused across all frames unless video dimensions change.
     */
    private convertVideoFrameToImageData(frame: HTMLVideoElement): ImageData {
        const width = frame.videoWidth;
        const height = frame.videoHeight;

        // Handle invalid dimensions (video not ready)
        if (width === 0 || height === 0) {
            return new ImageData(1, 1); // Return empty safety dummy
        }

        // Create canvas only once or when dimensions change
        if (!this.cachedCanvas || this.cachedCanvas.width !== width || this.cachedCanvas.height !== height) {
            this.cachedCanvas = document.createElement('canvas');
            this.cachedCanvas.width = width;
            this.cachedCanvas.height = height;

            // willReadFrequently hint optimizes for repeated getImageData() calls
            this.cachedContext = this.cachedCanvas.getContext('2d', { willReadFrequently: true})!;
        }

        // Reuse existing canvas and context
        this.cachedContext!.drawImage(frame, 0, 0);
        return this.cachedContext!.getImageData(0, 0, width, height);
    }

    pauseProcessing(): void {
        this.frameCallback = undefined;
    }

    dispose(): void {
        this.isRunning = false;
        this.stopWebcam();
        this.frameCallback = undefined;

        // Clean up cached canvas resources
        this.cachedCanvas = null;
        this.cachedContext = null;

        this._disposed = true;
    }

    get isDisposed(): boolean {
        return this._disposed;
    }
}
