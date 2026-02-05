import type {TrackingContext, VideoFrameData} from '../WebEyeTrack/types.ts'; // Note move these up?

export default class WebcamClient {
    public onStreamEnded?: () => void;

    private videoElement: HTMLVideoElement;
    private stream?: MediaStream;
    private frameCallback?: (frame: VideoFrameData, context: TrackingContext) => Promise<void>;
    private isRunning: boolean = false; // Flag to kill the loop
    private animationFrameId: number | null = null;
    private videoFrameId: number | null = null;
    private _disposed: boolean = false;
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
                    width: { ideal: 1920 },
                    height: { ideal: 1080 },
                    frameRate: { ideal: 60 },
                    facingMode: "user"
                },
                audio: false
            };

            // Request webcam access
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.videoElement.srcObject = this.stream;
            this.isRunning = true;

            // Disconnect handler
            const track = this.stream.getVideoTracks()[0];
            track.onended = () => {
                console.warn("Camera disconnected (Lid closed or device removed)");
                this.stopWebcam();
                if (this.onStreamEnded) this.onStreamEnded();
            };

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
        } catch (e) {
            // Reset state on failure so the user can try again
            this.isRunning = false;
            this.stream = undefined;
            // Handle errors
            if (e instanceof DOMException && (e.name === 'NotFoundError' || e.name === 'NotReadableError')) {
                console.warn("Camera not found. It might be disconnected or the laptop lid is closed.");
            } else {
                console.error("Error accessing the webcam:", e);
            }
            throw e; // Re-throw so the UI knows it failed
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
        this.isRunning = false;
        this.stopWebcam();
        this.frameCallback = undefined;

        this._disposed = true;
    }

    get isDisposed(): boolean {
        return this._disposed;
    }
}
