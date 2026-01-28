import { convertVideoFrameToImageData } from './utils/misc';
import type { TrackingContext } from './types';

export default class WebcamClient {
    private videoElement: HTMLVideoElement;
    private stream?: MediaStream;
    private frameCallback?: (frame: ImageData, context: TrackingContext) => Promise<void>;
    private fallbackFrameCount = 0;
    private isRunning: boolean = false; // Flag to kill the loop
    private handleLoadedData = () => this._processFrames();

    constructor(videoElementId: string) {
        const videoElement = document.getElementById(videoElementId) as HTMLVideoElement;
        if (!videoElement) {
            throw new Error(`Video element with id '${videoElementId}' not found`);
        }
        this.videoElement = videoElement;
    }

    async startWebcam(frameCallback?: (frame: ImageData, context: TrackingContext) => Promise<void>): Promise<void> {
        try {
            const constraints: MediaStreamConstraints = {
                video: { // TODO: check these constraints
                    width: { min: 640, ideal: 1920 },
                    height: { min: 400, ideal: 1080 },
                    // width: { ideal: 640 },
                    // height: { ideal: 480 },
                    frameRate: { ideal: 60, min: 30 },
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

            // Start video playback
            this.videoElement.onloadedmetadata = () => {
                this.videoElement.play();
            };
            // Use reference for removal later
            this.videoElement.addEventListener('loadeddata', this.handleLoadedData);

        } catch (error) {
            console.error("Error accessing the webcam:", error);
        }
    }

    stopWebcam(): void {
        this.isRunning = false; // Stop the rvfc/raf loop

        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = undefined;
        }

        // Clean up listeners to prevent memory leaks
        this.videoElement.removeEventListener('loadeddata', this.handleLoadedData);
        this.videoElement.srcObject = null;
    }

    private _processFrames(): void {
        if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) { // Higher Precision for if available
            console.log('Using requestVideoFrameCallback')
            // 'now' and 'metadata' are provided by the browser API
            const process = (now: number, metadata: VideoFrameCallbackMetadata) => {
                if (!this.isRunning || !this.videoElement || this.videoElement.paused || this.videoElement.ended) return;

                const imageData = convertVideoFrameToImageData(this.videoElement);

                const context: TrackingContext = {
                    videoTime: (metadata.mediaTime * 1000) || 0.0001, // Convert s to ms
                    systemTime: now,
                    frameId: metadata.presentedFrames,
                    rawMetadata: metadata // TODO: consider how expectedDisplayTime might be useful?
                };

                if (this.frameCallback) void this.frameCallback(imageData, context);
                this.videoElement.requestVideoFrameCallback(process);
            };

            this.videoElement.requestVideoFrameCallback(process);

        } else { // Fallback (Firefox, Safari)
            console.warn("requestVideoFrameCallback missing. Falling back to requestAnimationFrame.");

            const process = () => {
                if (!this.videoElement || this.videoElement.paused || this.videoElement.ended) return;

                const imageData = convertVideoFrameToImageData(this.videoElement);
                const now = performance.now();

                const context: TrackingContext = {
                    videoTime: now,
                    systemTime: now,
                    frameId: ++this.fallbackFrameCount,
                };

                if (this.frameCallback) void this.frameCallback(imageData, context);
                requestAnimationFrame(process);
            }
            requestAnimationFrame(process);
        }
    }

    dispose(): void {
        this.stopWebcam();
    }
}
