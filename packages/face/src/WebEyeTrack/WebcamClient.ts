import { convertVideoFrameToImageData } from './utils/misc';
import type { TrackingContext } from './types';

export default class WebcamClient {
    private videoElement: HTMLVideoElement;
    private stream?: MediaStream;
    private frameCallback?: (frame: ImageData, context: TrackingContext) => Promise<void>;
    private fallbackFrameCount = 0;

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
                    // width: { ideal: 1280 },
                    // height: { ideal: 720 },
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: "user"
                },
                audio: false
            };

            // Request webcam access
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.videoElement.srcObject = this.stream;

            // Set the callback if provided
            if (frameCallback) {
                this.frameCallback = frameCallback;
            }

            // Start video playback
            this.videoElement.onloadedmetadata = () => {
                this.videoElement.play();
            };

            this.videoElement.addEventListener('loadeddata', () => {
                this._processFrames();
            });

        } catch (error) {
            console.error("Error accessing the webcam:", error);
        }
    }

    stopWebcam(): void {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = undefined;
        }
    }

    private _processFrames(): void {
        if ('requestVideoFrameCallback' in this.videoElement) { // Higher Precision for if available

            // 'now' and 'metadata' are provided by the browser API
            const process = (now: number, metadata: VideoFrameCallbackMetadata) => {
                if (!this.videoElement || this.videoElement.paused || this.videoElement.ended) return;

                const imageData = convertVideoFrameToImageData(this.videoElement);

                // Wrap them in your context
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
                    videoTime: now, // Best effort: use system time
                    systemTime: now,
                    frameId: ++this.fallbackFrameCount,
                };

                if (this.frameCallback) void this.frameCallback(imageData, context);
                requestAnimationFrame(process);
            }
            requestAnimationFrame(process);
        }
    }
}
