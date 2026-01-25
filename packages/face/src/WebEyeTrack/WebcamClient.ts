import { convertVideoFrameToImageData } from './utils/misc';
export default class WebcamClient {
    private videoElement: HTMLVideoElement;
    private stream?: MediaStream;
    private frameCallback?: (frame: ImageData, timestamp: number) => Promise<void>;

    constructor(videoElementId: string) {
        const videoElement = document.getElementById(videoElementId) as HTMLVideoElement;
        if (!videoElement) {
            throw new Error(`Video element with id '${videoElementId}' not found`);
        }
        this.videoElement = videoElement;
    }

    async startWebcam(frameCallback?: (frame: ImageData, timestamp: number) => Promise<void>): Promise<void> {
        try {
            const constraints: MediaStreamConstraints = {
                video: {
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
        // Check for API support (Chrome/Edge/Opera support this; Firefox/Safari may not)
        if ('requestVideoFrameCallback' in this.videoElement) {

            const process = (_now: number, metadata: VideoFrameCallbackMetadata) => {
                if (!this.videoElement || this.videoElement.paused || this.videoElement.ended) return;

                const imageData = convertVideoFrameToImageData(this.videoElement);

                // FIX: MediaPipe crashes on 0, so ensure we are always > 0
                let timestamp = metadata.mediaTime * 1000;
                if (timestamp === 0) timestamp = 0.0001;

                if (this.frameCallback) {
                    // Pass the precise camera timestamp
                    void this.frameCallback(imageData, timestamp);
                }

                // Register for the next specific frame
                this.videoElement.requestVideoFrameCallback(process);
            };

            this.videoElement.requestVideoFrameCallback(process);

        } else {
            // Fallback for browsers without requestVideoFrameCallback (e.g. older Safari)
            console.warn("requestVideoFrameCallback missing. Falling back to requestAnimationFrame.");

            const process = () => {
                if (!this.videoElement || this.videoElement.paused || this.videoElement.ended) return;

                const imageData = convertVideoFrameToImageData(this.videoElement);
                // Fallback to performance.now(), good enough for basic tracking, worse for rPPG
                if (this.frameCallback) void this.frameCallback(imageData, performance.now());

                requestAnimationFrame(process);
            }
            requestAnimationFrame(process);
        }
    }
}
