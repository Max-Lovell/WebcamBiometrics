// Frame Conversion Utilities
// Converts various video frame types (VideoFrame, ImageBitmap, HTMLVideoElement)
// into ImageData for CPU-based processing in the gaze pipeline.
// TODO: consider merging with similar stuff in rPPG package, maybe move up chain to pipeline?

import type { VideoFrameData } from "../types.ts";

// Frame width, handling both VideoFrame (displayWidth) and other CanvasImageSource types (width).
export function getFrameWidth(frame: VideoFrameData): number {
    if ('displayWidth' in frame) return frame.displayWidth;   // VideoFrame
    if ('videoWidth' in frame) return frame.videoWidth;        // HTMLVideoElement
    return frame.width;                                        // ImageData, ImageBitmap
}

export function getFrameHeight(frame: VideoFrameData): number {
    if ('displayHeight' in frame) return frame.displayHeight;   // VideoFrame
    if ('videoHeight' in frame) return frame.videoHeight;        // HTMLVideoElement
    return frame.height;                                        // ImageData, ImageBitmap
}

// Manages an OffscreenCanvas/HTMLCanvasElement for converting video frames to ImageData.
// Reuses the canvas across calls to avoid repeated allocation.
export class FrameConverter {
    private canvas: OffscreenCanvas | HTMLCanvasElement | null = null;
    private ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null = null;

    // Converts any supported frame type into ImageData for CPU-based processing. Also returns the frame dimensions.
    convert(frame: VideoFrameData): { imageData: ImageData; width: number; height: number } {
        if (frame instanceof ImageData) {
            return { imageData: frame, width: frame.width, height: frame.height };
        }

        const { width, height } = this.getFrameDimensions(frame);
        const ctx = this.drawToCanvas(frame, width, height);
        const imageData = ctx.getImageData(0, 0, width, height);

        return { imageData, width, height };
    }

    getCanvas(frame: VideoFrameData): OffscreenCanvas | HTMLCanvasElement | ImageData {
        if (frame instanceof ImageData) {
            return frame; // fromPixels handles ImageData directly
        }

        const { width, height } = this.getFrameDimensions(frame);
        this.drawToCanvas(frame, width, height);
        return this.canvas!;
    }

     getFrameDimensions(frame: Exclude<VideoFrameData, ImageData>): { width: number; height: number } {
        if (frame instanceof VideoFrame) {
            return { width: frame.displayWidth, height: frame.displayHeight };
        }
        // ImageBitmap or HTMLVideoElement
        return { width: frame.width, height: frame.height };
    }

    private drawToCanvas(
        frame: Exclude<VideoFrameData, ImageData>,
        width: number,
        height: number,
    ): OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D {
        const ctx = this.getContext(width, height);
        ctx.drawImage(frame as CanvasImageSource, 0, 0);
        return ctx;
    }

    private getContext(width: number, height: number) {
        if (!this.canvas) {
            if (typeof OffscreenCanvas !== "undefined") {
                this.canvas = new OffscreenCanvas(width, height);
            } else {
                this.canvas = document.createElement("canvas");
            }
        }

        if (this.canvas.width !== width || this.canvas.height !== height) {
            this.canvas.width = width;
            this.canvas.height = height;
            this.ctx = null;
        }

        if (!this.ctx) {
            this.ctx = this.canvas.getContext("2d", {
                willReadFrequently: true,
            }) as CanvasRenderingContext2D;
        }

        return this.ctx;
    }
}
