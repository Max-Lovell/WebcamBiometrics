// Frame Conversion Utilities
// Converts various video frame types (VideoFrame, ImageBitmap, HTMLVideoElement)
// into ImageData for CPU-based processing in the gaze pipeline.
// TODO: consider merging with similar stuff in rPPG package, maybe move up chain to pipeline?

import type { VideoFrameData } from "../../types.ts";

// Manages an OffscreenCanvas/HTMLCanvasElement for converting video frames to ImageData.
//Reuses the canvas across calls to avoid repeated allocation.
export class FrameConverter {
    private canvas: OffscreenCanvas | HTMLCanvasElement | null = null;
    private ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null = null;
    // Converts any supported frame type into ImageData for CPU-based processing. Also returns the frame dimensions.
    convert(frame: VideoFrameData): { imageData: ImageData; width: number; height: number } {
        if (frame instanceof ImageData) {
            return { imageData: frame, width: frame.width, height: frame.height };
        }

        let width: number;
        let height: number;

        if (frame instanceof VideoFrame) {
            width = frame.displayWidth;
            height = frame.displayHeight;
        } else {
            // ImageBitmap or HTMLVideoElement
            width = frame.width;
            height = frame.height;
        }

        const ctx = this.getContext(width, height);
        ctx.drawImage(frame as CanvasImageSource, 0, 0);
        const imageData = ctx.getImageData(0, 0, width, height);

        return { imageData, width, height };
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

    getCanvas(frame: VideoFrameData): OffscreenCanvas | HTMLCanvasElement | ImageData {
        if (frame instanceof ImageData) {
            return frame; // fromPixels handles ImageData directly
        }

        let width: number;
        let height: number;

        if (frame instanceof VideoFrame) {
            width = frame.displayWidth;
            height = frame.displayHeight;
        } else {
            width = frame.width;
            height = frame.height;
        }

        const ctx = this.getContext(width, height);
        ctx.drawImage(frame as CanvasImageSource, 0, 0);
        return this.canvas!;
    }
}
