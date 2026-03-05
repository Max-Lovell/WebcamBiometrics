export interface Point {
    x: number;
    y: number;
}

// Note HTMLVideoElement doesn't work in a worker, but that's up to the user to pass in the correct data
export type VideoFrameData = VideoFrame | ImageData | ImageBitmap | HTMLVideoElement;
