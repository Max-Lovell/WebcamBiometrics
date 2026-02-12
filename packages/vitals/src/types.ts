// TODO: these are duplicated from eye tracker - need some central location for them

export type VideoFrameData = VideoFrame | ImageData | ImageBitmap;

export interface TrackingContext {
    videoTime: number; // metadata.mediaTime (rVFC) OR performance.now() (fallback)
    systemTime: number; // Source: metadata.expectedDisplayTime (rVFC) OR performance.now() (fallback)
    frameId: number; // Source: metadata.presentedFrames (rVFC) OR incrementing counter (fallback)
    rawMetadata?: VideoFrameCallbackMetadata; // raw data for debugging (optional incase fallback)
    trace?: Array<{ step: string; timestamp: number }>;
}
