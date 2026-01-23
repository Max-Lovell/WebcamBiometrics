import WebEyeTrack from './WebEyeTrack';

let tracker: WebEyeTrack;

type InitMessage = {
  type: 'init';
  payload: {
    maxPoints?: number;
    clickTTL?: number;
    modelUrl?: string;
  };
};

type StepMessage = {
  type: 'step';
  payload: {
    frame: ImageData;
    timestamp: number;
  };
};

type ClickMessage = {
  type: 'click';
  payload: {
    x: number;
    y: number;
  };
};

type WorkerMessage = InitMessage | StepMessage | ClickMessage;

// const ctx: Worker = self as any;
let status: 'idle' | 'inference' | 'calib' = 'idle';
// let lastTimestamp: number | null = null;

self.onmessage = async (e: MessageEvent) => {
  const data = e.data as WorkerMessage;

  switch (data.type) {
    case 'init':
      tracker = new WebEyeTrack(
          data.payload.maxPoints,
          data.payload.clickTTL,
          data.payload.modelUrl
      );
      await tracker.initialize();
      self.postMessage({ type: 'ready' });
      status = 'idle';
      break;

    case 'step':
      if (status === 'idle') {

        status = 'inference';
        self.postMessage({ type: 'statusUpdate', status: status});

        const result = await tracker.step(data.payload.frame as ImageData, data.payload.timestamp);
        // lastTimestamp = data.payload.timestamp;
        self.postMessage({ type: 'stepResult', result });

        status = 'idle';
        self.postMessage({ type: 'statusUpdate', status: status});
      }
      break;

    case 'click':
      // Handle click event for re-calibration
      status = 'calib';
      self.postMessage({ type: 'statusUpdate', status: status});

      tracker.handleClick(data.payload.x, data.payload.y);

      status = 'idle';
      self.postMessage({ type: 'statusUpdate', status: status});
      break;

    default:
      console.warn(`[WebEyeTrackWorker] Unknown message data: ${data}`);
      break;
  }
};

export {}; // for TS module mode
