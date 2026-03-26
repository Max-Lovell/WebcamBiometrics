import WebEyeTrack from '../../webeyetrack/WebEyeTrack';
import type {Stage, Blackboard} from '../types';

export interface GazeStageConfig {
    maxPoints?: number;
    clickTTL?: number;
    maxCalibPoints?: number;
    maxClickPoints?: number;
    modelPath: string;
}

export class GazeStage implements Stage {
    readonly name = 'gaze';
    readonly dependsOn = ['face'];

    // Public so the worker/proxy can access calibration methods:
    //   gazeStage.tracker.handleClick(x, y)
    //   gazeStage.tracker.adapt(...)
    //   gazeStage.tracker.clearCalibrationBuffer()
    readonly tracker: WebEyeTrack;
    private modelPath: string;

    constructor(config: GazeStageConfig) {
        this.tracker = new WebEyeTrack(
            {
                maxCalibPoints: config.maxCalibPoints,
                maxClickPoints: config.maxClickPoints,
                clickTTL: config.clickTTL,
            });
        this.modelPath = config.modelPath;
    }

    async initialize(): Promise<void> {
        await this.tracker.initialize(this.modelPath);
    }

    async process(ctx: Blackboard): Promise<void> {
        // WebEyeTrack.step() handles the no-face case internally,
        // so we pass the result through even when face is not detected.
        ctx.gaze = await this.tracker.step(
            ctx.frame,
            ctx.frameMetadata.videoTime,
            ctx.face?.faceLandmarkerResult ?? null // TODO: consider handling no-face here instead and skipping step()
        );
    }

    dispose(): void {
        this.tracker.dispose();
    }

    handleCommand(command: string, payload: any): unknown {
        // Helpers to call the functions in WebEyeTrack.ts, implements functionality beyond generic process() call
        switch (command) {
            case 'click':
                return this.tracker.handleClick(payload.x, payload.y);
            case 'adapt':
                return this.tracker.adapt(
                    payload.eyePatches, payload.headVectors,
                    payload.faceOrigins3D, payload.normPogs,
                    payload.stepsInner, payload.innerLR, payload.ptType,
                );
            case 'clearCalibration':
                return this.tracker.clearCalibrationBuffer();
            case 'clearClickstream':
                return this.tracker.clearClickstreamPoints();
            case 'resetAllBuffers':
                return this.tracker.resetAllBuffers();
            default:
                throw new Error(`GazeStage: unknown command "${command}"`);
        }
    }
}
