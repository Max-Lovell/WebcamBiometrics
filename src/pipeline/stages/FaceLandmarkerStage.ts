import FaceLandmarkerClient from '../../Core/FaceLandmarkerClient';
import type {Stage, Blackboard} from '../types';

export class FaceLandmarkerStage implements Stage {
    readonly name = 'face';
    readonly dependsOn: string[] = [];
    private client: FaceLandmarkerClient;

    constructor() {
        this.client = new FaceLandmarkerClient();
    }

    async initialize(): Promise<void> {
        await this.client.initialize();
    }

    async process(ctx: Blackboard): Promise<void> {
        const result = await this.client.processFrame(ctx.frame, ctx.frameMetadata.videoTime);
        const detected = !!(result?.faceLandmarks?.length);

        ctx.face = result
            ? { faceLandmarkerResult: result, detected }
            : undefined;
    }

    dispose(): void {
        this.client.dispose();
    }
}
