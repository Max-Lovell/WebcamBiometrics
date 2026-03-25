import FaceLandmarkerClient from '../../Core/FaceLandmarkerClient';
import type {Stage, Blackboard} from '../types';

export interface FaceLandmarkerStageConfig {
    wasmBasePath: string;
    modelPath: string;
}

export class FaceLandmarkerStage implements Stage {
    readonly name = 'face';
    readonly dependsOn: string[] = [];
    private client: FaceLandmarkerClient;
    // Hate storing these as class properties to use once - consider passing to static factory create() method, although requires BClient refactor...
    private readonly wasmBasePath: string;
    private readonly modelPath: string;

    constructor(config: FaceLandmarkerStageConfig) {
        this.client = new FaceLandmarkerClient();
        this.wasmBasePath = config.wasmBasePath;
        this.modelPath = config.modelPath;
    }

    async initialize(): Promise<void> {
        await this.client.initialize(this.wasmBasePath, this.modelPath);
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
