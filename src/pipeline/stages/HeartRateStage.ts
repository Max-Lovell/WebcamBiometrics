import { HeartRateMonitor } from '../../rPPG';
import type { HeartRateMonitorConfig } from '../../rPPG';
import type {Stage, Blackboard} from '../types';

export class HeartRateStage implements Stage {
    readonly name = 'heart';
    readonly dependsOn = ['face'];
    private monitor: HeartRateMonitor;

    constructor(config?: Partial<HeartRateMonitorConfig>) {
        this.monitor = new HeartRateMonitor(config);
    }

    // Note this one isn't async
    process(ctx: Blackboard): void {
        if (!ctx.face?.detected) return;

        ctx.heart = this.monitor.processFrame(
            ctx.frame,
            ctx.face.faceLandmarkerResult,
            ctx.frameMetadata.videoTime,
        );
    }

    reset(): void {
        this.monitor.reset();
    }
}
