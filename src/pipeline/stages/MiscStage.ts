import { MiscProcessor } from '../../misc';
import type { Stage, Blackboard } from '../types';

export class MiscStage implements Stage {
    readonly name = 'misc';
    readonly dependsOn = ['face'];

    private processor = new MiscProcessor();

    async process(ctx: Blackboard): Promise<void> {
        if (!ctx.face?.detected) return;
        ctx.misc = this.processor.process(ctx.frame, ctx.face); //  ctx.frameMetadata,
    }

    dispose(): void {
        this.processor.dispose();
    }
}
