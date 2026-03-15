import { MiscProcessor } from '../../Misc';
import type { Stage, Blackboard } from '../types';

export class MiscStage implements Stage {
    readonly name = 'misc';
    readonly dependsOn = ['face'];

    private processor = new MiscProcessor();

    async process(ctx: Blackboard): Promise<void> {
        if (!ctx.face?.detected) return;
        ctx.misc = this.processor.process(ctx.frame, ctx.frameMetadata, ctx.face);
    }

    dispose(): void {
        this.processor.dispose();
    }
}
