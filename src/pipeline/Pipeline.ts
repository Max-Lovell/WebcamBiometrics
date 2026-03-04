/**
 * BiometricsPipeline
 *
 * Step after the 'Worker' which calls the Stages files.
 * Runs in the worker and calls the process function from each of the stages and returns the result.
 * Receives the frame and metadata and runs all registered stages (respecting dependencies), e.g. FaceLandmarker first, then Gaze, HeartRate, in parallel
 * When all stages finish, strips the frame from the blackboard and returns the `BiometricsResult`.
 * Technically designed as a 'DAG executor'
 *
 *  Example:
 * const pipeline = new BiometricsPipeline()
 * .addStage(faceLandmarkerStage)   // dependsOn: []
 * .addStage(gazeStage)             // dependsOn: ['faceLandmarkerStage']
 * .addStage(heartRateStage);       // dependsOn: ['faceLandmarkerStage']
 *
 * // gazeStage and heartRateStage run in parallel after faceLandmarkerStage
 * const ctx = await pipeline.processFrame(frame, metadata);
 *
 * // TO TEST:
 * const log: string[] = [];
 *
 * const faceStage: Stage = {
 *     name: 'face',
 *     dependsOn: [],
 *     async process() {
 *         log.push('face_start');
 *         await sleep(50); // simulate work
 *         log.push('face_end');
 *     }
 * };
 *
 * const gazeStage: Stage = {
 *     name: 'gaze',
 *     dependsOn: ['face'],
 *     async process() {
 *         log.push('gaze_start');
 *         await sleep(30);
 *         log.push('gaze_end');
 *     }
 * };
 *
 * Overengineered a bit, isn't it? No idea why I bothered doing this other than curiosity. May revert to more obvious system.
 *  But, if someone wants to add something other than heart and eyetracking, at least it'll work now out of the box.
 */

// Stages declare dependencies, pipeline resolves graph, runs independent stages in parallel.
import type {BiometricsResult, Stage, FrameMetadata, Blackboard} from './types';
import type { VideoFrameData } from '../types';

export class Pipeline {
    private stages: Stage[] = [];
    private stageMap: Map<string, Stage> = new Map();
    private needsValidation: boolean = false;

    // TODO: handle async initialisation: stages.filter(s => s.initialize); await Promise.all(stages.map(s => s.initialize!()));

    // Register stage - can be added in any order, but dependencies must be registered.
    addStage(stage: Stage): this {
        if (this.stageMap.has(stage.name)) {
            throw new Error(`Stage "${stage.name}" is already registered`);
        }
        this.stages.push(stage);
        this.stageMap.set(stage.name, stage);
        this.needsValidation = true;
        return this; // Returns `this` for chaining: pipeline.addStage(a).addStage(b)
    }

    getStage(name: string): Stage | undefined {
        return this.stageMap.get(name);
    }

    removeStage(name: string): this {
        // Check no other stage depends on this one
        for (const stage of this.stages) {
            if (stage.dependsOn.includes(name)) {
                throw new Error( // Throws error if other stages depend on this one
                    `Cannot remove stage "${name}": stage "${stage.name}" depends on it`
                );
            }
        }
        this.stages = this.stages.filter(s => s.name !== name);
        this.stageMap.delete(name);
        return this;
    }

    // Process frame through registered stages,
    async processFrame(
        frame: VideoFrameData,
        frameMetadata: FrameMetadata // TODO: or just exclude it above, should be optional anyway.
    ): Promise<BiometricsResult> {
        // Technically this is "a recursive function + memoisation map approach to DAG execution"
        // Note this approach is complicated to understand, but does allow developers to plug in a new custom stage pretty easily.
        // (e.g., BlinkDetector, EmotionAnalyzer, YawnCounter) and let the pipeline figure out the execution order automatically.
        // Simpler version of this function would be e.g. if (stages.gaze) parallelTasks.push(stages.gaze.process(blackboard)); await Promise.all(parallelTasks);

        // Validate all dependencies exist before executing
        // e.g. if heartrate depends on facelandmarker and that isn't registered, then error to user.
        if (this.needsValidation) { // TODO: still iterates all stages × all deps every time but fine for now...
            this.validateDependencies();
            this.needsValidation = false;
        }

        // Initialize the Blackboard with the frame and metadata
        const ctx: Blackboard = { frame, frameMetadata };
        ctx.frameMetadata.trace = ctx.frameMetadata.trace ?? [] // init if missing
        ctx.frameMetadata.trace.push({ step: 'pipeline_start', timestamp: performance.now() });

        // Memoised empty promise Map — ensure each stage runs only once per frame
        // e.g. Gaze and Heart Rate depend on Face, but don't want running twice
        const promises = new Map<string, Promise<void>>(); // Note: Map is insertion-ordered list of key:value pairs

        const run = (stage: Stage): Promise<void> => {
            // Return existing promise if already started (memoisation, avoids duplicates)
            // check this stage's promise is already in the map. If it is, return it.
            const existing = promises.get(stage.name);
            if (existing) return existing;

            // When a stage is triggered, await all dependency promises (which may already be resolved)
            // Build the promise: wait for deps, then execute
            const depPromises = stage.dependsOn.map(depName => {
                // For a stage, get the dependencies (dependsOn is array of names) and run this function on them.
                const dep = this.stageMap.get(depName)!; // validated above
                return run(dep); // Run returns a promise that will resolve and run app deps at the same time
            });

            // Independent stages get their own Promise.all chains and run concurrently.
            // map over dependencies and recursively call run() on them, wrapping them in a Promise.all()
                // Note if dependsOn/depPromises is empty [], resolves immediately to .then() and stores in map under stage name.
                // But e.g. if dependsOn is ['face'] and `run(face)` was already called memoisation check returns existing promise in map, e.g. depPromises = [facePromise]
                // so .then() won't fire until that face promise resolves. The gaze promise gets stored in the map.
            // TODO: note logging trace times, looks like GazeStage.process is async and tracker.step returns a promise which involves synchronous GPU work so event loop blocked and heart actually starts after.
            const promise = Promise.all(depPromises).then(async () => {
                ctx.frameMetadata.trace!.push({ step: `${stage.name}_start`, timestamp: performance.now() });
                try { // TODO: stages that depend on failed stage could skip rather than throw or check ctx.errors map or blackboard for deps
                    await stage.process(ctx);  // async run process function from stage
                } catch (err) { // TODO: stage.reset() on face detection fail?
                    ctx.errors ??= {};
                    ctx.errors[stage.name] = String(err);
                }
                ctx.frameMetadata.trace!.push({ step: `${stage.name}_end`, timestamp: performance.now() });
            });

            promises.set(stage.name, promise);
            return promise;
        };

        // Kick off all stages — the DAG resolves execution order automatically - Map all registered stages to the run function and add to Promise.all().
        // Recursive promise resolution - independent stages (Gaze/Heart Rate) will automatically execute concurrently once face is ready
        // e.g. calls run(face), run(gaze), run(heart)
        await Promise.all(this.stages.map(run));

        ctx.frameMetadata.trace!.push({ step: 'pipeline_end', timestamp: performance.now() });

        // Extract BiometricsResult and completely drop the frame from the output
        const { frame: _dropFrame, ...result } = ctx;

        // Automatically close the frame if it is a VideoFrame or ImageBitmap to prevent GPU memory leaks
        if (_dropFrame && typeof (_dropFrame as any).close === 'function') {
            (_dropFrame as any).close();
        }

        return result;
    }

    // Reset all stages (e.g., on session change or face lost).
    reset(): void {
        for (const stage of this.stages) {
            stage.reset?.();
        }
    }

    // Dispose all stages and clear the pipeline.
    dispose(): void {
        for (const stage of this.stages) {
            stage.dispose?.();
        }
        this.stages = [];
        this.stageMap.clear();
    }

    // Validate that all declared dependencies exist as registered stages. Called once per processFrame to fail fast on misconfiguration.
    private validateDependencies(): void {
        for (const stage of this.stages) {
            for (const dep of stage.dependsOn) {
                if (!this.stageMap.has(dep)) {
                    throw new Error(
                        `Stage "${stage.name}" depends on "${dep}", which is not registered. ` +
                        `Registered stages: [${[...this.stageMap.keys()].join(', ')}]`
                    );
                }
            }
        }
    }
}
