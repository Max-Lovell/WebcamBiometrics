/**
 * Default Pipeline Builder
 *
 * Note separate to BiometricsPipeline class so aren't forced to bundle TensorFlow.js or Mediapipe WASM files, etc. if not using them (Tree Shaking)
 * Pipeline is for running tasks, but is task agnostic/ignorant, factory knows which stages depend on which (face -> heart + gaze)
 * To add a stage, add a new flag to config and wire it up in factory.
 */

// TODO: what config is needed generally? Probably want options to config everything...
import { Pipeline } from './Pipeline.ts';
import { FaceLandmarkerStage } from './stages/FaceLandmarkerStage';
import { GazeStage } from './stages/GazeStage';
import { HeartRateStage } from './stages/HeartRateStage';

export interface DefaultPipelineConfig {
    face?: boolean; // | FaceStageConfig; //modelPathBasePath?: string; // e.g., '/public/models/'
    gaze?: boolean;// | GazeStageConfig;
    heart?: boolean;// | HeartRateStageConfig;
}

// Creates a pre-configured BiometricsPipeline with the requested stages.
// Handles instantiating the underlying AI models and wiring up dependencies.
export async function createDefaultPipeline(config: DefaultPipelineConfig = {}): Promise<Pipeline> {
    const pipeline = new Pipeline();

    // Default to true if not explicitly disabled
    const useGaze = config.gaze !== false;
    const useHeartRate = config.heart !== false;

    // const stages = [face, gaze, heart].filter(s => s.initialize);
    // await Promise.all(stages.map(s => s.initialize!()));

    // Face Landmarker is the foundation. If we need gaze or HR, we need the face.
    const initPromises: Promise<void>[] = [];

    const faceStage = new FaceLandmarkerStage();
    pipeline.addStage(faceStage);
    initPromises.push(faceStage.initialize());

    if (useGaze) {
        const gazeStage = new GazeStage();
        pipeline.addStage(gazeStage);
        initPromises.push(gazeStage.initialize());
    }

    if (useHeartRate) {
        pipeline.addStage(new HeartRateStage());
    }

    // Run init in parallel
    await Promise.all(initPromises);

    return pipeline;
}
