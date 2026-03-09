import * as tf from "@tensorflow/tfjs";
import { Matrix } from "ml-matrix";

import type { WebEyeTrackResult } from "./types.ts";
import type { Point } from "../types.ts";
import BlazeGaze from "./BlazeGaze.ts";
import { computeEAR } from "./utils/blink.ts";
import {computeEyeQuad} from "./utils/eyePatch.ts";
import {
  computeAffineMatrixML,
  applyAffineMatrix,
} from "./utils/affineTransformation.ts";
import {
  computeMetricFaceOrigin,
  createIntrinsicsMatrix,
  createPerspectiveMatrix,
  estimateFaceWidth,
  translateMatrix,
  getHeadVector,
} from "./utils/faceOrigin.ts";
import { KalmanFilter2D } from "./utils/filter.ts";
import { FrameConverter } from "./utils/frameUtils.ts";
import type {
  FaceLandmarkerResult,
  NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import type { VideoFrameData } from "../types.ts";
import {warpGPU} from "./utils/eyePatchWarp.ts";

// ============================================================================
// Support Tensor Types
// ============================================================================

interface SupportX {
  eyePatches: tf.Tensor;
  headVectors: tf.Tensor;
  faceOrigins3D: tf.Tensor;
}

// Convert JS objects into tensors - inputs for model learning/adaptation process
function generateSupport(
    eyePatches: tf.Tensor4D[],
    headVectors: number[][],
    faceOrigins3D: number[][],
    normPogs: number[][]
): { supportX: SupportX; supportY: tf.Tensor } {
  // tidy handles disposal of intermediate tensors from fromPixels/toFloat/div chain
  // TODO: does network expect [-1, 1]? e.g. .div(tf.scalar(127.5)).sub(tf.scalar(1.0))
  const batchPatches = tf.tidy(() =>
      tf.stack(
          eyePatches.map((patch) =>
              patch.squeeze([0])  // [1,H,W,3] → [H,W,3] for stacking
          ),
          0
      )
  );

  const supportX: SupportX = {
    eyePatches: batchPatches,
    headVectors: tf.tensor(headVectors, [headVectors.length, 3], "float32"),
    faceOrigins3D: tf.tensor(
        faceOrigins3D,
        [faceOrigins3D.length, 3],
        "float32"
    ),
  };

  const supportY = tf.tensor(normPogs, [normPogs.length, 2], "float32");
  return { supportX, supportY };
}

// Disposes all tensors inside a SupportX and its corresponding Y tensor.
function disposeSupportEntry(x: SupportX, y: tf.Tensor): void {
  tf.dispose([x.eyePatches, x.headVectors, x.faceOrigins3D, y]);
}

// ============================================================================
// Configuration
// ============================================================================

export interface WebEyeTrackOptions {
  // Max calibration points (typically 4 or 9). Default: 4.
  maxCalibPoints?: number;
  // Max clickstream points (FIFO + TTL). Default: 5.
  maxClickPoints?: number;
  // Time-to-live for click points in seconds. Default: 60.
  clickTTL?: number;
}

// ============================================================================
// WebEyeTrack
// ============================================================================

export default class WebEyeTrack {
  // Internal state
  private blazeGaze: BlazeGaze;
  private frameConverter = new FrameConverter();
  private kalmanFilter: KalmanFilter2D;
  private _disposed: boolean = false;

  // Camera matrices (lazily initialised on first frame)
  private perspectiveMatrix: Matrix | null = null;
  private intrinsicsMatrix: Matrix | null = null;

  // Face width (estimated once from iris ratio)
  private faceWidthCm: number = 13;
  private faceWidthComputed: boolean = false;

  // Affine correction matrix (computed after >=4 calibration points)
  private affineMatrix: tf.Tensor | null = null;

  // Public state
  public loaded: boolean = false;
  public latestMouseClick: {
    x: number;
    y: number;
    timestamp: number;
  } | null = null;
  public latestGazeResult: WebEyeTrackResult | null = null;
  private latestEyePatchTensor: tf.Tensor4D | null = null;

  // Separate buffers for calibration (persistent) vs clickstream (ephemeral) points
  public calibData: {
    // Persistent calibration buffer (never evicted by pruning)
    calibSupportX: SupportX[];
    calibSupportY: tf.Tensor[];
    calibTimestamps: number[];
    // Temporal clickstream buffer (TTL + FIFO eviction)
    clickSupportX: SupportX[];
    clickSupportY: tf.Tensor[];
    clickTimestamps: number[];
  } = {
    calibSupportX: [],
    calibSupportY: [],
    calibTimestamps: [],
    clickSupportX: [],
    clickSupportY: [],
    clickTimestamps: [],
  };

  // Configuration
  public maxCalibPoints: number;
  public maxClickPoints: number;
  public clickTTL: number;

  constructor(opts: WebEyeTrackOptions = {}) {
    this.blazeGaze = new BlazeGaze();
    this.kalmanFilter = new KalmanFilter2D();

    this.maxCalibPoints = opts.maxCalibPoints ?? 4;
    this.maxClickPoints = opts.maxClickPoints ?? 5;
    this.clickTTL = opts.clickTTL ?? 60;
  }

  // ==========================================================================
  // Lifecycle
  // ==========================================================================

  async initialize(modelPath?: string): Promise<void> {
    await this.blazeGaze.loadModel(modelPath);
    await this.warmup();
    this.loaded = true;
  }

  // Pre-warms the TensorFlow.js execution pipeline with dummy forward/backward
  // passes to compile WebGL shaders before first real usage.
  private async warmup(): Promise<void> {
    const warmupStart = performance.now();

    // Two iterations is enough to compile shaders for both forward and backward paths
    for (let iteration = 1; iteration <= 2; iteration++) {
      await tf.nextFrame();

      const dummyEyePatch = tf.randomUniform([1, 128, 512, 3], 0, 1);
      const dummyHeadVector = tf.randomUniform([1, 3], -1, 1);
      const dummyFaceOrigin3D = tf.randomUniform([1, 3], -100, 100);
      const dummyTarget = tf.randomUniform([1, 2], -0.5, 0.5);

      // Forward pass
      tf.tidy(() => {
        this.blazeGaze.predict(
            dummyEyePatch,
            dummyHeadVector,
            dummyFaceOrigin3D
        );
      });

      // Backward pass
      const opt = tf.train.adam(1e-5, 0.85, 0.9, 1e-8);
      tf.tidy(() => {
        const { grads } = tf.variableGrads(() => {
          const preds = this.blazeGaze.predict(
              dummyEyePatch,
              dummyHeadVector,
              dummyFaceOrigin3D
          );
          return tf.losses.meanSquaredError(dummyTarget, preds).asScalar();
        });
        opt.applyGradients(grads as Record<string, tf.Variable>);
      });

      opt.dispose();
      tf.dispose([dummyEyePatch, dummyHeadVector, dummyFaceOrigin3D, dummyTarget]);
    }

    console.log(
        `TensorFlow.js warmup complete in ${(performance.now() - warmupStart).toFixed(2)}ms`
    );
  }

  dispose(): void {
    if (this._disposed) return;
    this.latestEyePatchTensor?.dispose();
    this.latestEyePatchTensor = null;

    this.clearCalibrationBuffer();
    this.clearClickstreamPoints();

    if (this.affineMatrix) {
      tf.dispose(this.affineMatrix);
      this.affineMatrix = null;
    }

    this.blazeGaze.dispose();
    this._disposed = true;
  }

  get isDisposed(): boolean {
    return this._disposed;
  }

  // ==========================================================================
  // Buffer Management
  // ==========================================================================

  clearCalibrationBuffer(): void {
    this.calibData.calibSupportX.forEach((x, i) =>
        disposeSupportEntry(x, this.calibData.calibSupportY[i])
    );
    this.calibData.calibSupportX = [];
    this.calibData.calibSupportY = [];
    this.calibData.calibTimestamps = [];

    if (this.affineMatrix) {
      tf.dispose(this.affineMatrix);
      this.affineMatrix = null;
    }
  }

  clearClickstreamPoints(): void {
    this.calibData.clickSupportX.forEach((x, i) =>
        disposeSupportEntry(x, this.calibData.clickSupportY[i])
    );
    this.calibData.clickSupportX = [];
    this.calibData.clickSupportY = [];
    this.calibData.clickTimestamps = [];
  }

  resetAllBuffers(): void {
    this.clearCalibrationBuffer();
    this.clearClickstreamPoints();
  }

  // Prunes clickstream buffer by TTL and FIFO.
  // Calibration buffer is never pruned — points persist for the session.
  private pruneCalibData(): void {
    const currentTime = Date.now();
    const ttl = this.clickTTL * 1000;

    // TTL pruning
    const validIndices: number[] = [];
    this.calibData.clickTimestamps.forEach((timestamp, index) => {
      if (currentTime - timestamp <= ttl) {
        validIndices.push(index);
      } else {
        disposeSupportEntry(
            this.calibData.clickSupportX[index],
            this.calibData.clickSupportY[index]
        );
      }
    });

    this.calibData.clickSupportX = validIndices.map(
        (i) => this.calibData.clickSupportX[i]
    );
    this.calibData.clickSupportY = validIndices.map(
        (i) => this.calibData.clickSupportY[i]
    );
    this.calibData.clickTimestamps = validIndices.map(
        (i) => this.calibData.clickTimestamps[i]
    );

    // FIFO pruning
    if (this.calibData.clickSupportX.length > this.maxClickPoints) {
      const numToRemove =
          this.calibData.clickSupportX.length - this.maxClickPoints;

      for (let i = 0; i < numToRemove; i++) {
        disposeSupportEntry(
            this.calibData.clickSupportX[i],
            this.calibData.clickSupportY[i]
        );
      }

      this.calibData.clickSupportX =
          this.calibData.clickSupportX.slice(-this.maxClickPoints);
      this.calibData.clickSupportY =
          this.calibData.clickSupportY.slice(-this.maxClickPoints);
      this.calibData.clickTimestamps =
          this.calibData.clickTimestamps.slice(-this.maxClickPoints);
    }
  }

  // ==========================================================================
  // Click Handling
  // ==========================================================================

  handleClick(x: number, y: number): void {
    // Debounce: ignore clicks within 1s of the last
    if (
        this.latestMouseClick &&
        Date.now() - this.latestMouseClick.timestamp < 1000
    ) {
      return;
    }

    // Ignore clicks too close spatially to the last
    if (
        this.latestMouseClick &&
        Math.abs(x - this.latestMouseClick.x) < 0.05 &&
        Math.abs(y - this.latestMouseClick.y) < 0.05
    ) {
      return;
    }

    this.latestMouseClick = { x, y, timestamp: Date.now() };

    if (this.loaded && this.latestGazeResult && this.latestGazeResult.gazeState === "open" && this.latestEyePatchTensor) {
      // Don't await here - just allow click to be handled
      const eyePatchClone = this.latestEyePatchTensor.clone();
      this.adapt(
          [eyePatchClone],
          [this.latestGazeResult.headVector],
          [this.latestGazeResult.faceOrigin3D],
          [[x, y]],
          1,
          1e-5,
          "click"
      ).catch((e) => console.warn("Click adaptation failed:", e));
    }
  }

  // ==========================================================================
  // Input Preparation
  // ==========================================================================

  private ensureCameraMatrices(width: number, height: number): void {
    if (!this.perspectiveMatrix) {
      // 4x4 projection matrix mapping camera coords to NDC (w,h -> -1,1).
      // Used by faceReconstruction to unproject 2D landmarks back to 3D.
      this.perspectiveMatrix = createPerspectiveMatrix(width / height);
    }
    if (!this.intrinsicsMatrix) {
      // Camera intrinsics (K) describing focal length and optical center.
      // Maps 3D camera coordinates to 2D pixel coordinates.
      this.intrinsicsMatrix = createIntrinsicsMatrix(width, height);
    }
  }

  private prepareInput(
      frame: VideoFrameData,
      result: FaceLandmarkerResult
  ): [Point[] | null, number[], number[], ImageData] {
    const { imageData, width, height } = this.frameConverter.convert(frame);
    this.ensureCameraMatrices(width, height);

    const landmarks = result.faceLandmarks[0];
    const landmarks2d: Point[] = landmarks.map(
        (landmark: NormalizedLandmark) => ({
          x: Math.floor(landmark.x * width),
          y: Math.floor(landmark.y * height)
        })
    );

    const faceRT = translateMatrix(result.facialTransformationMatrixes[0]);
    const eyeQuad = computeEyeQuad(landmarks2d);

    if (!this.faceWidthComputed) {
      this.faceWidthCm = estimateFaceWidth(landmarks2d);
      this.faceWidthComputed = true;
    }

    const faceOrigin3D = computeMetricFaceOrigin(
        this.perspectiveMatrix!,
        this.intrinsicsMatrix!,
        landmarks.map((l: NormalizedLandmark) => [l.x, l.y] as [number, number]),
        faceRT,
        this.faceWidthCm,
        width,
        height,
        this.latestGazeResult?.faceOrigin3D?.[2] ?? 60
    );

    const headVector = getHeadVector(faceRT);

    return [eyeQuad, headVector, faceOrigin3D, imageData];
  }

  // ==========================================================================
  // Gaze Estimation (per-frame)
  // ==========================================================================

  // TODO: step() returns a promise but does synchronous GPU work internally,
  // blocking the worker thread. Need to make TF.js inference actually async.
  async step(
      frame: VideoFrameData,
      timestamp: number,
      result: FaceLandmarkerResult | null
  ): Promise<WebEyeTrackResult> {
    const tic1 = performance.now();

    if (!result?.faceLandmarks?.length) {
      return {
        headVector: [0, 0, 0],
        faceOrigin3D: [0, 0, 0],
        gazeState: "closed",
        normPog: [0, 0],
        durations: { blazeGaze: 0, kalmanFilter: 0, total: 0 },
        timestamp,
      };
    }

    const [eyeQuad, headVector, faceOrigin3D, imageData] = this.prepareInput(
        frame,
        result
    );
    const tic3 = performance.now();

    // Blink detection via Eye Aspect Ratio
    const leftEAR = computeEAR(result.faceLandmarks[0], "left");
    const rightEAR = computeEAR(result.faceLandmarks[0], "right");
    const gazeState: "open" | "closed" =
        leftEAR < 0.2 || rightEAR < 0.2 ? "closed" : "open";

    if (gazeState === "closed") {
      return {
        headVector,
        faceOrigin3D,
        gazeState,
        normPog: [0, 0],
        durations: { blazeGaze: 0, kalmanFilter: 0, total: tic3 - tic1 },
        timestamp,
      };
    }

    // GPU warp: frame → eye patch tensor directly, no CPU pixel copy
    const frameTensor = tf.browser.fromPixels(imageData);
    const warpResult = await warpGPU(frameTensor, eyeQuad!, 512, 128, false);
    frameTensor.dispose();

    if (!warpResult) {
      return {
        headVector,
        faceOrigin3D,
        gazeState,
        normPog: [0, 0],
        durations: { blazeGaze: 0, kalmanFilter: 0, total: performance.now() - tic1 },
        timestamp,
      };
    }

    // warpResult.tensor is already [1, 128, 512, 3] float32 [0,1]
    const [predNormPog, tic4] = tf.tidy(() => {
      const headVectorTensor = tf.tensor2d(headVector, [1, 3]);
      const faceOriginTensor = tf.tensor2d(faceOrigin3D, [1, 3]);

      let output = this.blazeGaze.predict(
          warpResult.tensor,
          headVectorTensor,
          faceOriginTensor
      );

      if (this.affineMatrix) {
        output = applyAffineMatrix(this.affineMatrix, output);
      }

      if (!output || output.shape.length === 0) {
        throw new Error("BlazeGaze model did not return valid output");
      }

      return [output, performance.now()];
    });


    const normPog = (await predNormPog.array()) as number[][];
    tf.dispose(predNormPog);

    const kalmanOutput = this.kalmanFilter.step(normPog[0]);
    kalmanOutput[0] = Math.max(-0.5, Math.min(0.5, kalmanOutput[0]));
    kalmanOutput[1] = Math.max(-0.5, Math.min(0.5, kalmanOutput[1]));

    this.latestEyePatchTensor?.dispose();
    this.latestEyePatchTensor = warpResult.tensor.clone();
    warpResult.tensor.dispose();

    const tic5 = performance.now();
    const gazeResult: WebEyeTrackResult = {
      headVector,
      faceOrigin3D,
      gazeState,
      normPog: kalmanOutput,
      durations: {
        blazeGaze: tic4 - tic3,
        kalmanFilter: tic5 - tic4,
        total: tic5 - tic1,
      },
      timestamp,
      // debug: {eyePatch: await this.getDebugEyePatch()}
    };

    this.latestGazeResult = gazeResult;
    warpResult.tensor.dispose();
    return gazeResult;
  }

  // ==========================================================================
  // MAML-Style Adaptation
  // ==========================================================================

  // MAML-style adaptation — fine-tunes the gaze model on calibration/click data.
  // TODO: consider persistent optimiser (avoids recreating momentum buffers each call,
  // but risks stale momentum from old postures)
  async adapt(
      eyePatches: tf.Tensor4D[],
      headVectors: number[][],
      faceOrigins3D: number[][],
      normPogs: number[][],
      stepsInner: number = 5,    // matches Python webeyetrack.py:324
      innerLR: number = 1e-5,    // matches Python webeyetrack.py:325
      ptType: "calib" | "click" = "calib"
  ): Promise<void> {
    // Prune old clickstream data (calibration buffer is never pruned)
    this.pruneCalibData();

    const opt = tf.train.adam(innerLR, 0.85, 0.9, 1e-8);

    try {
      console.log(eyePatches)
      const { supportX, supportY } = generateSupport(
          eyePatches,
          headVectors,
          faceOrigins3D,
          normPogs
      );

      // Add to appropriate buffer
      if (ptType === "calib") {
        this.calibData.calibSupportX.push(supportX);
        this.calibData.calibSupportY.push(supportY);
        this.calibData.calibTimestamps.push(Date.now());
      } else {
        this.calibData.clickSupportX.push(supportX);
        this.calibData.clickSupportY.push(supportY);
        this.calibData.clickTimestamps.push(Date.now());
      }

      // Enforce total limit. Eviction strategy: prefer evicting old CALIBRATION data.
      // New correction clicks (reflecting current posture) should push out old calibration
      // points (which might reflect an old posture) — prevents "elastic banding".
      // TODO: implement auto-reset on movement for high-confidence calibration points
      const TOTAL_LIMIT = this.maxCalibPoints + this.maxClickPoints;
      while (
          this.calibData.calibSupportX.length +
          this.calibData.clickSupportX.length >
          TOTAL_LIMIT
          ) {
        if (this.calibData.calibSupportX.length > 0) {
          disposeSupportEntry(
              this.calibData.calibSupportX.shift()!,
              this.calibData.calibSupportY.shift()!
          );
          this.calibData.calibTimestamps.shift();
        } else {
          disposeSupportEntry(
              this.calibData.clickSupportX.shift()!,
              this.calibData.clickSupportY.shift()!
          );
          this.calibData.clickTimestamps.shift();
        }
      }

      // Concatenate all buffers for training
      const allSupportX = [
        ...this.calibData.calibSupportX,
        ...this.calibData.clickSupportX,
      ];
      const allSupportY = [
        ...this.calibData.calibSupportY,
        ...this.calibData.clickSupportY,
      ];

      let tfEyePatches: tf.Tensor;
      let tfHeadVectors: tf.Tensor;
      let tfFaceOrigins3D: tf.Tensor;
      let tfSupportY: tf.Tensor;
      let needsDisposal: boolean;

      if (allSupportX.length > 1) {
        tfEyePatches = tf.concat(
            allSupportX.map((s) => s.eyePatches),
            0
        );
        tfHeadVectors = tf.concat(
            allSupportX.map((s) => s.headVectors),
            0
        );
        tfFaceOrigins3D = tf.concat(
            allSupportX.map((s) => s.faceOrigins3D),
            0
        );
        tfSupportY = tf.concat(allSupportY, 0);
        needsDisposal = true;
      } else {
        tfEyePatches = supportX.eyePatches;
        tfHeadVectors = supportX.headVectors;
        tfFaceOrigins3D = supportX.faceOrigins3D;
        tfSupportY = supportY;
        needsDisposal = false;
      }

      // Compute affine transformation (requires >=4 points — affine has 6 DOF:
      // 2 scale, 2 rotation/shear, 2 translation)
      if (tfEyePatches.shape[0] > 3) {
        const supportPreds = tf.tidy(() =>
            this.blazeGaze.predict(tfEyePatches, tfHeadVectors, tfFaceOrigins3D)
        );
        const supportPredsNumber = (await supportPreds.array()) as number[][];
        const supportYNumber = (await tfSupportY.array()) as number[][];
        tf.dispose(supportPreds);

        try {
          const affineMatrixML = computeAffineMatrixML(
              supportPredsNumber,
              supportYNumber
          );
          if (this.affineMatrix) {
            tf.dispose(this.affineMatrix);
          }
          this.affineMatrix = tf.tensor2d(affineMatrixML, [2, 3], "float32");
        } catch (e) {
          // Swallow error so the tracker keeps running with the previous valid matrix
          console.warn(
              "Skipping affine update: calibration points likely too similar (rank deficient).",
              e
          );
        }
      }

      // MAML-style gradient steps on the gaze MLP (CNN encoder is frozen)
      tf.tidy(() => {
        for (let i = 0; i < stepsInner; i++) {
          const { grads, value: loss } = tf.variableGrads(() => {
            const preds = this.blazeGaze.predict(
                tfEyePatches,
                tfHeadVectors,
                tfFaceOrigins3D
            );
            const predsTransformed = this.affineMatrix
                ? applyAffineMatrix(this.affineMatrix, preds)
                : preds;
            return tf.losses
                .meanSquaredError(tfSupportY, predsTransformed)
                .asScalar();
          });
          opt.applyGradients(grads as Record<string, tf.Variable>);
          // Defensive dispose — tidy should handle this, but gradients are
          // returned as a NamedTensorMap which can confuse tidy's tracking
          Object.values(grads).forEach((g) => g.dispose());
          loss.dispose();
        }
      });

      // Only dispose if we created new concatenated tensors (not buffer references)
      if (needsDisposal) {
        tf.dispose([tfEyePatches, tfHeadVectors, tfFaceOrigins3D, tfSupportY]);
      }
    } finally {
      // Optimizer creates internal variables (momentum buffers, variance accumulators)
      // that persist until explicitly disposed
      opt.dispose();
    }
  }

  async getDebugEyePatch(): Promise<ImageData | null> {
    if (!this.latestEyePatchTensor) return null;
    const squeezed = this.latestEyePatchTensor.squeeze([0]) as tf.Tensor3D;
    const pixels = await tf.browser.toPixels(squeezed);
    squeezed.dispose();
    return new ImageData(new Uint8ClampedArray(pixels), 512, 128);
  }
}
