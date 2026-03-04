import * as tf from '@tensorflow/tfjs';
import {Matrix} from 'ml-matrix';

import type {Point, WebEyeTrackResult} from "./types.ts";
import BlazeGaze from "./BlazeGaze.ts";
import {
  applyAffineMatrix,
  computeAffineMatrixML,
  computeEAR,
  computeFaceOrigin3D,
  createIntrinsicsMatrix,
  createPerspectiveMatrix,
  estimateFaceWidth,
  faceReconstruction,
  getHeadVector,
  obtainEyePatch,
  translateMatrix,
} from "./utils/mathUtils.ts";
import {KalmanFilter2D} from "./utils/filter.ts";
import type {FaceLandmarkerResult, NormalizedLandmark} from "@mediapipe/tasks-vision";
import type {VideoFrameData} from "../types.ts";

// Reference
// https://mediapipe-studio.webapps.google.com/demo/face_landmarker
// TODO: This needs heavy fixing - very inefficient
interface SupportX {
  eyePatches: tf.Tensor;
  headVectors: tf.Tensor;
  faceOrigins3D: tf.Tensor;
}

function generateSupport(
    eyePatches: ImageData[],
    headVectors: number[][],
    faceOrigins3D: number[][],
    normPogs: number[][]
): { supportX: SupportX, supportY: tf.Tensor } {
  // Convert JS objects into tensors - inputs for model learning/adaptation process

  // Implementation for generating support samples
  return tf.tidy(() => {
    const batchPatches = tf.stack( // creates new combined tensor from pixel tensors below
        eyePatches.map(patch => tf.browser.fromPixels(patch)), 0) // fromPixels creates new tensor for every image patch in loop.
        .toFloat() // toFloat, div, scalar also create intermediate tensors
        .div(tf.scalar(255.0)); // TODO: does network expect [-1, 1]? e.g. .div(tf.scalar(127.5)).sub(tf.scalar(1.0));

    const supportX: SupportX = {
      eyePatches: batchPatches,
      headVectors: tf.tensor(headVectors, [headVectors.length, 3], 'float32'),
      faceOrigins3D: tf.tensor(faceOrigins3D, [faceOrigins3D.length, 3], 'float32')
    };

    const supportY = tf.tensor(normPogs, [normPogs.length, 2], 'float32');

    // Note this double casting seems a bit hacky but appeases typescript and seems to work...
    return { supportX, supportY } as any; // ScopeFn<TensorContainer> requirement
  }) as { supportX: SupportX, supportY: tf.Tensor }; // restore strict typing
}

export default class WebEyeTrack {

  // Instance variables
  private blazeGaze: BlazeGaze;
  private faceWidthCm: number = 13;
  private faceWidthComputed: boolean = false;
  private perspectiveMatrixSet: boolean = false;
  private perspectiveMatrix: Matrix = new Matrix(4, 4);
  private intrinsicsMatrixSet: boolean = false;
  private intrinsicsMatrix: Matrix = new Matrix(3, 3);
  private affineMatrix: tf.Tensor | null = null;
  private kalmanFilter: KalmanFilter2D;
  private _disposed: boolean = false;

  // Public variables
  public loaded: boolean = false;
  public latestMouseClick: { x: number, y: number, timestamp: number } | null = null;
  public latestGazeResult: WebEyeTrackResult | null = null;

  // private smoothedFaceOrigin: number[] = [0, 0, 60]; // Default start guess
  // private originAlpha: number = 0.1; // Smoothing factor (0.1 = heavy smoothing, 0.9 = reactive)
  // private widthAlpha: number = 0.05; // Smoothing factor (lower = smoother but slower to adapt)

  // Separate buffers for calibration (persistent) vs clickstream (ephemeral) points
  public calibData: {
    // === PERSISTENT CALIBRATION BUFFER (never evicted) ===
    calibSupportX: SupportX[],
    calibSupportY: tf.Tensor[],
    calibTimestamps: number[],

    // === TEMPORAL CLICKSTREAM BUFFER (TTL + FIFO eviction) ===
    clickSupportX: SupportX[],
    clickSupportY: tf.Tensor[],
    clickTimestamps: number[],
  } = {
    calibSupportX: [],
    calibSupportY: [],
    calibTimestamps: [],
    clickSupportX: [],
    clickSupportY: [],
    clickTimestamps: [],
  };

  // Configuration
  public maxCalibPoints: number = 4;    // Max calibration points (4-point or 9-point calibration)
  public maxClickPoints: number = 5;    // Max clickstream points (FIFO + TTL)
  public clickTTL: number = 60;         // Time-to-live for click points in seconds

  // Cached canvas for efficient VideoFrame -> ImageData conversion
  private tempCanvas: OffscreenCanvas | HTMLCanvasElement | null = null;
  private tempCtx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null = null;

  constructor(
      maxPoints: number = 5,              // Deprecated: use maxClickPoints instead
      clickTTL: number = 60,              // Time-to-live for click points in seconds
      maxCalibPoints?: number,            // Max calibration points (4 or 9 typically)
      maxClickPoints?: number             // Max clickstream points
    ) {

    // Initialize services
    this.blazeGaze = new BlazeGaze();
    this.kalmanFilter = new KalmanFilter2D();

    // Storing configs with backward compatibility
    this.maxCalibPoints = maxCalibPoints ?? 4;           // Default: 4-point calibration
    this.maxClickPoints = maxClickPoints ?? maxPoints;   // Use maxClickPoints if provided, else maxPoints
    this.clickTTL = clickTTL;
  }

  private getTempContext(width: number, height: number) {
    if (!this.tempCanvas) {
      if (typeof OffscreenCanvas !== 'undefined') {
        this.tempCanvas = new OffscreenCanvas(width, height);
      } else {
        // Fallback if worker doesn't support OffscreenCanvas (rare)
        this.tempCanvas = document.createElement('canvas');
      }
    }

    // Ensure dimensions match - resize if changed
    if (this.tempCanvas.width !== width || this.tempCanvas.height !== height) {
      this.tempCanvas.width = width;
      this.tempCanvas.height = height;
      // Invalidate context if we resized (safest approach)
      this.tempCtx = null;
    }

    if(!this.tempCtx){
      this.tempCtx = this.tempCanvas.getContext('2d', { willReadFrequently: true }) as CanvasRenderingContext2D;
    }

    return this.tempCtx;
  }

  async initialize(modelPath?: string): Promise<void> {
    await this.blazeGaze.loadModel(modelPath);
    await this.warmup(); //TODO - consider ditching this approach?
    this.loaded = true;
  }

  /**
   * Pre-warms TensorFlow.js execution pipeline by running dummy forward/backward passes.
   * This compiles WebGL shaders and optimizes computation graphs before first real usage.
   */
  async warmup(): Promise<void> {
    // TODO: check this is correct... maybe delete?
    const warmupStart = performance.now();
    // Warmup iterations match total buffer capacity to exercise all code paths
    const numWarmupIterations = this.maxCalibPoints + this.maxClickPoints;

    for (let iteration = 1; iteration <= numWarmupIterations; iteration++) {
      await tf.nextFrame(); // Yield to prevent blocking

      const iterationStart = performance.now();

      // Create dummy tensors matching expected shapes
      // Eye patch: ImageData(width=512, height=128) -> tensor [batch, height, width, channels]
      const dummyEyePatch = tf.randomUniform([1, 128, 512, 3], 0, 1); // [batch, H=128, W=512, channels]
      const dummyHeadVector = tf.randomUniform([1, 3], -1, 1);
      const dummyFaceOrigin3D = tf.randomUniform([1, 3], -100, 100);
      const dummyTarget = tf.randomUniform([1, 2], -0.5, 0.5);

      // Warmup forward pass
      tf.tidy(() => {
        this.blazeGaze.predict(dummyEyePatch, dummyHeadVector, dummyFaceOrigin3D);
      });

      // Warmup backward pass (gradient computation)
      const opt = tf.train.adam(1e-5, 0.85, 0.9, 1e-8);
      tf.tidy(() => {
        const { grads } = tf.variableGrads(() => {
          const preds = this.blazeGaze.predict(dummyEyePatch, dummyHeadVector, dummyFaceOrigin3D);
          const loss = tf.losses.meanSquaredError(dummyTarget, preds);
          return loss.asScalar();
        });
        opt.applyGradients(grads as Record<string, tf.Variable>);
      });

      // Warmup affine matrix computation path (kicks in at iteration 4)
      if (iteration >= 4) {
        tf.tidy(() => {
          // Simulate multiple calibration points [batch, H=128, W=512, channels]
          const multiEyePatches = tf.randomUniform([iteration, 128, 512, 3], 0, 1);
          const multiHeadVectors = tf.randomUniform([iteration, 3], -1, 1);
          const multiFaceOrigins3D = tf.randomUniform([iteration, 3], -100, 100);

          const preds = this.blazeGaze.predict(multiEyePatches, multiHeadVectors, multiFaceOrigins3D);

          // Trigger affine transformation path
          if (this.affineMatrix) {
            applyAffineMatrix(this.affineMatrix, preds);
          }
        });
      }

      // Clean up iteration tensors
      opt.dispose();
      tf.dispose([dummyEyePatch, dummyHeadVector, dummyFaceOrigin3D, dummyTarget]);

      const iterationTime = performance.now() - iterationStart;
      console.log(`  Iteration ${iteration}/${numWarmupIterations}: ${iterationTime.toFixed(2)}ms`);
    }

    console.log(`TensorFlow.js warmup complete in ${(performance.now() - warmupStart).toFixed(2)}ms`);
  }

  /**
   * Clears the calibration buffer and resets affine matrix.
   * Call this when starting a new calibration session (e.g., user clicks "Calibrate" button again).
   * Properly disposes all calibration tensors to prevent memory leaks.
   */
  clearCalibrationBuffer() {
    console.log('Clearing calibration buffer');
    // Dispose all calibration tensors
    this.calibData.calibSupportX.forEach(item => tf.dispose([item.eyePatches, item.headVectors, item.faceOrigins3D]));
    this.calibData.calibSupportY.forEach(tensor => tf.dispose(tensor));
    this.calibData.calibSupportX = [];
    this.calibData.calibSupportY = [];
    this.calibData.calibTimestamps = [];
    // Reset affine matrix (will be recomputed with new calibration)
    if (this.affineMatrix) {
      tf.dispose(this.affineMatrix);
      this.affineMatrix = null;
    }
  }

  /**
   * Clears the clickstream buffer while preserving calibration points.
   * Use this to remove stale clickstream data without affecting calibration.
   * Properly disposes all clickstream tensors to prevent memory leaks.
   *
   * @example
   * // Clear stale clicks while keeping calibration
   * tracker.clearClickstreamPoints();
   */
  clearClickstreamPoints() {
    console.log('Clearing clickstream buffer');
    this.calibData.clickSupportX.forEach(item => tf.dispose([item.eyePatches, item.headVectors, item.faceOrigins3D]));
    this.calibData.clickSupportY.forEach(tensor => tf.dispose(tensor));
    this.calibData.clickSupportX = [];
    this.calibData.clickSupportY = [];
    this.calibData.clickTimestamps = [];
  }

  /**
   * Resets both calibration and clickstream buffers for a completely fresh start.
   * This is the recommended method to call when initiating re-calibration.
   * Properly disposes all tensors and resets affine matrix.
   *
   * @example
   * // User clicks "Recalibrate" button
   * tracker.resetAllBuffers();
   * tracker.adapt(...); // Start fresh calibration
   */
  resetAllBuffers() {
    this.clearCalibrationBuffer();
    this.clearClickstreamPoints();
  }

  /**
   * Prunes the clickstream buffer based on TTL and maxClickPoints.
   * Calibration buffer is NEVER pruned - calibration points persist for the entire session.
   */
  pruneCalibData() {
    // === CALIBRATION BUFFER: No pruning ===
    // Calibration points are permanent and never evicted
    // Overflow is handled in adapt() method with user-visible error

    // === CLICKSTREAM BUFFER: TTL + FIFO pruning ===
    const currentTime = Date.now();
    const ttl = this.clickTTL * 1000;

    // Step 1: Remove expired click points (TTL pruning)
    const validIndices: number[] = [];
    const expiredIndices: number[] = [];

    this.calibData.clickTimestamps.forEach((timestamp, index) => {
      if (currentTime - timestamp <= ttl) {
        validIndices.push(index);
      } else {
        expiredIndices.push(index);
      }
    });

    // Dispose expired tensors
    expiredIndices.forEach(index => {
      const item = this.calibData.clickSupportX[index];
      tf.dispose([item.eyePatches, item.headVectors, item.faceOrigins3D]);
      tf.dispose(this.calibData.clickSupportY[index]);
    });

    // Filter to keep only non-expired clicks
    this.calibData.clickSupportX = validIndices.map(i => this.calibData.clickSupportX[i]);
    this.calibData.clickSupportY = validIndices.map(i => this.calibData.clickSupportY[i]);
    this.calibData.clickTimestamps = validIndices.map(i => this.calibData.clickTimestamps[i]);

    // Step 2: Apply FIFO if still over maxClickPoints
    if (this.calibData.clickSupportX.length > this.maxClickPoints) {
      // Calculate how many to remove
      const numToRemove = this.calibData.clickSupportX.length - this.maxClickPoints;

      // Dispose oldest click tensors
      const itemsToRemove = this.calibData.clickSupportX.slice(0, numToRemove);
      itemsToRemove.forEach(item => {
        tf.dispose([item.eyePatches, item.headVectors, item.faceOrigins3D]);
      });

      const tensorsToRemove = this.calibData.clickSupportY.slice(0, numToRemove);
      tensorsToRemove.forEach(tensor => {
        tf.dispose(tensor);
      });

      // Keep only last maxClickPoints
      this.calibData.clickSupportX = this.calibData.clickSupportX.slice(-this.maxClickPoints);
      this.calibData.clickSupportY = this.calibData.clickSupportY.slice(-this.maxClickPoints);
      this.calibData.clickTimestamps = this.calibData.clickTimestamps.slice(-this.maxClickPoints);
    }
  }

  handleClick(x: number, y: number) {
    console.log(`🖱️ Global click at: (${x}, ${y}), ${this.loaded}`);

    // Remove time/space close clicks - but only for most recent actually used click!
    // Debounce clicks based on the latest click timestamp
    if (this.latestMouseClick && (Date.now() - this.latestMouseClick.timestamp < 1000)) {
      console.log("🖱️ Click ignored due to debounce");
      return;
    }

    // Avoid pts that are too close to the last click
    if (this.latestMouseClick &&
        Math.abs(x - this.latestMouseClick.x) < 0.05 &&
        Math.abs(y - this.latestMouseClick.y) < 0.05) {
        console.log("🖱️ Click ignored due to proximity to last click");
        return;
    }

    this.latestMouseClick = { x, y, timestamp: Date.now() };

    if (this.loaded && this.latestGazeResult) {
      // Adapt the model based on the click position
      // Use Python default parameters (main.py:183-185) for click calibration
      this.adapt(
        [this.latestGazeResult.eyePatch],
        [this.latestGazeResult.headVector],
        [this.latestGazeResult.faceOrigin3D],
        [[x, y]],
          1,      // stepsInner: matches Python main.py:183
        1e-5,    // innerLR: matches Python main.py:184
        'click'  // ptType: matches Python main.py:185
      );
    }
  }


  computeFaceOrigin3D(frame: ImageData, normFaceLandmarks: Point[], faceLandmarks: Point[], faceRT: Matrix): number[] {
    // Re-estimate face width to stop wobble
    // const currentFaceWidth = estimateFaceWidth(faceLandmarks);

    if (!this.faceWidthComputed) {
      this.faceWidthCm = estimateFaceWidth(faceLandmarks);
      this.faceWidthComputed = true;
    }

    // Perform 3D face reconstruction and determine the pose in 3d cm space
    // Metric Solving - force 3D model into cm units and predict angle of gaze.
    const [_, metricFace] = faceReconstruction(
      this.perspectiveMatrix,
      normFaceLandmarks as [number, number][],
      faceRT,
      this.intrinsicsMatrix,
      this.faceWidthCm,
      frame.width,
      frame.height,
      this.latestGazeResult?.faceOrigin3D?.[2] ?? 60
    );

    return computeFaceOrigin3D(metricFace); // Note this isnt recursion - calls the import from mathUtils.
  }

  createNewRegion(frame: VideoFrameData, result: FaceLandmarkerResult) {
    let width, height;
    if (frame instanceof VideoFrame) {
      width = frame.displayWidth;
      height = frame.displayHeight;
    } else {
      width = frame.width;
      height = frame.height;
    }

    // -- Get landmarks
    const landmarks = result.faceLandmarks[0]
    // right 34 (127?), left 264 (356). Move in
    // TODO: don't center on noseBridge, but on midpoint between eyes

    // Get midpoint between eyes
    // 133, 362
    landmarks[133] // right
    landmarks[362] // left
    const anchorLandmarks = [landmarks[168], landmarks[264], landmarks[34]]; // center, left, right - use object?
    // {centre: landmarks[6], left: landmarks[264], right: landmarks[34]}
    const normLandmarks: Point[] = anchorLandmarks.map((landmark: NormalizedLandmark) => {
      return [
        Math.round(landmark.x * width),
        Math.round(landmark.y * height),
        // landmark.z
      ];
    });

    // -- Get distance
    const dx = normLandmarks[1][0] - normLandmarks[2][0];
    const dy = normLandmarks[1][1] - normLandmarks[2][1];
    const pixelWidth = Math.round(Math.sqrt(dx * dx + dy * dy));

    const regionSize = [512,128]
    const aspectRatio = regionSize[0]/regionSize[1] // 512/128=4

    const pixelHeight = Math.round(pixelWidth/aspectRatio)
    const halfSrcBox = [pixelWidth/2, pixelHeight/2]

    return [
      [normLandmarks[0][0] - halfSrcBox[0], normLandmarks[0][1] - halfSrcBox[1]], // Top Left
      [normLandmarks[0][0] - halfSrcBox[0], normLandmarks[0][1] + halfSrcBox[1]], // Bottom Left
      [normLandmarks[0][0] + halfSrcBox[0], normLandmarks[0][1] + halfSrcBox[1]], // Bottom Right
      [normLandmarks[0][0] + halfSrcBox[0], normLandmarks[0][1] - halfSrcBox[1]]  // Top Right
    ]

  }

  prepareInput(frame: VideoFrameData, result: FaceLandmarkerResult):  [ImageData, number[], number[]] {
    // TODO: Note this is likely the culprit of the variable frame rate, see below
    let width: number;
    let height: number;
    let frameImageData: ImageData | ImageBitmap;

    if (frame instanceof ImageData) {
      // Legacy/Fallback path
      frameImageData = frame;
      width = frame.width;
      height = frame.height;
    } else {
      // VideoFrame or ImageBitmap (Worker path)
      if (frame instanceof VideoFrame) {
        width = frame.displayWidth; // TODO: Should be visible, display, or coded width??
        height = frame.displayHeight;
      } else {
        // ImageBitmap
        width = frame.width;
        height = frame.height;
      }
      // Draw to OffscreenCanvas to get pixel data
      const ctx = this.getTempContext(width, height);
      // Both VideoFrame and ImageBitmap are valid CanvasImageSource
      ctx.drawImage(frame, 0, 0);
      // This is the unavoidable CPU readback cost for using CPU-based mathUtils
      // With 'willReadFrequently', this is somewhat optimized
      frameImageData = ctx.getImageData(0, 0, width, height); // TODO: triggers a GPU→CPU readback, which is inherently variable in timing depending on GPU pipeline state.
    }

    // If perspective matrix is not set, initialize it
    if (!this.perspectiveMatrixSet) {
      // 4*4 projection matrix mapping camera coords to NDC (i.e. w,h -> -1,1). used by faceReconstruction to unproject.
      const aspectRatio = width / height;
      // TODO: cleanup how many new matrices are created here to stop triggering GC? preallocate into single ImageData buffer using data.set()
      this.perspectiveMatrix = createPerspectiveMatrix(aspectRatio); //
      this.perspectiveMatrixSet = true;
    }

    // If intrinsics matrix is not set, initialize it
    if (!this.intrinsicsMatrixSet) {
      // Computer Vision matrix (K) describing internal properties of the camera (focal length/ optical center).
      this.intrinsicsMatrix = createIntrinsicsMatrix(width, height);
      this.intrinsicsMatrixSet = true;
    }

    // Convert the normalized landmarks to non-normalized coordinates
    const landmarks = result.faceLandmarks[0];
    const landmarks2d: Point[] = landmarks.map((landmark: NormalizedLandmark) => {
      return [
        Math.floor(landmark.x * width),
        Math.floor(landmark.y * height),
      ];
    });

    // Convert from MediaPipeMatrix to ml-matrix Matrix
    const faceRT = translateMatrix(result.facialTransformationMatrixes[0]);

    // First, extract the eye patch
    // takes tilted/rotated face from camera and unwarps to aligned rectangular image of eyes
    // TODO: see https://github.com/google-ai-edge/mediapipe/issues/3495
    const eyePatch = obtainEyePatch(
        frameImageData,
      landmarks2d,
    );

    // Second, compute the face origin in 3D space
    // Uses 15cm face width to estimate angle of gaze/head
    // specific coordinate xyz origin for gaze vector.
    const face_origin_3d = this.computeFaceOrigin3D(
        frameImageData,
      landmarks.map((l: NormalizedLandmark) => [l.x, l.y]), // TODO: what is the diff between this and landmarks2d?!
      landmarks2d,
      faceRT
    )

    // Third, compute the head vector
    // converts mediapipe rotation matrix to Euler angles and swaps/negates axes for different coordinate system
    const head_vector = getHeadVector(
      faceRT
    );

    return [
      eyePatch,
      head_vector,
      face_origin_3d
    ];
  }

  async adapt(
    eyePatches: ImageData[],
    headVectors: number[][],
    faceOrigins3D: number[][],
    normPogs: number[][],
    stepsInner: number = 5,    // Default: 5 (matches Python webeyetrack.py:324)
    innerLR: number = 1e-5,    // Default: 1e-5 (matches Python webeyetrack.py:325)
    ptType: 'calib' | 'click' = 'calib'
  ) {
    // TODO: reduce how many small objects, matrices, and tensors are created here. (e.g. in adapt, createPerspectiveMatrix and step)

    // Prune old clickstream data (calibration buffer is never pruned)
    this.pruneCalibData();
    // TODO: consider persistent optimiser, but haas downsides...
    const opt = tf.train.adam(innerLR, 0.85, 0.9, 1e-8);

    try {
      let { supportX, supportY } = generateSupport(
        eyePatches,
        headVectors,
        faceOrigins3D,
        normPogs
      );

      // add the new data to the appropriate buffer first
      if (ptType === 'calib') {
        this.calibData.calibSupportX.push(supportX);
        this.calibData.calibSupportY.push(supportY);
        this.calibData.calibTimestamps.push(Date.now());
      } else {
        this.calibData.clickSupportX.push(supportX);
        this.calibData.clickSupportY.push(supportY);
        this.calibData.clickTimestamps.push(Date.now());
      }

      // Enforce Total Max Points Limit - if points exceed the limit, we evict data to make room.
      // TODO: implement Auto-Reset on Movement for high-confidence calibration points
      const TOTAL_LIMIT = this.maxCalibPoints + this.maxClickPoints;

      while ((this.calibData.calibSupportX.length + this.calibData.clickSupportX.length) > TOTAL_LIMIT) {
        // Eviction Strategy: Prevent "Elastic Banding"
        // If we are over the limit, we prefer to evict old CALIBRATION data.
        // This ensures that new correction clicks (which reflect your current posture)
        // push out the old calibration points (which might reflect an old posture).

        let evictedItem: SupportX | undefined;
        let evictedY: tf.Tensor | undefined;

        if (this.calibData.calibSupportX.length > 0) {
          // Evict oldest calibration point (Fixes elastic band effect)
          evictedItem = this.calibData.calibSupportX.shift();
          evictedY = this.calibData.calibSupportY.shift();
          this.calibData.calibTimestamps.shift();
          // console.log('♻️ Evicted old calibration point to make room for new data.');
        } else {
          // Fallback: If no calibration data left, evict oldest click point
          evictedItem = this.calibData.clickSupportX.shift();
          evictedY = this.calibData.clickSupportY.shift();
          this.calibData.clickTimestamps.shift();
        }

        // CRITICAL: Dispose memory to prevent leaks
        if (evictedItem && evictedY) {
          tf.dispose([
            evictedItem.eyePatches,
            evictedItem.headVectors,
            evictedItem.faceOrigins3D,
            evictedY
          ]);
        }
      }

      // === CONCATENATE FROM BOTH BUFFERS FOR TRAINING ===
      let tfEyePatches: tf.Tensor;
      let tfHeadVectors: tf.Tensor;
      let tfFaceOrigins3D: tf.Tensor;
      let tfSupportY: tf.Tensor;
      let needsDisposal: boolean; // Track if we created new tensors that need disposal

      const allSupportX = [...this.calibData.calibSupportX, ...this.calibData.clickSupportX];
      const allSupportY = [...this.calibData.calibSupportY, ...this.calibData.clickSupportY];

      if (allSupportX.length > 1) {
        // Create concatenated tensors from both buffers
        tfEyePatches = tf.concat(allSupportX.map(s => s.eyePatches), 0);
        tfHeadVectors = tf.concat(allSupportX.map(s => s.headVectors), 0);
        tfFaceOrigins3D = tf.concat(allSupportX.map(s => s.faceOrigins3D), 0);
        tfSupportY = tf.concat(allSupportY, 0);
        needsDisposal = true; // We created new concatenated tensors
      } else {
        // Only one point total, use it directly (no concatenation needed)
        tfEyePatches = supportX.eyePatches;
        tfHeadVectors = supportX.headVectors;
        tfFaceOrigins3D = supportX.faceOrigins3D;
        tfSupportY = supportY;
        needsDisposal = false; // These are references to buffer tensors, don't dispose
      }

      // === COMPUTE AFFINE TRANSFORMATION ===
      // Requires at least 4 points (affine has 6 DOF: 2 scale, 2 rotation/shear, 2 translation)
      if (tfEyePatches.shape[0] > 3) {
        const supportPreds = tf.tidy(() => {
          return this.blazeGaze.predict(tfEyePatches, tfHeadVectors, tfFaceOrigins3D);
        });
        const supportPredsNumber = await supportPreds.array() as number[][];
        const supportYNumber = await tfSupportY.array() as number[][];
        // Dispose the prediction tensor after extracting values
        tf.dispose(supportPreds);
        try { // TODO: Already added this try catch.
          const affineMatrixML = computeAffineMatrixML(supportPredsNumber, supportYNumber);
          // Dispose old affine matrix before creating new one
          if (this.affineMatrix) {
            tf.dispose(this.affineMatrix);
          }
          this.affineMatrix = tf.tensor2d(affineMatrixML, [2, 3], 'float32');
        } catch (e) {
          console.warn("Skipping affine update: Calibration points are likely too similar (rank deficient).", e);
          // swallow the error here so the tracker keeps running with the previous valid matrix.
        }
      }

      // === MAML-STYLE ADAPTATION TRAINING ===
      tf.tidy(() => {
        for (let i = 0; i < stepsInner; i++) {
          const { grads, value: loss } = tf.variableGrads(() => {
            const preds = this.blazeGaze.predict(tfEyePatches, tfHeadVectors, tfFaceOrigins3D);
            const predsTransformed = this.affineMatrix ? applyAffineMatrix(this.affineMatrix, preds) : preds;
            const loss = tf.losses.meanSquaredError(tfSupportY, predsTransformed);
            return loss.asScalar();
          });
          // variableGrads returns NamedTensorMap where values are gradients of Variables
          // Type assertion is safe because variableGrads computes gradients w.r.t. Variables
          opt.applyGradients(grads as Record<string, tf.Variable>);
          // Explicitly dispose gradients (defensive, tf.tidy should handle this)
          Object.values(grads).forEach(g => g.dispose());

          // Synchronous logging to avoid race condition with tf.tidy() cleanup
          // const lossValue = loss.dataSync()[0];
          // console.log(`Loss = ${lossValue.toFixed(4)}`);
          loss.dispose();
        }
      });

      // === CLEANUP: Dispose concatenated tensors ===
      // Only dispose if we created new tensors via concatenation
      if (needsDisposal) {
        tf.dispose([tfEyePatches, tfHeadVectors, tfFaceOrigins3D, tfSupportY]);
      }
    } finally {
      // CRITICAL: Dispose optimizer to prevent memory leak
      // Optimizer creates internal variables (momentum buffers, variance accumulators)
      // that persist until explicitly disposed, causing ~1-5 MB leak per adapt() call
      opt.dispose();
      let mem = tf.memory();
      console.log(`End adapt() n Tensors: ${mem.numTensors}, TF Memory: ${(mem.numBytes / 1024 / 1024).toFixed(2)} MB`);
    }
  }

  async step(frame: VideoFrameData, timestamp: number, result: FaceLandmarkerResult | null): Promise<WebEyeTrackResult> {
    // TODO: Note this returns a promise but does synchronous GPU work internally so blocks worker thread until that yields.
    //  need to make the TF.js inference actually async at some point.
    const tic1 = performance.now();
    // result = null; // For testing purposes, we can set result to null to simulate no face detected
    // TODO: move up to worker
    // TODO: cleanup how many new matrices are created here to stop triggering GC? preallocate into single ImageData buffer using data.set()
    if (!result || !result.faceLandmarks || result.faceLandmarks.length === 0) {
      return {
        eyePatch: new ImageData(1, 1), // Placeholder for eye patch
        headVector: [0, 0, 0], // Placeholder for head vector
        faceOrigin3D: [0, 0, 0], // Placeholder for face
        metric_transform: {rows: 3, columns: 3, data: [1, 0, 0, 1, 0, 0, 1, 0, 0]}, // Placeholder for metric transform
        gazeState: 'closed', // Default to closed state if no landmarks
        normPog: [0, 0], // Placeholder for normalized point of gaze
        durations: {
          blazeGaze: 0,
          kalmanFilter: 0,
          total: 0
        },
        timestamp: timestamp // Include the timestamp
      };
    }

    // Perform preprocessing to obtain the eye patch, head_vector, and face_origin_3d
    const [eyePatch, headVector, faceOrigin3D] = this.prepareInput(frame, result);
    const tic3 = performance.now();

    // Compute the EAR ratio to determine if the eyes are open or closed
    let gaze_state: 'open' | 'closed' = 'open';
    const leftEAR = computeEAR(result.faceLandmarks[0], 'left');
    const rightEAR = computeEAR(result.faceLandmarks[0], 'right');
    if ( leftEAR < 0.2 || rightEAR < 0.2) {
      gaze_state = 'closed';
    }

    // If 'closed' return (0, 0)
    // console.log(result)
    if (gaze_state === 'closed') {
      return {
        eyePatch: eyePatch,
        headVector: headVector,
        faceOrigin3D: faceOrigin3D,
        metric_transform: {rows: 3, columns: 3, data: [1, 0, 0, 1, 0, 0, 1, 0, 0]}, // Placeholder, should be computed
        gazeState: gaze_state,
        normPog: [0, 0],
        durations: {
          blazeGaze: 0, // No BlazeGaze inference if eyes are closed
          kalmanFilter: 0, // No Kalman filter step if eyes are closed
          total: tic3 - tic1
        },
        timestamp: timestamp // Include the timestamp
      };
    }

    const [predNormPog, tic4] = tf.tidy(() => {

      // Perform the gaze estimation via BlazeGaze Model (tensorflow.js)
      const inputTensor = tf.browser.fromPixels(eyePatch).toFloat().expandDims(0); // TODO: CPU→GPU transfer creates variable timing

      // Divide the inputTensor by 255 to normalize pixel values
      const normalizedInputTensor = inputTensor.div(tf.scalar(255.0));
      const headVectorTensor = tf.tensor2d(headVector, [1, 3]);
      const faceOriginTensor = tf.tensor2d(faceOrigin3D, [1, 3]);
      // Note blazeGaze predicts gaze location from the extracted eye patch
      // Deals with ambiguity: faceLandmarker jitter, oval shaped iris, iris squashed by camera lens, etc
      let outputTensor = this.blazeGaze.predict(normalizedInputTensor, headVectorTensor, faceOriginTensor);
      // Defensive dispose - but probably handled by tidy()
      tf.dispose([inputTensor, normalizedInputTensor, headVectorTensor, faceOriginTensor]);

      // If affine transformation is available, apply it
      if (this.affineMatrix) {
        outputTensor = applyAffineMatrix(this.affineMatrix, outputTensor);
      }

      // Extract the 2D gaze point data from the output tensor
      if (!outputTensor || outputTensor.shape.length === 0) {
        throw new Error("BlazeGaze model did not return valid output");
      }
      return [outputTensor, performance.now()];
    });

    const normPog = await predNormPog.array() as number[][]; // TODO: async GPU→CPU readback here as well, depends on GPU state.
    tf.dispose(predNormPog);

    // Apply Kalman filter to smooth the gaze point
    const kalmanOutput = this.kalmanFilter.step(normPog[0]);
    // Clip the output to the range of [-0.5, 0.5]
    kalmanOutput[0] = Math.max(-0.5, Math.min(0.5, kalmanOutput[0]));
    kalmanOutput[1] = Math.max(-0.5, Math.min(0.5, kalmanOutput[1]));

    // Return GazeResult
    const tic5 = performance.now();
    let gaze_result: WebEyeTrackResult = {
      eyePatch: eyePatch,
      headVector: headVector,
      faceOrigin3D: faceOrigin3D,
      metric_transform: {rows: 3, columns: 3, data: [1, 0, 0, 1, 0, 0, 1, 0, 0]}, // Placeholder, should be computed
      gazeState: gaze_state,
      normPog: kalmanOutput,
      durations: {
        blazeGaze: tic4 - tic3,
        kalmanFilter: tic5 - tic4,
        total: tic5 - tic1
      },
      timestamp: timestamp
    };

    // Debug: Printout the tf.Memory
    // console.log(`[WebEyeTrack] tf.Memory: ${JSON.stringify(tf.memory().numTensors)} tensors, ${JSON.stringify(tf.memory().unreliable)} unreliable, ${JSON.stringify(tf.memory().numBytes)} bytes`);

    // Update the latest gaze result
    this.latestGazeResult = gaze_result;
    return gaze_result;
  }

  /**
   * Disposes all TensorFlow.js tensors and resources held by this tracker.
   * After calling dispose(), this object should not be used.
   */
  dispose(): void {
    if (this._disposed) {
      return;
    }
    this.clearCalibrationBuffer();
    this.clearClickstreamPoints();

    // Dispose all calibration buffer tensors
    this.calibData.calibSupportX.forEach(item => {
      tf.dispose([item.eyePatches, item.headVectors, item.faceOrigins3D]);
    });

    this.calibData.calibSupportY.forEach(tensor => {
      tf.dispose(tensor);
    });

    // Dispose all clickstream buffer tensors
    this.calibData.clickSupportX.forEach(item => {
      tf.dispose([item.eyePatches, item.headVectors, item.faceOrigins3D]);
    });

    this.calibData.clickSupportY.forEach(tensor => {
      tf.dispose(tensor);
    });

    // Clear all buffer arrays
    this.calibData.calibSupportX = [];
    this.calibData.calibSupportY = [];
    this.calibData.calibTimestamps = [];
    this.calibData.clickSupportX = [];
    this.calibData.clickSupportY = [];
    this.calibData.clickTimestamps = [];

    // Dispose affine matrix
    if (this.affineMatrix) {
      tf.dispose(this.affineMatrix);
      this.affineMatrix = null;
    }

    // Dispose child components if they have dispose methods
    if (this.blazeGaze && typeof this.blazeGaze.dispose === 'function') {
      this.blazeGaze.dispose();
    }

    this._disposed = true;
  }

  /**
   * Returns true if dispose() has been called on this tracker.
   */
  get isDisposed(): boolean {
    return this._disposed;
  }
}
