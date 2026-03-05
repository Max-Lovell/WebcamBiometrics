// ============================================================================
// Compute Affine Transformation Matrix
// ============================================================================

import * as tf from '@tensorflow/tfjs';
import {Matrix, solve} from "ml-matrix";

export function computeAffineMatrixML(src: number[][], dst: number[][]): number[][] {
    // const N = src.length;
    const srcAug = src.map(row => [...row, 1]); // [N, 3]

    const X = new Matrix(srcAug);   // [N, 3]
    const Y = new Matrix(dst);      // [N, 2]

    const A = solve(X, Y); // [3, 2]
    return A.transpose().to2DArray(); // [2, 3]
}

export function applyAffineMatrix(A: tf.Tensor, V: tf.Tensor): tf.Tensor {
    const reshapedOutput = V.reshape([-1, 2]);        // [B, 2]
    const ones = tf.ones([reshapedOutput.shape[0], 1]);          // [B, 1]
    const homog = tf.concat([reshapedOutput, ones], 1);          // [B, 3]
    const affineT = A.transpose();                // [3, 2]
    const transformed = tf.matMul(homog, affineT);                // [B, 2]
    tf.dispose([reshapedOutput, ones, homog, affineT]); // Clean up intermediate tensors
    return transformed.reshape(V.shape);       // reshape back
}
