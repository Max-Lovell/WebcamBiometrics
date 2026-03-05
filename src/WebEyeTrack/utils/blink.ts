// ============================================================================
// Gaze State
// ============================================================================
import type { NormalizedLandmark} from '@mediapipe/tasks-vision';

const LEFT_EYE_EAR_LANDMARKS = [362, 385, 387, 263, 373, 380]
const RIGHT_EYE_EAR_LANDMARKS = [133, 158, 160, 33, 144, 153]

export function computeEAR(eyeLandmarks: NormalizedLandmark[], side: 'left' | 'right'): number {
    const EYE_EAR_LANDMARKS = side === 'left' ? LEFT_EYE_EAR_LANDMARKS : RIGHT_EYE_EAR_LANDMARKS;
    const [p1, p2, p3, p4, p5, p6] = EYE_EAR_LANDMARKS.map(idx => [eyeLandmarks[idx].x, eyeLandmarks[idx].y]);

    const a = Math.sqrt(Math.pow(p2[0] - p6[0], 2) + Math.pow(p2[1] - p6[1], 2));
    const b = Math.sqrt(Math.pow(p3[0] - p5[0], 2) + Math.pow(p3[1] - p5[1], 2));
    const c = Math.sqrt(Math.pow(p1[0] - p4[0], 2) + Math.pow(p1[1] - p4[1], 2));

    return (a + b) / (2.0 * c);
}
