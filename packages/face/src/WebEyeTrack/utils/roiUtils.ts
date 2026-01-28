// MediaPipe Face Mesh Indices for rPPG signal
// Optimized ROIs based on GRGB rPPG research
export const FACE_ROIS = {
    // Center forehead (Avoids hair/eyebrows)
    forehead: [107, 66, 69, 109, 10, 338, 299, 296, 336, 9],

    // Left Cheek (Subject's Left - Indices > 200)
    leftCheek: [347, 348, 329, 355, 429, 279, 358, 423, 425, 280],

    // Right Cheek (Subject's Right - Indices < 200)
    rightCheek: [118, 119, 100, 126, 209, 49, 129, 203, 205, 50]
};

// Helper to get points from indices if you need it elsewhere
// (Assuming you pass the full landmarks array)
export function getPoints(indices: number[], landmarks: any[]) {
    return indices.map(i => landmarks[i]);
}
