import type {FaceLandmarkerResult} from "@mediapipe/tasks-vision";

function faceOrigin(faceLandmarkerResult: FaceLandmarkerResult){
    const facialTransformation = faceLandmarkerResult.facialTransformationMatrixes[0]
    // take relationship to facial transformation matrix

    // Why can't you get the metric location of the facial transform

}
