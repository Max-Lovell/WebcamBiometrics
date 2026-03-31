import { BiometricsClient } from 'webcam-biometrics';

const client = new BiometricsClient('webcam', {
        workerUrl: '/worker.js',
        assets: {
            gazeModelPath: '/models/model.json',
        },
        pipeline: {
            heart: false
        }
    }
);

const cursor = document.getElementById('cursor');
client.onResult = (result) => {
    console.log(result);

    //
    const normPog = result.gaze?.normPog;
    if (normPog && cursor) {
        const vw = document.documentElement.clientWidth || window.innerWidth;
        const vh = document.documentElement.clientHeight || window.innerHeight;
        cursor.style.left = `${(normPog[0] + 0.5) * vw}px`;
        cursor.style.top = `${(normPog[1] + 0.5) * vh}px`;
        cursor.style.backgroundColor = result.gaze!.gazeState === 'closed' ? 'gray' : 'purple';
    }


};
await client.start();
