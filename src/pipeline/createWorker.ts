// Worker Factory
// Allows passing a url for a hosted version of the worker if loading in a script tag
declare const __LIB_BUILD__: boolean;

export function createWorker(workerUrl?: string): Worker {
    if (workerUrl) {
        return new Worker(workerUrl, { type: 'module' });
    }

    if (typeof __LIB_BUILD__ !== 'undefined' && __LIB_BUILD__) {
        throw new Error(
            'webcam-biometrics: workerUrl is required. ' +
            'See https://github.com/Max-Lovell/WebcamBiometrics#setup'
        );
    }

    // Dev/demo mode — Vite handles .ts workers natively
    return new Worker(
        new URL('./Worker.ts', import.meta.url),
        { type: 'module' },
    );
}
