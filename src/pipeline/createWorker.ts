// Worker Factory
// Allows passing a url for a hosted version of the worker if loading in a script tag
declare const __LIB_BUILD__: boolean;

export function createWorker(workerUrl?: string): Worker {
    if (workerUrl) {
        // Cross-origin workers must be proxied through a blob
        if (workerUrl.startsWith('http') && typeof location !== 'undefined'
            && !workerUrl.startsWith(location.origin)) {
            const blob = new Blob(
                [`import "${workerUrl}"`],
                { type: 'application/javascript' }
            );
            return new Worker(URL.createObjectURL(blob), { type: 'module' });
        }
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
        new URL('./Worker.ts', import.meta.url), // or fallback to './worker.js' for others.
        { type: 'module' },
    );
}
