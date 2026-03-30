// Worker Factory
// Isolates worker creation so inline-worker plugin can swap entire module during build via load hook.
// See vite-plugin-inline-worker.ts in root for where this is replaced.
export function createWorker(): Worker {
    return new Worker(
        new URL('./Worker.ts', import.meta.url),
        { type: 'module' },
    );
}
