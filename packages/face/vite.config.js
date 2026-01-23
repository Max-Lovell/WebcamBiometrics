import { defineConfig } from 'vite';

export default defineConfig({
    optimizeDeps: {
        // Tell Vite to exclude MediaPipe from pre-bundling - turns import() into a wrapper (self.import), causes error
        exclude: ['@mediapipe/tasks-vision']
    },
    worker: {
        // Ensure Workers are compiled as ES Modules, allows 'import()' to work natively.
        format: 'es'
    },
    server: {
        // Optional: sometimes helps with strict MIME type checking for WASM
        headers: {
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Cross-Origin-Embedder-Policy': 'require-corp',
        }
    }
});
