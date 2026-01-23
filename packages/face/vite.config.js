import { defineConfig } from 'vite';

export default defineConfig({
    optimizeDeps: {
        // Tell Vite to exclude MediaPipe from pre-bundling - turns import() into a wrapper (self.import), causes error
        exclude: ['@mediapipe/tasks-vision']
    },
    worker: {
        // Ensure Workers are compiled as ES Modules
        format: 'es'
    }
});
