import { build, type Plugin, type Rollup } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// This plugin() is called by vite in vite.config.ts in library/prod mode
// Vite's dev server can serve workers on the fly from new Worker()...
// BUT if installing from NPM (Vite 'library' mode) import.meta.url points to the wrong place
// This function will quickly intercept the createWorker() code and replace it with a big string containing the entire worker's code.
export function inlineWorkerPlugin(): Plugin {
    let workerCode: string = '';

    return {
        name: 'inline-worker',
        enforce: 'pre', // Run before Vite

        async buildStart() { // runs once before any modules are processed.
            // Build the worker into a self-contained bundle.
            // Uses 'es' format so top-level self.onmessage works as-is.
            const result = await build({
                configFile: false,
                build: {
                    write: false,
                    lib: {
                        entry: path.resolve(__dirname, 'src/pipeline/Worker.ts'), // Highest parent file to replace
                        formats: ['es'], // Module type
                        fileName: () => 'worker.js', // Output file name
                    },
                    rollupOptions: { external: [] },
                    minify: true,
                    sourcemap: false,
                },
                define: {
                    __PACKAGE_VERSION__: JSON.stringify(
                        process.env.npm_package_version ?? '0.0.0'
                    ),
                },
            });

            const outputs = Array.isArray(result) ? result : [result];
            const output = outputs[0] as Rollup.RollupOutput;
            const chunk = output.output.find(
                (o) => o.type === 'chunk' && o.isEntry
            );
            if (!chunk || chunk.type !== 'chunk') {
                throw new Error('Worker build produced no entry chunk');
            }
            workerCode = chunk.code;
        },

        // Replace the entire createWorker module before Vite's worker detection ever sees the new URL() pattern.
        // Note define() doesn't work here for this unfortunately
        load(id) { // Load() is custom loader hook for replacing an entire module's contents
            if (!id.includes('createWorker')) return; // Called on every module so stop those running - could be more specific...

            return `
export function createWorker() {
    const b = new Blob([${JSON.stringify(workerCode)}], { type: 'application/javascript' });
    const u = URL.createObjectURL(b);
    const w = new Worker(u);
    URL.revokeObjectURL(u);
    return w;
}`;
        },
    };
}

// WHEN SCREWING WITH THIS RUN CHECKS AFTER BUILD:
// grep "new Blob" dist/webcam-biometrics.js | head -1 // Confirm the Blob pattern is in the output
// grep -c "assets/Worker" dist/webcam-biometrics.js //  Confirm there's NO asset URL pattern (the thing we're avoiding)
// grep -c "new URL" dist/webcam-biometrics.js // Confirm no new URL() worker pattern survived
// ls -lh dist/webcam-biometrics.js // Confirm the worker code is actually substantial (not empty) - should be ~2MB
// grep -oP 'new Blob\(\[.{0,80}' dist/webcam-biometrics.js | head -1 // Extract a snippet of what's inside the Blob constructor - should contain minified worker code
// npm run dev — should start without errors and the webcam should work // Confirm dev still works (Vite should handle the URL pattern natively)
