/**
 * Vite Plugin: Inline Worker Builder
 *
 * Two-phase build approach:
 * 1. Before the main library build, compile Worker.ts into a self-contained ES bundle
 * 2. Inject the compiled code as a compile-time constant (__INLINE_WORKER__) via Vite's `define`
 *
 * BiometricsClient checks `typeof __INLINE_WORKER__` at runtime:
 * - Library build: the constant exists, worker is created from a Blob URL
 * - Dev/demo build: the constant is undefined, Vite handles the worker via URL
 *
 * Usage in vite.config.ts:
 *   import { inlineWorkerPlugin } from './vite-plugin-inline-worker'
 *   // Only add to the library build's plugins array
 *   plugins: [inlineWorkerPlugin()]
 */

import { build, type Plugin, type Rollup } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export function inlineWorkerPlugin(): Plugin {
    let workerCode: string = '';

    return {
        name: 'inline-worker',
        enforce: 'pre',

        async buildStart() {
            // Build the worker into a self-contained bundle. note config() below runs first
            // Uses 'es' format so top-level self.onmessage works as-is.
            const result = await build({
                configFile: false,
                build: {
                    write: false, // Don't write to disk — keep in memory
                    lib: {
                        entry: path.resolve(__dirname, 'src/pipeline/Worker.ts'),
                        formats: ['es'],
                        fileName: () => 'worker.js',
                    },
                    rollupOptions: {
                        // Everything must be bundled INTO the worker — no externals.
                        // MediaPipe and TF.js JS code is included here; the heavy
                        // WASM/model binaries are still fetched at runtime via URLs.
                        external: [],
                    },
                    minify: true,
                    sourcemap: false,
                },
                define: {
                    __PACKAGE_VERSION__: JSON.stringify(
                        process.env.npm_package_version ?? '0.0.0'
                    ),
                },

            });

            // Extract the generated code string
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

        // Inject the worker code as a compile-time constant.
        // replaces every occurrence of __INLINE_WORKER__ with the string literal.
        transform(code, _) {
            if (code.includes('__INLINE_WORKER__')) {
                return code.replaceAll(
                    '__INLINE_WORKER__',
                    JSON.stringify(workerCode)
                );
            }
        },
    };
}
