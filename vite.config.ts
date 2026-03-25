import { defineConfig } from 'vite'
import path from 'path'
import { fileURLToPath } from 'url'
import pkg from './package.json' with { type: 'json' };

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig(({ mode }) => {
    const shared = {
        define: {
            __PACKAGE_VERSION__: JSON.stringify(pkg.version),
        },
    };

    // Dev mode (npm run dev) -----------------
    // Serves index.html from demo/, hot-reloads, uses local model files
    if (mode === 'development') {
        return {
            ...shared,
            root: 'demo',
            publicDir: path.resolve(__dirname, 'public'),
            resolve: {
                alias: {
                    // If still need the local vitals alias:
                    // '@webcambiometrics/vitals': path.resolve(__dirname, '../vitals/src/index.ts'),
                },
            },
        }
    }

    // Demo build (npm run build:demo) -----------------
    // Produces a static site in dist-demo/ for GitHub Pages. normal Vite app build, not a library build.
    if (mode === 'demo') {
        return {
            ...shared,
            root: 'demo',
            publicDir: path.resolve(__dirname, 'public'),
            build: {
                outDir: path.resolve(__dirname, 'dist-demo'),
                emptyOutDir: true,
            },
            // base sets the URL prefix for assets. GitHub Pages serves at https://USERNAME.github.io/REPO_NAME/, so assets need that prefix.
            // Change this to match repo name if forking
            base: '/webcam-biometrics/',
        }
    }

    // Library build (npm run build)
    // Produces the NPM package in dist/.
    return {
        ...shared,
        build: {
            lib: {
                entry: path.resolve(__dirname, 'src/index.ts'),
                name: 'WebcamBiometrics',
                formats: ['es', 'cjs', 'umd'],
                fileName: (format) => {
                    if (format === 'es') return 'webcam-biometrics.js'
                    if (format === 'cjs') return 'webcam-biometrics.cjs'
                    return 'webcam-biometrics.umd.js'
                },
            },
            outDir: 'dist',
            sourcemap: true,
            rollupOptions: {
                external: [
                    '@mediapipe/tasks-vision',
                    '@tensorflow/tfjs',
                    'mathjs',
                    'ml-matrix',
                ],
                output: {
                    globals: {
                        '@mediapipe/tasks-vision': 'MediaPipeVision',
                        '@tensorflow/tfjs': 'tf',
                        'mathjs': 'math',
                        'ml-matrix': 'mlMatrix',
                    },
                },
            },
        },
        worker: {
            format: 'es',
            plugins: () => [],
        },
    }
})