import { defineConfig } from 'vite'
import path from 'path'
import { fileURLToPath } from 'url'
import pkg from './package.json' with { type: 'json' };
import basicSsl from '@vitejs/plugin-basic-ssl';

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
            plugins: [basicSsl()],
            optimizeDeps: {
                exclude: ['@mediapipe/tasks-vision'],
            },
            worker: {
                format: 'es',
            },
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
                target: 'esnext',
            },
            // base sets the URL prefix for assets. GitHub Pages serves at https://USERNAME.github.io/REPO_NAME/, so assets need that prefix.
            // Change this to match repo name if forking
            base: '/WebcamBiometrics/',
        }
    }

    // Library build (npm run build)
    // Produces the NPM package in dist/.
    return {
        ...shared,
        publicDir: false,
        build: {
            lib: {
                entry: 'src/index.ts',
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
            rollupOptions: {          // Move inside build
                output: {
                    chunkFileNames: '[name].js',
                },
            },
        },
        worker: {
            format: 'es',
            rollupOptions: {
                output: {
                    entryFileNames: 'worker.js',
                },
            },
        },
        define: {
            __PACKAGE_VERSION__: JSON.stringify(pkg.version),
            __LIB_BUILD__: 'true',
        },
    }
})
