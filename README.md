# webcam-biometrics

Real-time webcam biometrics: face landmarks, gaze tracking, and heart rate estimation.

## Quick Start

```bash
npm install webcam-biometrics
```

Copy the worker file to your project's public directory:

```bash
cp node_modules/webcam-biometrics/dist/worker.js public/
```

```js
import { BiometricsClient } from 'webcam-biometrics';

const client = new BiometricsClient('webcam', {
    workerUrl: '/worker.js',
});
client.onResult = (result) => { console.log(result) };
await client.start();
```

The library handles everything internally — webcam access, face detection, gaze estimation, and heart rate extraction — and returns results via a callback on every processed frame. All heavy computation runs in a Web Worker so the main thread stays responsive.
See note on the workerURL parameter in `examples/bundler/readme.md`
### Script Tag Usage
To use in your own (non-node) javascript page you can just run:
```html
<script src="https://cdn.jsdelivr.net/npm/webcam-biometrics@0.1.0/dist/webcam-biometrics.umd.js"></script>
<script>
    const client = new WebcamBiometrics.BiometricsClient('webcam', {
        workerUrl: 'https://cdn.jsdelivr.net/npm/webcam-biometrics@0.1.0/dist/worker.js'
    });
    client.onResult = (result) => { console.log(result) };
    client.start();
</script>
```

### Configuration

```js
const client = new BiometricsClient('webcam', {
    workerUrl: '/worker.js',
    pipeline: {
        gaze: { maxCalibPoints: 50, maxClickPoints: 5, clickTTL: 60000 },
        heart: {}, // or `false` to disable
        misc: true,
    },
    assets: {
        // Override default CDN URLs to self-host models
        // wasmBasePath: '/wasm',
        // faceLandmarkerModelPath: '/wasm/face_landmarker.task',
        // gazeModelPath: '/models/model.json',
    },
});
```

All pipeline stages are enabled by default. Set any stage to `false` to disable it. Asset URLs default to public CDNs (jsDelivr for the gaze model, Google Storage for MediaPipe). Consumers can override asset URLs via the `assets` config option to self-host models.

Misc is currently a playground for new ideas — the irisDistance seems quite accurate on testing — idea from here: https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/ with minor improvements. Consider using blendshapes for a FACS style emotion analysis.

### Gaze Calibration

The gaze tracker calibrates from pointer clicks. Each click records the normalised screen position and the corresponding eye patch, building a calibration buffer. By default the last 5 clicks are used and the buffer clears every 60 seconds.

### Lifecycle

```js
await client.start();  // Start webcam and processing
client.stop();          // Pause capture (resumable)
client.dispose();       // Permanent teardown — not resumable
```

---

## Development

### Prerequisites

- Node.js ≥ 18
- npm

### Setup

```bash
git clone https://github.com/max-lovell/webcam-biometrics.git
cd webcam-biometrics
npm install
```

### Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start Vite dev server with hot reload (serves `demo/`) |
| `npm run build` | Build the library for npm (`dist/`) |
| `npm run build:demo` | Build the demo as a static site (`dist-demo/`) |
| `npm run build:check` | Type-check without emitting |
| `npm test` | Run tests |

---

## Build Architecture

The library uses a Web Worker for all heavy computation. The worker is shipped as a separate file (`dist/worker.js`) alongside the main bundle. Consumers provide the worker URL via config — the same pattern used by libraries like PDF.js and Monaco Editor.

### Why `workerUrl`?

Vite's library mode doesn't reliably handle `new URL('./worker.js', import.meta.url)` across the bundler boundary — the path gets resolved at build time and bakes in an absolute URL that breaks in the consumer's environment. Rather than relying on fragile bundler interop, the library lets consumers host the worker file however suits their setup and point to it explicitly.

### How it works

There are three build modes:

**`npm run dev`** — Vite dev server. Vite detects `new URL('./Worker.ts', import.meta.url)` and serves the worker as a module on the fly. No `workerUrl` needed — the fallback path in `createWorker` handles this automatically.

**`npm run build:demo`** — Static site build for GitHub Pages. Same as dev — Vite handles the worker natively, bundling it as a separate chunk.

**`npm run build`** — Library build for npm. The worker is compiled into a self-contained ES module (`dist/worker.js`) with all dependencies (MediaPipe, TensorFlow.js, pipeline logic) bundled in. The main library bundle (`dist/webcam-biometrics.js`) does not contain the worker code — consumers load it via the `workerUrl` config option.

### Model and WASM files

The heavy binaries — MediaPipe WASM (~4 MB), the face landmarker model (~15 MB), and the BlazeGaze model (~1 MB) — are **not** bundled into the library. They are fetched at runtime from public CDNs via the URLs in `assetDefaults.ts`. The worker file contains only the JavaScript code for MediaPipe, TensorFlow.js, and the pipeline logic.

Consumers can override asset URLs via the `assets` config option to self-host the models.

### MediaPipe Compatibility

MediaPipe's vision bundle has compatibility issues with module workers — it calls `self.import()` (which doesn't exist in module workers), references `document` (unavailable in workers), and has a strict-mode scoping bug. The worker includes three small shims at the top of `Worker.ts` to patch these. These are upstream MediaPipe bugs, not related to the build setup.

---

## Serving the Demo Locally

The demo build is configured for GitHub Pages deployment with `base: '/WebcamBiometrics/'`. To test locally after building:

```bash
npm run build:demo
npx serve dist-demo
```

Then visit `http://localhost:3000`.

> **Note:** If the `base` setting is present in `vite.config.ts` for the demo build, assets will be prefixed with `/WebcamBiometrics/`. Either remove the `base` line for local testing, or navigate to `http://localhost:3000/WebcamBiometrics/`.

---

## License

AGPL-3.0-or-later. See [LICENSE](./LICENSE).

Free for open-source and academic/research use. If you're building something
closed-source or commercial, a commercial license is available —
get in touch at [max_lovell [at] hotmail [dot] co [dot] uk].

### Third-Party Notices

- `src/webeyetrack/` — MIT License (Davalos, Zhang, Goodwin, Biswas)
- MediaPipe WASM — Apache 2.0 (Google)
