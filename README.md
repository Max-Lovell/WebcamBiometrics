# webcam-biometrics

Real-time webcam biometrics: face landmarks, gaze tracking, and heart rate estimation.

## Quick Start

```bash
npm install webcam-biometrics
```

```js
import { BiometricsClient } from 'webcam-biometrics';
const client = new BiometricsClient('webcam'); // where 'webcam' is the id of an HTML video element
client.onResult = (result) => { console.log(result) }; // log result
await client.start(); // start processing
```

The library handles everything internally — webcam access, face detection, gaze estimation, and heart rate extraction — and returns results via a callback on every processed frame. All heavy computation runs in a Web Worker so the main thread stays responsive.

### Configuration

```js
const client = new BiometricsClient('webcam', {
    pipeline: {
        gaze: { maxCalibPoints: 50, maxClickPoints: 5, clickTTL: 60000 },
        heart: {}, // or `false` to disable
        misc: true,
    },
});
```

All pipeline stages are enabled by default. Set any stage to `false` to disable it. Asset URLs default to public CDNs (jsDelivr for the gaze model, Google Storage for MediaPipe) and fall back to local paths on `localhost`.
Misc is currently a playground for new ideas - the irisDistance seems quite accurate on testing - idea from here: https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/ with minor improvements. Consider using blendshapes for a FACS style emotion analysis.

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

The library uses a Web Worker for all heavy computation. Distributing a worker in an npm package is tricky — `new URL('./Worker.ts', import.meta.url)` works in Vite but breaks in Webpack, Rollup, esbuild, and plain `<script>` tags. To make the library work everywhere with zero config, the worker is **inlined** into the main bundle at build time.

### How it works

There are three build modes, each handled differently:

**`npm run dev`** — Vite dev server. Vite detects `new URL('./Worker.ts', import.meta.url)` in `BiometricsClient.ts`, compiles the worker on the fly, and serves it as a separate module. Standard Vite behaviour, no plugins involved.

**`npm run build:demo`** — Static site build for GitHub Pages. Same as dev — Vite handles the worker natively, bundling it as a separate chunk. The demo is a normal Vite app, not a library.

**`npm run build`** — Library build for npm. This is where the inlining happens:

1. A custom Vite plugin (`vite-plugin-inline-worker.ts`) runs **before** the main build.
2. It compiles `Worker.ts` into a self-contained ES bundle with all dependencies (MediaPipe, TensorFlow.js, etc.) bundled in. No externals.
3. The compiled worker code is injected as a compile-time string constant (`__INLINE_WORKER__`) via Vite's `define`.
4. In `BiometricsClient.ts`, `typeof __INLINE_WORKER__` is checked at the point of worker creation:
    - **Library build:** The constant exists → worker is created from a Blob URL. No separate file needed.
    - **Dev/demo:** The constant is undefined → falls back to the standard `new URL()` pattern that Vite handles natively.
5. A small `drop-worker-chunk` plugin removes the redundant worker chunk that Vite emits from the `new URL()` pattern (it's dead code in the library build but Vite's static analysis still emits it).

The result: consumers get a single JS file that creates the worker from an embedded string. Works with any bundler or none at all.

### Model and WASM files

The heavy binaries — MediaPipe WASM (~4 MB), the face landmarker model (~15 MB), and the BlazeGaze model (~1 MB) — are **not** bundled into the library. They are fetched at runtime from public CDNs via the URLs in `assetDefaults.ts`. The inlined worker only contains the JavaScript code (~1.8 MB minified) for MediaPipe, TensorFlow.js, and the pipeline logic.

Consumers can override asset URLs via the `assets` config option if they want to self-host the models.

### CSP Note

The inlined worker uses a Blob URL. If the consuming page has a strict Content Security Policy, it must include `blob:` in its `worker-src` directive:

```
Content-Security-Policy: worker-src 'self' blob:;
```

---

## Serving the Demo Locally

The demo build is configured for GitHub Pages deployment with `base: '/webcam-biometrics/'`. To test locally after building:

```bash
npm run build:demo
npx serve dist-demo
```

Then visit `http://localhost:3000`.

> **Note:** If the `base` setting is present in `vite.config.ts` for the demo build, assets will be prefixed with `/webcam-biometrics/`. Either remove the `base` line for local testing, or navigate to `http://localhost:3000/webcam-biometrics/`.

---

## License

AGPL-3.0-or-later — see [LICENSE.txt](LICENSE.txt) for details.

## License

AGPL-3.0-or-later. See [LICENSE](./LICENSE).

Free for open-source and academic/research use. If you're building something
closed-source or commercial, a commercial license is available —
get in touch at [your-email@sussex.ac.uk].

### Third-Party Notices

- `src/webeyetrack/` — MIT License (Davalos, Zhang, Goodwin, Biswas)
- MediaPipe WASM — Apache 2.0 (Google)
