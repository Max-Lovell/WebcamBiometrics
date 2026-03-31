# Testing the bundled npm package
Minimal test file to check that imports resolve and types work. Intended to be used in a new project to test the packaged tarball before uploading to NPM

## Create project
Either create a Vite app with `npm create vite@latest test-app -- --template vanilla-ts`
or a Node app in a new directory with:
```
npm init
npm install vite
```
add `"scripts": {"dev": "vite"}` to `package.json` and 
    ```
    "compilerOptions": {
        "target": "es2022",
        "module": "esnext"
    }
    ```
to `tsconfig.json`

## Install the Webcam Biometrics Package
Install the package from npm `npm install webcam-biometrics` or to install from a local build (after `npm run build`) using `npm install [PATH TO DIR]/webcam-biometrics-0.1.0.tgz`

Then copy the worker file - no good fixes to avoid this with current Vite that don't cause their own issues ([1](https://github.com/vitejs/vite/discussions/15547), [2](https://github.com/vitejs/vite/discussions/1736)), and these problems interact with how MediaPipe handles importing modules in workers:
```
cp node_modules/webcam-biometrics/dist/worker.js public/worker.js
```

and optionally copy the BlazeGaze model if running that locally:
```
cp -r node_modules/webcam-biometrics/models/blazegaze public/models
```

To reinstall and test a new build in a test project use:
```
npm cache clean --force
npm uninstall webcam-biometrics
npm install ~/Code/WebcamBiometrics/webcam-biometrics-0.1.0.tgz
```

## Note on inlining the worker for future dev work
See `vite-plugin-inline-worker` in git history. Rollup screws with this line's import() statement somehow:
`const module = await import(/* @vite-ignore */ url);`
could be fixed with `new Function('url', 'return import(url)')` to hide the dynamic import() from Rollup's static analysis. 
But seems like too many fragile, hacky, hard to debug (particularly as blob doesn't have sourcemaps) issues. Also increases bundle size, and may clash with Content Security Policy headers at unis.
Looks like there's a fix for Vite late 7 or maybe 8 releasing soon: https://github.com/vitejs/vite/issues/11672#issuecomment-1415820673, https://github.com/vitejs/vite/blob/main/packages/vite/CHANGELOG.md





