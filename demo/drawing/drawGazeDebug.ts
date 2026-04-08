// demo/drawing/drawGazeDebug.ts
import type { Point } from '../../src/types';

interface Coordinate3D { x: number; y: number; z: number; }

interface GazeInfo {
    origin: Coordinate3D;
    vector: Coordinate3D;
}

interface IrisGazeOutput {
    left: GazeInfo;
    right: GazeInfo;
    cyclopean: GazeInfo;
    screenPog: Point | null;
    fx?: number;
}

function metric2Pixel(
    p: Coordinate3D,
    frameWidth: number,
    frameHeight: number,
    fx: number,
): Point {
    if (p.z <= 0) return { x: -1, y: -1 }; // sentinel for "behind camera"
    const pixelX = (p.x / p.z) * fx;
    const pixelY = (p.y / p.z) * fx;
    return {
        x: pixelX + frameWidth / 2,
        y: -pixelY + frameHeight / 2,
    };
}

// ── Primitive helpers ──────────────────────────────────────────────
function dot(
    ctx: CanvasRenderingContext2D,
    p: Point,
    color: string,
    label?: string,
    radius = 4,
): void {
    if (p.x < 0 || p.y < 0) return;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fill();
    if (label) {
        ctx.fillStyle = color;
        ctx.font = '12px monospace';
        ctx.fillText(label, p.x + radius + 2, p.y - radius - 2);
    }
}

function line(
    ctx: CanvasRenderingContext2D,
    a: Point,
    b: Point,
    color: string,
    width = 2,
): void {
    if (a.x < 0 || b.x < 0) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
}

// ── Debug config — toggle things on/off here ──────────────────────
export interface GazeDebugOptions {
    showLandmarks?: boolean;      // iris + corners
    showEyeballCenters?: boolean; // projected 3D eyeball centers
    showPupilOrigins?: boolean;   // projected 3D pupil positions
    showGazeRays?: boolean;       // left/right/cyclopean rays
    showReadout?: boolean;        // text stats in corner
    rayLengthCm?: number;
}

const DEFAULTS: Required<GazeDebugOptions> = {
    showLandmarks: true,
    showEyeballCenters: true,
    showPupilOrigins: true,
    showGazeRays: true,
    showReadout: true,
    rayLengthCm: 25,
};

// ── Main debug draw ────────────────────────────────────────────────
export function drawGazeDebug(
    gaze: IrisGazeOutput,
    landmarks: { x: number; y: number }[] | null,
    ctx: CanvasRenderingContext2D,
    frameWidth: number,
    frameHeight: number,
    fx: number,
    opts: GazeDebugOptions = {},
): void {
    const o = { ...DEFAULTS, ...opts };

    // Landmark IDs (must match irisUnit.ts)
    const LM = {
        left:  { pupil: 473, innerCorner: 362, outerCorner: 263,
            irisTop: 475, irisBottom: 477, irisInner: 476, irisOuter: 474 },
        right: { pupil: 468, innerCorner: 133, outerCorner: 33,
            irisTop: 470, irisBottom: 472, irisInner: 469, irisOuter: 471 },
    };

    // ── Raw landmarks (from mediapipe, normalized → pixels) ──
    if (o.showLandmarks && landmarks) {
        const denorm = (id: number): Point => ({
            x: landmarks[id].x * frameWidth,
            y: landmarks[id].y * frameHeight,
        });

        // Left eye — cyan family
        dot(ctx, denorm(LM.left.pupil),       'cyan', 'Lp');
        dot(ctx, denorm(LM.left.innerCorner), '#0088ff', 'Li', 3);
        dot(ctx, denorm(LM.left.outerCorner), '#0088ff', 'Lo', 3);
        dot(ctx, denorm(LM.left.irisTop),     '#66ddff', undefined, 2);
        dot(ctx, denorm(LM.left.irisBottom),  '#66ddff', undefined, 2);
        dot(ctx, denorm(LM.left.irisInner),   '#66ddff', undefined, 2);
        dot(ctx, denorm(LM.left.irisOuter),   '#66ddff', undefined, 2);

        // Right eye — magenta family
        dot(ctx, denorm(LM.right.pupil),       'magenta', 'Rp');
        dot(ctx, denorm(LM.right.innerCorner), '#ff0088', 'Ri', 3);
        dot(ctx, denorm(LM.right.outerCorner), '#ff0088', 'Ro', 3);
        dot(ctx, denorm(LM.right.irisTop),     '#ff66dd', undefined, 2);
        dot(ctx, denorm(LM.right.irisBottom),  '#ff66dd', undefined, 2);
        dot(ctx, denorm(LM.right.irisInner),   '#ff66dd', undefined, 2);
        dot(ctx, denorm(LM.right.irisOuter),   '#ff66dd', undefined, 2);
    }

    // ── 3D points projected back to pixels ──
    // These should land ON TOP of their corresponding landmarks if the math is right.
    // If they don't, your metric2Pixel / landmark2Metric inverse is off.
    if (o.showEyeballCenters) {
        const lEye = metric2Pixel(gaze.left.origin, frameWidth, frameHeight, fx);
        const rEye = metric2Pixel(gaze.right.origin, frameWidth, frameHeight, fx);
        const cEye = metric2Pixel(gaze.cyclopean.origin, frameWidth, frameHeight, fx);
        dot(ctx, lEye, '#00ff00', 'L-eyeball', 5);
        dot(ctx, rEye, '#00ff00', 'R-eyeball', 5);
        dot(ctx, cEye, '#ffff00', 'C', 5);
    }

    // ── Gaze rays ──
    if (o.showGazeRays) {
        const drawRay = (info: GazeInfo, color: string, label: string) => {
            const { origin, vector } = info;

            // Cap ray at (or just before) screen plane
            const tToScreen = Math.abs(vector.z) > 1e-6 ? -origin.z / vector.z : Infinity;
            const t = Math.min(o.rayLengthCm, Math.max(0, tToScreen - 1));

            const end: Coordinate3D = {
                x: origin.x + t * vector.x,
                y: origin.y + t * vector.y,
                z: origin.z + t * vector.z,
            };
            const originPx = metric2Pixel(origin, frameWidth, frameHeight, fx);
            const endPx = metric2Pixel(end, frameWidth, frameHeight, fx);
            line(ctx, originPx, endPx, color, 2);
            dot(ctx, endPx, color, label, 3);
        };

        drawRay(gaze.left,      'cyan',    'L→');
        drawRay(gaze.right,     'magenta', 'R→');
        drawRay(gaze.cyclopean, 'yellow',  'C→');
    }

    // ── Text readout in corner ──
    if (o.showReadout) {
        const lines = [
            `fx=${fx.toFixed(0)}px  frame=${frameWidth}x${frameHeight}`,
            `L-eye origin: (${gaze.left.origin.x.toFixed(1)}, ${gaze.left.origin.y.toFixed(1)}, ${gaze.left.origin.z.toFixed(1)})`,
            `R-eye origin: (${gaze.right.origin.x.toFixed(1)}, ${gaze.right.origin.y.toFixed(1)}, ${gaze.right.origin.z.toFixed(1)})`,
            `L-gaze vec:   (${gaze.left.vector.x.toFixed(2)}, ${gaze.left.vector.y.toFixed(2)}, ${gaze.left.vector.z.toFixed(2)})`,
            `R-gaze vec:   (${gaze.right.vector.x.toFixed(2)}, ${gaze.right.vector.y.toFixed(2)}, ${gaze.right.vector.z.toFixed(2)})`,
            `C-gaze vec:   (${gaze.cyclopean.vector.x.toFixed(2)}, ${gaze.cyclopean.vector.y.toFixed(2)}, ${gaze.cyclopean.vector.z.toFixed(2)})`,
            `screenPog cm: ${gaze.screenPog ? `(${gaze.screenPog.x.toFixed(1)}, ${gaze.screenPog.y.toFixed(1)})` : 'null'}`,
        ];

        ctx.font = '12px monospace';
        const pad = 6;
        const lineH = 14;
        const boxW = 320;
        const boxH = lines.length * lineH + pad * 2;

        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(pad, pad, boxW, boxH);

        ctx.fillStyle = 'white';
        lines.forEach((l, i) => ctx.fillText(l, pad + 4, pad + lineH * (i + 1) - 2));
    }
}