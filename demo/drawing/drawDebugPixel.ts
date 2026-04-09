// drawing/drawDebugPixel.ts

interface Point {
    x: number;
    y: number;
}

/**
 * Draw a single point on the overlay canvas for debugging.
 *
 * Coordinate convention: image-space pixels matching the video frame.
 * Origin is top-left, +x right, +y down — same as the canvas itself.
 * If your point is in centered-pixel space (origin at image center, +y up),
 * convert before calling: { x: cx + W/2, y: -cy + H/2 }.
 */
export function drawDebugPixel(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    options: {
        color?: string;
        radius?: number;
        label?: string;
    } = {},
): void {
    const { color = 'lime', radius = 6, label } = options;

    ctx.save();
    // Outline so it's visible against any background.
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;
    ctx.fillStyle = color;

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    if (label) {
        ctx.font = '12px sans-serif';
        ctx.fillStyle = 'white';
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 3;
        ctx.strokeText(label, x + radius + 4, y - radius);
        ctx.fillText(label, x + radius + 4, y - radius);
    }
    ctx.restore();
}

export function drawHeadAxes(
    ctx: CanvasRenderingContext2D,
    axes: { origin: Point; xAxis: Point; yAxis: Point; zAxis: Point },
): void {
    const drawLine = (to: Point, color: string, label: string) => {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(axes.origin.x, axes.origin.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();

        ctx.fillStyle = color;
        ctx.font = 'bold 14px sans-serif';
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 3;
        ctx.strokeText(label, to.x + 4, to.y);
        ctx.fillText(label, to.x + 4, to.y);
        ctx.restore();
    };

    drawLine(axes.xAxis, 'red', 'X');
    drawLine(axes.yAxis, 'lime', 'Y');
    drawLine(axes.zAxis, 'blue', 'Z');

    // Origin dot
    ctx.save();
    ctx.fillStyle = 'white';
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(axes.origin.x, axes.origin.y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
}