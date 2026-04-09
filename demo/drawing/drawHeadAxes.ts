interface Point {
    x: number;
    y: number;
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