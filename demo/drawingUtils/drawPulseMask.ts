import type {RegionDetail} from "../../src/rPPG";

export function drawPulseMask(
    regions: Record<string, RegionDetail>,
    canvasContext: CanvasRenderingContext2D,
    width: number,
    height: number,
    minPulse: number = .008,
    maxPulse: number = .019){
    canvasContext.clearRect(0, 0, width, height);
    for (const regionName in regions) {
        const regionData = regions[regionName];
        const polygon = regionData.polygon;
        if (polygon.length === 0) continue;
        canvasContext.beginPath();
        canvasContext.moveTo(polygon[0].x, polygon[0].y);
        for (let i = 1; i < polygon.length; i++) canvasContext.lineTo(polygon[i].x, polygon[i].y);
        canvasContext.closePath();
        const pulse = regionData.pulse ?? 0;
        const alpha = Math.min(1, Math.max(0, (pulse + minPulse) / maxPulse)); // 0.009) / 0.015
        canvasContext.fillStyle = `rgba(255, 0, 0, ${alpha})`;
        canvasContext.fill();
    }
}
