export interface Point {
    x: number;
    y: number;
}

export interface RGBSample {
    r: number;
    g: number;
    b: number;
    timestamp: number;
}

/**
 * Computes the average RGB value of a specific region of interest (ROI).
 * Supports generic ImageData (browser or node-canvas).
 */
export function extractAverageRGB(
    frame: ImageData,
    region: Point[],
    timestamp: number
): RGBSample {
    const data = frame.data;
    const width = frame.width;
    const height = frame.height;

    // Calculate Bounding Box - note this will work for any polygons passed in (incl triangle)
        // consider using point-in-polygon to calculate actual region
    // Use infinity to ensure first comparison is always stored as min/max.
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

    for (const p of region) {
        if (p.x < minX) minX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.x > maxX) maxX = p.x;
        if (p.y > maxY) maxY = p.y;
    }

    // Clamp to image boundaries
        // Note facemesh maintains tracking if outside image so coords can be <0 or >width
        // Floor() to convert float coords to pixel index.
    minX = Math.max(0, Math.floor(minX));
    minY = Math.max(0, Math.floor(minY));
    maxX = Math.min(width - 1, Math.floor(maxX));
    maxY = Math.min(height - 1, Math.floor(maxY));

    // Calculate total R,G,B values across all pixels
    let rSum = 0, gSum = 0, bSum = 0, count = 0;

    // Iterate pixels
    for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
            const index = (y * width + x) * 4;
            // Check if index is within buffer bounds
            if (index + 2 < data.length) {
                rSum += data[index];
                gSum += data[index + 1];
                bSum += data[index + 2];
                count++;
            }
        }
    }

    if (count === 0) {
        return { r: 0, g: 0, b: 0, timestamp };
    }

    // Return average by dividing sum by pixel count.
    return {
        r: rSum / count,
        g: gSum / count,
        b: bSum / count,
        timestamp
    };
}
