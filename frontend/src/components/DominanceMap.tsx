/**
 * DominanceMap component - Visualizes which cluster dominates each spatial feature.
 */

import React from 'react';
import Plot from 'react-plotly.js';

interface DominanceMapProps {
    diff: number[];
    metadata: any;
}

// Helpers for court drawing (copied from RawDataExplorer to avoid circular dependencies or need for common lib refactor yet)
function getBasketballCourtShapes() {
    // 3-point intersection calculation
    const radius = 237.5;
    const cornerX = 220;
    const yBreak = Math.sqrt(radius * radius - cornerX * cornerX);

    return [
        // Court background
        { type: 'rect', x0: -250, y0: -47.5, x1: 250, y1: 422.5, line: { color: 'rgba(255,255,255,0.9)', width: 3 }, fillcolor: 'rgba(20, 30, 50, 0.3)', layer: 'below' },
        // Paint
        { type: 'rect', x0: -80, y0: -47.5, x1: 80, y1: 142.5, line: { color: 'rgba(255,255,255,0.7)', width: 2 }, fillcolor: 'rgba(0,0,0,0)', layer: 'above' },
        // Lane
        { type: 'rect', x0: -60, y0: -47.5, x1: 60, y1: 142.5, line: { color: 'rgba(255,255,255,0.7)', width: 2 }, fillcolor: 'rgba(0,0,0,0)', layer: 'above' },
        // Hoop
        { type: 'circle', x0: -7.5, y0: -7.5, x1: 7.5, y1: 7.5, line: { color: 'rgba(255, 140, 0, 1)', width: 2.5 }, fillcolor: 'rgba(0,0,0,0)', layer: 'above' },
        // FT Circle
        { type: 'circle', x0: -60, y0: 82.5, x1: 60, y1: 202.5, line: { color: 'rgba(255,255,255,0.7)', width: 2 }, fillcolor: 'rgba(0,0,0,0)', layer: 'above' },
        // Restricted Arc
        { type: 'path', path: 'M -40 0 A 40 40 0 0 1 40 0', line: { color: 'rgba(255,255,255,0.6)', width: 2 }, layer: 'above' },
        // Backboard
        { type: 'line', x0: -30, y0: -7.5, x1: 30, y1: -7.5, line: { color: 'rgba(255,255,255,0.9)', width: 4 }, layer: 'above' },
        // Corner Lines
        { type: 'line', x0: -220, y0: -47.5, x1: -220, y1: yBreak, line: { color: 'rgba(255,255,255,0.8)', width: 2.5 }, layer: 'above' },
        { type: 'line', x0: 220, y0: -47.5, x1: 220, y1: yBreak, line: { color: 'rgba(255,255,255,0.8)', width: 2.5 }, layer: 'above' },
    ] as any;
}

function get3PTArcTrace() {
    const radius = 237.5;
    const cornerX = 220;
    const yBreak = Math.sqrt(radius * radius - cornerX * cornerX);
    const thetaLeft = Math.atan2(yBreak, -cornerX);
    const thetaRight = Math.atan2(yBreak, cornerX);
    const numPoints = 200;
    const theta = [];
    for (let i = 0; i <= numPoints; i++) theta.push(thetaLeft + (thetaRight - thetaLeft) * (i / numPoints));

    return {
        x: theta.map(t => radius * Math.cos(t)),
        y: theta.map(t => radius * Math.sin(t)),
        mode: 'lines',
        line: { color: 'white', width: 1 },
        showlegend: false,
        hoverinfo: 'skip',
    };
}

const DominanceMap: React.FC<DominanceMapProps> = ({ diff, metadata }) => {
    // Create grid coordinates from metadata
    // Metadata usually contains x_edges and y_edges. We need centers.
    // Assuming diff is flattened array corresponding to (y, x) grid

    if (!metadata || !diff || diff.length === 0) return null;

    const xEdges = metadata.x_edges;
    const yEdges = metadata.y_edges;

    if (!xEdges || !yEdges) return null;

    const xCenters = [];
    for (let i = 0; i < xEdges.length - 1; i++) xCenters.push((xEdges[i] + xEdges[i + 1]) / 2);

    const yCenters = [];
    for (let i = 0; i < yEdges.length - 1; i++) yCenters.push((yEdges[i] + yEdges[i + 1]) / 2);

    const x: number[] = [];
    const y: number[] = [];
    const color: string[] = [];
    const size: number[] = [];
    const text: string[] = [];
    const customdata: number[] = [];

    // Flatten logic: row-major (y changes, then x changes? or vice versa?)
    // Usually numpy flatten is Row-Major (C-style): last index changes fastest.
    // Tensor is (T, S, V, C). S=Time, V=Space.
    // Normalized tensor: games x time x space x channels.
    // We summed over time, so we have (space,).
    // Space is usually y * x grid flattened using meshgrid or similar.
    // The standard way in this app seems to be row-major (i*gridX + j).

    const gridXBins = xCenters.length;
    const gridYBins = yCenters.length;

    // Normalize diff for sizing? Or just binary color?
    // Request says: "Color circle with dominant cluster color".
    // Let's use Red (Cluster 1, positive diff) and Blue (Cluster 2, negative diff).

    for (let i = 0; i < gridYBins; i++) {
        for (let j = 0; j < gridXBins; j++) {
            const idx = i * gridXBins + j;
            if (idx >= diff.length) continue;

            const val = diff[idx];

            // Only show if there is a meaningful difference?
            // Or show all? Let's show all but scale opacity/size potentially?
            // For now, simple binary color.

            x.push(xCenters[j]);
            y.push(yCenters[i]);
            customdata.push(val);

            if (val > 0) {
                // Cluster 1 Dominant -> Red
                color.push('#e74c3c'); // Red
            } else {
                // Cluster 2 Dominant -> Blue
                color.push('#3498db'); // Blue
            }

            // Size could be fixed or proportional to abs(val)
            // Let's try fixed for clarity as requested "circles", maybe scale slightly
            // But if val is near 0, maybe smaller?
            // "Standardized" values can be small.
            // Let's use a base size.
            size.push(8);

            text.push(`Diff: ${val.toFixed(4)}`);
        }
    }

    const data: any[] = [
        get3PTArcTrace(),
        {
            x: x,
            y: y,
            mode: 'markers',
            marker: {
                color: color,
                size: size,
                opacity: 0.8,
                line: { color: 'rgba(255,255,255,0.3)', width: 1 }
            },
            type: 'scatter',
            text: text,
            hoverinfo: 'text',
            showlegend: false
        }
    ];

    return (
        <Plot
            data={data}
            layout={{
                autosize: true,
                height: 500, // Taller comparison map
                margin: { l: 20, r: 20, t: 30, b: 20 },
                paper_bgcolor: 'black',
                plot_bgcolor: 'black',
                xaxis: { range: [-250, 250], showgrid: false, zeroline: false, showticklabels: false },
                yaxis: { range: [-47.5, 422.5], showgrid: false, zeroline: false, scaleanchor: 'x', scaleratio: 1.0, showticklabels: false },
                shapes: getBasketballCourtShapes(),
                title: { text: 'Dominance Map (Red=Cluster 1, Blue=Cluster 2)', font: { color: 'white' } }
            } as any}
            config={{ displayModeBar: false }}
            style={{ width: '100%', height: '500px' }}
            useResizeHandler
        />
    );
};

export default DominanceMap;
