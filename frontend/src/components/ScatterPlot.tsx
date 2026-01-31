/**
 * ScatterPlot component for 2D embedding visualization.
 */

import React, { useEffect, useState } from 'react';
import { Box, Spinner, Center } from '@chakra-ui/react';
import Plot from 'react-plotly.js';
import { useAppContext } from '../context/AppContext';
import type { PlotSelectionEvent } from 'plotly.js';

// Extended color palette for multiple players
// Extended color palette for multiple players
const PLAYER_COLORS = [
    '#e74c3c', // Red
    '#3498db', // Blue
    '#2ecc71', // Green
    '#f39c12', // Orange
    '#9b59b6', // Purple
    '#1abc9c', // Turquoise
    '#e67e22', // Carrot
    '#34495e', // Dark gray
    '#16a085', // Green sea
    '#c0392b', // Dark red
];

const COLOR_CLUSTER1 = '#C0392B';
const COLOR_CLUSTER2 = '#2874A6';

const ScatterPlot: React.FC = () => {
    const {
        embedding,
        playerLabels,
        playerNames,
        gameIds,
        cluster1,
        cluster2,
        setCluster1,
        setCluster2,
        isLoading,
    } = useAppContext();

    const [plotData, setPlotData] = useState<any[]>([]);

    useEffect(() => {
        if (embedding.length === 0 || playerLabels.length === 0) return;

        const cluster1Set = new Set(cluster1 || []);
        const cluster2Set = new Set(cluster2 || []);

        // Get unique player classes
        const uniqueClasses = Array.from(new Set(playerLabels)).sort((a, b) => a - b);

        // Create data structures for each player class
        const playerData: Record<number, {
            x: number[];
            y: number[];
            text: string[];
            indices: number[];
        }> = {};

        uniqueClasses.forEach(playerClass => {
            playerData[playerClass] = { x: [], y: [], text: [], indices: [] };
        });

        // Cluster points
        const c1X: number[] = [];
        const c1Y: number[] = [];
        const c1Text: string[] = [];
        const c1Indices: number[] = [];

        const c2X: number[] = [];
        const c2Y: number[] = [];
        const c2Text: string[] = [];
        const c2Indices: number[] = [];

        // Distribute points to player classes or clusters
        embedding.forEach((point, idx) => {
            const playerClass = playerLabels[idx];
            const gameId = gameIds[idx];

            if (cluster1Set.has(idx)) {
                c1X.push(point[0]);
                c1Y.push(point[1]);
                c1Text.push(`Game ${gameId}`);
                c1Indices.push(idx);
            } else if (cluster2Set.has(idx)) {
                c2X.push(point[0]);
                c2Y.push(point[1]);
                c2Text.push(`Game ${gameId}`);
                c2Indices.push(idx);
            } else {
                playerData[playerClass].x.push(point[0]);
                playerData[playerClass].y.push(point[1]);
                playerData[playerClass].text.push(`Game ${gameId}`);
                playerData[playerClass].indices.push(idx);
            }
        });

        const traces: any[] = [];

        // Create a trace for each player class
        uniqueClasses.forEach((playerClass, classIdx) => {
            const data = playerData[playerClass];
            if (data.x.length > 0) {
                const playerName = playerNames[playerClass] || `Player ${playerClass}`;
                traces.push({
                    x: data.x,
                    y: data.y,
                    mode: 'markers',
                    type: 'scatter',
                    marker: {
                        color: PLAYER_COLORS[playerClass % PLAYER_COLORS.length],
                        size: 8,
                        opacity: 0.7,
                        line: { width: 0.5, color: 'white' },
                    },
                    text: data.text,
                    hoverinfo: 'text',
                    customdata: data.indices,
                    name: playerName,
                    legendgroup: `player${playerClass}`,
                });
            }
        });

        // Cluster 1 trace
        if (c1X.length > 0) {
            traces.push({
                x: c1X,
                y: c1Y,
                mode: 'markers',
                type: 'scatter',
                marker: {
                    color: COLOR_CLUSTER1,
                    size: 12,
                    opacity: 1.0,
                    line: { width: 1, color: 'white' },
                },
                text: c1Text,
                hoverinfo: 'text',
                customdata: c1Indices,
                name: 'Cluster 1',
                legendgroup: 'cluster1',
            });
        }

        // Cluster 2 trace
        if (c2X.length > 0) {
            traces.push({
                x: c2X,
                y: c2Y,
                mode: 'markers',
                type: 'scatter',
                marker: {
                    color: COLOR_CLUSTER2,
                    size: 12,
                    opacity: 1.0,
                    line: { width: 1, color: 'white' },
                },
                text: c2Text,
                hoverinfo: 'text',
                customdata: c2Indices,
                name: 'Cluster 2',
                legendgroup: 'cluster2',
            });
        }

        setPlotData(traces);
    }, [embedding, playerLabels, playerNames, gameIds, cluster1, cluster2]);

    const handleSelection = (event: Readonly<PlotSelectionEvent>) => {
        if (!event || !event.points) return;

        const selectedIndices = event.points
            .map((p: any) => p.customdata)
            .filter((idx: any) => idx !== undefined) as number[];

        if (selectedIndices.length === 0) return;

        // Logic: first selection -> cluster1, second -> cluster2, third -> reset and new cluster1
        if (!cluster1 && !cluster2) {
            setCluster1(selectedIndices);
        } else if (cluster1 && !cluster2) {
            setCluster2(selectedIndices);
        } else {
            // Both exist, reset and set new cluster1
            setCluster1(selectedIndices);
            setCluster2(null);
        }
    };

    if (isLoading) {
        return (
            <Center h="100%">
                <Spinner size="xl" />
            </Center>
        );
    }

    if (embedding.length === 0) {
        return (
            <Center h="100%">
                <Box>No data loaded</Box>
            </Center>
        );
    }

    return (
        <Box h="100%" w="100%">
            <Plot
                data={plotData}
                layout={{
                    title: '2D Embedding',
                    autosize: true,
                    dragmode: 'select',
                    hovermode: 'closest',
                    showlegend: true,
                    margin: { l: 40, r: 40, t: 40, b: 40 },
                    xaxis: { title: '', showticklabels: false, showline: false, zeroline: false, showgrid: false },
                    yaxis: { title: '', showticklabels: false, showline: false, zeroline: false, showgrid: false },
                    paper_bgcolor: '#1A202C', // gray.900
                    plot_bgcolor: '#1A202C',  // gray.900
                    font: { color: 'white' },
                } as any}
                config={{
                    displayModeBar: true,
                    displaylogo: false,
                }}
                onSelected={handleSelection}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler
            />
        </Box>
    );
};

export default ScatterPlot;
