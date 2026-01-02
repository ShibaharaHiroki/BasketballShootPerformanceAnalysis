/**
 * SpatialHeatmap component for court visualization with importance overlay.
 */

import React, { useEffect, useState } from 'react';
import { Box, HStack, VStack, Divider, Select, Radio, RadioGroup, Text, useToast, Stack } from '@chakra-ui/react';
import Plot from 'react-plotly.js';
import { useAppContext } from '../context/AppContext';
import { apiClient } from '../services/api';

const SpatialHeatmap: React.FC = () => {
    const { cluster1, cluster2, metadata, tensorShape } = useAppContext();
    const toast = useToast();

    const [timeBin, setTimeBin] = useState<string>('0');
    const [channel, setChannel] = useState<string>('attempts');
    const [contribData, setContribData] = useState<number[][][] | null>(null);
    const [plotData, setPlotData] = useState<any[]>([]);

    const S_bins = tensorShape.length > 0 ? tensorShape[1] : 4;
    const gridXBins = metadata?.grid_x_bins || 17;
    const gridYBins = metadata?.grid_y_bins || 16;

    // Fetch contribution tensor when clusters change
    useEffect(() => {
        const fetchContribution = async () => {
            console.log('SpatialHeatmap: cluster1=', cluster1);
            console.log('SpatialHeatmap: cluster2=', cluster2);

            if (!cluster1 || !cluster2 || cluster1.length === 0 || cluster2.length === 0) {
                console.log('SpatialHeatmap: clusters not ready');
                setContribData(null);
                return;
            }

            console.log('SpatialHeatmap: fetching contribution tensor...');
            try {
                const response = await apiClient.analyzeClusters(cluster1, cluster2);
                console.log('SpatialHeatmap: got contribution tensor', response.contrib_tensor);
                setContribData(response.contrib_tensor);
            } catch (err: any) {
                console.error('SpatialHeatmap: error fetching contribution', err);
                toast({
                    title: 'Error',
                    description: 'Failed to compute contribution tensor',
                    status: 'error',
                    duration: 3000,
                    isClosable: true,
                });
            }
        };

        fetchContribution();
    }, [cluster1, cluster2]);

    // Generate plot data
    useEffect(() => {
        if (!contribData || !metadata) {
            setPlotData([]);
            return;
        }

        const xEdges = metadata.x_edges || [];
        const yEdges = metadata.y_edges || [];

        if (xEdges.length === 0 || yEdges.length === 0) {
            setPlotData([]);
            return;
        }

        const cellW = xEdges[1] - xEdges[0];
        const cellH = yEdges[1] - yEdges[0];

        const xs: number[] = [];
        const ys: number[] = [];
        const vals: number[] = [];

        // Aggregate contribution based on channel selection
        for (let iy = 0; iy < gridYBins; iy++) {
            for (let ix = 0; ix < gridXBins; ix++) {
                const cellIdx = iy * gridXBins + ix;
                xs.push(xEdges[ix] + cellW / 2);
                ys.push(yEdges[iy] + cellH / 2);

                let val = 0;
                if (timeBin === 'all') {
                    // Sum across all time bins
                    for (let t = 0; t < S_bins; t++) {
                        if (channel === 'all') {
                            val += contribData[t][cellIdx][0] + contribData[t][cellIdx][1] + contribData[t][cellIdx][2];  // all channels
                        } else if (channel === 'attempts') {
                            val += contribData[t][cellIdx][0];  // attempts
                        } else if (channel === 'points') {
                            val += contribData[t][cellIdx][2];  // weighted_makes (points)
                        } else {  // made
                            val += contribData[t][cellIdx][1];  // makes
                        }
                    }
                } else {
                    const t = parseInt(timeBin);
                    if (channel === 'all') {
                        val = contribData[t][cellIdx][0] + contribData[t][cellIdx][1] + contribData[t][cellIdx][2];  // all channels
                    } else if (channel === 'attempts') {
                        val = contribData[t][cellIdx][0];  // attempts
                    } else if (channel === 'points') {
                        val = contribData[t][cellIdx][2];  // weighted_makes (points)
                    } else {  // made
                        val = contribData[t][cellIdx][1];  // makes
                    }
                }
                vals.push(val);
            }
        }

        // Calculate grid cell diagonal for maximum marker size
        const cellDiagonal = Math.sqrt(cellW * cellW + cellH * cellH);

        // Use absolute scale for marker size (not normalized to current selection)
        // This allows consistent comparison across different time/space selections
        const FIXED_MAX_IMPORTANCE = 2.5; // Adjusted for individual time bins/channels (not aggregated 'All')
        const sizes = vals.map((v) => Math.min(v / FIXED_MAX_IMPORTANCE, 1.0) * cellDiagonal * 24);

        const traces: any[] = [
            // Court marker
            {
                x: xs,
                y: ys,
                mode: 'markers',
                type: 'scatter',
                marker: {
                    size: sizes,
                    color: 'rgba(192, 57, 43, 0.3)',
                    line: { color: 'rgba(192, 57, 43, 0.8)', width: 1 },
                },
                customdata: vals,
                hovertemplate: 'x=%{x:.1f}, y=%{y:.1f}<br>importance=%{customdata:.3f}<extra></extra>',
                showlegend: false,
            },
        ];

        // 3ポイントアーク生成（完全版）
        const radius = 237.5; // 23.75 ft
        const cornerX = 220;  // コーナーの 3P ライン x
        const yBreak = Math.sqrt(radius * radius - cornerX * cornerX); // ≒ 89.48

        // 左右コーナーに対応する角度を計算
        const thetaLeft = Math.atan2(yBreak, -cornerX); // ≒ 157.9°
        const thetaRight = Math.atan2(yBreak, cornerX); // ≒  22.1°

        const numPoints = 200;
        const thetaArr: number[] = [];
        for (let i = 0; i <= numPoints; i++) {
            // 左端から右端までを滑らかにスイープする
            const t = thetaLeft + (thetaRight - thetaLeft) * (i / numPoints);
            thetaArr.push(t);
        }

        const xArc = thetaArr.map((t) => radius * Math.cos(t));
        const yArc = thetaArr.map((t) => radius * Math.sin(t));

        // Plotly のトレースに追加
        traces.push({
            x: xArc,
            y: yArc,
            mode: 'lines',
            type: 'scatter',
            line: { color: 'white', width: 1 },
            showlegend: false,
            hoverinfo: 'skip',
        });

        setPlotData(traces);
    }, [contribData, timeBin, channel, metadata, gridXBins, gridYBins, S_bins]);

    // NBA half-court visualization - Based on official NBA dimensions
    // Coordinate system: X from -250 to 250 (50 feet wide), Y from -47.5 to 422.5 (47 feet deep)
    // Scale: 10 units = 1 foot

    // 3ポイントアークとコーナーラインの接続点を計算
    const radius = 237.5; // 23.75 ft
    const cornerX = 220;  // コーナーの 3P ライン x
    const yBreak = Math.sqrt(radius * radius - cornerX * cornerX); // ≒ 89.48

    const courtShapes = [
        // Court background
        {
            type: 'rect',
            x0: -250,
            y0: -47.5,
            x1: 250,
            y1: 422.5,
            line: { color: 'rgba(255,255,255,0.9)', width: 3 },
            fillcolor: 'rgba(20, 30, 50, 0.3)',
            layer: 'below',
        },
        // Paint area (the key) - 16 feet wide (160 units), 19 feet deep (190 units)
        {
            type: 'rect',
            x0: -80,
            y0: -47.5,
            x1: 80,
            y1: 142.5,
            line: { color: 'rgba(255,255,255,0.7)', width: 2 },
            fillcolor: 'rgba(0,0,0,0)',
            layer: 'above',
        },
        // Free throw lane - 12 feet wide (120 units)
        {
            type: 'rect',
            x0: -60,
            y0: -47.5,
            x1: 60,
            y1: 142.5,
            line: { color: 'rgba(255,255,255,0.7)', width: 2 },
            fillcolor: 'rgba(0,0,0,0)',
            layer: 'above',
        },
        // Basket rim - 18 inches diameter (1.5 feet = 15 units)
        // バスケット中心を (0, 0) として描画
        {
            type: 'circle',
            x0: -7.5,
            y0: -7.5,
            x1: 7.5,
            y1: 7.5,
            line: { color: 'rgba(255, 140, 0, 1)', width: 2.5 },
            fillcolor: 'rgba(0,0,0,0)',
            layer: 'above',
        },
        // Free throw circle - 6 feet radius (60 units), center at Y=142.5
        {
            type: 'circle',
            x0: -60,
            y0: 82.5,
            x1: 60,
            y1: 202.5,
            line: { color: 'rgba(255,255,255,0.7)', width: 2 },
            fillcolor: 'rgba(0,0,0,0)',
            layer: 'above',
        },
        // Restricted area arc - 4 feet radius (40 units) from basket center (0,0)
        // 中心を正しくバスケット位置に合わせる
        {
            type: 'path',
            path: 'M -40 0 A 40 40 0 0 1 40 0',
            line: { color: 'rgba(255,255,255,0.6)', width: 2 },
            layer: 'above',
        },
        // Backboard - 6 feet wide (60 units), 4 feet from baseline
        // ベースライン -47.5 から 4ft（40units）離れた y = -7.5 に修正
        {
            type: 'line',
            x0: -30,
            y0: -7.5,
            x1: 30,
            y1: -7.5,
            line: { color: 'rgba(255,255,255,0.9)', width: 4 },
            layer: 'above',
        },
        // 3-point line corner - straight line part
        // 3PT corner distance: 22 feet from basket center = 220 units in X
        // Arc begins at y = yBreak (≒ 89.5)
        {
            type: 'line',
            x0: -220,
            y0: -47.5,
            x1: -220,
            y1: yBreak,
            line: { color: 'rgba(255,255,255,0.8)', width: 2.5 },
            layer: 'above',
        },
        {
            type: 'line',
            x0: 220,
            y0: -47.5,
            x1: 220,
            y1: yBreak,
            line: { color: 'rgba(255,255,255,0.8)', width: 2.5 },
            layer: 'above',
        },
        // Lane space marks (hash marks) 以下はそのままでも OK
        {
            type: 'line',
            x0: -80,
            y0: -17.5,
            x1: -73,
            y1: -17.5,
            line: { color: 'rgba(255,255,255,0.6)', width: 1.5 },
            layer: 'above',
        },
        {
            type: 'line',
            x0: 80,
            y0: -17.5,
            x1: 73,
            y1: -17.5,
            line: { color: 'rgba(255,255,255,0.6)', width: 1.5 },
            layer: 'above',
        },
        {
            type: 'line',
            x0: -80,
            y0: 32.5,
            x1: -73,
            y1: 32.5,
            line: { color: 'rgba(255,255,255,0.6)', width: 1.5 },
            layer: 'above',
        },
        {
            type: 'line',
            x0: 80,
            y0: 32.5,
            x1: 73,
            y1: 32.5,
            line: { color: 'rgba(255,255,255,0.6)', width: 1.5 },
            layer: 'above',
        },
        {
            type: 'line',
            x0: -80,
            y0: 62.5,
            x1: -73,
            y1: 62.5,
            line: { color: 'rgba(255,255,255,0.6)', width: 1.5 },
            layer: 'above',
        },
        {
            type: 'line',
            x0: 80,
            y0: 62.5,
            x1: 73,
            y1: 62.5,
            line: { color: 'rgba(255,255,255,0.6)', width: 1.5 },
            layer: 'above',
        },
        {
            type: 'line',
            x0: -80,
            y0: 92.5,
            x1: -73,
            y1: 92.5,
            line: { color: 'rgba(255,255,255,0.6)', width: 1.5 },
            layer: 'above',
        },
        {
            type: 'line',
            x0: 80,
            y0: 92.5,
            x1: 73,
            y1: 92.5,
            line: { color: 'rgba(255,255,255,0.6)', width: 1.5 },
            layer: 'above',
        },
        {
            type: 'path',
            // 左コーナー (-cornerX, yBreak) から右コーナー (cornerX, yBreak) へ
            path: `M ${-cornerX} ${yBreak} A ${radius} ${radius} 0 0 1 ${cornerX} ${yBreak}`,
            line: { color: 'rgba(255,255,255,0.9)', width: 3 },
            layer: 'above',
        },
    ];


    const timeBinOptions: { value: string; label: string }[] = [];
    const quarterLabels = ['1Q', '2Q', '3Q', '4Q'];
    for (let t = 0; t < S_bins && t < 4; t++) {
        timeBinOptions.push({
            value: String(t),
            label: quarterLabels[t],
        });
    }

    // Get player names for clusters
    const { playerLabels, gameIds, playerNames } = useAppContext();

    const getClusterPlayers = (clusterIndices: number[] | null) => {
        if (!clusterIndices || !playerLabels || !gameIds) return [];

        return clusterIndices.map(idx => {
            const playerIdx = playerLabels[idx];
            const gameId = gameIds[idx];
            const playerName = playerNames[playerIdx] || `Player ${playerIdx}`;
            return { gameId, playerName, playerIdx };
        }).sort((a, b) => a.gameId - b.gameId); // Sort by game ID (chronological)
    };

    const cluster1Players = getClusterPlayers(cluster1);
    const cluster2Players = getClusterPlayers(cluster2);

    // Calculate timeline range
    const allGames = [...cluster1Players, ...cluster2Players];
    const minGameId = allGames.length > 0 ? Math.min(...allGames.map(g => g.gameId)) : 0;
    const maxGameId = allGames.length > 0 ? Math.max(...allGames.map(g => g.gameId)) : 1;
    const gameRange = maxGameId - minGameId || 1;

    // Player colors (same as ScatterPlot)
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

    return (
        <HStack h="100%" w="100%" spacing={2} align="stretch">
            {/* Player Timeline Panel */}
            <Box flex="3" h="100%" overflowY="auto" bg="gray.800" p={3} borderRadius="md">
                <VStack align="stretch" spacing={4}>
                    {/* Cluster 1 Players */}
                    <Box>
                        <Text fontSize="sm" fontWeight="bold" color="red.400" mb={3}>
                            Cluster 1 ({cluster1Players.length} games)
                        </Text>
                        {cluster1Players.length > 0 ? (
                            <VStack align="stretch" spacing={3}>
                                {/* Group by player */}
                                {Array.from(new Set(cluster1Players.map(p => p.playerIdx))).map(playerIdx => {
                                    const playerGames = cluster1Players.filter(p => p.playerIdx === playerIdx);
                                    const playerColor = PLAYER_COLORS[playerIdx % PLAYER_COLORS.length];
                                    const playerName = playerGames[0].playerName;

                                    return (
                                        <Box key={playerIdx}>
                                            <Text fontSize="xs" color="white" mb={1} fontWeight="semibold">
                                                {playerName} ({playerGames.length})
                                            </Text>
                                            <Box position="relative" h="12px" bg="gray.700" borderRadius="full">
                                                {playerGames.map((p, idx) => {
                                                    const position = ((p.gameId - minGameId) / gameRange) * 100;
                                                    return (
                                                        <Box
                                                            key={idx}
                                                            position="absolute"
                                                            left={`${position}%`}
                                                            top="50%"
                                                            transform="translate(-50%, -50%)"
                                                            w="8px"
                                                            h="8px"
                                                            bg={playerColor}
                                                            borderRadius="full"
                                                            title={`Game #${p.gameId}`}
                                                            cursor="pointer"
                                                        />
                                                    );
                                                })}
                                            </Box>
                                        </Box>
                                    );
                                })}
                            </VStack>
                        ) : (
                            <Text fontSize="xs" color="gray.500">No games selected</Text>
                        )}
                    </Box>

                    <Divider borderColor="gray.600" />

                    {/* Cluster 2 Players */}
                    <Box>
                        <Text fontSize="sm" fontWeight="bold" color="blue.400" mb={3}>
                            Cluster 2 ({cluster2Players.length} games)
                        </Text>
                        {cluster2Players.length > 0 ? (
                            <VStack align="stretch" spacing={3}>
                                {/* Group by player */}
                                {Array.from(new Set(cluster2Players.map(p => p.playerIdx))).map(playerIdx => {
                                    const playerGames = cluster2Players.filter(p => p.playerIdx === playerIdx);
                                    const playerColor = PLAYER_COLORS[playerIdx % PLAYER_COLORS.length];
                                    const playerName = playerGames[0].playerName;

                                    return (
                                        <Box key={playerIdx}>
                                            <Text fontSize="xs" color="white" mb={1} fontWeight="semibold">
                                                {playerName} ({playerGames.length})
                                            </Text>
                                            <Box position="relative" h="12px" bg="gray.700" borderRadius="full">
                                                {playerGames.map((p, idx) => {
                                                    const position = ((p.gameId - minGameId) / gameRange) * 100;
                                                    return (
                                                        <Box
                                                            key={idx}
                                                            position="absolute"
                                                            left={`${position}%`}
                                                            top="50%"
                                                            transform="translate(-50%, -50%)"
                                                            w="8px"
                                                            h="8px"
                                                            bg={playerColor}
                                                            borderRadius="full"
                                                            title={`Game #${p.gameId}`}
                                                            cursor="pointer"
                                                        />
                                                    );
                                                })}
                                            </Box>
                                        </Box>
                                    );
                                })}
                            </VStack>
                        ) : (
                            <Text fontSize="xs" color="gray.500">No games selected</Text>
                        )}
                    </Box>
                </VStack>
            </Box>

            {/* Spatial Heatmap */}
            <Box flex="7" h="100%" w="100%">
                <HStack spacing={2} mb={2} fontSize="xs" flexWrap="wrap">
                    <Select
                        size="xs"
                        value={timeBin}
                        onChange={(e) => setTimeBin(e.target.value)}
                        w="120px"
                        color="white"
                        bg="gray.800"
                    >
                        {timeBinOptions.map((opt) => (
                            <option key={opt.value} value={opt.value} style={{ color: 'black' }}>
                                {opt.label}
                            </option>
                        ))}
                    </Select>

                </HStack>

                <Box h="calc(100% - 40px)" w="100%">
                    <Plot
                        data={plotData}
                        layout={{
                            autosize: true,
                            template: 'plotly_white',
                            margin: { l: 40, r: 40, t: 40, b: 40 },
                            xaxis: {
                                range: [-250, 250],
                                showgrid: false,
                                zeroline: false,
                                showticklabels: false,
                            },
                            yaxis: {
                                range: [-47.5, 422.5],
                                showgrid: false,
                                zeroline: false,
                                showticklabels: false,
                                scaleanchor: 'x',
                                scaleratio: 1,
                            },
                            shapes: courtShapes as any,
                            paper_bgcolor: '#1A202C', // gray.900
                            plot_bgcolor: '#1A202C',  // gray.900
                            font: { color: 'white' },
                        } as any}
                        config={{
                            displayModeBar: false,
                        }}
                        style={{ width: '100%', height: '100%' }}
                        useResizeHandler
                    />
                </Box>
            </Box>
        </HStack>
    );
};

export default SpatialHeatmap;
