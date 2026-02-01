/**
 * SpatialHeatmap component for court visualization with importance overlay.
 */

import React, { useEffect, useState } from 'react';
import { Box, HStack, VStack, Divider, Radio, RadioGroup, Text, useToast, Stack, Menu, MenuButton, MenuList, MenuItem, Button } from '@chakra-ui/react';
import { ChevronDownIcon } from '@chakra-ui/icons';
import Plot from 'react-plotly.js';
import { useAppContext } from '../context/AppContext';
import { apiClient } from '../services/api';

const SpatialHeatmap: React.FC = () => {
    const { cluster1, cluster2, metadata, tensorShape } = useAppContext();
    const toast = useToast();

    // Constant colors for clusters
    const COLOR_C1 = 'rgba(231, 76, 60, 0.7)'; // Red
    const COLOR_C2 = 'rgba(52, 152, 219, 0.7)'; // Blue

    const [timeBin, setTimeBin] = useState<string>('0');
    const [contribData, setContribData] = useState<number[][] | null>(null);
    const [domData, setDomData] = useState<number[][] | null>(null);
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
                setDomData(null);
                return;
            }

            console.log('SpatialHeatmap: fetching contribution tensor...');
            try {
                const response = await apiClient.analyzeClusters(cluster1, cluster2);
                console.log('SpatialHeatmap: got contribution tensor', response.contrib_tensor);
                setContribData(response.contrib_tensor);
                setDomData(response.dominance_tensor);
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
        // Helper function to generate 3PT arc trace
        const generate3PtArcTrace = () => {
            const radius = 237.5;
            const cornerX = 220;
            const yBreak = Math.sqrt(radius * radius - cornerX * cornerX);
            const thetaLeft = Math.atan2(yBreak, -cornerX);
            const thetaRight = Math.atan2(yBreak, cornerX);
            const numPoints = 200;
            const thetaArr: number[] = [];
            for (let i = 0; i <= numPoints; i++) {
                const t = thetaLeft + (thetaRight - thetaLeft) * (i / numPoints);
                thetaArr.push(t);
            }
            return {
                x: thetaArr.map((t) => radius * Math.cos(t)),
                y: thetaArr.map((t) => radius * Math.sin(t)),
                mode: 'lines' as const,
                type: 'scatter' as const,
                line: { color: 'white', width: 1 },
                showlegend: false,
                hoverinfo: 'skip' as const,
            };
        };

        if (!contribData || !metadata) {
            // Still show the 3PT arc even without data
            setPlotData([generate3PtArcTrace()]);
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
        const domVals: number[] = [];

        // Aggregate contribution based on time bin selection
        // contrib_tensor is now 2D: (time_bins, spatial_cells)
        for (let iy = 0; iy < gridYBins; iy++) {
            for (let ix = 0; ix < gridXBins; ix++) {
                const cellIdx = iy * gridXBins + ix;

                let val = 0;
                let dom = 0;

                if (timeBin === 'all') {
                    // Sum across all time bins
                    for (let t = 0; t < S_bins; t++) {
                        val += contribData[t][cellIdx];
                        if (domData) {
                            dom += domData[t][cellIdx]; // Summing difference (valid for mean diff)
                        }
                    }
                } else {
                    const t = parseInt(timeBin);
                    val = contribData[t][cellIdx];
                    if (domData) {
                        dom = domData[t][cellIdx];
                    }
                }

                // Only push points if they have non-zero importance or just to keep grid aligned?
                // The original code pushed everything.
                xs.push(xEdges[ix] + cellW / 2);
                ys.push(yEdges[iy] + cellH / 2);
                vals.push(val);
                domVals.push(dom);
            }
        }

        // --- 【修正】特徴量重要度に基づく相対的スケーリング ---

        // 1. 特徴量重要度の最大値を取得
        // 画面の表示範囲（ズーム等）には依存せず、
        // 計算された「特徴量重要度」データ全体の中での最大値を基準とします。
        let maxImportance = 0;
        if (vals.length > 0) {
            maxImportance = Math.max(...vals);
        }

        // 全て0の場合やエラー回避のためのガード
        if (maxImportance === 0) maxImportance = 1.0;

        // 2. 図形サイズの計算
        // 論文記述: "表示されている特徴量の中で最も重要度が高い要素の円のサイズを...動的に固定する"
        // 論文記述: "その他の領域については...比率に応じて線形に図形サイズを調整する"
        const MAX_CIRCLE_DIAMETER = 28; // Max circle diameter in pixels (approximate cell diagonal)
        const sizes = vals.map((v) => {
            // 最大値に対する比率 (0.0 ～ 1.0)
            const ratio = v / maxImportance;

            // 比率に基づいてサイズを決定
            // MAX_CIRCLE_DIAMETER は定数定義されている前提 (例: 28)
            return ratio * MAX_CIRCLE_DIAMETER;
        });

        // 3. Color mapping: Dominance > 0 ? Red : Blue, Near 0 ? Gray
        const COLOR_NEUTRAL = 'rgba(200, 200, 200, 0.7)'; // Neutral Gray
        const DOMINANCE_THRESHOLD = 0.0001; // Threshold for zero

        const colors = domVals.map(d => {
            // Check for neutral (near zero)
            if (Math.abs(d) < DOMINANCE_THRESHOLD) {
                return COLOR_NEUTRAL;
            }
            // Positive -> Cluster 1 (Red), Negative -> Cluster 2 (Blue)
            return d > 0 ? COLOR_C1 : COLOR_C2;
        });

        const traces: any[] = [
            // Court marker
            {
                x: xs,
                y: ys,
                mode: 'markers',
                type: 'scatter',
                marker: {
                    size: sizes,
                    color: colors,
                    opacity: 0.8,
                    line: { color: 'white', width: 1 }, // White border
                    sizemode: 'diameter',
                },
                text: vals.map((v, i) => {
                    let clusterName = 'Neutral';
                    if (domVals[i] > DOMINANCE_THRESHOLD) clusterName = 'Cluster 1';
                    else if (domVals[i] < -DOMINANCE_THRESHOLD) clusterName = 'Cluster 2';

                    return `Importance: ${v.toFixed(4)}<br>Dominant: ${clusterName}<br>Val: ${domVals[i].toFixed(4)}`;
                }),
                hoverinfo: 'text',
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
    }, [contribData, domData, timeBin, metadata, gridXBins, gridYBins, S_bins]);

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
    // For team_season mode, game IDs are encoded as season*1000000 + original_id
    // Normalize by extracting original game ID for timeline positioning
    const normalizeGameId = (gameId: number) => {
        // If gameId is > 1000000, it's encoded with season prefix
        if (gameId >= 1000000) {
            return gameId % 1000000; // Extract original game ID
        }
        return gameId;
    };

    // Detect team_season mode by checking if playerNames look like seasons
    const isTeamSeasonMode = playerNames.length === 2 &&
        (playerNames[0] === '2022-23' || playerNames[0] === '2023-24');

    const allGames = [...cluster1Players, ...cluster2Players];

    // Calculate min/max per season for team_season mode
    const getSeasonRange = (seasonIdx: number) => {
        const seasonGames = allGames.filter(g => g.playerIdx === seasonIdx);
        const normalizedIds = seasonGames.map(g => normalizeGameId(g.gameId));
        if (normalizedIds.length === 0) return { min: 0, max: 1, range: 1 };
        const min = Math.min(...normalizedIds);
        const max = Math.max(...normalizedIds);
        return { min, max, range: max - min || 1 };
    };

    // For non-team_season mode, use global range
    const normalizedGameIds = allGames.map(g => normalizeGameId(g.gameId));
    const minGameId = normalizedGameIds.length > 0 ? Math.min(...normalizedGameIds) : 0;
    const maxGameId = normalizedGameIds.length > 0 ? Math.max(...normalizedGameIds) : 1;
    const gameRange = maxGameId - minGameId || 1;

    // Helper to get normalized position
    const getGamePosition = (gameId: number, playerIdx?: number) => {
        const normalized = normalizeGameId(gameId);

        if (isTeamSeasonMode && playerIdx !== undefined) {
            // For team_season mode, use per-season range
            const { min, range } = getSeasonRange(playerIdx);
            return ((normalized - min) / range) * 100;
        }

        // Default: use global range
        return ((normalized - minGameId) / gameRange) * 100;
    };

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
                                                    const position = getGamePosition(p.gameId, playerIdx);
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
                        ) : null}
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
                                                    const position = getGamePosition(p.gameId, playerIdx);
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
                        ) : null}
                    </Box>
                </VStack>
            </Box>

            {/* Spatial Heatmap */}
            <Box flex="7" h="100%" w="100%">
                <HStack spacing={2} mb={2} fontSize="xs" flexWrap="wrap">
                    <Menu>
                        <MenuButton
                            as={Button}
                            size="xs"
                            rightIcon={<ChevronDownIcon />}
                            bg="gray.800"
                            color="white"
                            _hover={{ bg: 'gray.700' }}
                            _active={{ bg: 'gray.600' }}
                            w="120px"
                            textAlign="left"
                            fontWeight="normal"
                        >
                            {timeBinOptions.find(opt => opt.value === timeBin)?.label || '1Q'}
                        </MenuButton>
                        <MenuList bg="gray.800" borderColor="gray.600">
                            {timeBinOptions.map((opt) => (
                                <MenuItem
                                    key={opt.value}
                                    bg="gray.800"
                                    color="white"
                                    _hover={{ bg: 'gray.700' }}
                                    onClick={() => setTimeBin(opt.value)}
                                >
                                    {opt.label}
                                </MenuItem>
                            ))}
                        </MenuList>
                    </Menu>


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