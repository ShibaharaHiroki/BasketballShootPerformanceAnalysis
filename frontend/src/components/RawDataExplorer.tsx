/**
 * Enhanced RawDataExplorer component - Side-by-side cluster comparison with black theme.
 */

import React, { useState, useEffect } from 'react';
import {
    Box,
    VStack,
    HStack,
    SimpleGrid,
    Heading,
    RadioGroup,
    Radio,
    Stack,
    Text,
    Divider,
    Spinner,
    Table,
    Thead,
    Tbody,
    Tr,
    Th,
    Td,
} from '@chakra-ui/react';
import Plot from 'react-plotly.js';
import { useAppContext } from '../context/AppContext';
import { ShotData, ShotTypeStats } from '../types';

// Helper functions
function categorizeShotTypes(shots: ShotData[]): ShotTypeStats[] {
    const categories: { [key: string]: { attempts: number, makes: number, weighted_makes: number } } = {
        'Jump Shot': { attempts: 0, makes: 0, weighted_makes: 0 },
        'Layup': { attempts: 0, makes: 0, weighted_makes: 0 },
        'Floater': { attempts: 0, makes: 0, weighted_makes: 0 },
        'Hook Shot': { attempts: 0, makes: 0, weighted_makes: 0 },
        'Dunk': { attempts: 0, makes: 0, weighted_makes: 0 }
    };

    shots.forEach(shot => {
        const action = shot.ACTION_TYPE.toLowerCase();
        const shotType = shot.SHOT_TYPE || '';
        let category = 'Jump Shot'; // Default category
        let is3PT = shotType.includes('3PT');

        if (action.includes('dunk')) {
            category = 'Dunk';
        } else if (action.includes('layup') || action.includes('lay up') || action.includes('lay-up')) {
            category = 'Layup';
        } else if (action.includes('floater') || action.includes('floating')) {
            category = 'Floater';
        } else if (action.includes('hook')) {
            category = 'Hook Shot';
        } else if (action.includes('jump') || action.includes('jumper') || action.includes('pullup') || action.includes('pull-up') || action.includes('fadeaway') || action.includes('turnaround') || action.includes('step back')) {
            category = 'Jump Shot';
        }

        categories[category].attempts++;
        if (shot.SHOT_MADE_FLAG === 1) {
            categories[category].makes++;
            // For EFG%, 3PT shots count as 1.5x
            categories[category].weighted_makes += is3PT ? 1.5 : 1.0;
        }
    });

    return Object.entries(categories)
        .filter(([_, stats]) => stats.attempts > 0)
        .map(([category, stats]) => ({
            category,
            attempts: stats.attempts,
            makes: stats.makes,
            weighted_makes: stats.weighted_makes
        }))
        .sort((a, b) => b.attempts - a.attempts); // Sort by attempts descending
}

function getBasketballCourtShapes() {
    // NBA half-court - Based on official NBA dimensions
    // Scale: 10 units = 1 foot
    // Hoop center: (0, 0)
    // Baseline: y = -47.5

    // 3ポイントアークとコーナーラインの接続点を計算
    const radius = 237.5; // 23.75 ft from basket center
    const cornerX = 220;  // 22 ft from basket center in X
    const yBreak = Math.sqrt(radius * radius - cornerX * cornerX); // ≒ 89.48

    return [
        // Court background
        {
            type: 'rect' as const,
            x0: -250,
            y0: -47.5,
            x1: 250,
            y1: 422.5,
            line: { color: 'rgba(255,255,255,0.9)', width: 3 },
            fillcolor: 'rgba(20, 30, 50, 0.3)',
            layer: 'below' as const,
        },
        // Paint area - 16 feet wide, 19 feet deep
        {
            type: 'rect' as const,
            x0: -80,
            y0: -47.5,
            x1: 80,
            y1: 142.5,
            line: { color: 'rgba(255,255,255,0.7)', width: 2 },
            fillcolor: 'rgba(0,0,0,0)',
            layer: 'above' as const,
        },
        // Free throw lane - 12 feet wide
        {
            type: 'rect' as const,
            x0: -60,
            y0: -47.5,
            x1: 60,
            y1: 142.5,
            line: { color: 'rgba(255,255,255,0.7)', width: 2 },
            fillcolor: 'rgba(0,0,0,0)',
            layer: 'above' as const,
        },
        // Basket rim - 18 inches diameter, バスケット中心を (0, 0) として描画
        {
            type: 'circle' as const,
            x0: -7.5,
            y0: -7.5,
            x1: 7.5,
            y1: 7.5,
            line: { color: 'rgba(255, 140, 0, 1)', width: 2.5 },
            fillcolor: 'rgba(0,0,0,0)',
            layer: 'above' as const,
        },
        // Free throw circle - 6 feet radius, center at Y=142.5
        {
            type: 'circle' as const,
            x0: -60,
            y0: 82.5,
            x1: 60,
            y1: 202.5,
            line: { color: 'rgba(255,255,255,0.7)', width: 2 },
            fillcolor: 'rgba(0,0,0,0)',
            layer: 'above' as const,
        },
        // Restricted area arc - 4 feet radius from basket center (0,0)
        {
            type: 'path' as const,
            path: 'M -40 0 A 40 40 0 0 1 40 0',
            line: { color: 'rgba(255,255,255,0.6)', width: 2 },
            layer: 'above' as const,
        },
        // Backboard - 6 feet wide, ベースライン -47.5 から 4ft（40units）離れた y = -7.5
        {
            type: 'line' as const,
            x0: -30,
            y0: -7.5,
            x1: 30,
            y1: -7.5,
            line: { color: 'rgba(255,255,255,0.9)', width: 4 },
            layer: 'above' as const,
        },
        // 3-point corner lines - 22 feet from basket, Arc begins at y = yBreak
        {
            type: 'line' as const,
            x0: -220,
            y0: -47.5,
            x1: -220,
            y1: yBreak,
            line: { color: 'rgba(255,255,255,0.8)', width: 2.5 },
            layer: 'above' as const,
        },
        {
            type: 'line' as const,
            x0: 220,
            y0: -47.5,
            x1: 220,
            y1: yBreak,
            line: { color: 'rgba(255,255,255,0.8)', width: 2.5 },
            layer: 'above' as const,
        },
        // Lane space marks - 3 feet from baseline
        {
            type: 'line' as const,
            x0: -80,
            y0: -17.5,
            x1: -73,
            y1: -17.5,
            line: { color: 'rgba(255,255,255,0.6)', width: 1.5 },
            layer: 'above' as const,
        },
        {
            type: 'line' as const,
            x0: 80,
            y0: -17.5,
            x1: 73,
            y1: -17.5,
            line: { color: 'rgba(255,255,255,0.6)', width: 1.5 },
            layer: 'above' as const,
        },
        // 8 feet from baseline
        {
            type: 'line' as const,
            x0: -80,
            y0: 32.5,
            x1: -73,
            y1: 32.5,
            line: { color: 'rgba(255,255,255,0.6)', width: 1.5 },
            layer: 'above' as const,
        },
        {
            type: 'line' as const,
            x0: 80,
            y0: 32.5,
            x1: 73,
            y1: 32.5,
            line: { color: 'rgba(255,255,255,0.6)', width: 1.5 },
            layer: 'above' as const,
        },
        // 11 feet from baseline
        {
            type: 'line' as const,
            x0: -80,
            y0: 62.5,
            x1: -73,
            y1: 62.5,
            line: { color: 'rgba(255,255,255,0.6)', width: 1.5 },
            layer: 'above' as const,
        },
        {
            type: 'line' as const,
            x0: 80,
            y0: 62.5,
            x1: 73,
            y1: 62.5,
            line: { color: 'rgba(255,255,255,0.6)', width: 1.5 },
            layer: 'above' as const,
        },
        // 14 feet from baseline
        {
            type: 'line' as const,
            x0: -80,
            y0: 92.5,
            x1: -73,
            y1: 92.5,
            line: { color: 'rgba(255,255,255,0.6)', width: 1.5 },
            layer: 'above' as const,
        },
        {
            type: 'line' as const,
            x0: 80,
            y0: 92.5,
            x1: 73,
            y1: 92.5,
            line: { color: 'rgba(255,255,255,0.6)', width: 1.5 },
            layer: 'above' as const,
        },
    ];
}


function get3PTArcTrace() {
    // 3PT arc params (NBA)
    const radius = 237.5; // 23.75 ft
    const cornerX = 220;  // 22 ft
    const yBreak = Math.sqrt(radius * radius - cornerX * cornerX); // コーナーとアークの接続 y

    // 左右コーナーに対応する角度を計算
    const thetaLeft = Math.atan2(yBreak, -cornerX); // 左端 (-220, yBreak)
    const thetaRight = Math.atan2(yBreak, cornerX); // 右端 (220, yBreak)

    const numPoints = 200;
    const theta: number[] = [];
    for (let i = 0; i <= numPoints; i++) {
        // 左コーナー -> 右コーナー を上側でスイープ
        const t = thetaLeft + (thetaRight - thetaLeft) * (i / numPoints);
        theta.push(t);
    }

    const x_arc = theta.map(t => radius * Math.cos(t));
    const y_arc = theta.map(t => radius * Math.sin(t));

    return {
        x: x_arc,
        y: y_arc,
        mode: 'lines' as const,
        line: { color: 'white', width: 1 },
        showlegend: false,
        hoverinfo: 'skip' as const,
    };
}

function renderHeatmap(spatialData: number[], metadata: any, metric: 'attempts' | 'fg' | 'wfg') {
    const gridYBins = metadata.grid_y_bins || 16;
    const gridXBins = metadata.grid_x_bins || 17;
    const z = [];
    for (let i = 0; i < gridYBins; i++) {
        z.push(spatialData.slice(i * gridXBins, (i + 1) * gridXBins));
    }
    const label = metric === 'attempts' ? 'Attempts' : metric === 'fg' ? 'FG%' : 'EFG%';
    const data: any[] = [{ x: metadata.x_edges, y: metadata.y_edges, z: z, type: 'heatmap', colorscale: 'Viridis', showscale: true, colorbar: { title: label, len: 0.7, tickfont: { color: 'white' }, titlefont: { color: 'white' } } }];
    data.push(get3PTArcTrace() as any);
    return <Plot data={data} layout={{ autosize: true, height: 280, margin: { l: 20, r: 40, t: 10, b: 20 }, paper_bgcolor: 'black', plot_bgcolor: 'black', xaxis: { range: [-250, 250], showgrid: false, zeroline: false, showticklabels: false }, yaxis: { range: [-47.5, 422.5], showgrid: false, zeroline: false, scaleanchor: 'x', scaleratio: 1.0, showticklabels: false }, shapes: getBasketballCourtShapes() } as any} config={{ displayModeBar: false }} style={{ width: '100%', height: '280px' }} useResizeHandler />;
}

function renderShotMap(shotData: ShotData[]) {
    const made = shotData.filter(s => s.SHOT_MADE_FLAG === 1);
    const miss = shotData.filter(s => s.SHOT_MADE_FLAG === 0);
    const data: any[] = [get3PTArcTrace() as any];
    if (miss.length > 0) data.push({ x: miss.map(s => s.LOC_X), y: miss.map(s => s.LOC_Y), mode: 'markers', marker: { size: 5, color: 'rgba(200,200,200,0.55)' }, name: 'Miss', hovertemplate: 'x=%{x:.1f}, y=%{y:.1f}<extra></extra>' } as any);
    if (made.length > 0) data.push({ x: made.map(s => s.LOC_X), y: made.map(s => s.LOC_Y), mode: 'markers', marker: { size: 5, color: 'rgba(50,220,120,0.70)' }, name: 'Made', hovertemplate: 'x=%{x:.1f}, y=%{y:.1f}<extra></extra>' } as any);
    return <Plot data={data} layout={{ autosize: true, height: 280, margin: { l: 20, r: 40, t: 10, b: 20 }, paper_bgcolor: 'black', plot_bgcolor: 'black', xaxis: { range: [-250, 250], showgrid: false, zeroline: false, showticklabels: false }, yaxis: { range: [-47.5, 422.5], showgrid: false, zeroline: false, scaleanchor: 'x', scaleratio: 1.0, showticklabels: false }, shapes: getBasketballCourtShapes(), showlegend: true, legend: { x: 0.5, y: 1.1, xanchor: 'center', orientation: 'h', font: { color: 'white' } } } as any} config={{ displayModeBar: false }} style={{ width: '100%', height: '280px' }} useResizeHandler />;
}

function renderShotTypesChart(stats: ShotTypeStats[], clusterColor: string) {
    if (stats.length === 0) return null;

    // Calculate total attempts for frequency percentage
    const totalAttempts = stats.reduce((sum, s) => sum + s.attempts, 0);

    return (
        <Box width="100%" overflowX="auto">
            <Table size="sm" variant="simple">
                <Thead>
                    <Tr bg="gray.800">
                        <Th color="white" fontSize="xs">Shot Type</Th>
                        <Th color="white" fontSize="xs">Frequency Bar</Th>
                        <Th color="white" fontSize="xs" isNumeric>Frequency</Th>
                        <Th color="white" fontSize="xs" isNumeric>FG/FGA</Th>
                        <Th color="white" fontSize="xs" isNumeric>FG%</Th>
                        <Th color="white" fontSize="xs" isNumeric>EFG%</Th>
                    </Tr>
                </Thead>
                <Tbody>
                    {stats.map((stat, idx) => {
                        const fgPct = stat.attempts > 0 ? (stat.makes / stat.attempts * 100) : 0;
                        const efgPct = stat.attempts > 0 ? (stat.weighted_makes / stat.attempts * 100) : 0;
                        const freqPct = totalAttempts > 0 ? (stat.attempts / totalAttempts * 100) : 0;
                        const barWidth = `${Math.min(freqPct * 2, 100)}%`; // Scale for visibility

                        return (
                            <Tr key={idx} _hover={{ bg: 'gray.800' }}>
                                <Td color="white" fontSize="xs">{stat.category}</Td>
                                <Td>
                                    <Box
                                        bg={clusterColor}
                                        height="16px"
                                        width={barWidth}
                                        borderRadius="sm"
                                    />
                                </Td>
                                <Td color="white" fontSize="xs" isNumeric>{freqPct.toFixed(1)}%</Td>
                                <Td color="gray.300" fontSize="xs" isNumeric fontWeight="semibold">
                                    {stat.makes}/{stat.attempts}
                                </Td>
                                <Td color="white" fontSize="xs" isNumeric fontWeight="semibold">
                                    {fgPct.toFixed(1)}%
                                </Td>
                                <Td color="gray.400" fontSize="xs" isNumeric>
                                    {efgPct.toFixed(1)}%
                                </Td>
                            </Tr>
                        );
                    })}
                </Tbody>
            </Table>
        </Box>
    );
}

function renderTimeProfile(data: { attempts: number[], fg: number[], wfg: number[] }, clusterColor: string) {
    if (data.attempts.length === 0) return null;

    const quarters = ['1Q', '2Q', '3Q', '4Q'];

    // Calculate total attempts for frequency percentage
    const totalAttempts = data.attempts.reduce((sum, a) => sum + a, 0);

    return (
        <Box width="100%" overflowX="auto">
            <Table size="sm" variant="simple">
                <Thead>
                    <Tr bg="gray.800">
                        <Th color="white" fontSize="xs">Quarter</Th>
                        <Th color="white" fontSize="xs">Frequency Bar</Th>
                        <Th color="white" fontSize="xs" isNumeric>Frequency</Th>
                        <Th color="white" fontSize="xs" isNumeric>FG/FGA</Th>
                        <Th color="white" fontSize="xs" isNumeric>FG%</Th>
                        <Th color="white" fontSize="xs" isNumeric>EFG%</Th>
                    </Tr>
                </Thead>
                <Tbody>
                    {quarters.map((quarter, idx) => {
                        const attempts = data.attempts[idx] || 0;
                        const fgPct = data.fg[idx] || 0;
                        const efgPct = data.wfg[idx] || 0;
                        const freqPct = totalAttempts > 0 ? (attempts / totalAttempts * 100) : 0;
                        const barWidth = `${Math.min(freqPct * 2, 100)}%`; // Scale for visibility

                        return (
                            <Tr key={idx} _hover={{ bg: 'gray.800' }}>
                                <Td color="white" fontSize="xs">{quarter}</Td>
                                <Td>
                                    <Box
                                        bg={clusterColor}
                                        height="16px"
                                        width={barWidth}
                                        borderRadius="sm"
                                    />
                                </Td>
                                <Td color="white" fontSize="xs" isNumeric>{freqPct.toFixed(1)}%</Td>
                                <Td color="gray.300" fontSize="xs" isNumeric fontWeight="semibold">
                                    {Math.round(attempts * fgPct / 100)}/{attempts.toFixed(0)}
                                </Td>
                                <Td color="white" fontSize="xs" isNumeric fontWeight="semibold">
                                    {fgPct.toFixed(1)}%
                                </Td>
                                <Td color="gray.400" fontSize="xs" isNumeric>
                                    {efgPct.toFixed(1)}%
                                </Td>
                            </Tr>
                        );
                    })}
                </Tbody>
            </Table>
        </Box>
    );
}

// Cluster Panel Component
interface ClusterPanelProps {
    clusterNumber: '1' | '2';
    clusterIndices: number[] | null;
    metadata: any;
}

const ClusterPanel: React.FC<ClusterPanelProps> = ({ clusterNumber, clusterIndices, metadata }) => {
    const [spatialMode, setSpatialMode] = useState<'heatmap' | 'shotmap'>('heatmap');
    const [metric, setMetric] = useState<'attempts' | 'fg' | 'wfg'>('attempts');
    const [isLoading, setIsLoading] = useState(false);
    const [spatialData, setSpatialData] = useState<number[]>([]);
    const [shotData, setShotData] = useState<ShotData[]>([]);
    const [shotTypeStats, setShotTypeStats] = useState<ShotTypeStats[]>([]);
    const [timeProfileData, setTimeProfileData] = useState<{ attempts: number[], fg: number[], wfg: number[] }>({ attempts: [], fg: [], wfg: [] });

    const clusterColor = clusterNumber === '1' ? 'red.500' : 'blue.500';
    const clusterName = `Cluster ${clusterNumber}`;

    useEffect(() => {
        if (!clusterIndices || clusterIndices.length === 0) { setSpatialData([]); return; }
        setIsLoading(true);
        const channel = metric === 'attempts' ? 0 : metric === 'fg' ? 1 : 2;
        const weighted = metric === 'wfg';  // Only wfg should use weighted calculation
        fetch('http://localhost:8000/api/aggregate-cluster', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cluster_idx: clusterIndices, channel, weighted, time_bin: null }) })
            .then(res => res.json()).then(data => { setSpatialData(data.values || []); setIsLoading(false); }).catch(err => { console.error('Failed to fetch spatial data:', err); setIsLoading(false); });
    }, [clusterIndices, metric]);

    useEffect(() => {
        if (!clusterIndices || clusterIndices.length === 0 || spatialMode !== 'shotmap') { setShotData([]); return; }
        fetch('http://localhost:8000/api/cluster-shots', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cluster_idx: clusterIndices, time_bin: null }) })
            .then(res => res.json()).then(data => setShotData(data.shots || [])).catch(err => console.error('Failed to fetch shot data:', err));
    }, [clusterIndices, spatialMode]);

    useEffect(() => {
        if (!clusterIndices || clusterIndices.length === 0) { setShotTypeStats([]); return; }
        fetch('http://localhost:8000/api/cluster-shots', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cluster_idx: clusterIndices, time_bin: null }) })
            .then(res => res.json()).then(data => { const shots = data.shots || []; setShotTypeStats(categorizeShotTypes(shots)); }).catch(err => console.error('Failed to fetch shot type data:', err));
    }, [clusterIndices]);

    useEffect(() => {
        if (!clusterIndices || clusterIndices.length === 0) { setTimeProfileData({ attempts: [], fg: [], wfg: [] }); return; }
        const promises = [];
        for (let q = 0; q < 4; q++) {
            promises.push(Promise.all([
                fetch('http://localhost:8000/api/aggregate-cluster', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cluster_idx: clusterIndices, channel: 0, weighted: false, time_bin: q }) }).then(res => res.json()),
                fetch('http://localhost:8000/api/aggregate-cluster', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cluster_idx: clusterIndices, channel: 1, weighted: false, time_bin: q }) }).then(res => res.json()),
                fetch('http://localhost:8000/api/aggregate-cluster', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cluster_idx: clusterIndices, channel: 2, weighted: true, time_bin: q }) }).then(res => res.json())
            ]));
        }
        Promise.all(promises).then(results => {
            // For attempts (channel 0), sum all spatial bins
            const attempts = results.map(([att]) => att.values.reduce((a: number, b: number) => a + b, 0));

            // For FG% and wFG%, the API returns probabilities (0-1 range)
            // We need to calculate weighted average based on attempts per spatial bin
            const fg = results.map(([attData, fgData], idx) => {
                const totalAttempts = attData.values.reduce((a: number, b: number) => a + b, 0);
                if (totalAttempts === 0) return 0;

                // Weighted average: sum(fg% * attempts) / total_attempts
                const weightedSum = fgData.values.reduce((sum: number, fgVal: number, idx: number) => {
                    return sum + (fgVal * (attData.values[idx] || 0));
                }, 0);
                return (weightedSum / totalAttempts) * 100;
            });

            const wfg = results.map(([attData, _, wfgData], idx) => {
                const totalAttempts = attData.values.reduce((a: number, b: number) => a + b, 0);
                if (totalAttempts === 0) return 0;

                // Weighted average: sum(EFG% * attempts) / total_attempts
                const weightedSum = wfgData.values.reduce((sum: number, wfgVal: number, binIdx: number) => {
                    return sum + (wfgVal * (attData.values[binIdx] || 0));
                }, 0);
                return (weightedSum / totalAttempts) * 100;
            });

            setTimeProfileData({ attempts, fg, wfg });
        }).catch(err => console.error('Failed to fetch time profile:', err));
    }, [clusterIndices]);

    const hasData = clusterIndices && clusterIndices.length > 0;

    return (
        <VStack spacing={2} align="stretch" p={2} h="100%" borderWidth="2px" borderColor={hasData ? clusterColor : 'gray.600'} borderRadius="md" bg="black" overflowY="auto">
            <Heading size="sm" color={hasData ? clusterColor : 'gray.400'}>{clusterName} {hasData && `(${clusterIndices.length} games)`}</Heading>

            {/* Spatial Visualization */}
            <Box>
                <HStack spacing={2} mb={2}>
                    <Text fontSize="xs" fontWeight="bold" color="white">Spatial:</Text>
                    <RadioGroup size="sm" value={spatialMode} onChange={(val) => setSpatialMode(val as 'heatmap' | 'shotmap')}>
                        <Stack direction="row" fontSize="xs" spacing={3}>
                            <Radio value="heatmap" colorScheme={clusterNumber === '1' ? 'red' : 'blue'}>
                                <Text color="white">Heatmap</Text>
                            </Radio>
                            <Radio value="shotmap" colorScheme={clusterNumber === '1' ? 'red' : 'blue'}>
                                <Text color="white">Shot Map</Text>
                            </Radio>
                        </Stack>
                    </RadioGroup>
                </HStack>
                <Box h="280px" borderRadius="md" bg="black" display="flex" alignItems="center" justifyContent="center">
                    {!hasData ? (
                        <Text fontSize="sm" color="white">No cluster selected</Text>
                    ) : isLoading ? (
                        <Spinner color={clusterColor} />
                    ) : spatialMode === 'heatmap' && spatialData.length > 0 && metadata ? (
                        renderHeatmap(spatialData, metadata, metric)
                    ) : spatialMode === 'shotmap' && shotData.length > 0 ? (
                        renderShotMap(shotData)
                    ) : (
                        <Text fontSize="sm" color="white">Loading...</Text>
                    )}
                </Box>
                {spatialMode === 'heatmap' && hasData && (
                    <RadioGroup size="sm" value={metric} onChange={(val) => setMetric(val as any)} mt={2}>
                        <Stack direction="row" fontSize="xs" spacing={3}>
                            <Radio value="attempts" colorScheme={clusterNumber === '1' ? 'red' : 'blue'}>
                                <Text color="white">Attempts</Text>
                            </Radio>
                            <Radio value="fg" colorScheme={clusterNumber === '1' ? 'red' : 'blue'}>
                                <Text color="white">FG%</Text>
                            </Radio>
                            <Radio value="wfg" colorScheme={clusterNumber === '1' ? 'red' : 'blue'}>
                                <Text color="white">EFG%</Text>
                            </Radio>
                        </Stack>
                    </RadioGroup>
                )}
            </Box>

            <Divider borderColor="gray.600" />

            {/* Shot Types */}
            <Box>
                <Text fontSize="xs" fontWeight="bold" color="white" mb={2}>Shot Types</Text>
                <Box borderRadius="md" bg="black" display="flex" alignItems="center" justifyContent="center">
                    {!hasData ? (
                        <Text fontSize="sm" color="gray.500">No cluster selected</Text>
                    ) : shotTypeStats.length > 0 ? (
                        renderShotTypesChart(shotTypeStats, clusterColor)
                    ) : (
                        <Spinner size="sm" color={clusterColor} />
                    )}
                </Box>
            </Box>

            <Divider borderColor="gray.600" />

            {/* Time Profile */}
            <Box>
                <Text fontSize="xs" fontWeight="bold" color="white" mb={2}>Time Profile</Text>
                <Box borderRadius="md" bg="black" display="flex" alignItems="center" justifyContent="center">
                    {!hasData ? (
                        <Text fontSize="sm" color="gray.500">No cluster selected</Text>
                    ) : timeProfileData.attempts.length > 0 ? (
                        renderTimeProfile(timeProfileData, clusterColor)
                    ) : (
                        <Spinner size="sm" color={clusterColor} />
                    )}
                </Box>
            </Box>
        </VStack>
    );
};

const RawDataExplorer: React.FC = () => {
    const { cluster1, cluster2, metadata } = useAppContext();

    return (
        <Box p={4} h="100%" display="flex" flexDirection="column" overflow="hidden" bg="gray.900">
            <SimpleGrid columns={2} spacing={4} flex="1" minH="0">
                <ClusterPanel clusterNumber="1" clusterIndices={cluster1} metadata={metadata} />
                <ClusterPanel clusterNumber="2" clusterIndices={cluster2} metadata={metadata} />
            </SimpleGrid>
        </Box>
    );
};

export default RawDataExplorer;
