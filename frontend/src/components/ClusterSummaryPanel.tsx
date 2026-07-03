/**
 * ClusterSummaryPanel - Gemini LLM を使ってクラスタ比較の自然言語サマリを表示する
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
    Box,
    Button,
    Text,
    Spinner,
    VStack,
    HStack,
    Divider,
    Alert,
    AlertIcon,
    AlertDescription,
    Badge,
    Collapse,
    IconButton,
    Tooltip,
} from '@chakra-ui/react';
import { apiClient } from '../services/api';
import { ShotTypeStats, ClusterStats } from '../types';

interface ClusterSummaryPanelProps {
    cluster1Indices: number[] | null;
    cluster2Indices: number[] | null;
    cluster1ShotTypeStats: ShotTypeStats[];
    cluster2ShotTypeStats: ShotTypeStats[];
    cluster1TimeProfile: { attempts: number[]; fg: number[]; wfg: number[] };
    cluster2TimeProfile: { attempts: number[]; fg: number[]; wfg: number[] };
    playerNames: string[];
}

/** ShotTypeStats[] → ClusterStats に変換 */
function buildClusterStats(
    clusterIndices: number[] | null,
    shotTypeStats: ShotTypeStats[],
    timeProfile: { attempts: number[]; fg: number[]; wfg: number[] }
): ClusterStats {
    return {
        game_count: clusterIndices?.length ?? 0,
        shot_type_stats: shotTypeStats.map(s => ({
            category: s.category,
            attempts: s.attempts,
            makes: s.makes,
            weighted_makes: s.weighted_makes,
        })),
        time_profile: timeProfile,
    };
}

/** Markdown の太字 (**text**) と見出し (# / ## / ###) を簡易レンダリング */
function renderMarkdownLine(line: string, idx: number): React.ReactElement {
    // 見出し
    const h3Match = line.match(/^###\s+(.*)/);
    const h2Match = line.match(/^##\s+(.*)/);
    const h1Match = line.match(/^#\s+(.*)/);
    if (h1Match) return <Text key={idx} fontWeight="bold" fontSize="md" color="white" mt={3}>{h1Match[1]}</Text>;
    if (h2Match) return <Text key={idx} fontWeight="bold" fontSize="sm" color="white" mt={2}>{h2Match[1]}</Text>;
    if (h3Match) return <Text key={idx} fontWeight="semibold" fontSize="sm" color="white" mt={2}>{h3Match[1]}</Text>;

    // 箇条書き
    const bulletMatch = line.match(/^[-*•]\s+(.*)/);
    if (bulletMatch) {
        const content = bulletMatch[1].replace(/\*\*(.*?)\*\*/g, '$1'); // 太字を除去（テキストのみ）
        return (
            <HStack key={idx} align="flex-start" spacing={2} pl={2}>
                <Text color="white" flexShrink={0}>•</Text>
                <Text fontSize="sm" color="white">{content}</Text>
            </HStack>
        );
    }

    // 番号リスト
    const numMatch = line.match(/^(\d+)\.\s+(.*)/);
    if (numMatch) {
        const content = numMatch[2].replace(/\*\*(.*?)\*\*/g, '$1');
        return (
            <HStack key={idx} align="flex-start" spacing={2} pl={2}>
                <Badge colorScheme="whiteAlpha" flexShrink={0} mt="1px">{numMatch[1]}</Badge>
                <Text fontSize="sm" color="white">{content}</Text>
            </HStack>
        );
    }

    // 空行
    if (line.trim() === '') return <Box key={idx} h={1} />;

    // 通常テキスト（太字インライン対応）
    const parts = line.split(/(\*\*.*?\*\*)/g);
    return (
        <Text key={idx} fontSize="sm" color="white">
            {parts.map((part, i) => {
                const boldMatch = part.match(/^\*\*(.*)\*\*$/);
                return boldMatch
                    ? <Text as="span" key={i} fontWeight="bold" color="white">{boldMatch[1]}</Text>
                    : part;
            })}
        </Text>
    );
}

const ClusterSummaryPanel: React.FC<ClusterSummaryPanelProps> = ({
    cluster1Indices,
    cluster2Indices,
    cluster1ShotTypeStats,
    cluster2ShotTypeStats,
    cluster1TimeProfile,
    cluster2TimeProfile,
    playerNames,
}) => {
    const [summary, setSummary] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const hasData =
        cluster1Indices && cluster1Indices.length > 0 &&
        cluster2Indices && cluster2Indices.length > 0;

    const handleSummarize = useCallback(async () => {
        if (!hasData) return;
        setIsLoading(true);
        setError(null);
        setSummary(null);
        try {
            const cluster1 = buildClusterStats(cluster1Indices, cluster1ShotTypeStats, cluster1TimeProfile);
            const cluster2 = buildClusterStats(cluster2Indices, cluster2ShotTypeStats, cluster2TimeProfile);
            const result = await apiClient.summarize({
                cluster1,
                cluster2,
                player_names: playerNames,
                cluster1_idx: cluster1Indices ?? [],
                cluster2_idx: cluster2Indices ?? [],
            });
            setSummary(result.summary);
        } catch (err: any) {
            const detail = err?.response?.data?.detail || err?.message || 'An unknown error occurred';
            setError(detail);
        } finally {
            setIsLoading(false);
        }
    }, [hasData, cluster1Indices, cluster2Indices, cluster1ShotTypeStats, cluster2ShotTypeStats, cluster1TimeProfile, cluster2TimeProfile, playerNames]);

    // Auto-generate when both clusters are selected
    useEffect(() => {
        if (hasData) {
            handleSummarize();
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [cluster1Indices, cluster2Indices]);

    return (
        <Box
            h="100%"
            p={4}
            borderWidth="1px"
            borderColor="white"
            borderRadius="md"
            bg="gray.900"
            display="flex"
            flexDirection="column"
            overflow="hidden"
        >
            {/* ヘッダー */}
            <HStack justify="space-between" mb={3} flexShrink={0}>
                <Text fontSize="sm" fontWeight="bold" color="white">
                    AI Analysis Summary
                </Text>
                <Button
                    size="sm"
                    colorScheme="whiteAlpha"
                    variant="outline"
                    isLoading={isLoading}
                    loadingText="Generating..."
                    isDisabled={!hasData || isLoading}
                    onClick={handleSummarize}
                    _hover={{ bg: 'blue.900' }}
                >
                    Regenerate
                </Button>
            </HStack>

            <Divider borderColor="white" mb={3} flexShrink={0} />

            {/* スクロール可能なコンテンツエリア */}
            <Box flex="1" overflowY="auto">
                {/* 未選択時 */}
                {!hasData && (
                    <Text fontSize="xs" color="whiteAlpha.700" textAlign="center">
                        Select both Cluster 1 and Cluster 2 to generate a summary
                    </Text>
                )}

                {/* ローディング */}
                {isLoading && (
                    <VStack spacing={2} py={4}>
                        <Spinner color="white" size="md" />
                        <Text fontSize="xs" color="white">Gemini is analyzing...</Text>
                    </VStack>
                )}

                {/* エラー */}
                {error && (
                    <Alert status="error" bg="red.900" borderRadius="md" mt={2}>
                        <AlertIcon />
                        <AlertDescription fontSize="xs" color="red.200">{error}</AlertDescription>
                    </Alert>
                )}

                {/* サマリ表示 */}
                {summary && !isLoading && (
                    <VStack align="stretch" spacing={1} mt={1}>
                        {summary.split('\n').map((line, idx) => renderMarkdownLine(line, idx))}
                    </VStack>
                )}


            </Box>
        </Box>
    );
};

export default ClusterSummaryPanel;
