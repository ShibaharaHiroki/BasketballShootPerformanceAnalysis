/**
 * API client for NBA Shot Pattern Explorer backend.
 */

import axios from 'axios';
import type {
    ClassWeight,
    InitializeResponse,
    RecomputeTulcaResponse,
    AnalyzeClustersResponse,
    AggregateClusterResponse,
    GetPlayersResponse,
} from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: `${API_BASE_URL}/api`,
    headers: {
        'Content-Type': 'application/json',
    },
    timeout: 300000, // 5 minutes for data loading and TULCA computation
});

export const apiClient = {
    /**
     * Health check
     */
    health: async () => {
        const response = await api.get('/health');
        return response.data;
    },

    /**
     * Get available players for specified seasons
     */
    getPlayers: async (seasons: number[] = [2022]): Promise<GetPlayersResponse> => {
        const response = await api.post('/players', {
            seasons,
        });
        return response.data;
    },

    /**
     * Initialize data and compute initial TULCA
     */
    initialize: async (
        playerIds: number[] = [203999, 203507, 203954],
        seasons: number[] = [2022],
        sDim: number = 4,
        vDim: number = 160,
        cDim: number = 3
    ): Promise<InitializeResponse> => {
        const response = await api.post('/initialize', {
            player_ids: playerIds,
            seasons,
            s_dim: sDim,
            v_dim: vDim,
            c_dim: cDim,
        });
        return response.data;
    },

    /**
     * Recompute TULCA with new parameters
     */
    recomputeTulca: async (
        classWeights: ClassWeight[],
        sDim: number,
        vDim: number,
        cDim: number
    ): Promise<RecomputeTulcaResponse> => {
        const response = await api.post('/recompute-tulca', {
            class_weights: classWeights,
            s_dim: sDim,
            v_dim: vDim,
            c_dim: cDim,
        });
        return response.data;
    },

    /**
     * Analyze two clusters with RandomForest
     */
    analyzeClusters: async (
        cluster1Idx: number[],
        cluster2Idx: number[]
    ): Promise<AnalyzeClustersResponse> => {
        const response = await api.post('/analyze-clusters', {
            cluster1_idx: cluster1Idx,
            cluster2_idx: cluster2Idx,
        });
        return response.data;
    },

    /**
     * Aggregate cluster data
     */
    aggregateCluster: async (
        clusterIdx: number[],
        channel: number = 0,
        weighted: boolean = false,
        timeBin: number | null = null
    ): Promise<AggregateClusterResponse> => {
        const response = await api.post('/aggregate-cluster', {
            cluster_idx: clusterIdx,
            channel,
            weighted,
            time_bin: timeBin,
        });
        return response.data;
    },
};
