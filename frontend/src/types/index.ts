/**
 * TypeScript type definitions for NBA Shot Pattern Explorer.
 */

export interface PlayerInfo {
    id: number;
    name: string;
    game_count?: number;
}

export interface TensorMetadata {
    x_edges: number[];
    y_edges: number[];
    game_ids: number[];
    num_time_bins: number;
    grid_size: number;
    grid_x_bins: number;
    grid_y_bins: number;
}

export interface ClassWeight {
    w_tg: number;
    w_bw: number;
    w_bg: number;
}

export interface AppState {
    embedding: number[][];
    scaledData: number[][];
    projMats: number[][][];
    playerLabels: number[];
    gameIds: number[];
    playerNames: string[];
    tensorShape: number[];
    metadata: TensorMetadata;
    cluster1: number[] | null;
    cluster2: number[] | null;
    contribTensor: number[][][] | null;
    isLoading: boolean;
    error: string | null;
}

export interface InitializeResponse {
    embedding: number[][];
    scaled_data: number[][];
    proj_mats: number[][][];
    player_labels: number[];
    game_ids: number[];
    player_names: string[];
    tensor_shape: number[];
    metadata: TensorMetadata;
}

export interface RecomputeTulcaResponse {
    embedding: number[][];
    scaled_data: number[][];
    proj_mats: number[][][];
}

export interface AnalyzeClustersResponse {
    contrib_tensor: number[][][];
}

export interface AggregateClusterResponse {
    values: number[];
    attempts?: number[];
}

export interface ShotData {
    LOC_X: number;
    LOC_Y: number;
    SHOT_MADE_FLAG: number;
    ACTION_TYPE: string;
    SHOT_TYPE: string;
    ELAPSED_SEC: number;
}

export interface ShotTypeStats {
    category: string;
    attempts: number;
    makes: number;
    weighted_makes: number;  // For EFG% calculation (3PT = 1.5x)
}

export interface ClusterShotsResponse {
    shots: ShotData[];
}

export interface AvailablePlayer {
    player_id: number;
    player_name: string;
    game_count: number;
}

export interface GetPlayersResponse {
    players: AvailablePlayer[];
}
