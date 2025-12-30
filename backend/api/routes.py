"""
API routes for NBA Shot Pattern Explorer.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import numpy as np
import pandas as pd

from models import (
    InitializeRequest,
    InitializeResponse,
    RecomputeTulcaRequest,
    RecomputeTulcaResponse,
    AnalyzeClustersRequest,
    AnalyzeClustersResponse,
    AggregateClusterRequest,
    AggregateClusterResponse,
    ClusterShotsRequest,
    ShotDataPoint,
    ClusterShotsResponse,
    HealthResponse,
    PlayerInfo,
    GetPlayersRequest,
    GetPlayersResponse,
)
from core.data_loader import load_nba_data, make_game_time_space_tensor_both
from core.analysis import (
    standardize_tensor_for_tulca,
    compute_embedding_and_projections,
    recalc_tulca_with_weights,
    compute_contribution_tensor,
)
from core.aggregations import aggregate_cluster_counts_raw, aggregate_cluster_prob_raw

router = APIRouter()

# Global state (in production, use Redis or similar)
app_state: Dict[str, Any] = {}

# Configuration constants
PLAYER_NAMES_MAP = {
    203999: "Jokic",
    203507: "Antetokounmpo",
    203954: "Embiid"
}

GRID_X_BINS = 17
GRID_Y_BINS = 16
TIME_BIN_SECONDS = 720

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 3,
    "min_samples_split": 6,
    "max_features": 0.7,
    "bootstrap": True,
    "n_jobs": -1,
    "random_state": 42,
}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="NBA Shot Pattern Explorer API is running"
    )


@router.post("/players", response_model=GetPlayersResponse)
async def get_players(request: GetPlayersRequest):
    """
    Get available players for specified seasons.
    """
    try:
        # Load NBA data for specified seasons
        df_all = load_nba_data(
            seasons=tuple(request.seasons),
            data=("shotdetail",),
            seasontype="rg"
        )
        
        # Get unique players with game counts
        player_stats = df_all.groupby('PLAYER_ID').agg({
            'PLAYER_NAME': 'first',
            'GAME_ID': 'nunique'
        }).reset_index()
        
        player_stats.columns = ['PLAYER_ID', 'PLAYER_NAME', 'GAME_COUNT']
        
        # Sort by game count descending
        player_stats = player_stats.sort_values('GAME_COUNT', ascending=False)
        
        # Convert to PlayerInfo models
        players = [
            PlayerInfo(
                player_id=int(row['PLAYER_ID']),
                player_name=str(row['PLAYER_NAME']),
                game_count=int(row['GAME_COUNT'])
            )
            for _, row in player_stats.iterrows()
        ]
        
        return GetPlayersResponse(players=players)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch players: {str(e)}")


@router.post("/initialize", response_model=InitializeResponse)
async def initialize(request: InitializeRequest):
    """
    Initialize the application by loading data and computing initial TULCA + PaCMAP.
    """
    try:
        # Load NBA data
        df_all = load_nba_data(
            seasons=tuple(request.seasons),
            data=("shotdetail",),
            seasontype="rg"
        )

        # Filter by player IDs
        df_player = df_all[df_all["PLAYER_ID"].isin(request.player_ids)].copy()
        df_player = df_player[df_player["PERIOD"] <= 4].copy()
        
        # Calculate elapsed seconds
        df_player["ELAPSED_SEC"] = (
            (df_player["PERIOD"] - 1) * 720
            + (720 - (df_player["MINUTES_REMAINING"] * 60 + df_player["SECONDS_REMAINING"]))
        )

        # Build tensors for each player
        player_tensors = []
        player_labels = []
        all_game_ids = []
        player_of_game = []
        meta_any = None

        for p_idx, pid in enumerate(request.player_ids):
            df_sub = df_player[df_player["PLAYER_ID"] == pid].copy()
            if df_sub.empty:
                continue

            tensor_p, meta = make_game_time_space_tensor_both(
                df_sub,
                grid_x_bins=GRID_X_BINS,
                grid_y_bins=GRID_Y_BINS,
                time_bin_seconds=TIME_BIN_SECONDS,
            )

            player_tensors.append(tensor_p)
            player_labels.extend([p_idx] * tensor_p.shape[0])
            all_game_ids.extend(meta["game_ids"])
            player_of_game.extend([PLAYER_NAMES_MAP.get(pid, f"Player_{pid}")] * tensor_p.shape[0])

            if meta_any is None:
                meta_any = meta

        if not player_tensors:
            raise HTTPException(status_code=400, detail="No data found for specified players")

        # Concatenate tensors
        tensor_raw = np.concatenate(player_tensors, axis=0)
        tensor = standardize_tensor_for_tulca(tensor_raw)

        player_labels = np.array(player_labels, dtype=int)
        all_game_ids = np.array(all_game_ids)

        # Compute TULCA + PaCMAP
        proj_mats, scaled_data, embedding = compute_embedding_and_projections(
            tensor,
            player_labels,
            s_dim=request.s_dim,
            v_dim=request.v_dim,
            c_dim=request.c_dim,
        )

        # Store in global state
        app_state["tensor_raw"] = tensor_raw
        app_state["tensor_standardized"] = tensor
        app_state["player_labels"] = player_labels
        app_state["all_game_ids"] = all_game_ids
        app_state["player_of_game"] = player_of_game
        app_state["df_player"] = df_player
        app_state["metadata"] = meta_any
        app_state["player_ids"] = request.player_ids
        app_state["proj_mats"] = proj_mats
        app_state["scaled_data"] = scaled_data
        app_state["embedding"] = embedding

        T_games, S_bins, V_cells, C_channels = tensor.shape

        return InitializeResponse(
            embedding=embedding.tolist(),
            scaled_data=scaled_data.tolist(),
            proj_mats=[m.tolist() for m in proj_mats],
            player_labels=player_labels.tolist(),
            game_ids=[int(g) for g in all_game_ids],
            player_names=[PLAYER_NAMES_MAP.get(pid, f"Player_{pid}") for pid in request.player_ids],
            tensor_shape=[int(x) for x in [T_games, S_bins, V_cells, C_channels]],
            metadata={
                **meta_any,
                "x_edges": [float(x) for x in meta_any.get("x_edges", [])],
                "y_edges": [float(y) for y in meta_any.get("y_edges", [])],
                "game_ids": [int(g) for g in meta_any.get("game_ids", [])],
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@router.post("/recompute-tulca", response_model=RecomputeTulcaResponse)
async def recompute_tulca(request: RecomputeTulcaRequest):
    """
    Recompute TULCA with new dimensions and class weights.
    """
    try:
        if "tensor_standardized" not in app_state:
            raise HTTPException(status_code=400, detail="Data not initialized. Call /initialize first.")

        tensor = app_state["tensor_standardized"]
        player_labels = app_state["player_labels"]
        
        # Convert class weights to list of dicts
        class_weights_list = [w.dict() for w in request.class_weights]
        n_classes = len(class_weights_list)

        proj_mats, scaled_data, embedding = recalc_tulca_with_weights(
            tensor,
            player_labels,
            class_weights_list,
            n_classes,
            request.s_dim,
            request.v_dim,
            request.c_dim,
        )

        # Update state
        app_state["proj_mats"] = proj_mats
        app_state["scaled_data"] = scaled_data
        app_state["embedding"] = embedding

        return RecomputeTulcaResponse(
            embedding=embedding.tolist(),
            scaled_data=scaled_data.tolist(),
            proj_mats=[m.tolist() for m in proj_mats],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TULCA recomputation failed: {str(e)}")


@router.post("/analyze-clusters", response_model=AnalyzeClustersResponse)
async def analyze_clusters(request: AnalyzeClustersRequest):
    """
    Analyze two clusters using RandomForest and return contribution tensor.
    """
    try:
        if "scaled_data" not in app_state:
            raise HTTPException(status_code=400, detail="Data not initialized. Call /initialize first.")

        scaled_data = app_state["scaled_data"]
        proj_mats = app_state["proj_mats"]
        tensor_raw = app_state["tensor_raw"]

        _, S_bins, V_cells, C_channels = tensor_raw.shape

        # Set normalize_zscore=True to enable Z-score normalization
        # Set normalize_zscore=False to use original implementation (no normalization)
        contrib_tensor = compute_contribution_tensor(
            request.cluster1_idx,
            request.cluster2_idx,
            scaled_data,
            proj_mats,
            S_bins,
            V_cells,
            C_channels,
            RF_PARAMS,
            normalize_zscore=False,  # Change to True to enable normalization
        )

        return AnalyzeClustersResponse(
            contrib_tensor=contrib_tensor.tolist()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cluster analysis failed: {str(e)}")


@router.post("/aggregate-cluster", response_model=AggregateClusterResponse)
async def aggregate_cluster(request: AggregateClusterRequest):
    """
    Aggregate cluster data for specified channel.
    """
    try:
        if "tensor_raw" not in app_state:
            raise HTTPException(status_code=400, detail="Data not initialized. Call /initialize first.")

        tensor_raw = app_state["tensor_raw"]

        # If channel is 0 (attempts), return counts
        if request.channel == 0:
            counts = aggregate_cluster_counts_raw(tensor_raw, request.cluster_idx, channel=0)
            
            if request.time_bin is not None:
                values = counts[request.time_bin, :].flatten().tolist()
            else:
                values = counts.sum(axis=0).flatten().tolist()
            
            return AggregateClusterResponse(
                values=values,
                attempts=None
            )
        
        # For channels 1 (FG%) or 2 (EFG%), calculate probabilities
        # weighted flag determines which channel to use for numerator
        prob, attempts = aggregate_cluster_prob_raw(
            tensor_raw,
            request.cluster_idx,
            weighted=request.weighted
        )
        
        if request.time_bin is not None:
            values = prob[request.time_bin, :].flatten().tolist()
            attempts_list = attempts[request.time_bin, :].flatten().tolist()
        else:
            # Aggregate across time
            num_channel = 2 if request.weighted else 1
            num = aggregate_cluster_counts_raw(tensor_raw, request.cluster_idx, channel=num_channel).sum(axis=0)
            att = attempts.sum(axis=0)
            values = np.divide(num, att, out=np.zeros_like(num, dtype=np.float32), where=(att > 0)).flatten().tolist()
            attempts_list = att.flatten().tolist()
        
        return AggregateClusterResponse(
            values=values,
            attempts=attempts_list
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cluster aggregation failed: {str(e)}")



@router.post("/cluster-shots", response_model=ClusterShotsResponse)
async def cluster_shots(request: ClusterShotsRequest):
    """
    Get raw shot data for a cluster.
    """
    try:
        if "df_player" not in app_state or "all_game_ids" not in app_state:
            raise HTTPException(status_code=400, detail="Data not initialized. Call /initialize first.")
        
        df_player = app_state["df_player"]
        all_game_ids = app_state["all_game_ids"]
        metadata = app_state.get("metadata", {})
        
        if not request.cluster_idx:
            return ClusterShotsResponse(shots=[])
        
        # Get game IDs for cluster
        cluster_idx = np.array(request.cluster_idx, dtype=int)
        game_ids_cluster = all_game_ids[cluster_idx]
        
        # Filter shots by cluster games
        sub = df_player[df_player["GAME_ID"].isin(game_ids_cluster)].copy()
        
        # Apply time filtering if specified
        if request.time_bin is not None:
            time_bin_seconds = 720  # This should match TIME_BIN_SECONDS constant
            t_start = request.time_bin * time_bin_seconds
            t_end = (request.time_bin + 1) * time_bin_seconds
            sub = sub[(sub["ELAPSED_SEC"] >= t_start) & (sub["ELAPSED_SEC"] < t_end)].copy()
        
        # Convert to shot data points
        shots = []
        for _, row in sub.iterrows():
            shots.append(ShotDataPoint(
                LOC_X=float(row["LOC_X"]),
                LOC_Y=float(row["LOC_Y"]),
                SHOT_MADE_FLAG=int(row["SHOT_MADE_FLAG"]),
                ACTION_TYPE=str(row.get("ACTION_TYPE", "Unknown")),
                SHOT_TYPE=str(row.get("SHOT_TYPE", "Unknown")),
                ELAPSED_SEC=float(row["ELAPSED_SEC"])
            ))
        
        return ClusterShotsResponse(shots=shots)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve cluster shots: {str(e)}")
