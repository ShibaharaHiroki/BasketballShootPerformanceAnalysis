"""
API routes for Basketball Shoot Performance Analysis.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file if present
load_dotenv()

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
    SummarizeRequest,
    SummarizeResponse,
    ClusterStats,
)
from core.data_loader import load_nba_data, make_game_time_space_tensor_both
from core.bleague_data_loader import (
    load_bleague_data,
    get_bleague_players,
    make_bleague_tensor,
    load_bleague_team_data,
    make_bleague_team_tensor,
)
from core.analysis import (
    standardize_tensor_for_tulca,
    compute_embedding_and_projections,
    recalc_tulca_with_weights,
    compute_contribution_tensor,
)
from core.aggregations import aggregate_cluster_counts_raw, aggregate_cluster_prob_raw
from core.external_stats import fetch_all_player_stats

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
TIME_BIN_SECONDS_NBA = 720      # 12-minute periods (NBA)
TIME_BIN_SECONDS_BLEAGUE = 600  # 10-minute periods (B.League)

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
        message="Basketball Shoot Performance Analysis API is running"
    )


@router.post("/players", response_model=GetPlayersResponse)
async def get_players(request: GetPlayersRequest):
    """
    Get available players for specified seasons.
    """
    try:
        # Handle B.League
        if request.league == "bleague":
            players_data = get_bleague_players()
            players = [
                PlayerInfo(
                    player_id=int(p['player_id']),
                    player_name=str(p['player_name']),
                    game_count=int(p['game_count'])
                )
                for p in players_data
            ]
            return GetPlayersResponse(players=players)

        # Handle NBA (default)
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
        # Determine time bin based on league
        time_bin_seconds = TIME_BIN_SECONDS_BLEAGUE if request.league == "bleague" else TIME_BIN_SECONDS_NBA
        
        # Store league in app_state for later use
        app_state["league"] = request.league
        app_state["time_bin_seconds"] = time_bin_seconds
        app_state["tulca_channel"] = request.tulca_channel
        app_state["analysis_mode"] = request.analysis_mode

        # ── B.League branch ──────────────────────────────────────────────────
        if request.league == "bleague":

            # ── Team Season Comparison (三遠 2022-23 vs 2023-24) ────────────
            if request.analysis_mode == "team_season":
                df_2022_23, df_2023_24 = load_bleague_team_data()
                tensor_raw, meta_any, season_labels = make_bleague_team_tensor(
                    df_2022_23,
                    df_2023_24,
                    grid_x_bins=GRID_X_BINS,
                    grid_y_bins=GRID_Y_BINS,
                    time_bin_seconds=time_bin_seconds,
                )

                player_labels = np.array(season_labels, dtype=int)
                all_game_ids = np.array(meta_any["game_ids"])

                # Combine both seasons' DataFrames for shot lookup
                df_2022_23["SEASON_LABEL"] = 0
                df_2023_24["SEASON_LABEL"] = 1
                df_player = pd.concat([df_2022_23, df_2023_24], ignore_index=True)

                tensor = standardize_tensor_for_tulca(tensor_raw)

                proj_mats, scaled_data, embedding = compute_embedding_and_projections(
                    tensor,
                    player_labels,
                    s_dim=request.s_dim,
                    v_dim=request.v_dim,
                    tulca_channel=request.tulca_channel,
                )

                app_state["tensor_raw"] = tensor_raw
                app_state["tensor_standardized"] = tensor
                app_state["player_labels"] = player_labels
                app_state["all_game_ids"] = all_game_ids
                app_state["player_of_game"] = season_labels  # season index per game
                app_state["df_player"] = df_player
                app_state["metadata"] = meta_any
                app_state["player_ids"] = [0, 1]
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
                    player_names=["2022-23", "2023-24"],
                    tensor_shape=[int(x) for x in [T_games, S_bins, V_cells, C_channels]],
                    metadata={
                        **meta_any,
                        "x_edges": [float(x) for x in meta_any.get("x_edges", [])],
                        "y_edges": [float(y) for y in meta_any.get("y_edges", [])],
                        "game_ids": [int(g) for g in meta_any.get("game_ids", [])],
                    },
                )

            # ── Player Analysis (B.League) ───────────────────────────────────
            df_all = load_bleague_data()

            # Filter by player IDs if specified (empty list = all players)
            if request.player_ids:
                df_player = df_all[df_all["PLAYER_ID"].isin(request.player_ids)].copy()
            else:
                df_player = df_all.copy()

            player_ids = request.player_ids if request.player_ids else sorted(df_player["PLAYER_ID"].unique().tolist())

            player_tensors = []
            player_labels = []
            all_game_ids = []
            player_of_game = []
            meta_any = None

            for p_idx, pid in enumerate(player_ids):
                df_sub = df_player[df_player["PLAYER_ID"] == pid].copy()
                if df_sub.empty:
                    continue

                tensor_p, meta = make_bleague_tensor(
                    df_sub,
                    grid_x_bins=GRID_X_BINS,
                    grid_y_bins=GRID_Y_BINS,
                    time_bin_seconds=time_bin_seconds,
                )

                player_tensors.append(tensor_p)
                player_labels.extend([p_idx] * tensor_p.shape[0])
                all_game_ids.extend(meta["game_ids"])
                player_of_game.extend([pid] * tensor_p.shape[0])

                if meta_any is None:
                    meta_any = meta

            if not player_tensors:
                raise HTTPException(status_code=400, detail="No B.League data found for specified players")

            tensor_raw = np.concatenate(player_tensors, axis=0)
            tensor = standardize_tensor_for_tulca(tensor_raw)
            player_labels = np.array(player_labels, dtype=int)
            all_game_ids = np.array(all_game_ids)

            proj_mats, scaled_data, embedding = compute_embedding_and_projections(
                tensor,
                player_labels,
                s_dim=request.s_dim,
                v_dim=request.v_dim,
                tulca_channel=request.tulca_channel,
            )

            app_state["tensor_raw"] = tensor_raw
            app_state["tensor_standardized"] = tensor
            app_state["player_labels"] = player_labels
            app_state["all_game_ids"] = all_game_ids
            app_state["player_of_game"] = player_of_game
            app_state["df_player"] = df_player
            app_state["metadata"] = meta_any
            app_state["player_ids"] = player_ids
            app_state["proj_mats"] = proj_mats
            app_state["scaled_data"] = scaled_data
            app_state["embedding"] = embedding

            T_games, S_bins, V_cells, C_channels = tensor.shape
            # Build player name map from B.League data
            bleague_name_map = dict(zip(df_player["PLAYER_ID"], df_player["PLAYER_NAME"]))

            return InitializeResponse(
                embedding=embedding.tolist(),
                scaled_data=scaled_data.tolist(),
                proj_mats=[m.tolist() for m in proj_mats],
                player_labels=player_labels.tolist(),
                game_ids=[int(g) for g in all_game_ids],
                player_names=[bleague_name_map.get(pid, f"Player_{pid}") for pid in player_ids],
                tensor_shape=[int(x) for x in [T_games, S_bins, V_cells, C_channels]],
                metadata={
                    **meta_any,
                    "x_edges": [float(x) for x in meta_any.get("x_edges", [])],
                    "y_edges": [float(y) for y in meta_any.get("y_edges", [])],
                    "game_ids": [int(g) for g in meta_any.get("game_ids", [])],
                },
            )


        # ── NBA branch ───────────────────────────────────────────────────────
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
                time_bin_seconds=time_bin_seconds,
            )

            player_tensors.append(tensor_p)
            player_labels.extend([p_idx] * tensor_p.shape[0])
            all_game_ids.extend(meta["game_ids"])
            player_of_game.extend([pid] * tensor_p.shape[0])

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
            tulca_channel=request.tulca_channel,
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
        # Compatible with both Pydantic v1 and v2
        class_weights_list = [
            w.model_dump() if hasattr(w, "model_dump") else w.dict() 
            for w in request.class_weights
        ]
        n_classes = len(class_weights_list)

        proj_mats, scaled_data, embedding = recalc_tulca_with_weights(
            tensor,
            player_labels,
            class_weights_list,
            n_classes,
            request.s_dim,
            request.v_dim,
            request.tulca_channel,
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
        # TULCA now operates on 3D (time × space), so contrib_tensor is 2D (S, V)
        contrib_tensor = compute_contribution_tensor(
            request.cluster1_idx,
            request.cluster2_idx,
            scaled_data,
            proj_mats,
            S_bins,
            V_cells,
            RF_PARAMS,
            normalize_zscore=False,  # Change to True to enable normalization
        )

        # Calculate dominance (Cluster 1 - Cluster 2) using standardized tensor
        tensor = app_state["tensor_standardized"]
        tulca_channel = app_state.get("tulca_channel", 0)
        
        # Get indices
        idx1 = np.array(request.cluster1_idx, dtype=int)
        idx2 = np.array(request.cluster2_idx, dtype=int)
        
        # Calculate means for the specific channel
        # tensor shape: (Games, Time, Space, Channels)
        # We need to average over Games (axis 0)
        if len(idx1) > 0:
            mean_c1 = tensor[idx1, :, :, tulca_channel].mean(axis=0)
        else:
            mean_c1 = np.zeros((S_bins, V_cells))
            
        if len(idx2) > 0:
            mean_c2 = tensor[idx2, :, :, tulca_channel].mean(axis=0)
        else:
            mean_c2 = np.zeros((S_bins, V_cells))
            
        dominance_tensor = mean_c1 - mean_c2

        # Save to app_state for use in /summarize
        app_state["contrib_tensor"] = contrib_tensor
        app_state["dominance_tensor"] = dominance_tensor

        return AnalyzeClustersResponse(
            contrib_tensor=contrib_tensor.tolist(),
            dominance_tensor=dominance_tensor.tolist()
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
            # For EFG%, use channel 3 (EFG weights: 1.0/1.5), else use channel 1 (regular makes)
            num_channel = 3 if request.weighted else 1
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
        time_bin_seconds = app_state.get("time_bin_seconds", TIME_BIN_SECONDS_NBA)
        
        if not request.cluster_idx:
            return ClusterShotsResponse(shots=[])
        
        # Get game IDs and player IDs for cluster
        cluster_idx = np.array(request.cluster_idx, dtype=int)
        game_ids_cluster = all_game_ids[cluster_idx]
        
        # B.League team_season mode is disabled
        analysis_mode = app_state.get("analysis_mode", "player")
        league = app_state.get("league", "nba")

        if league == "bleague" and analysis_mode == "team_season":
            # team_season: player_of_game contains season labels (0 or 1)
            # all_game_ids are combined IDs: season * 1000000 + original_gid
            # df_player has original GAME_ID and SEASON_LABEL columns
            player_of_game = app_state["player_of_game"]
            seasons_cluster = [player_of_game[i] for i in cluster_idx]
            # Recover original game IDs by stripping season prefix
            original_game_ids = [int(gid) % 1000000 for gid in game_ids_cluster]
            valid_pairs = pd.DataFrame({
                "GAME_ID": original_game_ids,
                "SEASON_LABEL": seasons_cluster
            }).drop_duplicates()
            sub = df_player.merge(valid_pairs, on=["GAME_ID", "SEASON_LABEL"], how="inner")
        elif len(app_state.get("player_of_game", [])) > 0:
            # Player analysis mode with player_of_game
            player_of_game = app_state["player_of_game"]
            players_cluster = [player_of_game[i] for i in cluster_idx]
            valid_pairs = pd.DataFrame({
                "GAME_ID": game_ids_cluster,
                "PLAYER_ID": players_cluster
            }).drop_duplicates()
            
            # Use merge for fast filtering
            sub = df_player.merge(valid_pairs, on=["GAME_ID", "PLAYER_ID"], how="inner")
        else:
            # Fallback to just game ID filter
            sub = df_player[df_player["GAME_ID"].isin(game_ids_cluster)].copy()
        
        # Apply time filtering if specified
        if request.time_bin is not None:
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


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_clusters(request: SummarizeRequest):
    """
    Summarize two cluster comparisons using Gemini LLM with ReAct prompting.

    Uses the ReAct (Reasoning + Acting) methodology (Yao et al., 2022) where
    the LLM alternates between Thought, Action, and Observation steps to
    selectively retrieve information from Wikipedia and NBA stats tools.
    """
    try:
        llm_provider = os.environ.get("LLM_PROVIDER", "gemini")
        
        if llm_provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise HTTPException(
                    status_code=500,
                    detail="GEMINI_API_KEY is not set. Please set it in the environment or backend/.env file."
                )
            genai.configure(api_key=api_key)
            model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
        else:
            # For Ollama, the model config is read directly in the agent or from here
            model_name = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

        quarters = ["1Q", "2Q", "3Q", "4Q"]

        def format_cluster(label: str, stats: ClusterStats) -> str:
            lines = [f"【{label} Stats】Games: {stats.game_count}"]

            # Shot type stats
            if stats.shot_type_stats:
                lines.append("By Shot Type:")
                for s in stats.shot_type_stats:
                    attempts = s.get("attempts", 0)
                    makes = s.get("makes", 0)
                    weighted_makes = s.get("weighted_makes", 0)
                    fg_pct = (makes / attempts * 100) if attempts > 0 else 0
                    efg_pct = (weighted_makes / attempts * 100) if attempts > 0 else 0
                    lines.append(
                        f"  - {s.get('category', 'Unknown')}: "
                        f"{attempts} attempts, FG% {fg_pct:.1f}%, EFG% {efg_pct:.1f}%"
                    )

            # Time profile
            attempts_list = stats.time_profile.get("attempts", [])
            fg_list = stats.time_profile.get("fg", [])
            wfg_list = stats.time_profile.get("wfg", [])
            if attempts_list:
                lines.append("By Quarter:")
                for i, q in enumerate(quarters):
                    if i < len(attempts_list):
                        a = attempts_list[i]
                        fg = fg_list[i] if i < len(fg_list) else 0
                        wfg = wfg_list[i] if i < len(wfg_list) else 0
                        lines.append(
                            f"  - {q}: {a:.0f} attempts, FG% {fg:.1f}%, EFG% {wfg:.1f}%"
                        )
            return "\n".join(lines)

        cluster1_text = format_cluster("Cluster 1", request.cluster1)
        cluster2_text = format_cluster("Cluster 2", request.cluster2)

        # ── Feature importance summary from app_state ──
        importance_text = ""
        if "contrib_tensor" in app_state and "dominance_tensor" in app_state:
            try:
                contrib = app_state["contrib_tensor"]  # shape: (S_bins, V_cells)
                dom = app_state["dominance_tensor"]     # shape: (S_bins, V_cells)
                meta = app_state.get("metadata", {})
                grid_x_bins = meta.get("grid_x_bins", 17)
                grid_y_bins = meta.get("grid_y_bins", 16)
                x_edges = meta.get("x_edges", [])
                y_edges = meta.get("y_edges", [])

                # Aggregate across time bins: sum importance, mean dominance
                total_contrib = contrib.sum(axis=0)  # (V_cells,)
                mean_dom = dom.mean(axis=0)           # (V_cells,)

                # Define court zones based on NBA coordinates
                def get_zone(cx, cy):
                    """Map (cx, cy) center coordinates to a named court zone."""
                    dist = (cx**2 + cy**2)**0.5
                    if dist < 50:
                        return "Near Rim"
                    elif dist < 100:
                        return "Paint Area"
                    elif abs(cx) > 220:
                        if cy < 90:
                            return "Corner Three"
                        else:
                            return "Wing Three"
                    elif dist > 230:
                        return "Top Three"
                    elif cy < 142:
                        return "Midrange (Low)"
                    else:
                        return "Midrange (High)/Elbow"

                # Compute zone-level importance
                zone_data = {}  # zone -> {importance, dom_sum, count}
                for iy in range(grid_y_bins):
                    for ix in range(grid_x_bins):
                        cell_idx = iy * grid_x_bins + ix
                        if cell_idx >= len(total_contrib):
                            break
                        imp = float(total_contrib[cell_idx])
                        d = float(mean_dom[cell_idx])
                        # Calculate cell center
                        cx = (x_edges[ix] + x_edges[min(ix + 1, len(x_edges) - 1)]) / 2 if len(x_edges) > ix else 0
                        cy = (y_edges[iy] + y_edges[min(iy + 1, len(y_edges) - 1)]) / 2 if len(y_edges) > iy else 0
                        zone = get_zone(cx, cy)
                        if zone not in zone_data:
                            zone_data[zone] = {"importance": 0.0, "dom_sum": 0.0, "count": 0}
                        zone_data[zone]["importance"] += imp
                        zone_data[zone]["dom_sum"] += d
                        zone_data[zone]["count"] += 1

                # Sort by importance descending
                sorted_zones = sorted(zone_data.items(), key=lambda x: x[1]["importance"], reverse=True)

                lines = ["\n[Court Area Feature Importance]"]
                lines.append("(Key court areas distinguishing the 2 clusters calculated by RandomForest)")
                for zone, data in sorted_zones:
                    if data["importance"] < 0.001:
                        continue
                    dominant = "Leans Cluster 1" if data["dom_sum"] > 0 else "Leans Cluster 2"
                    lines.append(f"  - {zone}: Importance {data['importance']:.3f} ({dominant})")
                importance_text = "\n".join(lines)
            except Exception:
                importance_text = ""  # Silently skip if computation fails

        # ── Player composition per cluster ──
        composition_text = ""
        if request.cluster1_idx and request.cluster2_idx and "player_of_game" in app_state:
            try:
                player_of_game = app_state["player_of_game"]
                player_ids = app_state.get("player_ids", [])
                analysis_mode = app_state.get("analysis_mode", "player")

                # Build name map
                if analysis_mode == "team_season":
                    name_map = {0: "2022-23", 1: "2023-24"}
                elif app_state.get("league") == "bleague":
                    df_p = app_state.get("df_player")
                    if df_p is not None:
                        name_map = dict(zip(df_p["PLAYER_ID"], df_p["PLAYER_NAME"]))
                    else:
                        name_map = {pid: f"Player_{pid}" for pid in player_ids}
                else:
                    name_map = PLAYER_NAMES_MAP

                def _composition(indices):
                    from collections import Counter
                    labels = [player_of_game[i] for i in indices if i < len(player_of_game)]
                    counts = Counter(labels)
                    total = sum(counts.values())
                    if total == 0:
                        return ""
                    lines = []
                    for pid, cnt in counts.most_common():
                        pct = cnt / total * 100
                        pname = name_map.get(pid, f"Player_{pid}")
                        lines.append(f"  - {pname}: {cnt} games ({pct:.1f}%)")
                    return "\n".join(lines)

                c1_comp = _composition(request.cluster1_idx)
                c2_comp = _composition(request.cluster2_idx)
                if c1_comp or c2_comp:
                    comp_lines = ["\n[Player Composition per Cluster (For context only)]"]
                    comp_lines.append("Cluster 1:")
                    comp_lines.append(c1_comp if c1_comp else "  - No data")
                    comp_lines.append("Cluster 2:")
                    comp_lines.append(c2_comp if c2_comp else "  - No data")
                    composition_text = "\n".join(comp_lines)
            except Exception:
                composition_text = ""  # Silently skip if computation fails

        is_bleague = request.league == "bleague"
        league_label = "B.League (San-en NeoPhoenix)" if is_bleague else "NBA"

        # ── ReAct Agent: load external data caches for tool access ──
        from core.react_agent import (
            run_react_agent,
            load_stats_cache,
            load_wiki_cache,
        )

        player_ids = app_state.get("player_ids", [])
        stats_cache = load_stats_cache() if not is_bleague else {}
        wiki_cache = load_wiki_cache(
            stats_cache, player_ids, request.player_names
        ) if not is_bleague else {}

        # Build external_stats_text for the response (backward compat)
        external_stats_text = ""
        if not is_bleague:
            try:
                external_stats_text = fetch_all_player_stats(
                    player_ids=player_ids,
                    player_names=request.player_names,
                    league=request.league,
                )
            except Exception:
                external_stats_text = ""

        # ── Run ReAct agent loop ──
        llm_provider = os.environ.get("LLM_PROVIDER", "gemini")
        ollama_host = os.environ.get("OLLAMA_HOST", None)

        summary_text = run_react_agent(
            cluster1_text=cluster1_text,
            cluster2_text=cluster2_text,
            importance_text=importance_text,
            composition_text=composition_text,
            player_names=request.player_names,
            league_label=league_label,
            stats_cache=stats_cache,
            wiki_cache=wiki_cache,
            model_name=model_name,
            llm_provider=llm_provider,
            ollama_host=ollama_host,
        )

        return SummarizeResponse(summary=summary_text, external_stats=external_stats_text)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ReAct summarization failed: {str(e)}")
