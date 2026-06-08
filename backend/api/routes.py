"""
API routes for NBA Shot Pattern Explorer.
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
# B.League data loader is disabled (requires local Excel file)
# from core.bleague_data_loader import (
#     load_bleague_data,
#     get_bleague_players,
#     make_bleague_tensor,
#     load_bleague_team_data,
#     make_bleague_team_tensor,
# )
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
        message="NBA Shot Pattern Explorer API is running"
    )


@router.post("/players", response_model=GetPlayersResponse)
async def get_players(request: GetPlayersRequest):
    """
    Get available players for specified seasons. (NBA only)
    """
    try:
        # B.League is disabled (requires local Excel file)
        if request.league == "bleague":
            raise HTTPException(status_code=400, detail="B.League data is not available in this deployment.")
        
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
        # B.League is disabled (requires local Excel file)
        if request.league == "bleague":
            raise HTTPException(status_code=400, detail="B.League data is not available in this deployment.")

        # Determine time bin based on league
        time_bin_seconds = TIME_BIN_SECONDS_NBA
        
        # Store league in app_state for later use
        app_state["league"] = request.league
        app_state["time_bin_seconds"] = time_bin_seconds
        app_state["tulca_channel"] = request.tulca_channel
        
        # B.League support is disabled in this deployment (requires local Excel file)
        # Handle NBA (default)
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
        class_weights_list = [w.dict() for w in request.class_weights]
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
        
        if len(app_state.get("player_of_game", [])) > 0:
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
    Summarize two cluster comparisons using Gemini LLM.
    """
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY is not set. Please set it in the environment or backend/.env file."
            )

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        players_str = "、".join(request.player_names) if request.player_names else "（未指定）"
        quarters = ["1Q", "2Q", "3Q", "4Q"]

        def format_cluster(label: str, stats: ClusterStats) -> str:
            lines = [f"【{label}】ゲーム数: {stats.game_count}"]

            # Shot type stats
            if stats.shot_type_stats:
                lines.append("シュートタイプ別:")
                for s in stats.shot_type_stats:
                    attempts = s.get("attempts", 0)
                    makes = s.get("makes", 0)
                    weighted_makes = s.get("weighted_makes", 0)
                    fg_pct = (makes / attempts * 100) if attempts > 0 else 0
                    efg_pct = (weighted_makes / attempts * 100) if attempts > 0 else 0
                    lines.append(
                        f"  - {s.get('category', 'Unknown')}: "
                        f"{attempts}本試投, FG% {fg_pct:.1f}%, EFG% {efg_pct:.1f}%"
                    )

            # Time profile
            attempts_list = stats.time_profile.get("attempts", [])
            fg_list = stats.time_profile.get("fg", [])
            wfg_list = stats.time_profile.get("wfg", [])
            if attempts_list:
                lines.append("クォーター別:")
                for i, q in enumerate(quarters):
                    if i < len(attempts_list):
                        a = attempts_list[i]
                        fg = fg_list[i] if i < len(fg_list) else 0
                        wfg = wfg_list[i] if i < len(wfg_list) else 0
                        lines.append(
                            f"  - {q}: {a:.0f}本, FG% {fg:.1f}%, EFG% {wfg:.1f}%"
                        )
            return "\n".join(lines)

        cluster1_text = format_cluster("クラスタ1", request.cluster1)
        cluster2_text = format_cluster("クラスタ2", request.cluster2)

        prompt = f"""あなたはNBAのシュートパターン分析の専門家です。
以下は、TULCA（Tensor-based Unsupervised Learning for Cluster Analysis）を使って識別された2つのシュートパターンクラスタの統計データです。
対象選手: {players_str}

{cluster1_text}

{cluster2_text}

上記2つのクラスタを比較して、以下の点についてプレイヤー視点で日本語で簡潔にサマリしてください（箇条書きと見出しを使って読みやすく）:
1. 各クラスタのシュートパターンの特徴
2. 両クラスタの主な違い（シュートタイプ・効率・時間帯など）
3. どちらのクラスタがより効率的か、その理由
4. 改善点や戦術的示唆（あれば）"""

        response = model.generate_content(prompt)
        summary_text = response.text.strip()

        return SummarizeResponse(summary=summary_text)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini summarization failed: {str(e)}")
