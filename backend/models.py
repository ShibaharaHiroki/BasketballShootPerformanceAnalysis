"""
Pydantic models for API request/response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ClassWeight(BaseModel):
    """Weight parameters for a single class."""
    w_tg: float = Field(default=1.0, ge=0.0, le=1.0)
    w_bw: float = Field(default=1.0, ge=0.0, le=1.0)
    w_bg: float = Field(default=1.0, ge=0.0, le=1.0)


class InitializeRequest(BaseModel):
    """Request to initialize data and compute initial TULCA."""
    player_ids: List[int] = Field(default=[203999, 203507, 203954])
    seasons: List[int] = Field(default=[2022])
    s_dim: int = Field(default=4, ge=1)
    v_dim: int = Field(default=160, ge=1)
    tulca_channel: int = Field(default=0, ge=0, le=4)  # 0=attempts, 1=makes, 2=points, 3=efg_weights, 4=misses
    league: str = Field(default="nba")  # "nba" or "bleague"
    analysis_mode: str = Field(default="player")  # "player" or "team_season"


class InitializeResponse(BaseModel):
    """Response from initialization."""
    embedding: List[List[float]]
    scaled_data: List[List[float]]
    proj_mats: List[List[List[float]]]
    player_labels: List[int]
    game_ids: List[int]
    player_names: List[str]
    tensor_shape: List[int]
    metadata: Dict[str, Any]


class RecomputeTulcaRequest(BaseModel):
    """Request to recompute TULCA with new parameters."""
    class_weights: List[ClassWeight]
    s_dim: int = Field(ge=1)
    v_dim: int = Field(ge=1)
    tulca_channel: int = Field(default=0, ge=0, le=4)  # 0=attempts, 1=makes, 2=points, 3=efg_weights, 4=misses


class RecomputeTulcaResponse(BaseModel):
    """Response from TULCA recomputation."""
    embedding: List[List[float]]
    scaled_data: List[List[float]]
    proj_mats: List[List[List[float]]]


class AnalyzeClustersRequest(BaseModel):
    """Request to analyze two clusters with RandomForest."""
    cluster1_idx: List[int]
    cluster2_idx: List[int]


class AnalyzeClustersResponse(BaseModel):
    """Response from cluster analysis."""
    contrib_tensor: List[List[float]]  # (S, V) - 2D since TULCA operates on single channel


class AggregateClusterRequest(BaseModel):
    """Request to aggregate cluster data."""
    cluster_idx: List[int]
    channel: int = Field(default=0, ge=0, le=2)
    weighted: bool = Field(default=False)
    time_bin: Optional[int] = None


class AggregateClusterResponse(BaseModel):
    """Response from cluster aggregation."""
    values: List[float]  # Flattened spatial array or time series
    attempts: Optional[List[float]] = None


class ClusterShotsRequest(BaseModel):
    """Request to get raw shot data for a cluster."""
    cluster_idx: List[int]
    time_bin: Optional[int] = None


class ShotDataPoint(BaseModel):
    """Individual shot data point."""
    LOC_X: float
    LOC_Y: float
    SHOT_MADE_FLAG: int
    ACTION_TYPE: str
    SHOT_TYPE: str
    ELAPSED_SEC: float


class ClusterShotsResponse(BaseModel):
    """Response with raw shot data."""
    shots: List[ShotDataPoint]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str


class PlayerInfo(BaseModel):
    """Information about an available player."""
    player_id: int
    player_name: str
    game_count: int


class GetPlayersRequest(BaseModel):
    """Request to get available players."""
    seasons: List[int] = Field(default=[2022])
    league: str = Field(default="nba")  # "nba" or "bleague"


class GetPlayersResponse(BaseModel):
    """Response with available players."""
    players: List[PlayerInfo]
