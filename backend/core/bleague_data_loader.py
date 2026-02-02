"""
B.League data loading utilities.
Loads shot data from Excel files for B.League (三遠ネオフェニックス).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Any


# Action ID mappings for shots (アクション1)
SHOT_ACTION_IDS = [1, 2, 3, 4, 5, 6, 44, 45]
MADE_SHOT_IDS = [1, 3, 4, 44]  # ○ = success
THREE_POINT_IDS = [1, 2]       # 3P shots

# Shot type action IDs (アクション2) - based on バスケデータ仕様書
# 27: ジャンプショット -> Jump Shot
# 28: レイアップ -> Layup
# 29: ダンク -> Dunk
# 91: フェイドアウェイ -> Jump Shot
# 92: チップイン -> Excluded
# 93: アリウープ -> Excluded
# 94: ドライビングレイアップ -> Layup
# 95: フックショット -> Hook Shot
# 96: フローティングジャンプショット -> Floater
# 97: ステップバックジャンプショット -> Jump Shot
# 98: プルアップジャンプショット -> Jump Shot
# 99: ターンアラウンドジャンプショット -> Jump Shot
SHOT_TYPE_ACTION2_MAP = {
    27: 'Jump Shot',
    28: 'Layup',
    29: 'Dunk',
    91: 'Jump Shot',
    94: 'Layup',
    95: 'Hook Shot',
    96: 'Floater',
    97: 'Jump Shot',
    98: 'Jump Shot',
    99: 'Jump Shot',
}
EXCLUDED_ACTION2_IDS = [92, 93]  # チップイン, アリウープ

# B.League period duration (10 minutes)
PERIOD_SECONDS = 600

# Data file path (relative to project root)
DATA_FILE = r"D:\data\3.1_イベントデータ(座標付き)_三遠2022-23_2023-24シーズン.xlsx"


def parse_time_remaining(time_str: str) -> int:
    """
    Parse period remaining time string (mm:ss) to seconds.
    
    Args:
        time_str: Time string in "mm:ss" or "m:ss" format
        
    Returns:
        Remaining seconds in the period
    """
    if pd.isna(time_str):
        return 0
    
    time_str = str(time_str).strip()
    if ':' not in time_str:
        return 0
    
    try:
        parts = time_str.split(':')
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    except (ValueError, IndexError):
        return 0


def load_bleague_data(data_path: Path = None) -> pd.DataFrame:
    """
    Load B.League shot data from Excel file.
    
    Args:
        data_path: Path to the Excel file. If None, uses default location.
        
    Returns:
        DataFrame with shot data, filtered and standardized.
    """
    if data_path is None:
        # Look in project root
        data_path = Path(__file__).parent.parent.parent / DATA_FILE
    
    # Read Excel file
    df = pd.read_excel(data_path, engine='openpyxl')
    
    # Filter to shot events only
    df = df[df['アクション1'].isin(SHOT_ACTION_IDS)].copy()
    
    # Filter to 三遠ネオフェニックス only (team code or name column)
    # Check if 'チーム名' column exists and filter for 三遠
    if 'チーム名' in df.columns:
        df = df[df['チーム名'].str.contains('三遠', na=False)].copy()
    elif 'チームID' in df.columns:
        # Alternative: filter by team ID if available
        pass  # Add specific team ID filter if needed
    
    # Note: Data is separated by sheets in Excel file
    # Default (first sheet) = 2022-2023 season
    # To load 2023-24 season, use: pd.read_excel(..., sheet_name=1)
    
    # Filter out rows without coordinates
    df = df.dropna(subset=['X座標', 'Y座標'])
    
    # Add SHOT_MADE_FLAG (NBA-compatible column)
    df['SHOT_MADE_FLAG'] = df['アクション1'].isin(MADE_SHOT_IDS).astype(int)
    
    # Add IS_3PT flag
    df['IS_3PT'] = df['アクション1'].isin(THREE_POINT_IDS)
    
    # Parse remaining time to seconds
    df['REMAINING_SEC'] = df['ピリオド残時間'].apply(parse_time_remaining)
    
    # Calculate elapsed seconds from game start
    # B.League: 4 periods x 10 minutes each
    df['ELAPSED_SEC'] = (
        (df['ピリオド'] - 1) * PERIOD_SECONDS +
        (PERIOD_SECONDS - df['REMAINING_SEC'])
    )
    
    # B.League coordinate system (horizontal court view):
    # - Origin (0,0) at bottom-left, (100,100) at top-right
    # - X座標 (0-100): Along court length (0 = left basket, 100 = right basket)
    # - Y座標 (0-100): Lateral (0 = bottom sideline, 100 = top sideline)
    # - サイド: 'left' or 'right' - indicates which basket the team is attacking
    #
    # Normalization strategy:
    # - 'left' side: attacking left basket (X=0), use coordinates as-is
    # - 'right' side: attacking right basket (X=100), flip both X and Y
    #   X_norm = 100 - X, Y_norm = 100 - Y
    #
    # After normalization, attacking half is X: 0-50, Y: 0-100
    # Exclude shots with X > 50 (behind half-court line)
    
    # Normalize coordinates based on attack direction (サイド)
    df['X座標_norm'] = df.apply(
        lambda row: 100 - row['X座標'] if row['サイド'] == 'right' else row['X座標'],
        axis=1
    )
    df['Y座標_norm'] = df.apply(
        lambda row: 100 - row['Y座標'] if row['サイド'] == 'right' else row['Y座標'],
        axis=1
    )
    
    # Filter out shots beyond half-court (X > 50)
    df = df[df['X座標_norm'] <= 50].copy()
    
    # Filter out チップイン (98) and アリウープ (99)
    if 'アクション2' in df.columns:
        df = df[~df['アクション2'].isin(EXCLUDED_ACTION2_IDS)].copy()
    
    # Map half-court to NBA coordinate system:
    # B.League half-court: X_norm (0-50), Y_norm (0-100)
    # NBA half-court: LOC_X (-250 to 250), LOC_Y (-47.5 to 422.5)
    #
    # Mapping:
    # - X_norm (0-50) -> LOC_Y: 0 = near basket (-47.5), 50 = half-court (422.5)
    #   LOC_Y = X_norm * 9.4 - 47.5  (0->-47.5, 50->422.5)
    # - Y_norm (0-100) -> LOC_X: 0 = left side (-250), 100 = right side (250)
    #   LOC_X = (Y_norm - 50) * 5  (0->-250, 50->0, 100->250)
    
    df['LOC_X'] = (df['Y座標_norm'] - 50) * 5  # 0-100 -> -250 to 250
    df['LOC_Y'] = df['X座標_norm'] * 9.4 - 47.5  # 0-50 -> -47.5 to 422.5
    
    # Add SHOT_TYPE (2PT/3PT) for compatibility
    df['SHOT_TYPE'] = df['IS_3PT'].apply(lambda x: '3PT Field Goal' if x else '2PT Field Goal')
    
    # Add ACTION_TYPE based on アクション2 (shot type classification)
    # Map: 91-94 -> Jump Shot, 95 -> Floater, 96 -> Hook Shot, 97 -> Layup
    def get_action_type(action2):
        if pd.isna(action2):
            return 'Other'
        action2_int = int(action2)
        return SHOT_TYPE_ACTION2_MAP.get(action2_int, 'Other')
    
    if 'アクション2' in df.columns:
        df['ACTION_TYPE'] = df['アクション2'].apply(get_action_type)
    else:
        df['ACTION_TYPE'] = 'Other'
    
    # Rename columns for compatibility
    df = df.rename(columns={
        '試合ID': 'GAME_ID',
        'ピリオド': 'PERIOD',
        '選手ID1': 'PLAYER_ID',
        '選手名1': 'PLAYER_NAME',
    })
    
    # Filter to first 4 periods only
    df = df[df['PERIOD'] <= 4].copy()
    
    return df


def get_bleague_players(data_path: Path = None) -> List[Dict[str, Any]]:
    """
    Get list of players with game counts from B.League data.
    
    Args:
        data_path: Path to the Excel file.
        
    Returns:
        List of dicts with player_id, player_name, game_count.
    """
    df = load_bleague_data(data_path)
    
    # Get unique players with game counts
    player_stats = df.groupby('PLAYER_ID').agg({
        'PLAYER_NAME': 'first',
        'GAME_ID': 'nunique'
    }).reset_index()
    
    player_stats.columns = ['PLAYER_ID', 'PLAYER_NAME', 'GAME_COUNT']
    
    # Sort by game count descending
    player_stats = player_stats.sort_values('GAME_COUNT', ascending=False)
    
    players = [
        {
            'player_id': int(row['PLAYER_ID']),
            'player_name': str(row['PLAYER_NAME']),
            'game_count': int(row['GAME_COUNT'])
        }
        for _, row in player_stats.iterrows()
    ]
    
    return players


def make_bleague_tensor(
    df: pd.DataFrame,
    grid_x_bins: int = 17,
    grid_y_bins: int = 16,
    time_bin_seconds: int = 600,  # 1 period = 10 minutes
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create game × time × position × channels tensor from B.League shot data.
    
    Args:
        df: DataFrame with shot data (from load_bleague_data)
        grid_x_bins: Number of spatial bins in X direction
        grid_y_bins: Number of spatial bins in Y direction
        time_bin_seconds: Duration of each time bin in seconds
        
    Returns:
        tuple: (tensor, metadata_dict)
            tensor shape: (games, time_bins, spatial_cells, 5)
            channels: 0=attempts, 1=makes, 2=points, 3=efg_weights, 4=misses
    """
    # Total game duration (4 periods x 10 minutes)
    total_duration = 4 * PERIOD_SECONDS
    num_time_bins = int(np.ceil(total_duration / time_bin_seconds))
    
    # Grid edges for court (NBA coordinate system)
    x_edges = np.linspace(-250, 250, grid_x_bins + 1)
    y_edges = np.linspace(-47.5, 422.5, grid_y_bins + 1)
    
    game_ids = sorted(df['GAME_ID'].unique())
    
    # 4D tensor (games, time, y, x, 4 channels)
    data_4d = np.zeros(
        (len(game_ids), num_time_bins, grid_y_bins, grid_x_bins, 4),
        dtype=np.float32,
    )
    
    for g_idx, gid in enumerate(game_ids):
        game_df = df[df['GAME_ID'] == gid].copy()
        game_df['time_bin'] = (game_df['ELAPSED_SEC'] // time_bin_seconds).astype(int)
        game_df['x_bin'] = np.digitize(game_df['LOC_X'], x_edges) - 1
        game_df['y_bin'] = np.digitize(game_df['LOC_Y'], y_edges) - 1
        
        # Filter out-of-grid shots
        game_df = game_df[
            (game_df['x_bin'] >= 0) & (game_df['x_bin'] < grid_x_bins) &
            (game_df['y_bin'] >= 0) & (game_df['y_bin'] < grid_y_bins)
        ]
        
        for _, row in game_df.iterrows():
            t = int(row['time_bin'])
            y = int(row['y_bin'])
            x = int(row['x_bin'])
            if t >= num_time_bins:
                continue
            
            # Channel 0: Attempts
            data_4d[g_idx, t, y, x, 0] += 1.0
            
            # Channel 1, 2, 3: Makes, points, EFG weights
            if int(row['SHOT_MADE_FLAG']) == 1:
                data_4d[g_idx, t, y, x, 1] += 1.0
                
                is_3pt = row['IS_3PT']
                pts = 3.0 if is_3pt else 2.0
                efg_weight = 1.5 if is_3pt else 1.0
                data_4d[g_idx, t, y, x, 2] += pts
                data_4d[g_idx, t, y, x, 3] += efg_weight
    
    # Add channel 4: Misses (Attempts - Makes)
    # ★変更: チャンネル数を 5 から 6 に変更
    data_6ch = np.zeros(
        (len(game_ids), num_time_bins, grid_y_bins, grid_x_bins, 6),
        dtype=np.float32,
    )
    data_6ch[:, :, :, :, :4] = data_4d
    data_6ch[:, :, :, :, 4] = data_4d[:, :, :, :, 0] - data_4d[:, :, :, :, 1]
    
    # ★追加: Channel 5: Frequency (初期値としてAttemptsをコピー。後で正規化してFrequencyにする)
    data_6ch[:, :, :, :, 5] = data_4d[:, :, :, :, 0]
    
    # Reshape
    tensor = data_6ch.reshape(
        len(game_ids),
        num_time_bins,
        grid_y_bins * grid_x_bins,
        6,  # 5 -> 6
    )
    
    meta = {
        'x_edges': x_edges.tolist(),
        'y_edges': y_edges.tolist(),
        'game_ids': [int(g) for g in game_ids],
        'num_time_bins': num_time_bins,
        'grid_size': grid_y_bins * grid_x_bins,
        'grid_x_bins': grid_x_bins,
        'grid_y_bins': grid_y_bins,
    }
    
    return tensor, meta


def load_bleague_team_data(data_path: Path = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load B.League team shot data from both season sheets.
    
    Args:
        data_path: Path to the Excel file.
        
    Returns:
        Tuple of (df_2022_23, df_2023_24) - DataFrames for each season
    """
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent / DATA_FILE
    
    # Load both season sheets
    df_2022_23 = pd.read_excel(data_path, sheet_name='2022-23シーズン', engine='openpyxl')
    df_2023_24 = pd.read_excel(data_path, sheet_name='2023-24シーズン', engine='openpyxl')
    
    # Process each season
    def process_season_df(df: pd.DataFrame, season_label: int) -> pd.DataFrame:
        # Filter to shot events only
        df = df[df['アクション1'].isin(SHOT_ACTION_IDS)].copy()
        
        # Filter to 三遠ネオフェニックス only
        if 'チーム名' in df.columns:
            df = df[df['チーム名'].str.contains('三遠', na=False)].copy()
        
        # Filter out rows without coordinates
        df = df.dropna(subset=['X座標', 'Y座標'])
        
        # Add SHOT_MADE_FLAG
        df['SHOT_MADE_FLAG'] = df['アクション1'].isin(MADE_SHOT_IDS).astype(int)
        df['IS_3PT'] = df['アクション1'].isin(THREE_POINT_IDS)
        
        # Parse remaining time
        df['REMAINING_SEC'] = df['ピリオド残時間'].apply(parse_time_remaining)
        df['ELAPSED_SEC'] = (
            (df['ピリオド'] - 1) * PERIOD_SECONDS +
            (PERIOD_SECONDS - df['REMAINING_SEC'])
        )
        
        # Normalize coordinates
        df['X座標_norm'] = df.apply(
            lambda row: 100 - row['X座標'] if row['サイド'] == 'right' else row['X座標'],
            axis=1
        )
        df['Y座標_norm'] = df.apply(
            lambda row: 100 - row['Y座標'] if row['サイド'] == 'right' else row['Y座標'],
            axis=1
        )
        
        # Filter out shots beyond half-court
        df = df[df['X座標_norm'] <= 50].copy()
        
        # Filter out チップイン and アリウープ
        if 'アクション2' in df.columns:
            df = df[~df['アクション2'].isin(EXCLUDED_ACTION2_IDS)].copy()
        
        # Map to NBA coordinates
        df['LOC_X'] = (df['Y座標_norm'] - 50) * 5
        df['LOC_Y'] = df['X座標_norm'] * 9.4 - 47.5
        
        # Add shot type columns
        df['SHOT_TYPE'] = df['IS_3PT'].apply(lambda x: '3PT Field Goal' if x else '2PT Field Goal')
        
        def get_action_type(action2):
            if pd.isna(action2):
                return 'Other'
            return SHOT_TYPE_ACTION2_MAP.get(int(action2), 'Other')
        
        if 'アクション2' in df.columns:
            df['ACTION_TYPE'] = df['アクション2'].apply(get_action_type)
        else:
            df['ACTION_TYPE'] = 'Other'
        
        # Rename columns
        df = df.rename(columns={
            '試合ID': 'GAME_ID',
            'ピリオド': 'PERIOD',
            '選手ID1': 'PLAYER_ID',
            '選手名1': 'PLAYER_NAME',
        })
        
        # Filter to first 4 periods
        df = df[df['PERIOD'] <= 4].copy()
        
        # Add season label
        df['SEASON_LABEL'] = season_label
        
        return df
    
    df_2022_23 = process_season_df(df_2022_23, 0)  # Label 0 for 2022-23
    df_2023_24 = process_season_df(df_2023_24, 1)  # Label 1 for 2023-24
    
    return df_2022_23, df_2023_24


def make_bleague_team_tensor(
    df_2022_23: pd.DataFrame,
    df_2023_24: pd.DataFrame,
    grid_x_bins: int = 17,
    grid_y_bins: int = 16,
    time_bin_seconds: int = 600,
) -> Tuple[np.ndarray, Dict[str, Any], List[int]]:
    """
    Create game × time × position × channels tensor for team season comparison.
    All players combined per game, labeled by season.
    
    Returns:
        tuple: (tensor, metadata, season_labels)
            tensor shape: (games, time_bins, spatial_cells, 5)
            season_labels: 0 = 2022-23, 1 = 2023-24
    """
    total_duration = 4 * PERIOD_SECONDS
    num_time_bins = int(np.ceil(total_duration / time_bin_seconds))
    
    x_edges = np.linspace(-250, 250, grid_x_bins + 1)
    y_edges = np.linspace(-47.5, 422.5, grid_y_bins + 1)
    
    # Get unique game IDs from each season
    game_ids_2022 = sorted(df_2022_23['GAME_ID'].unique())
    game_ids_2023 = sorted(df_2023_24['GAME_ID'].unique())
    
    # Combine game IDs (add prefix to avoid collision)
    all_game_ids = [(0, gid) for gid in game_ids_2022] + [(1, gid) for gid in game_ids_2023]
    season_labels = [0] * len(game_ids_2022) + [1] * len(game_ids_2023)
    
    # Tensor
    data_4d = np.zeros(
        (len(all_game_ids), num_time_bins, grid_y_bins, grid_x_bins, 4),
        dtype=np.float32,
    )
    
    for g_idx, (season, gid) in enumerate(all_game_ids):
        if season == 0:
            game_df = df_2022_23[df_2022_23['GAME_ID'] == gid].copy()
        else:
            game_df = df_2023_24[df_2023_24['GAME_ID'] == gid].copy()
        
        game_df['time_bin'] = (game_df['ELAPSED_SEC'] // time_bin_seconds).astype(int)
        game_df['x_bin'] = np.digitize(game_df['LOC_X'], x_edges) - 1
        game_df['y_bin'] = np.digitize(game_df['LOC_Y'], y_edges) - 1
        
        game_df = game_df[
            (game_df['x_bin'] >= 0) & (game_df['x_bin'] < grid_x_bins) &
            (game_df['y_bin'] >= 0) & (game_df['y_bin'] < grid_y_bins)
        ]
        
        for _, row in game_df.iterrows():
            t = int(row['time_bin'])
            y = int(row['y_bin'])
            x = int(row['x_bin'])
            if t >= num_time_bins:
                continue
            
            data_4d[g_idx, t, y, x, 0] += 1.0
            
            if int(row['SHOT_MADE_FLAG']) == 1:
                data_4d[g_idx, t, y, x, 1] += 1.0
                is_3pt = row['IS_3PT']
                data_4d[g_idx, t, y, x, 2] += 3.0 if is_3pt else 2.0
                data_4d[g_idx, t, y, x, 3] += 1.5 if is_3pt else 1.0
    
    # Add misses channel
    # ★変更: チャンネル数を 5 から 6 に変更
    data_6ch = np.zeros(
        (len(all_game_ids), num_time_bins, grid_y_bins, grid_x_bins, 6),
        dtype=np.float32,
    )
    data_6ch[:, :, :, :, :4] = data_4d
    data_6ch[:, :, :, :, 4] = data_4d[:, :, :, :, 0] - data_4d[:, :, :, :, 1]

    # ★追加: Channel 5: Frequency (初期値としてAttemptsをコピー)
    data_6ch[:, :, :, :, 5] = data_4d[:, :, :, :, 0]
    
    # Reshape
    tensor = data_6ch.reshape(
        len(all_game_ids),
        num_time_bins,
        grid_y_bins * grid_x_bins,
        6,  # 5 -> 6
    )
    
    # Create combined game IDs (season * 1000000 + original_id for uniqueness)
    combined_game_ids = [season * 1000000 + gid for season, gid in all_game_ids]
    
    meta = {
        'x_edges': x_edges.tolist(),
        'y_edges': y_edges.tolist(),
        'game_ids': combined_game_ids,
        'num_time_bins': num_time_bins,
        'grid_size': grid_y_bins * grid_x_bins,
        'grid_x_bins': grid_x_bins,
        'grid_y_bins': grid_y_bins,
    }
    
    return tensor, meta, season_labels

