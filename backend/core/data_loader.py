"""
Data loading utilities for NBA shot data.
Ported from the original Dash app.
"""

import tarfile
from io import BytesIO
from itertools import product
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd


def load_nba_data(
    path: "Path | str" = Path.cwd(),
    seasons=(2022,),
    data=("shotdetail",),
    seasontype: str = "rg",
    league: str = "nba",
    in_memory: bool = True,
    use_pandas: bool = True,
) -> pd.DataFrame:
    """
    Load NBA data from public GitHub repository.
    
    Args:
        path: Working directory path
        seasons: Tuple of season years
        data: Tuple of data types to load
        seasontype: "rg" (regular), "po" (playoffs), or "all"
        league: "nba" or "wnba"
        in_memory: Whether to load data in memory
        use_pandas: Whether to use pandas DataFrame
        
    Returns:
        DataFrame with NBA shot data
    """
    if isinstance(path, str):
        path = Path(path).expanduser()
    if isinstance(seasons, int):
        seasons = (seasons,)
    if isinstance(data, str):
        data = (data,)

    if (len(data) > 1) and in_memory:
        raise ValueError("When in_memory=True, please specify only one dataset type in 'data'.")

    if seasontype == "rg":
        need_data = tuple(
            ["_".join([d, str(season)]) for (d, season) in product(data, seasons)]
        )
    elif seasontype == "po":
        need_data = tuple(
            ["_".join([d, seasontype, str(season)])
             for (d, seasontype, season) in product(data, (seasontype,), seasons)]
        )
    else:
        need_data_rg = tuple(
            ["_".join([d, str(season)]) for (d, season) in product(data, seasons)]
        )
        need_data_po = tuple(
            ["_".join([d, seasontype, str(season)])
             for (d, seasontype, season) in product(data, ("po",), seasons)]
        )
        need_data = need_data_rg + need_data_po

    if league.lower() == "wnba":
        need_data = ["wnba_" + x for x in need_data]

    # Fetch list of available datasets
    with urlopen("https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt") as f:
        v = f.read().decode("utf-8").strip()

    name_v = [string.split("=")[0] for string in v.split("\n")]
    element_v = [string.split("=")[1] for string in v.split("\n")]

    need_name = [name for name in name_v if name in need_data]
    need_element = [
        element for (name, element) in zip(name_v, element_v) if name in need_data
    ]

    if not need_name:
        raise RuntimeError(
            f"Required data not found in list_data.txt. "
            f"Try changing 'seasons' or 'seasontype'."
        )

    if in_memory:
        table = pd.DataFrame() if use_pandas else []

        for name, url in zip(need_name, need_element):
            with urlopen(url) as response:
                file_content = response.read()
                with tarfile.open(fileobj=BytesIO(file_content), mode="r:xz") as tar:
                    csv_file_name = "".join([name, ".csv"])
                    csv_file = tar.extractfile(csv_file_name)
                    if csv_file is None:
                        continue
                    if use_pandas:
                        df_part = pd.read_csv(csv_file)
                        table = pd.concat([table, df_part], axis=0, ignore_index=True)
                    else:
                        raise NotImplementedError("use_pandas=False is not supported here.")
        return table

    raise NotImplementedError("in_memory=False is not implemented in this app.")


def make_game_time_space_tensor_both(
    df: pd.DataFrame,
    grid_x_bins: int = 17,
    grid_y_bins: int = 16,
    time_bin_seconds: int = 720,
):
    """
    Create game × time × position × channels tensor.
    
    Args:
        df: DataFrame with NBA shot data
        grid_x_bins: Number of spatial bins in X direction
        grid_y_bins: Number of spatial bins in Y direction
        time_bin_seconds: Duration of each time bin in seconds
        
    Returns:
        tuple: (tensor, metadata_dict)
            tensor shape: (games, time_bins, spatial_cells, 5)
            channels: 0=attempts, 1=makes, 2=points, 3=efg_weights, 4=misses
    """
    required_cols = {
        "LOC_X", "LOC_Y",
        "PERIOD",
        "MINUTES_REMAINING", "SECONDS_REMAINING",
        "GAME_ID",
        "SHOT_MADE_FLAG",
        "SHOT_TYPE",
    }
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

    # Limit to first 4 quarters
    df = df[df["PERIOD"] <= 4].copy()

    # Calculate elapsed time from tip-off
    df["ELAPSED_SEC"] = (
        (df["PERIOD"] - 1) * 720 +
        (720 - (df["MINUTES_REMAINING"] * 60 + df["SECONDS_REMAINING"]))
    )

    total_duration = 4 * 12 * 60  # 48 minutes
    num_time_bins = int(np.ceil(total_duration / time_bin_seconds))

    # Grid edges for court
    x_edges = np.linspace(-250, 250, grid_x_bins + 1)
    y_edges = np.linspace(-47.5, 422.5, grid_y_bins + 1)

    game_ids = sorted(df["GAME_ID"].unique())

    # 4D tensor (games, time, y, x, 4 channels: attempts, makes, points, efg_weights)
    data_4d = np.zeros(
        (len(game_ids), num_time_bins, grid_y_bins, grid_x_bins, 4),
        dtype=np.float32,
    )

    for g_idx, gid in enumerate(game_ids):
        game_df = df[df["GAME_ID"] == gid].copy()
        game_df["time_bin"] = (game_df["ELAPSED_SEC"] // time_bin_seconds).astype(int)
        game_df["x_bin"] = np.digitize(game_df["LOC_X"], x_edges) - 1
        game_df["y_bin"] = np.digitize(game_df["LOC_Y"], y_edges) - 1

        # Filter out-of-grid shots
        game_df = game_df[
            (game_df["x_bin"] >= 0) & (game_df["x_bin"] < grid_x_bins) &
            (game_df["y_bin"] >= 0) & (game_df["y_bin"] < grid_y_bins)
        ]

        for _, row in game_df.iterrows():
            t = int(row["time_bin"])
            y = int(row["y_bin"])
            x = int(row["x_bin"])
            if t >= num_time_bins:
                continue

            # Channel 0: Attempts
            data_4d[g_idx, t, y, x, 0] += 1.0

            # Channel 1, 2, 3: Makes, actual points, and EFG weights
            if int(row["SHOT_MADE_FLAG"]) == 1:
                data_4d[g_idx, t, y, x, 1] += 1.0
                
                shot_type = str(row["SHOT_TYPE"])
                is_3pt = "3PT" in shot_type
                pts = 3.0 if is_3pt else 2.0  # Actual point values
                efg_weight = 1.5 if is_3pt else 1.0  # EFG weights for EFG% calculation
                data_4d[g_idx, t, y, x, 2] += pts
                data_4d[g_idx, t, y, x, 3] += efg_weight

    # Add channel 4: Misses (Attempts - Makes)
    data_6ch = np.zeros(
        (len(game_ids), num_time_bins, grid_y_bins, grid_x_bins, 5),
        dtype=np.float32,
    )
    data_6ch[:, :, :, :, :4] = data_4d
    data_6ch[:, :, :, :, 4] = data_4d[:, :, :, :, 0] - data_4d[:, :, :, :, 1]  # misses = attempts - makes

    # Reshape: (games, time, y, x, 5) → (games, time, y*x, 5)
    tensor = data_6ch.reshape(
        len(game_ids),
        num_time_bins,
        grid_y_bins * grid_x_bins,
        5,
    )

    meta = {
        "x_edges": x_edges.tolist(),
        "y_edges": y_edges.tolist(),
        "game_ids": game_ids,
        "num_time_bins": num_time_bins,
        "grid_size": grid_y_bins * grid_x_bins,
        "grid_x_bins": grid_x_bins,
        "grid_y_bins": grid_y_bins,
    }

    return tensor, meta
