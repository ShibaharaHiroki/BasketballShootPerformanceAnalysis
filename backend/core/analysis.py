"""
TULCA and machine learning analysis utilities.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pacmap
from tulca import TULCA


def unfold_and_scale(low_dim_tensor: np.ndarray) -> np.ndarray:
    """
    Flatten (T, …) into (T, feature_dim) without additional scaling.
    
    Args:
        low_dim_tensor: TULCA output tensor (3D or 4D)
        
    Returns:
        2D array (num_games, feature_dim)
    """
    T_ = low_dim_tensor.shape[0]
    feature_dim = int(np.prod(low_dim_tensor.shape[1:]))
    return low_dim_tensor.reshape(T_, feature_dim)


def standardize_tensor_for_tulca(tensor: np.ndarray) -> np.ndarray:
    """
    Apply Two-Step Normalization:
    1. Volume Normalization: Divide by total attempts (ONLY for Channel 5: Frequency).
    2. Z-score Standardization: Standardize each grid/channel independently.
    
    Args:
        tensor: Input tensor (T, S, V, C) 
                Channel 0: Attempts (Raw)
                Channel 1: Makes (Raw)
                Channel 2: Points (Raw)
                Channel 3: EFG Weights
                Channel 4: Misses (Raw)
                Channel 5: Frequency (Source is Attempts, needs normalization)
        
    Returns:
        Normalized and Standardized tensor
    """
    T_, S_, V_, C_ = tensor.shape
    
    # --- Step 1: Volume Normalization ---
    # Channel 0 is Attempts (Raw).
    total_attempts = tensor[:, :, :, 0].sum(axis=(1, 2)) # Shape (T,)
    
    # Avoid division by zero
    total_attempts[total_attempts == 0] = 1.0
    
    # Reshape for broadcasting: (T, 1, 1, 1)
    total_attempts_expanded = total_attempts[:, np.newaxis, np.newaxis, np.newaxis]
    
    # Create a copy
    tensor_norm = tensor.copy()

    # ★変更点: Channel 5 (Frequency) のみにボリューム正規化を適用
    # Channel 0 (Attempts) など他のチャンネルは生のカウント値のまま維持
    if C_ > 5:
        tensor_norm[:, :, :, 5:6] = tensor[:, :, :, 5:6] / total_attempts_expanded
    
    # --- Step 2: Z-score Standardization ---
    result = np.zeros_like(tensor_norm)
    
    # Standardize each channel independently
    # Channel 5 は正規化済みの値(Frequency)、他は生の値に対して標準化が行われる
    for c in range(C_):
        channel_data = tensor_norm[:, :, :, c].reshape(T_, -1)
        scaler = StandardScaler()
        channel_scaled = scaler.fit_transform(channel_data)
        result[:, :, :, c] = channel_scaled.reshape(T_, S_, V_)
    
    return result


def compute_embedding_and_projections(
    tensor: np.ndarray,
    labels: np.ndarray,
    s_dim: int = 4,
    v_dim: int = 150,
    tulca_channel: int = 0,
):
    """
    Run TULCA and PaCMAP for initial embedding.
    
    Args:
        tensor: Standardized input tensor (games, time, space, channels)
        labels: Class labels for each game
        s_dim: Time mode latent dimension
        v_dim: Space mode latent dimension
        tulca_channel: Which channel to use for TULCA (0=attempts, 1=makes, 2=weighted, 3=misses)
        
    Returns:
        tuple: (projection_matrices, scaled_data, embedding)
    """
    # Extract only the specified channel for TULCA as a 3D array
    tensor_3d = tensor[:, :, :, tulca_channel]  # Shape: (games, time, space)
    
    tulca = TULCA(
        n_components=np.array([s_dim, v_dim]),  # 2D: time and space only
        optimization_method="evd",
    )

    low_dim_tensor = tulca.fit_transform(tensor_3d, labels)

    # ★追加: 再計算時と同じ挙動にするため、デフォルト重みで再fitする
    n_classes = len(np.unique(labels))
    w_tgs = [0.0] * n_classes
    w_bgs = [1.0] * n_classes
    w_bws = [1.0] * n_classes
    
    tulca.fit_with_new_weights(w_tgs, w_bgs, w_bws)
    low_dim_tensor = tulca.transform(tensor_3d)  # 結果を更新

    proj_mats = tulca.get_projection_matrices()

    scaled_data = unfold_and_scale(low_dim_tensor)
    embedding = pacmap.PaCMAP(
        n_components=2,
        random_state=42,
    ).fit_transform(scaled_data)

    return proj_mats, scaled_data, embedding


def recalc_tulca_with_weights(
    tensor: np.ndarray,
    labels: np.ndarray,
    class_weights: list,
    n_classes: int,
    s_dim: int,
    v_dim: int,
    tulca_channel: int = 0,
):
    """
    Recompute TULCA with specified class weights and dimensions.
    
    Args:
        tensor: Standardized input tensor
        labels: Class labels
        class_weights: List of dicts with w_tg, w_bw, w_bg for each class
        n_classes: Number of classes
        s_dim: Time dimension
        v_dim: Space dimension
        tulca_channel: Which channel to use for TULCA (0=attempts, 1=makes, 2=weighted, 3=misses)
        
    Returns:
        tuple: (projection_matrices, scaled_data, embedding)
    """
    # Extract weights
    w_tgs = [class_weights[i]["w_tg"] for i in range(n_classes)]
    w_bgs = [class_weights[i]["w_bg"] for i in range(n_classes)]
    w_bws = [class_weights[i]["w_bw"] for i in range(n_classes)]

    # Extract only the specified channel for TULCA as a 3D array
    tensor_3d = tensor[:, :, :, tulca_channel]  # Shape: (games, time, space)

    n_components = np.array([s_dim, v_dim])  # 2D: time and space only

    tulca = TULCA(
        n_components=n_components,
        optimization_method="evd",
    )

    # Initial fit
    low_dim_tensor = tulca.fit_transform(tensor_3d, labels)

    # Apply weights and refit
    tulca.fit_with_new_weights(w_tgs, w_bgs, w_bws)
    low_dim_tensor = tulca.transform(tensor_3d)

    proj_mats = tulca.get_projection_matrices()

    scaled_new = unfold_and_scale(low_dim_tensor)

    embedding_new = pacmap.PaCMAP(
        n_components=2,
        random_state=42,
    ).fit_transform(scaled_new)

    return proj_mats, scaled_new, embedding_new


def create_binary_classification_data(cluster1_idx, cluster2_idx, scaled_data):
    """
    Create (X, y) for binary RF classifier from two index lists.
    
    Args:
        cluster1_idx: Indices for cluster 1
        cluster2_idx: Indices for cluster 2
        scaled_data: Flattened TULCA output (num_games, features)
        
    Returns:
        tuple: (X, y) for RandomForest
    """
    cluster1_idx = np.array(cluster1_idx, dtype=int)
    cluster2_idx = np.array(cluster2_idx, dtype=int)

    X1 = scaled_data[cluster1_idx]
    X2 = scaled_data[cluster2_idx]

    X = np.concatenate([X1, X2], axis=0)
    y = np.concatenate([
        np.ones(len(cluster1_idx), dtype=int),
        np.zeros(len(cluster2_idx), dtype=int),
    ])
    return X, y


def compute_feature_importance(X: np.ndarray, y: np.ndarray, rf_params: dict) -> np.ndarray:
    """
    Train RandomForest and return feature importance.
    
    Args:
        X: Feature matrix
        y: Labels
        rf_params: RandomForest hyperparameters
        
    Returns:
        Feature importance array
    """
    if X.shape[0] < 5 or len(np.unique(y)) < 2:
        return np.zeros(X.shape[1])

    rf = RandomForestClassifier(**rf_params)
    rf.fit(X, y)

    return rf.feature_importances_


def compute_contribution_tensor(
    cluster1_idx,
    cluster2_idx,
    scaled_data: np.ndarray,
    proj_mats,
    S: int,
    V: int,
    rf_params: dict,
    normalize_zscore: bool = True,
) -> np.ndarray:
    """
    Compute contribution tensor by mapping RF feature importance back to original space.
    
    Args:
        cluster1_idx: Cluster 1 indices
        cluster2_idx: Cluster 2 indices
        scaled_data: Flattened TULCA output
        proj_mats: TULCA projection matrices (time, space)
        S: Original time bins
        V: Original spatial cells
        rf_params: RandomForest parameters
        normalize_zscore: If True, apply Z-score normalization (mean=0, std=1)
        
    Returns:
        Contribution tensor (S, V)
    """
    X, y = create_binary_classification_data(cluster1_idx, cluster2_idx, scaled_data)
    importance = compute_feature_importance(X, y, rf_params)

    Mt, Mv = [np.asarray(m) for m in proj_mats]
    
    # Get dimensions of low-dimensional space
    s, v = Mt.shape[1], Mv.shape[1]
    
    # --- Transpose matrix approach (memory efficient) ---
    # Reshape importance vector to matrix form
    importance_mat = importance.reshape(s, v)  # (s, v)
    # Map back to original space using matrix multiplication
    # Mt: (S, s), importance_mat: (s, v), Mv.T: (v, V) -> (S, V)
    contrib_tensor = Mt @ importance_mat @ Mv.T  # (S, V)
    contrib_tensor = np.abs(contrib_tensor)
    
    # --- Original Kronecker product approach (commented out) ---
    #　kron_mat = np.kron(Mt, Mv)  # (S*V, s*v)
    # contrib_flat = kron_mat @ importance  # (S*V,)
    # contrib_tensor = contrib_flat.reshape(S, V)  # (S, V)
    # contrib_tensor = np.abs(contrib_tensor)
    
    # Optional Z-score normalization
    # Optional Z-score normalization
    # if normalize_zscore:
    #     mean = contrib_tensor.mean()
    #     std = contrib_tensor.std()
    #     if std > 0:  # Avoid division by zero
    #         contrib_tensor = (contrib_tensor - mean) / std
    return contrib_tensor
