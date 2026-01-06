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
    Standardize the original shot tensor before feeding into TULCA.
    Each channel is standardized independently.
    
    Args:
        tensor: Input tensor (T, S, V, C)  - games × time × space × channels
        
    Returns:
        Standardized tensor with same shape
    """
    T_, S_, V_, C_ = tensor.shape
    result = np.zeros_like(tensor)
    
    # Standardize each channel independently
    for c in range(C_):
        channel_data = tensor[:, :, :, c].reshape(T_, -1)  # (T, S*V)
        scaler = StandardScaler()
        channel_scaled = scaler.fit_transform(channel_data)
        result[:, :, :, c] = channel_scaled.reshape(T_, S_, V_)
    
    return result


def compute_embedding_and_projections(
    tensor: np.ndarray,
    labels: np.ndarray,
    s_dim: int = 4,
    v_dim: int = 160,
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
    # Extract only the specified channel for TULCA
    tensor_single_channel = tensor[:, :, :, tulca_channel:tulca_channel+1]
    
    tulca = TULCA(
        n_components=np.array([s_dim, v_dim, 1]),  # c_dim is always 1
        optimization_method="evd",
    )

    low_dim_tensor = tulca.fit_transform(tensor_single_channel, labels)
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

    # Extract only the specified channel for TULCA
    tensor_single_channel = tensor[:, :, :, tulca_channel:tulca_channel+1]

    n_components = np.array([s_dim, v_dim, 1])  # c_dim is always 1

    tulca = TULCA(
        n_components=n_components,
        optimization_method="evd",
    )

    # Initial fit
    low_dim_tensor = tulca.fit_transform(tensor_single_channel, labels)

    # Apply weights and refit
    tulca.fit_with_new_weights(w_tgs, w_bgs, w_bws)
    low_dim_tensor = tulca.transform(tensor_single_channel)

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
    C: int,
    rf_params: dict,
    normalize_zscore: bool = False,
) -> np.ndarray:
    """
    Compute contribution tensor by mapping RF feature importance back to original space.
    
    Args:
        cluster1_idx: Cluster 1 indices
        cluster2_idx: Cluster 2 indices
        scaled_data: Flattened TULCA output
        proj_mats: TULCA projection matrices (time, space, channel)
        S: Original time bins
        V: Original spatial cells
        C: Original channels
        rf_params: RandomForest parameters
        normalize_zscore: If True, apply Z-score normalization (mean=0, std=1)
        
    Returns:
        Contribution tensor (S, V, C)
    """
    X, y = create_binary_classification_data(cluster1_idx, cluster2_idx, scaled_data)
    importance = compute_feature_importance(X, y, rf_params)

    Mt, Mv, Mc = [np.asarray(m) for m in proj_mats]
    
    # Get dimensions of low-dimensional space
    s, v, c = Mt.shape[1], Mv.shape[1], Mc.shape[1]
    
    # Original implementation using Kronecker product
    kron_mat = np.kron(np.kron(Mt, Mv), Mc)  # (S*V*C, s*v*c)
    contrib_flat = kron_mat @ importance  # (S*V*C,)
    contrib_tensor = contrib_flat.reshape(S, V, C)  # (S, V, C)
    contrib_tensor = np.abs(contrib_tensor)
    
    # Optional Z-score normalization
    if normalize_zscore:
        mean = contrib_tensor.mean()
        std = contrib_tensor.std()
        if std > 0:  # Avoid division by zero
            contrib_tensor = (contrib_tensor - mean) / std
    
    return contrib_tensor
