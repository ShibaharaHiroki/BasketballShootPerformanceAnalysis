"""
Data aggregation utilities for cluster statistics.
"""

import numpy as np


def aggregate_cluster_counts_raw(tensor_raw: np.ndarray, cluster_idx, channel: int = 0):
    """
    Aggregate counts across specified games for a channel.
    
    Args:
        tensor_raw: Raw (unstandardized) tensor (games, time, space, channels)
        cluster_idx: List of game indices to aggregate
        channel: 0=attempts, 1=makes, 2=weighted_makes
        
    Returns:
        Aggregated array (time, space)
    """
    if cluster_idx is None or len(cluster_idx) == 0:
        return tensor_raw[:, :, :, channel].sum(axis=0)
    
    idx = np.array(cluster_idx, dtype=int)
    return tensor_raw[idx, :, :, channel].sum(axis=0)


def aggregate_cluster_prob_raw(tensor_raw: np.ndarray, cluster_idx, weighted: bool = False):
    """
    Calculate FG% or EFG% (Effective Field Goal %) for cluster.
    
    Args:
        tensor_raw: Raw tensor
        cluster_idx: Game indices
        weighted: If True, use weighted makes (channel 2), else regular makes (channel 1)
        
    Returns:
        tuple: (probabilities, attempts) both shaped (time, space)
    """
    attempts = aggregate_cluster_counts_raw(tensor_raw, cluster_idx, channel=0)
    num = aggregate_cluster_counts_raw(tensor_raw, cluster_idx, channel=2 if weighted else 1)

    # Avoid division by zero
    prob = np.divide(num, attempts, out=np.zeros_like(num, dtype=np.float32), where=(attempts > 0))
    return prob, attempts
