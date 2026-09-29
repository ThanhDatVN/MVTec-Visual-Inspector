# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Excerpted from anomalib v2.3.0, commit 091ca6aca92c8d0e416394f79e52f5a3cea3db73:
#   src/anomalib/models/image/patchcore/torch_model.py  (euclidean_dist,
#       nearest_neighbors, compute_anomaly_score)
#   src/anomalib/models/components/sampling/k_center_greedy.py  (KCenterGreedy
#       selection loop)
# The full licence text is in LICENSE-APACHE-2.0.txt next to this file.
#
# MODIFIED by this project, and only as follows: methods became module-level
# functions that take the memory bank as an argument; anomalib imports, tqdm,
# XPU synchronisation, docstrings and the random projection were removed; the
# k-center loop takes its start index as an argument instead of drawing it, so
# two implementations can be compared from the same start. The arithmetic of
# every retained line is unchanged. This file is a test oracle, not part of the
# `inspector` package, and is never imported by it.

import torch
from torch.nn import functional as F  # noqa: N812

DEFAULT_CHUNK_SIZE = 1024


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = x.pow(2).sum(dim=-1, keepdim=True)
    y_norm = y.pow(2).sum(dim=-1, keepdim=True)
    res = torch.matmul(x, y.transpose(-2, -1))
    res.mul_(-2)
    res.add_(x_norm)
    res.add_(y_norm.transpose(-2, -1))
    return res.clamp_min_(0).sqrt_()


def nearest_neighbors(
    memory_bank: torch.Tensor, embedding: torch.Tensor, n_neighbors: int
) -> tuple[torch.Tensor, torch.Tensor]:
    n = embedding.shape[0]
    chunk_size = DEFAULT_CHUNK_SIZE

    if n <= chunk_size:
        distances = euclidean_dist(embedding, memory_bank)
        if n_neighbors == 1:
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
    else:
        all_scores = []
        all_locations = []
        for start_idx in range(0, n, chunk_size):
            end_idx = min(start_idx + chunk_size, n)
            embedding_chunk = embedding[start_idx:end_idx]
            distances = euclidean_dist(embedding_chunk, memory_bank)
            if n_neighbors == 1:
                chunk_scores, chunk_locations = distances.min(1)
            else:
                chunk_scores, chunk_locations = distances.topk(k=n_neighbors, largest=False, dim=1)
            all_scores.append(chunk_scores)
            all_locations.append(chunk_locations)
            del distances
        patch_scores = torch.cat(all_scores, dim=0)
        locations = torch.cat(all_locations, dim=0)

    return patch_scores, locations


def compute_anomaly_score(
    memory_bank: torch.Tensor,
    num_neighbors: int,
    patch_scores: torch.Tensor,
    locations: torch.Tensor,
    embedding: torch.Tensor,
) -> torch.Tensor:
    if num_neighbors == 1:
        return patch_scores.amax(1)
    batch_size, num_patches = patch_scores.shape
    max_patches = torch.argmax(patch_scores, dim=1)
    max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]
    score = patch_scores[torch.arange(batch_size), max_patches]
    nn_index = locations[torch.arange(batch_size), max_patches]
    nn_sample = memory_bank[nn_index, :]
    memory_bank_effective_size = memory_bank.shape[0]
    _, support_samples = nearest_neighbors(
        memory_bank,
        nn_sample,
        n_neighbors=min(num_neighbors, memory_bank_effective_size),
    )
    distances = euclidean_dist(max_patches_features.unsqueeze(1), memory_bank[support_samples])
    weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
    return weights * score


def score_images(
    memory_bank: torch.Tensor, embedding: torch.Tensor, batch_size: int, num_neighbors: int = 9
) -> tuple[torch.Tensor, torch.Tensor]:
    """The inference half of `PatchcoreModel.forward`, minus feature extraction
    and map upsampling: (patch_scores (B, P), pred_score (B,))."""
    patch_scores, locations = nearest_neighbors(memory_bank, embedding=embedding, n_neighbors=1)
    patch_scores = patch_scores.reshape((batch_size, -1))
    locations = locations.reshape((batch_size, -1))
    pred_score = compute_anomaly_score(memory_bank, num_neighbors, patch_scores, locations, embedding)
    return patch_scores, pred_score


def k_center_greedy(features: torch.Tensor, coreset_size: int, start: int) -> list[int]:
    """`KCenterGreedy.select_coreset_idxs` on already-projected features."""
    min_distances = None
    idx = torch.tensor(start)
    selected_coreset_idxs: list[int] = []
    for _ in range(coreset_size):
        center = features[idx].squeeze()
        distances = torch.linalg.norm(features - center, ord=2, dim=1, keepdim=True)
        min_distances = distances if min_distances is None else torch.minimum(min_distances, distances)
        _, idx = torch.max(min_distances.squeeze(1), dim=0)
        min_distances.scatter_(0, idx.unsqueeze(0).unsqueeze(1), 0.0)
        selected_coreset_idxs.append(int(idx.item()))
    return selected_coreset_idxs
