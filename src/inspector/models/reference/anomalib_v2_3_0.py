# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Excerpted from anomalib v2.3.0, commit 091ca6aca92c8d0e416394f79e52f5a3cea3db73:
#   src/anomalib/models/image/patchcore/torch_model.py              (PatchcoreModel)
#   src/anomalib/models/components/feature_extractors/timm.py       (TimmFeatureExtractor)
#   src/anomalib/models/components/sampling/k_center_greedy.py      (KCenterGreedy)
#   src/anomalib/models/components/dimensionality_reduction/random_projection.py
#                                                                   (SparseRandomProjection)
# The full licence text is in tests/reference/LICENSE-APACHE-2.0.txt.
#
# MODIFIED by this project, and only as follows, so that the reference pipeline runs without
# installing anomalib (whose dependency chain pulls Lightning, pandas and its own torch pins):
#   - docstrings, logging, tqdm, XPU synchronisation and the nn.Module-backbone branch of the
#     feature extractor were removed;
#   - `DynamicBufferMixin` / registered buffers became plain attributes;
#   - `InferenceBatch` became a tuple, and the forward pass returns the raw patch-score grid
#     instead of calling `AnomalyMapGenerator` (upsampling and blurring are done by this
#     project's evaluator for every model alike);
#   - sklearn's `sample_without_replacement` became NumPy's `choice(replace=False)`, drawing from
#     the global state when `random_state` is None, as sklearn does;
#   - `subsample_embedding` takes the device the coreset is selected on, so the embedding store
#     can be kept in host memory (the arithmetic is unchanged).
# Every other retained line is anomalib's.

from collections.abc import Sequence

import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


class TimmFeatureExtractor(nn.Module):
    def __init__(self, backbone: str, layers: Sequence[str], pre_trained: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.layers = list(layers)
        self.idx = self._map_layer_to_idx()
        self.feature_extractor = timm.create_model(
            backbone,
            pretrained=pre_trained,
            pretrained_cfg=None,
            features_only=True,
            exportable=True,
            out_indices=self.idx,
        )
        self.out_dims = self.feature_extractor.feature_info.channels()

    def _map_layer_to_idx(self) -> list[int]:
        idx = []
        model = timm.create_model(self.backbone, pretrained=False, features_only=True, exportable=True)
        layer_names = [info["module"] for info in model.feature_info.info]
        for layer in self.layers:
            idx.append(layer_names.index(layer))
        return idx

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(inputs)
        if not isinstance(features, dict):
            features = dict(zip(self.layers, features, strict=True))
        return features


class SparseRandomProjection:
    def __init__(self, eps: float = 0.1, random_state: int | None = None) -> None:
        self.n_components: int
        self.sparse_random_matrix: torch.Tensor
        self.eps = eps
        self.random_state = random_state

    def _sparse_random_matrix(self, n_features: int) -> torch.Tensor:
        density = 1 / np.sqrt(n_features)
        if density == 1:
            binomial = torch.distributions.Binomial(total_count=1, probs=0.5)
            components = binomial.sample((self.n_components, n_features)) * 2 - 1
            components = 1 / np.sqrt(self.n_components) * components
        else:
            components = torch.zeros((self.n_components, n_features), dtype=torch.float32)
            # random_state=None draws from NumPy's global state, as sklearn does upstream
            rng = np.random if self.random_state is None else np.random.RandomState(self.random_state)
            for i in range(self.n_components):
                nnz_idx = torch.distributions.Binomial(total_count=n_features, probs=density).sample()
                c_idx = torch.tensor(
                    rng.choice(n_features, size=int(nnz_idx), replace=False),
                    dtype=torch.int32,
                )
                data = torch.distributions.Binomial(total_count=1, probs=0.5).sample(sample_shape=c_idx.size()) * 2 - 1
                components[i, c_idx] = data
            components *= np.sqrt(1 / density) / np.sqrt(self.n_components)
        return components

    @staticmethod
    def _johnson_lindenstrauss_min_dim(n_samples: int, eps: float = 0.1) -> int | np.integer:
        denominator = (eps**2 / 2) - (eps**3 / 3)
        return (4 * np.log(n_samples) / denominator).astype(np.int64)

    def fit(self, embedding: torch.Tensor) -> "SparseRandomProjection":
        n_samples, n_features = embedding.shape
        device = embedding.device
        self.n_components = self._johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=self.eps)
        self.sparse_random_matrix = self._sparse_random_matrix(n_features=n_features).to(device)
        return self

    def transform(self, embedding: torch.Tensor) -> torch.Tensor:
        return embedding @ self.sparse_random_matrix.T.type(embedding.dtype)


class KCenterGreedy:
    def __init__(self, embedding: torch.Tensor, sampling_ratio: float) -> None:
        self.embedding = embedding
        self.coreset_size = int(embedding.shape[0] * sampling_ratio)
        self.model = SparseRandomProjection(eps=0.9)
        self.features: torch.Tensor
        self.min_distances: torch.Tensor = None
        self.n_observations = self.embedding.shape[0]

    def reset_distances(self) -> None:
        self.min_distances = None

    def update_distances(self, cluster_center: int | torch.Tensor | None) -> None:
        if cluster_center is not None:
            center = self.features[cluster_center]
            center = center.squeeze()
            distances = torch.linalg.norm(self.features - center, ord=2, dim=1, keepdim=True)
            if self.min_distances is None:
                self.min_distances = distances
            else:
                self.min_distances = torch.minimum(self.min_distances, distances)

    def get_new_idx(self) -> torch.Tensor:
        _, idx = torch.max(self.min_distances.squeeze(1), dim=0)
        return idx

    def select_coreset_idxs(self) -> list[int]:
        if self.embedding.ndim == 2:
            self.model.fit(self.embedding)
            self.features = self.model.transform(self.embedding)
            self.reset_distances()
        else:
            self.features = self.embedding.reshape(self.embedding.shape[0], -1)
            self.reset_distances()
        idx = torch.randint(high=self.n_observations, size=(1,), device=self.features.device).squeeze()
        selected_coreset_idxs: list[int] = []
        for _ in range(self.coreset_size):
            self.update_distances(cluster_center=idx)
            idx = self.get_new_idx()
            self.min_distances.scatter_(0, idx.unsqueeze(0).unsqueeze(1), 0.0)
            selected_coreset_idxs.append(int(idx.item()))
        return selected_coreset_idxs

    def sample_coreset(self) -> torch.Tensor:
        idxs = self.select_coreset_idxs()
        return self.embedding[idxs]


DEFAULT_CHUNK_SIZE = 1024


class PatchcoreModel(nn.Module):
    def __init__(
        self,
        layers: Sequence[str],
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.num_neighbors = num_neighbors
        self.feature_extractor = TimmFeatureExtractor(
            backbone=self.backbone,
            pre_trained=pre_trained,
            layers=self.layers,
        ).eval()
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.memory_bank: torch.Tensor = torch.empty(0)
        self.embedding_store: list[torch.Tensor] = []

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        input_tensor = input_tensor.type(self.memory_bank.dtype)
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)
        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)
        if self.training:
            self.embedding_store.append(embedding)
            return embedding
        if self.memory_bank.size(0) == 0:
            msg = "Memory bank is empty. Cannot provide anomaly scores"
            raise ValueError(msg)
        patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))
        pred_score = self.compute_anomaly_score(patch_scores, locations, embedding)
        patch_scores = patch_scores.reshape((batch_size, 1, width, height))
        return pred_score, patch_scores

    def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="bilinear")
            embeddings = torch.cat((embeddings, layer_embedding), 1)
        return embeddings

    @staticmethod
    def reshape_embedding(embedding: torch.Tensor) -> torch.Tensor:
        embedding_size = embedding.size(1)
        return embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)

    def subsample_embedding(self, sampling_ratio: float, device: torch.device | str = "cpu") -> None:
        if len(self.embedding_store) == 0:
            msg = "Embedding store is empty. Cannot perform coreset selection."
            raise ValueError(msg)
        self.memory_bank = torch.vstack(self.embedding_store).to(device)
        self.embedding_store.clear()
        sampler = KCenterGreedy(embedding=self.memory_bank, sampling_ratio=sampling_ratio)
        self.memory_bank = sampler.sample_coreset()

    @staticmethod
    def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)
        res = torch.matmul(x, y.transpose(-2, -1))
        res.mul_(-2)
        res.add_(x_norm)
        res.add_(y_norm.transpose(-2, -1))
        return res.clamp_min_(0).sqrt_()

    def nearest_neighbors(self, embedding: torch.Tensor, n_neighbors: int) -> tuple[torch.Tensor, torch.Tensor]:
        n = embedding.shape[0]
        chunk_size = DEFAULT_CHUNK_SIZE
        if n <= chunk_size:
            distances = self.euclidean_dist(embedding, self.memory_bank)
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
                distances = self.euclidean_dist(embedding_chunk, self.memory_bank)
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
        self,
        patch_scores: torch.Tensor,
        locations: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_neighbors == 1:
            return patch_scores.amax(1)
        batch_size, num_patches = patch_scores.shape
        max_patches = torch.argmax(patch_scores, dim=1)
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]
        score = patch_scores[torch.arange(batch_size), max_patches]
        nn_index = locations[torch.arange(batch_size), max_patches]
        nn_sample = self.memory_bank[nn_index, :]
        memory_bank_effective_size = self.memory_bank.shape[0]
        _, support_samples = self.nearest_neighbors(
            nn_sample,
            n_neighbors=min(self.num_neighbors, memory_bank_effective_size),
        )
        distances = self.euclidean_dist(max_patches_features.unsqueeze(1), self.memory_bank[support_samples])
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        return weights * score
