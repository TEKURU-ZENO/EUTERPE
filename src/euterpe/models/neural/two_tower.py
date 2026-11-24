"""Two-Tower Neural Recommendation Architecture for EUTERPE.

Implements symmetric dual-tower embedding projections:
- User Tower: Combines user latent ID embedding with historical listening sequence aggregation.
- Item Tower: Projects track ID embedding combined with artist identity representation.
- Similarity: Inner product similarity in shared 64-dimensional space with LayerNorm.
"""

from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """User representation tower projecting user ID and interaction history to R^d."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        use_history: bool = True,
        dropout: float = 0.1,
        sparse: bool = False,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.use_history = use_history
        self.sparse = sparse

        # User ID embedding
        self.user_embedding = nn.Embedding(n_users, embedding_dim, sparse=sparse)
        nn.init.normal_(self.user_embedding.weight, std=0.02)

        if self.use_history:
            # Item history embedding (0 is reserved for padding)
            self.history_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0, sparse=sparse)
            nn.init.normal_(self.history_embedding.weight, std=0.02)
            self.fc = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim),
            )

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        user_ids: torch.Tensor,
        history_items: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            user_ids: Tensor of shape (batch_size,) with integer user indices.
            history_items: Optional tensor of shape (batch_size, seq_len) with historical track IDs (+1 indexed, 0 for pad).
        Returns:
            user_repr: Normalized user representation vector of shape (batch_size, embedding_dim).
        """
        u_emb = self.user_embedding(user_ids)  # (B, d)

        if self.use_history and history_items is not None:
            # Mask out padding (index 0)
            mask = (history_items > 0).unsqueeze(-1).float()  # (B, S, 1)
            h_emb = self.history_embedding(history_items) * mask  # (B, S, d)
            sum_emb = h_emb.sum(dim=1)  # (B, d)
            counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
            mean_history = sum_emb / counts  # (B, d)

            combined = torch.cat([u_emb, mean_history], dim=-1)  # (B, 2*d)
            out = self.fc(combined)
        else:
            out = self.fc(u_emb)

        out = self.layer_norm(out)
        return out


class ItemTower(nn.Module):
    """Item representation tower projecting track ID and artist ID to R^d."""

    def __init__(
        self,
        n_items: int,
        n_artists: int,
        embedding_dim: int = 64,
        use_artist: bool = True,
        dropout: float = 0.1,
        sparse: bool = False,
    ):
        super().__init__()
        self.n_items = n_items
        self.n_artists = n_artists
        self.embedding_dim = embedding_dim
        self.use_artist = use_artist
        self.sparse = sparse

        # Track ID embedding
        self.track_embedding = nn.Embedding(n_items, embedding_dim, sparse=sparse)
        nn.init.normal_(self.track_embedding.weight, std=0.02)

        if self.use_artist:
            # Artist ID embedding
            self.artist_embedding = nn.Embedding(n_artists, embedding_dim, sparse=sparse)
            nn.init.normal_(self.artist_embedding.weight, std=0.02)
            self.fc = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim),
            )

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        track_ids: torch.Tensor,
        artist_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            track_ids: Tensor of shape (batch_size,) with track integer indices.
            artist_ids: Optional tensor of shape (batch_size,) with artist integer indices.
        Returns:
            item_repr: Normalized item representation vector of shape (batch_size, embedding_dim).
        """
        t_emb = self.track_embedding(track_ids)  # (B, d)

        if self.use_artist and artist_ids is not None:
            a_emb = self.artist_embedding(artist_ids)  # (B, d)
            combined = torch.cat([t_emb, a_emb], dim=-1)  # (B, 2*d)
            out = self.fc(combined)
        else:
            out = self.fc(t_emb)

        out = self.layer_norm(out)
        return out


class TwoTowerRecommender(nn.Module):
    """Complete Two-Tower retrieval model mapping users and tracks to a shared 64D inner-product space."""

    def __init__(
        self,
        n_users: int = 884,
        n_items: int = 443375,
        n_artists: int = 51833,
        embedding_dim: int = 64,
        use_history: bool = True,
        use_artist: bool = True,
        temperature: float = 0.05,
        dropout: float = 0.1,
        sparse: bool = False,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_artists = n_artists
        self.embedding_dim = embedding_dim
        self.use_history = use_history
        self.use_artist = use_artist
        self.temperature = temperature
        self.sparse = sparse

        self.user_tower = UserTower(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            use_history=use_history,
            dropout=dropout,
            sparse=sparse,
        )

        self.item_tower = ItemTower(
            n_items=n_items,
            n_artists=n_artists,
            embedding_dim=embedding_dim,
            use_artist=use_artist,
            dropout=dropout,
            sparse=sparse,
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        track_ids: torch.Tensor,
        artist_ids: Optional[torch.Tensor] = None,
        history_items: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns tuple of (user_repr, item_repr), both shape (B, d)."""
        u_repr = self.user_tower(user_ids, history_items=history_items)
        i_repr = self.item_tower(track_ids, artist_ids=artist_ids)
        return u_repr, i_repr

    def encode_users(
        self,
        user_ids: torch.Tensor,
        history_items: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode users for downstream vector index querying."""
        return self.user_tower(user_ids, history_items=history_items)

    def encode_items(
        self,
        track_ids: torch.Tensor,
        artist_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode items for index population."""
        return self.item_tower(track_ids, artist_ids=artist_ids)

    def get_embedding_parameters(self):
        """Returns list of parameters belonging to sparse embedding layers."""
        params = list(self.user_tower.user_embedding.parameters()) + list(self.item_tower.track_embedding.parameters())
        if self.user_tower.use_history:
            params += list(self.user_tower.history_embedding.parameters())
        if self.item_tower.use_artist:
            params += list(self.item_tower.artist_embedding.parameters())
        return params

    def get_dense_parameters(self):
        """Returns list of parameters belonging to dense projection layers."""
        params = list(self.user_tower.fc.parameters()) + list(self.user_tower.layer_norm.parameters())
        params += list(self.item_tower.fc.parameters()) + list(self.item_tower.layer_norm.parameters())
        return params

    def save_checkpoint(self, path: Path) -> None:
        """Save model state dict and architecture config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "n_users": self.n_users,
                "n_items": self.n_items,
                "n_artists": self.n_artists,
                "embedding_dim": self.embedding_dim,
                "use_history": self.use_history,
                "use_artist": self.use_artist,
                "temperature": self.temperature,
                "sparse": self.sparse,
            }
        }, str(path))

    @classmethod
    def load_checkpoint(cls, path: Path, device: str = "cpu") -> "TwoTowerRecommender":
        """Load model from checkpoint."""
        ckpt = torch.load(str(path), map_location=device, weights_only=True)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        return model
