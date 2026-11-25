"""Set-valued presence and count prediction; no identity-score modification."""
import torch
from torch import nn


class PresenceHead(nn.Module):
    def __init__(self, dim=256, count_cap=10):
        super().__init__()
        self.pcd = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, dim))
        self.geometry = nn.Sequential(nn.Linear(5, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.sentence = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, dim))
        self.presence_token = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=512,
                                           dropout=.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.presence = nn.Linear(dim, 1)
        self.count = nn.Linear(dim, count_cap + 1)

    def forward(self, pcd, margin, box, sent, padding_mask=None):
        single = pcd.ndim == 2
        if single:
            pcd, margin, box, sent = pcd[None], margin[None], box[None], sent[None]
            if padding_mask is not None:
                padding_mask = padding_mask[None]
        b, n = pcd.shape[:2]
        candidates = self.pcd(pcd) + self.geometry(torch.cat([margin[..., None], box], -1))
        token = self.presence_token.expand(b, -1, -1) + self.sentence(sent)[:, None]
        if padding_mask is None:
            padding_mask = torch.zeros((b, n), device=pcd.device, dtype=torch.bool)
        padding_mask = torch.cat([torch.zeros((b, 1), device=pcd.device, dtype=torch.bool), padding_mask], 1)
        output = self.encoder(torch.cat([token, candidates], 1), src_key_padding_mask=padding_mask)[:, 0]
        presence, count = self.presence(output).squeeze(-1), self.count(output)
        return (presence[0], count[0]) if single else (presence, count)
